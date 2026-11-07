import Foundation

// Gallery's pet detection (RF-DETR) and pet recognition, ported from
// open-noodle/gallery machine-learning/immich_ml/models/pet_detection/detection.py
// and pet_recognition/recognition.py. Same models, same preprocessing, same
// post-processing, so the boxes and embeddings land where Gallery's own
// container puts them and its pet clusters keep working.

// RF-DETR's 91-class COCO ids for the animals a household library contains.
// The non-pets stay in the scoring subspace only as distractors: a horse must be
// able to win its own argmax so it is discarded, rather than being forced onto
// its best cat/dog score and filed as a dog.
private let animalIDs = [16, 17, 18, 19, 20, 21]  // bird, cat, dog, horse, sheep, cow
private let petLabels = [17: "cat", 18: "dog"]

// RF-DETR inherits DINOv2's ImageNet normalisation, and the recognizer was
// trained with the same.
private let imagenetMean: [Float] = [0.485, 0.456, 0.406]
private let imagenetStd: [Float] = [0.229, 0.224, 0.225]

private let recognizerInput = 224
// A crop thinner than this after clamping would be a degenerate resize the
// recognizer never saw in training. Gallery leaves its embedding out, and the
// server files such a pet under its species instead.
private let minCropSide = 2

struct DetectedPet {
    let x1: Int, y1: Int, x2: Int, y2: Int
    let score: Float
    let label: String
}

func detectPets(rgb: [UInt8], w: Int, h: Int, session: ORTSession, minScore: Float) throws -> [DetectedPet] {
    // nano is 384 and small is 512: read it from the model, never assume.
    let dims = session.inputDims(0)
    let size = dims.count == 4 && dims[2] > 0 ? dims[2] : 384
    let resized = resampleRGB(rgb, w: w, region: (0, 0, w, h), outW: size, outH: size, area: false)
    let blob = normalizedCHW(resized, w: size, h: size)
    guard let outs = session.runAll(blob, shape: [1, 3, Int64(size), Int64(size)]), outs.count == 2 else {
        throw PredictError(status: "500 Internal Server Error", message: "pet detection failed")
    }
    // Boxes vs logits by trailing dimension, not export order.
    let (boxes, logits) = outs[0].shape.last == 4 ? (outs[0], outs[1]) : (outs[1], outs[0])
    let queries = boxes.shape.count == 3 ? boxes.shape[1] : 0
    let classes = logits.shape.last ?? 0
    guard queries > 0, classes > animalIDs.max()!, logits.data.count == queries * classes else {
        throw PredictError(status: "500 Internal Server Error", message: "pet detection: unexpected output shape")
    }

    var pets: [DetectedPet] = []
    for q in 0..<queries {
        // Best class within the animal subspace, on sigmoid scores (RF-DETR
        // emits pre-sigmoid logits).
        var bestID = animalIDs[0]
        var best = -Float.infinity
        for id in animalIDs {
            let s = 1 / (1 + exp(-logits.data[q * classes + id]))
            if s > best { best = s; bestID = id }
        }
        guard best >= minScore, let label = petLabels[bestID] else { continue }
        let b = boxes.data[(q * 4)..<(q * 4 + 4)].map { Double($0) }  // cx, cy, w, h, normalised
        let (cx, cy, bw, bh) = (b[0], b[1], b[2], b[3])
        let W = Double(w), H = Double(h)
        // Python's round() is half-to-even; so is .toNearestOrEven.
        func px(_ v: Double, _ limit: Double) -> Int { Int(min(max(v * limit, 0), limit).rounded(.toNearestOrEven)) }
        let pet = DetectedPet(x1: px(cx - bw / 2, W), y1: px(cy - bh / 2, H),
                              x2: px(cx + bw / 2, W), y2: px(cy + bh / 2, H), score: best, label: label)
        // A box clipped away to nothing is not a detection. No NMS: RF-DETR's
        // decoder already deduplicates its queries.
        guard pet.x2 > pet.x1, pet.y2 > pet.y1 else { continue }
        pets.append(pet)
    }
    return pets
}

// One embedding per pet, nil where the crop is too small to embed.
func embedPets(rgb: [UInt8], w: Int, h: Int, pets: [DetectedPet], session: ORTSession) throws -> [[Float]?] {
    try pets.map { pet in
        var x1 = min(max(pet.x1, 0), w), y1 = min(max(pet.y1, 0), h)
        var x2 = min(max(pet.x2, 0), w), y2 = min(max(pet.y2, 0), h)
        if x2 <= x1 { x2 = min(x1 + 1, w); x1 = max(x2 - 1, 0) }
        if y2 <= y1 { y2 = min(y1 + 1, h); y1 = max(y2 - 1, 0) }
        let cw = x2 - x1, ch = y2 - y1
        guard cw >= minCropSide, ch >= minCropSide else { return nil }
        // Gallery picks INTER_AREA on downscale to match the antialiased PIL
        // resize the recognizer was trained with, INTER_LINEAR on upscale.
        let area = cw * ch > recognizerInput * recognizerInput
        let crop = resampleRGB(rgb, w: w, region: (x1, y1, cw, ch),
                               outW: recognizerInput, outH: recognizerInput, area: area)
        let blob = normalizedCHW(crop, w: recognizerInput, h: recognizerInput)
        let n = Int64(recognizerInput)
        guard let e = session.run(blob, shape: [1, 3, n, n]) else {
            throw PredictError(status: "500 Internal Server Error", message: "pet recognition failed")
        }
        return e
    }
}

// RGB uint8 HWC -> ImageNet-normalised float CHW.
private func normalizedCHW(_ rgb: [UInt8], w: Int, h: Int) -> [Float] {
    let plane = w * h
    var out = [Float](repeating: 0, count: 3 * plane)
    for i in 0..<plane {
        for c in 0..<3 {
            out[c * plane + i] = (Float(rgb[i * 3 + c]) / 255 - imagenetMean[c]) / imagenetStd[c]
        }
    }
    return out
}

// cv2.resize on an RGB uint8 region of a row-major image whose width is `w`.
// area: false is INTER_LINEAR (half-pixel centres, no antialiasing); true is
// INTER_AREA, which averages the source pixels each output pixel covers on an
// axis that shrinks. On an axis that grows it falls back to linear, which is
// where cv2's own INTER_AREA differs slightly; Gallery only asks for it on a
// crop larger than the output, so that is only a very thin crop's short side.
// Results can differ from cv2 by one level, since cv2 resizes uint8 in fixed point.
func resampleRGB(_ rgb: [UInt8], w: Int, region: (x: Int, y: Int, w: Int, h: Int),
                 outW: Int, outH: Int, area: Bool) -> [UInt8] {
    let xt = taps(region.w, outW, area: area)
    let yt = taps(region.h, outH, area: area)
    // Horizontal pass over just the rows the region covers.
    var tmp = [Float](repeating: 0, count: outW * region.h * 3)
    for y in 0..<region.h {
        let row = (region.y + y) * w
        for ox in 0..<outW {
            var acc: (Float, Float, Float) = (0, 0, 0)
            for (sx, wt) in xt[ox] {
                let s = (row + region.x + sx) * 3
                acc.0 += Float(rgb[s]) * wt; acc.1 += Float(rgb[s + 1]) * wt; acc.2 += Float(rgb[s + 2]) * wt
            }
            let d = (y * outW + ox) * 3
            tmp[d] = acc.0; tmp[d + 1] = acc.1; tmp[d + 2] = acc.2
        }
    }
    var out = [UInt8](repeating: 0, count: outW * outH * 3)
    for oy in 0..<outH {
        for i in 0..<(outW * 3) {
            var acc: Float = 0
            for (sy, wt) in yt[oy] { acc += tmp[sy * outW * 3 + i] * wt }
            out[oy * outW * 3 + i] = UInt8(min(max(acc.rounded(), 0), 255))
        }
    }
    return out
}

// Source indices and weights for each output index on one axis.
private func taps(_ inSize: Int, _ outSize: Int, area: Bool) -> [[(Int, Float)]] {
    let scale = Double(inSize) / Double(outSize)
    if area && scale >= 1 {
        return (0..<outSize).map { i in
            let lo = Double(i) * scale, hi = lo + scale
            var t: [(Int, Float)] = []
            var s = Int(lo)
            while s < inSize && Double(s) < hi {
                let overlap = min(Double(s + 1), hi) - max(Double(s), lo)
                if overlap > 1e-9 { t.append((s, Float(overlap / scale))) }
                s += 1
            }
            return t
        }
    }
    return (0..<outSize).map { i in
        var fx = (Double(i) + 0.5) * scale - 0.5
        var sx = Int(fx.rounded(.down))
        fx -= Double(sx)
        if sx < 0 { fx = 0; sx = 0 }
        if sx >= inSize - 1 { fx = 0; sx = inSize - 1 }
        return fx == 0 ? [(sx, Float(1))] : [(sx, Float(1 - fx)), (sx + 1, Float(fx))]
    }
}
