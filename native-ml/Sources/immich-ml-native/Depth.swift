import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// Gallery's monocular depth for Spatial Scene, ported from open-noodle/gallery
// machine-learning/immich_ml/models/depth_estimation/depth.py (its CPU path,
// which is the one this runs; see GalleryModels.int8Depth). Answers with an
// 8-bit depth map normalised per photo, 255 nearest, plus the metrics the
// server decides Spatial Scene eligibility from.

// Every supported model is Depth Anything V2 (DINOv2 encoder): a 518 px short
// side in multiples of the 14 px patch, at most 160 patches a side.
private let patch = 14
private let shortSide = 518
private let maxPatches = 160
private let imagenetMean: [Float] = [0.485, 0.456, 0.406]
private let imagenetStd: [Float] = [0.229, 0.224, 0.225]
// A disparity step per pixel on the normalised map that counts as an occlusion edge.
private let edgeStep: Float = 0.04

func estimateDepth(rgb: [UInt8], w: Int, h: Int, session: ORTSession) throws -> [String: Any] {
    // The photo's own aspect, as the CPU provider allows.
    let scale = Double(shortSide) / Double(min(w, h))
    func snap(_ side: Int) -> Int {
        // Python's round() is half-to-even.
        min(max(Int((Double(side) * scale / Double(patch)).rounded(.toNearestOrEven)), 8), maxPatches) * patch
    }
    let inH = snap(h), inW = snap(w)
    // PIL BICUBIC, as Gallery resizes with Image.resize.
    let resized = Resize.bicubic(rgb, w: w, h: h, outW: inW, outH: inH)
    let plane = inW * inH
    var blob = [Float](repeating: 0, count: 3 * plane)
    for i in 0..<plane {
        for c in 0..<3 {
            blob[c * plane + i] = (Float(resized[i * 3 + c]) / 255 - imagenetMean[c]) / imagenetStd[c]
        }
    }
    guard let out = session.runAll(blob, shape: [1, 3, Int64(inH), Int64(inW)])?.first,
          out.shape.count >= 2 else {
        throw PredictError(status: "500 Internal Server Error", message: "depth estimation failed")
    }
    let mh = out.shape[out.shape.count - 2], mw = out.shape[out.shape.count - 1]
    let disparity = normaliseDepth(out.data)

    // The PNG rounds; the metrics' uint8 view truncates. Both as Gallery does.
    let png = disparity.map { UInt8(($0 * 255).rounded(.toNearestOrEven)) }
    guard let encoded = encodeGreyPNG(png, w: mw, h: mh) else {
        throw PredictError(status: "500 Internal Server Error", message: "depth map could not be encoded")
    }
    let m = depthMetrics(disparity, w: mw, h: mh)
    return [
        "depth": encoded.base64EncodedString(),
        "width": mw,
        "height": mh,
        "boundarySharpness": m.boundarySharpness,
        "edgeDensity": m.edgeDensity,
        "foregroundRatio": m.foregroundRatio,
        "focus": m.focus,
    ]
}

// Relative depth has an unknown scale and shift per photo, so the map is
// stretched to 0..1 between its 1st and 99th percentiles: a few hot pixels must
// not squash the rest of the range.
func normaliseDepth(_ d: [Float]) -> [Float] {
    let sorted = d.sorted()
    // numpy's default (linear) percentile.
    func percentile(_ q: Double) -> Double {
        let pos = q / 100 * Double(sorted.count - 1)
        let lo = Int(pos.rounded(.down)), hi = min(lo + 1, sorted.count - 1)
        return Double(sorted[lo]) + (Double(sorted[hi]) - Double(sorted[lo])) * (pos - Double(lo))
    }
    let low = percentile(1), high = percentile(99)
    let range = max(high - low, 1e-6)
    return d.map { Float(min(max((Double($0) - low) / range, 0), 1)) }
}

struct DepthMetrics {
    let boundarySharpness: Double  // median depth gradient along the foreground boundary
    let edgeDensity: Double        // share of pixels with an occlusion-sized step
    let foregroundRatio: Double    // share of pixels in the Otsu foreground
    let focus: Double              // mean foreground disparity, the viewer's still plane
}

func depthMetrics(_ d: [Float], w: Int, h: Int) -> DepthMetrics {
    let n = w * h
    let u8 = d.map { UInt8($0 * 255) }  // numpy astype(uint8) truncates
    let threshold = otsu(u8)
    let fg = u8.map { Int($0) > threshold }
    let fgCount = fg.reduce(0) { $0 + ($1 ? 1 : 0) }
    let ratio = Double(fgCount) / Double(n)

    // cv2.Sobel ksize 3 on CV_32F with BORDER_REFLECT_101, each scaled by 1/8.
    func at(_ x: Int, _ y: Int) -> Float {
        let rx = x < 0 ? -x : (x >= w ? 2 * w - 2 - x : x)
        let ry = y < 0 ? -y : (y >= h ? 2 * h - 2 - y : y)
        return d[ry * w + rx]
    }
    var gradient = [Float](repeating: 0, count: n)
    for y in 0..<h {
        for x in 0..<w {
            let gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)
                - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1)) / 8
            let gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)
                - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1)) / 8
            gradient[y * w + x] = (gx * gx + gy * gy).squareRoot()
        }
    }

    // MORPH_GRADIENT with a 3x3 square: a pixel whose in-bounds neighbourhood
    // holds both foreground and background. cv2's default border value never
    // wins a dilate or an erode, so out-of-bounds neighbours do not count.
    var boundary: [Float] = []
    for y in 0..<h {
        for x in 0..<w {
            var anyFG = false, anyBG = false
            for yy in max(y - 1, 0)...min(y + 1, h - 1) {
                for xx in max(x - 1, 0)...min(x + 1, w - 1) {
                    if fg[yy * w + xx] { anyFG = true } else { anyBG = true }
                }
            }
            if anyFG && anyBG { boundary.append(gradient[y * w + x]) }
        }
    }

    var fgSum = 0.0, allSum = 0.0
    for i in 0..<n {
        allSum += Double(d[i])
        if fg[i] { fgSum += Double(d[i]) }
    }
    return DepthMetrics(
        boundarySharpness: boundary.isEmpty ? 0 : median(boundary),
        edgeDensity: Double(gradient.reduce(0) { $0 + ($1 > edgeStep ? 1 : 0) }) / Double(n),
        foregroundRatio: ratio,
        focus: ratio > 0 && ratio < 1 ? fgSum / Double(fgCount) : allSum / Double(n)
    )
}

// cv2's THRESH_OTSU on 8-bit data: the level maximising between-class variance.
private func otsu(_ px: [UInt8]) -> Int {
    var hist = [Double](repeating: 0, count: 256)
    for p in px { hist[Int(p)] += 1 }
    let scale = 1 / Double(px.count)
    var mu = 0.0
    for i in 0..<256 { mu += Double(i) * hist[i] * scale }
    var mu1 = 0.0, q1 = 0.0, maxSigma = 0.0, maxVal = 0
    let eps = Double(Float.ulpOfOne)
    for i in 0..<256 {
        let p = hist[i] * scale
        mu1 *= q1
        q1 += p
        let q2 = 1 - q1
        if min(q1, q2) < eps || max(q1, q2) > 1 - eps { continue }
        mu1 = (mu1 + Double(i) * p) / q1
        let mu2 = (mu - q1 * mu1) / q2
        let sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)
        if sigma > maxSigma { maxSigma = sigma; maxVal = i }
    }
    return maxVal
}

// numpy's median: the mean of the two middle values when the count is even.
private func median(_ v: [Float]) -> Double {
    let s = v.sorted()
    let mid = s.count / 2
    return s.count % 2 == 1 ? Double(s[mid]) : (Double(s[mid - 1]) + Double(s[mid])) / 2
}

private func encodeGreyPNG(_ px: [UInt8], w: Int, h: Int) -> Data? {
    guard let provider = CGDataProvider(data: Data(px) as CFData),
          let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w,
                              space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                              provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    else { return nil }
    let out = NSMutableData()
    guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else {
        return nil
    }
    CGImageDestinationAddImage(dest, image, nil)
    return CGImageDestinationFinalize(dest) ? out as Data : nil
}
