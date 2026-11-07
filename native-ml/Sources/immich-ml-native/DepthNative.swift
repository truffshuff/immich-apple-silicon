import Foundation
import MLX
import MLXNN

// Depth Anything V2 (DINOv2 backbone + DPT head) on the GPU through MLX,
// ported from transformers' modeling_dinov2.py and modeling_depth_anything.py.
// Gallery's distill-any-depth-large ONNX is an unmodified export of the same
// checkpoint, and onnxruntime runs it in about 13 s a photo on an M2's CPU
// (CoreML is slower still: 105 s, it never reaches the Neural Engine). This
// runs it in about 1 s, in fp32, and matches the ONNX output: correlation
// 1.000000 and at most one level apart on the 8-bit map across the parity set,
// with every Spatial Scene decision the same. bf16 drifted boundarySharpness
// by up to 0.02 against a 0.03 threshold, so the weights stay fp32.
final class DepthNative: DepthEngine {
    private static let patch = 14
    private static let eps: Float = 1e-6  // DINOv2's layer_norm_eps
    // Every Depth Anything V2 size reassembles its four taps at 4x, 2x, 1x and
    // 0.5x the patch grid: transposed convs, identity, then a stride-2 conv.
    private static let transposedResize = [0, 1]
    private static let stridedResize = 3

    private let W: [String: MLXArray]
    private let dim: Int
    private let heads: Int
    private let layers: Int
    private let taps: Set<Int>
    // CPU copy of the learned 37x37 position table, resampled once per grid.
    private let posTable: [Float]
    private var posCache: [String: MLXArray] = [:]

    init(weightsPath: String) throws {
        var raw = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
        guard let cls = raw["backbone.embeddings.cls_token"], let pos = raw["backbone.embeddings.position_embeddings"]
        else { throw PredictError(status: "500 Internal Server Error", message: "not a Depth Anything checkpoint") }
        dim = cls.dim(-1)
        heads = dim / 64  // DINOv2 heads are 64 wide at every size
        layers = raw.keys.filter { $0.hasPrefix("backbone.encoder.layer.") && $0.hasSuffix(".norm1.weight") }.count
        // out_indices from the released configs: Large taps 5/12/18/24, Small 3/6/9/12.
        taps = layers == 24 ? [5, 12, 18, 24] : [3, 6, 9, 12]
        posTable = pos.asType(.float32).asArray(Float.self)
        for (k, v) in raw where v.ndim == 4 {
            if k == "backbone.embeddings.patch_embeddings.projection.weight" {
                // Linear over patches flattened in (channel, row, col) order.
                raw[k] = v.reshaped([dim, -1]).transposed()
            } else if !Self.transposedResize.contains(where: { k == "neck.reassemble_stage.layers.\($0).resize.weight" }) {
                raw[k] = v.transposed(0, 2, 3, 1)  // OIHW -> MLX's OHWI
            }
        }
        W = raw.mapValues { $0.asType(.float32) }
        eval(Array(W.values))
    }

    private func w(_ k: String) -> MLXArray {
        guard let v = W[k] else { fatalError("DepthNative: missing weight \(k)") }
        return v
    }

    // CHW float blob, already resized and normalised, to [h, w] disparity.
    func depth(_ blob: [Float], h: Int, w width: Int) -> (data: [Float], h: Int, w: Int)? {
        let p = Self.patch, gh = h / p, gw = width / p
        return withMetalLock {
            Device.withDefaultDevice(.gpu) {
                let pixels = MLXArray(blob, [3, gh, p, gw, p]).transposed(1, 3, 0, 2, 4).reshaped([gh * gw, 3 * p * p])
                var x = matmul(pixels, w("backbone.embeddings.patch_embeddings.projection.weight"))
                    + w("backbone.embeddings.patch_embeddings.projection.bias")
                x = concatenated([w("backbone.embeddings.cls_token").reshaped([1, dim]), x]) + positions(gh, gw)
                x = x.expandedDimensions(axis: 0)

                var feats: [MLXArray] = []
                for l in 0 ..< layers {
                    x = block(x, "backbone.encoder.layer.\(l)")
                    if taps.contains(l + 1) {
                        feats.append(ln(x, "backbone.layernorm")[0, 1...].reshaped([gh, gw, dim]))
                    }
                }
                let out = head(neck(feats, gh, gw), gh * p, gw * p)
                eval(out)
                let result = (out.asArray(Float.self), gh * p, gw * p)
                // The head's full-resolution 3x3 convs take ~2 GB of scratch,
                // sized by the photo's aspect, and MLX keeps freed buffers for
                // reuse up to nearly all of memory. Across a library's mix of
                // aspects that cache reached 11 GB on a 16 GB Mac and pushed
                // it into swap, so hand the scratch back after every map.
                MLX.Memory.clearCache()
                return result
            }
        }
    }

    private func ln(_ x: MLXArray, _ p: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: w("\(p).weight"), bias: w("\(p).bias"), eps: Self.eps)
    }
    private func lin(_ x: MLXArray, _ p: String) -> MLXArray {
        matmul(x, w("\(p).weight").transposed()) + w("\(p).bias")
    }
    // x is [h, w, c]; MLX convolves NHWC. A 3x3 conv over a large map unfolds
    // its whole input nine times over (1.9 GB for the head's full-resolution
    // conv), so above bandBytes it runs in bands of rows, each evaluated before
    // the next: MLX's peak drops from 3.75 to 2.76 GB, and the output moves by
    // float reassociation only (1e-4 on a disparity range of 108).
    private static let bandBytes = 256 << 20
    private func conv(_ x: MLXArray, _ p: String, stride: Int = 1, bias: Bool = true) -> MLXArray {
        let k = w("\(p).weight")
        let pad = k.dim(1) / 2
        let (H, Wd, C) = (x.dim(0), x.dim(1), x.dim(2))
        let unfolded = Wd * C * k.dim(1) * k.dim(2) * 4  // bytes per output row
        var y: MLXArray
        if stride == 1, pad > 0, H * unfolded > Self.bandBytes {
            let xp = padded(x, widths: [.init((pad, pad)), .init((pad, pad)), 0])
            let rows = max(1, Self.bandBytes / unfolded)
            var parts: [MLXArray] = []
            for r in Swift.stride(from: 0, to: H, by: rows) {
                let part = conv2d(xp[r ..< min(r + rows, H) + 2 * pad].expandedDimensions(axis: 0), k)[0]
                eval(part)
                parts.append(part)
            }
            y = concatenated(parts, axis: 0)
        } else {
            y = conv2d(x.expandedDimensions(axis: 0), k, stride: .init(stride), padding: .init(pad))[0]
        }
        return bias ? y + w("\(p).bias") : y
    }

    private func block(_ x: MLXArray, _ p: String) -> MLXArray {
        let seq = x.dim(1)
        let a = "\(p).attention.attention"
        let h = ln(x, "\(p).norm1")
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([1, seq, heads, dim / heads]).transposed(0, 2, 1, 3) }
        let o = scaledDotProductAttention(
            queries: split(lin(h, "\(a).query")), keys: split(lin(h, "\(a).key")), values: split(lin(h, "\(a).value")),
            scale: 1 / Float(dim / heads).squareRoot(), mask: nil
        ).transposed(0, 2, 1, 3).reshaped([1, seq, dim])
        var r = x + lin(o, "\(p).attention.output.dense") * w("\(p).layer_scale1.lambda1")
        let m = lin(gelu(lin(ln(r, "\(p).norm2"), "\(p).mlp.fc1")), "\(p).mlp.fc2")
        r = r + m * w("\(p).layer_scale2.lambda1")
        return r
    }

    private func neck(_ feats: [MLXArray], _ gh: Int, _ gw: Int) -> MLXArray {
        // Reassemble each tap to its own resolution, then to the fusion width.
        var maps: [MLXArray] = []
        for (i, f) in feats.enumerated() {
            let p = "neck.reassemble_stage.layers.\(i)"
            var x = conv(f, "\(p).projection")
            if Self.transposedResize.contains(i) {
                // ConvTranspose2d with kernel == stride: every input pixel
                // writes its own k x k output block, so it is one matmul.
                let k = w("\(p).resize.weight")  // [in, out, k, k], PyTorch layout
                let (cin, cout, s) = (k.dim(0), k.dim(1), k.dim(2))
                x = matmul(x.reshaped([gh * gw, cin]), k.reshaped([cin, cout * s * s]))
                    .reshaped([gh, gw, cout, s, s]).transposed(0, 3, 1, 4, 2).reshaped([gh * s, gw * s, cout])
                    + w("\(p).resize.bias")
            } else if i == Self.stridedResize {
                x = conv(x, "\(p).resize", stride: 2)
            }
            maps.append(conv(x, "neck.convs.\(i)", bias: false))
        }
        // Fuse, deepest first, each stage upsampled to the next one's size.
        func residual(_ x: MLXArray, _ p: String) -> MLXArray {
            x + conv(relu(conv(relu(x), "\(p).convolution1")), "\(p).convolution2")
        }
        let deep = Array(maps.reversed())
        var fused = deep[0]
        for i in 0 ..< deep.count {
            let p = "neck.fusion_stage.layers.\(i)"
            var h = fused
            if i > 0 {
                var r = deep[i]
                if r.shape != fused.shape { r = resize(r, fused.dim(0), fused.dim(1), alignCorners: false) }
                h = fused + residual(r, "\(p).residual_layer1")
            }
            h = residual(h, "\(p).residual_layer2")
            let (th, tw) = i + 1 < deep.count ? (deep[i + 1].dim(0), deep[i + 1].dim(1)) : (h.dim(0) * 2, h.dim(1) * 2)
            fused = conv(resize(h, th, tw, alignCorners: true), "\(p).projection")
        }
        return fused
    }

    private func head(_ x: MLXArray, _ h: Int, _ width: Int) -> MLXArray {
        var d = resize(conv(x, "head.conv1"), h, width, alignCorners: true)
        d = relu(conv(relu(conv(d, "head.conv2")), "head.conv3"))
        return d.reshaped([h, width])
    }

    // Bilinear resize of [h, w, c] as two matmuls, with torch's sampling.
    private func resize(_ x: MLXArray, _ h: Int, _ width: Int, alignCorners: Bool) -> MLXArray {
        let (H, Wd, C) = (x.dim(0), x.dim(1), x.dim(2))
        func weights(_ n: Int, _ m: Int) -> MLXArray {
            var out = [Float](repeating: 0, count: m * n)
            for o in 0 ..< m {
                let src = alignCorners
                    ? (m > 1 ? Double(o) * Double(n - 1) / Double(m - 1) : 0)
                    : max((Double(o) + 0.5) * Double(n) / Double(m) - 0.5, 0)
                let i0 = min(Int(src), n - 1), i1 = min(i0 + 1, n - 1), t = Float(src - Double(i0))
                out[o * n + i0] += 1 - t
                out[o * n + i1] += t
            }
            return MLXArray(out, [m, n])
        }
        let rows = matmul(weights(H, h), x.reshaped([H, Wd * C])).reshaped([h, Wd, C])
        return matmul(weights(Wd, width), rows.transposed(1, 0, 2).reshaped([Wd, h * C]))
            .reshaped([width, h, C]).transposed(1, 0, 2)
    }

    // The position table resampled to a gh x gw grid with torch's bicubic
    // (align_corners=false, A=-0.75, edge-clamped taps), as Dinov2Embeddings'
    // interpolate_pos_encoding does. Only called under the Metal lock.
    private func positions(_ gh: Int, _ gw: Int) -> MLXArray {
        let key = "\(gh)x\(gw)"
        if let p = posCache[key] { return p }
        let n = Int(Double(posTable.count / dim - 1).squareRoot().rounded())
        func cubic(_ m: Int) -> [[(Int, Double)]] {
            let a = -0.75
            func k(_ x: Double) -> Double {
                let x = abs(x)
                if x <= 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
                if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
                return 0
            }
            return (0 ..< m).map { o in
                let src = (Double(o) + 0.5) * Double(n) / Double(m) - 0.5
                let i0 = Int(src.rounded(.down)), t = src - Double(i0)
                return (-1 ... 2).map { j in (min(max(i0 + j, 0), n - 1), k(Double(j) - t)) }
            }
        }
        let ry = cubic(gh), rx = cubic(gw)
        var out = [Float](repeating: 0, count: (1 + gh * gw) * dim)
        for c in 0 ..< dim { out[c] = posTable[c] }  // the class token's own entry
        var acc = [Double](repeating: 0, count: dim)
        for y in 0 ..< gh {
            for x in 0 ..< gw {
                for c in 0 ..< dim { acc[c] = 0 }
                for (sy, wy) in ry[y] {
                    for (sx, wx) in rx[x] {
                        let base = (1 + sy * n + sx) * dim, wt = wy * wx
                        for c in 0 ..< dim { acc[c] += Double(posTable[base + c]) * wt }
                    }
                }
                let o = (1 + y * gw + x) * dim
                for c in 0 ..< dim { out[o + c] = Float(acc[c]) }
            }
        }
        let p = MLXArray(out, [1 + gh * gw, dim])
        posCache[key] = p
        return p
    }
}
