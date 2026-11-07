import Foundation
import COnnxShim

// Swift wrapper over the onnxruntime C shim. Runs the identical .onnx the Python
// service uses (same engine) so face embeddings match by construction.
final class ORTSession {
    private let handle: UnsafeMutableRawPointer
    let outDim: Int

    // Whether CoreML is running this session; false on the CPU, including
    // when CoreML was asked for and could not build it.
    let usesCoreML: Bool

    // coreMLCache: run through CoreML (see ort_load_coreml), keeping the
    // compiled model there. nil stays on the CPU.
    init?(modelPath: String, outDim: Int = 512, coreMLCache: URL? = nil) {
        let loaded: UnsafeMutableRawPointer?
        var coreml: Int32 = 0
        if let cache = coreMLCache {
            try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
            loaded = ort_load_coreml(modelPath, cache.path, &coreml)
        } else {
            loaded = ort_load(modelPath)
        }
        guard let h = loaded else { return nil }
        handle = h
        usesCoreML = coreml != 0
        self.outDim = outDim
    }

    // Typed input for multi-input models (the CLIP zoo towers).
    enum Tensor {
        case float([Float], shape: [Int64])
        case int32([Int32], shape: [Int64])
        case int64([Int64], shape: [Int64])
    }

    // Element type (0=float32, 1=int32, 2=int64) of input `i`, per the model.
    func inputElemType(_ i: Int) -> Int { Int(ort_input_elem_type(handle, Int32(i))) }

    // Declared dims of input `i`, a free dimension as -1.
    func inputDims(_ i: Int) -> [Int] {
        var dims = [Int64](repeating: 0, count: 8)
        let n = ort_input_dims(handle, Int32(i), &dims, 8)
        return n > 0 ? dims.prefix(Int(n)).map { Int($0) } : []
    }

    // Number of inputs the model declares (multilingual textual towers take 2).
    func inputCount() -> Int { Int(ort_input_count(handle)) }

    // Run a model with typed inputs bound in declaration order; returns the
    // first output as floats (embedding models have a single output).
    func runMulti(_ inputs: [Tensor], outDim: Int) -> [Float]? {
        var out = [Float](repeating: 0, count: outDim)
        var types: [Int32] = []
        var ndims: [Int32] = []
        var shapes: [Int64] = []
        // Keep buffers alive across the C call.
        var holds: [Any] = []
        var ptrs: [UnsafeRawPointer?] = []
        for input in inputs {
            switch input {
            case .float(let v, let s):
                let buf = UnsafeMutableBufferPointer<Float>.allocate(capacity: v.count)
                _ = buf.initialize(from: v)
                holds.append(buf); ptrs.append(UnsafeRawPointer(buf.baseAddress))
                types.append(0); ndims.append(Int32(s.count)); shapes.append(contentsOf: s)
            case .int32(let v, let s):
                let buf = UnsafeMutableBufferPointer<Int32>.allocate(capacity: v.count)
                _ = buf.initialize(from: v)
                holds.append(buf); ptrs.append(UnsafeRawPointer(buf.baseAddress))
                types.append(1); ndims.append(Int32(s.count)); shapes.append(contentsOf: s)
            case .int64(let v, let s):
                let buf = UnsafeMutableBufferPointer<Int64>.allocate(capacity: v.count)
                _ = buf.initialize(from: v)
                holds.append(buf); ptrs.append(UnsafeRawPointer(buf.baseAddress))
                types.append(2); ndims.append(Int32(s.count)); shapes.append(contentsOf: s)
            }
        }
        defer {
            for h in holds {
                if let b = h as? UnsafeMutableBufferPointer<Float> { b.deallocate() }
                if let b = h as? UnsafeMutableBufferPointer<Int32> { b.deallocate() }
                if let b = h as? UnsafeMutableBufferPointer<Int64> { b.deallocate() }
            }
        }
        let n = ptrs.withUnsafeMutableBufferPointer { pp in
            types.withUnsafeBufferPointer { tp in
                shapes.withUnsafeBufferPointer { sp in
                    ndims.withUnsafeBufferPointer { np in
                        out.withUnsafeMutableBufferPointer { op in
                            ort_run_multi(handle, Int32(inputs.count), pp.baseAddress,
                                          tp.baseAddress, sp.baseAddress, np.baseAddress,
                                          op.baseAddress, Int32(outDim))
                        }
                    }
                }
            }
        }
        return n > 0 ? Array(out.prefix(Int(n))) : nil
    }

    // Run one tensor (row-major float) through the model; returns the output row.
    func run(_ input: [Float], shape: [Int64]) -> [Float]? {
        var out = [Float](repeating: 0, count: outDim)
        var sh = shape
        let n = input.withUnsafeBufferPointer { inp in
            sh.withUnsafeMutableBufferPointer { sp in
                out.withUnsafeMutableBufferPointer { op in
                    ort_run(handle, inp.baseAddress, sp.baseAddress, Int32(shape.count),
                            op.baseAddress, Int32(outDim))
                }
            }
        }
        return n > 0 ? Array(out.prefix(Int(n))) : nil
    }

    // Run one tensor and return every output with its shape, in the model's
    // declared output order.
    func runAll(_ input: [Float], shape: [Int64]) -> [(data: [Float], shape: [Int])]? {
        var datas = [UnsafeMutablePointer<Float>?](repeating: nil, count: 8)
        var shapes = [Int64](repeating: 0, count: 8 * 8)
        var ndims = [Int32](repeating: 0, count: 8)
        let n = input.withUnsafeBufferPointer { inp in
            ort_run_all(handle, inp.baseAddress, shape, Int32(shape.count), 8, &datas, &shapes, &ndims)
        }
        guard n > 0 else { return nil }
        return (0..<Int(n)).map { i in
            let dims = (0..<Int(ndims[i])).map { Int(shapes[i * 8 + $0]) }
            let count = dims.reduce(1, *)
            defer { free(datas[i]) }
            return (Array(UnsafeBufferPointer(start: datas[i], count: count)), dims)
        }
    }

    deinit { ort_free(handle) }
}
