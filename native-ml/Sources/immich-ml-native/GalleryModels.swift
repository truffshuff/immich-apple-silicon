import Foundation

// Models for the tasks Noodle Gallery (open-noodle/gallery, an Immich fork)
// adds to the ML contract: pet detection, pet recognition and depth
// estimation. They are plain ONNX exports in Gallery's own Hugging Face org,
// so they run through onnxruntime like the CLIP zoo, fetched on first use.
//
// The name sets mirror immich_ml/models/constants.py in Gallery. A model name
// arrives from the network and becomes a cache path and a download URL, so
// only these exact names are accepted.
enum GalleryModels {
    static let org = "open-noodle"
    static let dir = NATIVE_CACHE_DIR.appendingPathComponent("gallery")

    static let petDetectors: Set = ["rfdetr-nano", "rfdetr-small"]
    static let petRecognizers: Set = ["pet-recognition-small", "pet-recognition-base", "pet-recognition-large"]
    static let depthModels: Set = ["depth-anything-v2-small", "distill-any-depth-small", "distill-any-depth-large"]

    // Gallery runs the small depth models from an int8 copy on the CPU and the
    // large one in fp32 (depth.py's INT8_MODELS: int8 cost the large model 15%
    // of its Spatial Scene decisions). Depth stays on the CPU here because
    // CoreML gains nothing on it: twelve IsNaN nodes split the graph into
    // fourteen partitions, and even with them rewritten the Neural Engine ran
    // 518x686 in 698 ms against int8's 665 ms on an M2. So this is exactly the
    // path Gallery's own container takes on a CPU-only server.
    static let int8Depth: Set = ["depth-anything-v2-small", "distill-any-depth-small"]

    // Depth models that run on the GPU instead (DepthNative), with the Hugging
    // Face repo of the checkpoint Gallery exported its ONNX from unmodified.
    // fp32 large on the CPU took ~13 s a photo; this takes ~1 s.
    static let nativeDepth = ["distill-any-depth-large": "xingyang1/Distill-Any-Depth-Large-hf"]

    enum Kind {
        case petDetector, petRecognizer, depth

        var known: Set<String> {
            switch self {
            case .petDetector: return GalleryModels.petDetectors
            case .petRecognizer: return GalleryModels.petRecognizers
            case .depth: return GalleryModels.depthModels
            }
        }

        func file(for name: String) -> String {
            switch self {
            case .petDetector: return "detection/model.onnx"
            case .petRecognizer: return "recognition/model.onnx"
            case .depth: return GalleryModels.int8Depth.contains(name) ? "depth/model_int8.onnx" : "depth/model.onnx"
            }
        }

        // RF-DETR (static 512 or 384 square) and the recognizer (batch pinned
        // to 1, see ort_load_coreml) compile for CoreML: measured on an M2,
        // detection 213 -> 97 ms and recognition 53 -> 19 ms per crop.
        var coreML: Bool { self != .depth }
    }

    // Fetches the model if needed and opens a session on it. Blocking, and the
    // first fetch is 100-350 MB, so callers must not hold a lock across it.
    static func load(_ kind: Kind, name: String) throws -> ORTSession {
        guard kind.known.contains(name) else {
            throw PredictError(status: "422 Unprocessable Entity", message: "Unknown model: \(name)")
        }
        let modelDir = dir.appendingPathComponent(name)
        let file = kind.file(for: name)
        try ZooCLIP.ensureFiles(name: name, dir: modelDir, org: org, only: [file])
        let path = modelDir.appendingPathComponent(file).path
        let cache = kind.coreML
            ? Models.coreMLCache(for: path, in: modelDir.appendingPathComponent("coreml")) : nil
        guard let session = ORTSession(modelPath: path, coreMLCache: cache) else {
            throw PredictError(status: "500 Internal Server Error", message: "could not load \(name)")
        }
        if let cache, !session.usesCoreML {
            // Same as ArcFace: a cache that did not load is cleared so the next
            // load compiles afresh instead of failing on it forever.
            try? FileManager.default.removeItem(at: cache)
        }
        print("[native-ml] \(name) on \(session.usesCoreML ? "CoreML" : "CPU")")
        return session
    }

    // A depth model on the GPU where there is a native port, falling back to
    // the ONNX export on the CPU if its weights cannot be fetched or loaded.
    static func loadDepth(name: String) throws -> DepthEngine {
        if let repo = nativeDepth[name] {
            do {
                let path = try SigLIPNative.ensureWeights(hfRepo: repo, name: name, root: dir)
                let model = try DepthNative(weightsPath: path)
                print("[native-ml] \(name) on GPU (MLX)")
                return model
            } catch {
                print("[native-ml] \(name): MLX load failed (\(error)), using ONNX on the CPU")
            }
        }
        return try load(.depth, name: name)
    }
}
