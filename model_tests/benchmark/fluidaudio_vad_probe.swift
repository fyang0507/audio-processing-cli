import CoreML
import FluidAudio
import Foundation

@main
struct FluidAudioVADProbe {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard
            let audioIndex = arguments.firstIndex(of: "--audio"), audioIndex + 1 < arguments.count,
            let outputIndex = arguments.firstIndex(of: "--output"), outputIndex + 1 < arguments.count,
            let unitsIndex = arguments.firstIndex(of: "--compute-units"), unitsIndex + 1 < arguments.count
        else { throw ProbeError.usage }
        let computeUnits: MLComputeUnits = switch arguments[unitsIndex + 1] {
        case "all", "ane": .cpuAndNeuralEngine
        case "cpu-only": .cpuOnly
        default: throw ProbeError.usage
        }
        let audioURL = URL(fileURLWithPath: arguments[audioIndex + 1])
        let outputURL = URL(fileURLWithPath: arguments[outputIndex + 1])
        let samples = try AudioConverter().resampleAudioFile(audioURL)
        let initializationStart = Date()
        let manager = try await VadManager(config: VadConfig(defaultThreshold: 0.5, computeUnits: computeUnits))
        let initializationSeconds = Date().timeIntervalSince(initializationStart)
        let inferenceStart = Date()
        let results = try await manager.process(samples)
        let inferenceSeconds = Date().timeIntervalSince(inferenceStart)
        let frames: [[String: Any]] = results.enumerated().map { index, result in
            ["start_sample": index * VadManager.chunkSize, "sample_count": min(VadManager.chunkSize, samples.count - index * VadManager.chunkSize), "probability": result.probability, "processing_s": result.processingTime]
        }
        let payload: [String: Any] = [
            "schema_version": 1,
            "input": ["sample_count": samples.count, "sample_rate_hz": VadManager.sampleRate],
            "timing": ["model_initialization_s": initializationSeconds, "inference_s": inferenceSeconds],
            "output": ["frame_samples": VadManager.chunkSize, "frames": frames],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outputURL, options: .atomic)
    }

    enum ProbeError: Error { case usage }
}
