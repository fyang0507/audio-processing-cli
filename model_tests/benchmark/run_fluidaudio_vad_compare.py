#!/usr/bin/env python3
"""Compare pinned Silero ONNX VAD with FluidAudio's Core ML Silero VAD.

The runner feeds the same prepared 16 kHz mono PCM WAV to both implementations.
It generates a temporary Swift package pointing at a caller-supplied FluidAudio
checkout, runs the tracked probe, and applies the product's threshold,
hysteresis, padding, and merge policy to each backend's probability sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from audio_cli.vad import MODEL_SHA256, MODEL_URL, MODEL_VERSION, SileroOnnxVad


PROBE_SOURCE = Path(__file__).with_name("fluidaudio_vad_probe.swift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fluid-source", required=True, type=Path)
    parser.add_argument("--fluid-version", required=True)
    parser.add_argument("--fluid-commit", required=True)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-log", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--exit-threshold", type=float, default=0.35)
    parser.add_argument("--min-speech-ms", type=int, default=100)
    parser.add_argument("--min-silence-ms", type=int, default=300)
    parser.add_argument("--speech-pad-ms", type=int, default=120)
    parser.add_argument("--score-frame-ms", type=float, default=10.0)
    parser.add_argument(
        "--compute-units", choices=("all", "cpu-only", "ane"), default="all"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pcm16_mono_16k(path: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != 16_000
            or source.getcomptype() != "NONE"
        ):
            raise ValueError("--audio must be mono PCM16 16 kHz WAV")
        sample_count = source.getnframes()
        raw = source.readframes(sample_count)
    if len(raw) != sample_count * 2:
        raise ValueError("short PCM read")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sample_count / 16_000


def regions_from_probabilities(
    probabilities: list[float],
    *,
    frame_samples: int,
    total_samples: int,
    threshold: float,
    exit_threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> list[dict[str, float]]:
    """Apply the product VAD state machine at either backend's native frame grid."""
    sample_rate = 16_000
    min_speech = round(sample_rate * min_speech_ms / 1000)
    min_silence = round(sample_rate * min_silence_ms / 1000)
    pad = round(sample_rate * speech_pad_ms / 1000)
    raw_regions: list[tuple[int, int]] = []
    triggered = False
    start_sample = 0
    possible_end: int | None = None
    for index, probability in enumerate(probabilities):
        position = index * frame_samples
        if probability >= threshold:
            possible_end = None
            if not triggered:
                triggered = True
                start_sample = position
        elif triggered and probability < exit_threshold:
            if possible_end is None:
                possible_end = position
            if position - possible_end >= min_silence:
                if possible_end - start_sample >= min_speech:
                    raw_regions.append((start_sample, possible_end))
                triggered = False
                possible_end = None
    if triggered and total_samples - start_sample >= min_speech:
        raw_regions.append((start_sample, total_samples))

    padded: list[tuple[int, int]] = []
    for start, end in raw_regions:
        left = max(0, start - pad)
        right = min(total_samples, end + pad)
        if padded and left < padded[-1][1]:
            midpoint = (padded[-1][1] + left) // 2
            padded[-1] = (padded[-1][0], midpoint)
            left = midpoint
        padded.append((left, right))
    merged: list[tuple[int, int]] = []
    merge_gap = round(0.30 * sample_rate)
    for start, end in padded:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [
        {"start_s": start / sample_rate, "end_s": end / sample_rate}
        for start, end in merged
    ]


def _is_active(intervals: list[tuple[float, float]], time_s: float) -> bool:
    return any(start <= time_s < end for start, end in intervals)


def activity_score(
    reference: list[tuple[float, float]],
    hypothesis: list[tuple[float, float]],
    *,
    duration_s: float,
    frame_s: float,
    labels: tuple[str, str],
) -> dict[str, float | int | None]:
    if frame_s <= 0:
        raise ValueError("--score-frame-ms must be positive")
    first, second = labels
    counts = {"both": 0, first: 0, second: 0}
    total_frames = math.ceil(duration_s / frame_s)
    for index in range(total_frames):
        midpoint_s = min(duration_s, (index + 0.5) * frame_s)
        first_active = _is_active(reference, midpoint_s)
        second_active = _is_active(hypothesis, midpoint_s)
        if first_active and second_active:
            counts["both"] += 1
        elif first_active:
            counts[first] += 1
        elif second_active:
            counts[second] += 1
    both, first_only, second_only = counts["both"], counts[first], counts[second]
    precision = both / (both + second_only) if both + second_only else None
    recall = both / (both + first_only) if both + first_only else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "frame_ms": round(frame_s * 1000, 6),
        "total_frames": total_frames,
        "both_active_s": both * frame_s,
        f"{first}_only_s": first_only * frame_s,
        f"{second}_only_s": second_only * frame_s,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def make_probe_package(directory: Path, fluid_source: Path) -> None:
    source_dir = directory / "Sources" / "FluidAudioVADProbe"
    source_dir.mkdir(parents=True)
    (directory / "Package.swift").write_text(
        "// swift-tools-version: 6.0\n"
        "import PackageDescription\n"
        "let package = Package(\n"
        "    name: \"FluidAudioVADProbe\",\n"
        "    platforms: [.macOS(.v14)],\n"
        f"    dependencies: [.package(name: \"FluidAudio\", path: {json.dumps(str(fluid_source))})],\n"
        "    targets: [.executableTarget(\n"
        "        name: \"FluidAudioVADProbe\",\n"
        "        dependencies: [.product(name: \"FluidAudio\", package: \"FluidAudio\")]\n"
        "    )]\n"
        ")\n",
        encoding="utf-8",
    )
    shutil.copy2(PROBE_SOURCE, source_dir / "main.swift")


def run_fluid_probe(
    *, fluid_source: Path, audio: Path, compute_units: str, raw_log: Path
) -> tuple[dict[str, Any], float]:
    with tempfile.TemporaryDirectory(prefix="fluidaudio-vad-probe-") as temporary:
        package_dir = Path(temporary)
        probe_output = package_dir / "probe.json"
        make_probe_package(package_dir, fluid_source)
        command = [
            "swift", "run", "--package-path", str(package_dir), "-c", "release",
            "FluidAudioVADProbe", "--audio", str(audio), "--output", str(probe_output),
            "--compute-units", compute_units,
        ]
        start = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        wall_s = time.perf_counter() - start
        raw_log.write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"FluidAudio probe exited {process.returncode}; see {raw_log}")
        if not probe_output.is_file():
            raise RuntimeError(f"FluidAudio probe did not write JSON; see {raw_log}")
        return json.loads(probe_output.read_text(encoding="utf-8")), wall_s


def main() -> int:
    args = parse_args()
    fluid_source = args.fluid_source.resolve()
    audio = args.audio.resolve()
    reference_path = args.reference.resolve()
    output = args.output.resolve()
    raw_log = args.raw_log.resolve()
    for path, label in (
        (fluid_source / "Package.swift", "--fluid-source"),
        (audio, "--audio"),
        (reference_path, "--reference"),
        (PROBE_SOURCE, "probe source"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
    if output.exists() or raw_log.exists():
        raise FileExistsError("--output and --raw-log must be new paths")
    if output == raw_log:
        raise ValueError("--output and --raw-log must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    samples, duration_s = load_pcm16_mono_16k(audio)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if not math.isclose(
        duration_s, float(reference["clip"]["duration_ms"]) / 1000, abs_tol=1 / 16_000
    ):
        raise ValueError("audio duration does not match reference")

    onnx = SileroOnnxVad()
    onnx_start = time.perf_counter()
    onnx_probabilities = onnx.probabilities(samples, 16_000).tolist()
    onnx_inference_s = time.perf_counter() - onnx_start
    onnx_regions = regions_from_probabilities(
        onnx_probabilities,
        frame_samples=onnx.frame_samples,
        total_samples=samples.size,
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )
    fluid, fluid_wall_s = run_fluid_probe(
        fluid_source=fluid_source,
        audio=audio,
        compute_units=args.compute_units,
        raw_log=raw_log,
    )
    if fluid["input"]["sample_count"] != samples.size or fluid["input"]["sample_rate_hz"] != 16_000:
        raise RuntimeError("FluidAudio probe did not receive the exact prepared sample contract")
    fluid_probabilities = [float(item["probability"]) for item in fluid["output"]["frames"]]
    fluid_frame_samples = int(fluid["output"]["frame_samples"])
    fluid_regions = regions_from_probabilities(
        fluid_probabilities,
        frame_samples=fluid_frame_samples,
        total_samples=samples.size,
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )
    reference_intervals = [
        (item["start_ms"] / 1000, item["end_ms"] / 1000)
        for item in reference["segments"]
    ]
    onnx_intervals = [(item["start_s"], item["end_s"]) for item in onnx_regions]
    fluid_intervals = [(item["start_s"], item["end_s"]) for item in fluid_regions]
    frame_s = args.score_frame_ms / 1000
    result = {
        "schema_version": 1,
        "runner": "Silero ONNX vs FluidAudio Core ML VAD probability-and-region comparison",
        "epistemic_limits": [
            "Both backends use the same named Silero VAD version but different exported artifacts and native inference frame sizes.",
            "The product state machine is reapplied to each backend at its native frame grid; results are not a bitwise-equivalence claim.",
            "The FluidAudio wall includes a temporary Swift-package build and model initialization, so it is not a performance comparison with the ONNX Python call.",
            "CantoMap activity labels are the union of ELAN speaker intervals, not independent VAD ground truth.",
        ],
        "input": {"path": str(audio), "sha256": sha256(audio), "duration_s": duration_s, "sample_rate_hz": 16_000, "channels": 1},
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path), "scope": "union of CantoMap ELAN speaker tiers", "interval_count": len(reference["segments"])},
        "shared_policy": {"threshold": args.threshold, "exit_threshold": args.exit_threshold, "min_speech_ms": args.min_speech_ms, "min_silence_ms": args.min_silence_ms, "speech_pad_ms": args.speech_pad_ms},
        "onnx": {"model": {"name": "Silero VAD", "version": MODEL_VERSION, "source": MODEL_URL, "expected_sha256": MODEL_SHA256, "resolved_sha256": sha256(onnx.model_path)}, "timing": {"inference_s": onnx_inference_s}, "output": {"frame_samples": onnx.frame_samples, "probabilities": onnx_probabilities, "regions": onnx_regions}, "reference_activity": activity_score(reference_intervals, onnx_intervals, duration_s=duration_s, frame_s=frame_s, labels=("reference", "onnx"))},
        "fluidaudio": {"version": args.fluid_version, "commit": args.fluid_commit, "source": str(fluid_source), "source_package_sha256": sha256(fluid_source / "Package.swift"), "compute_units": args.compute_units, "probe_source_sha256": sha256(PROBE_SOURCE), "raw_log": {"path": str(raw_log), "sha256": sha256(raw_log)}, "timing": {"harness_wall_s": fluid_wall_s, "probe": fluid["timing"]}, "output": {"frame_samples": fluid_frame_samples, "frames": fluid["output"]["frames"], "regions": fluid_regions}, "reference_activity": activity_score(reference_intervals, fluid_intervals, duration_s=duration_s, frame_s=frame_s, labels=("reference", "fluid"))},
        "comparison": {"region_activity_agreement": activity_score(onnx_intervals, fluid_intervals, duration_s=duration_s, frame_s=frame_s, labels=("onnx", "fluid"))},
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"onnx_regions": len(onnx_regions), "fluid_regions": len(fluid_regions), "region_agreement_f1": result["comparison"]["region_activity_agreement"]["f1"], "fluid_harness_wall_s": fluid_wall_s}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
