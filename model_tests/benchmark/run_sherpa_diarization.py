#!/usr/bin/env python3
"""Run sherpa-onnx offline speaker diarization with benchmark evidence.

The runner intentionally accepts only 16 kHz mono audio.  This keeps media
conversion outside the timed region and makes cross-run timing comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

SCRIPT_START = time.perf_counter()

import psutil
import sherpa_onnx
import soundfile as sf


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sherpa-onnx offline speaker diarization"
    )
    parser.add_argument("--audio", required=True, help="16 kHz mono input audio")
    parser.add_argument("--output", required=True, help="JSON evidence artifact")
    parser.add_argument("--segmentation-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=-1,
        help="Known speaker count; -1 uses clustering threshold",
    )
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    parser.add_argument("--min-duration-on", type=float, default=0.3)
    parser.add_argument("--min-duration-off", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--provider",
        choices=("cpu", "coreml"),
        default="cpu",
        help="Requested sherpa-onnx execution provider",
    )
    parser.add_argument("--memory-sample-ms", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_speakers == 0 or args.num_speakers < -1:
        raise ValueError("--num-speakers must be -1 or a positive integer")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.memory_sample_ms <= 0:
        raise ValueError("--memory-sample-ms must be positive")

    audio_path = Path(args.audio).resolve()
    segmentation_path = Path(args.segmentation_model).resolve()
    embedding_path = Path(args.embedding_model).resolve()
    output_path = Path(args.output).resolve()
    for path in (audio_path, segmentation_path, embedding_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    process = psutil.Process(os.getpid())
    memory_samples: list[dict[str, int | float]] = []
    sampler_errors: list[str] = []
    stop_sampler = threading.Event()
    sample_interval_s = args.memory_sample_ms / 1000.0

    def sample_memory() -> None:
        while not stop_sampler.is_set():
            try:
                memory_samples.append({
                    "elapsed_s": time.perf_counter() - SCRIPT_START,
                    "rss_bytes": process.memory_info().rss,
                })
            except Exception as exc:  # evidence should expose sampler failure
                sampler_errors.append(f"{type(exc).__name__}: {exc}")
            stop_sampler.wait(sample_interval_s)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    load_audio_start = time.perf_counter()
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio_load_s = time.perf_counter() - load_audio_start
    if sample_rate != 16000:
        raise ValueError(f"expected 16000 Hz input, got {sample_rate}")
    if samples.shape[1] != 1:
        raise ValueError(f"expected mono input, got {samples.shape[1]} channels")
    samples = samples[:, 0]
    duration_s = len(samples) / sample_rate

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_path)
            ),
            num_threads=args.threads,
            provider=args.provider,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_path),
            num_threads=args.threads,
            provider=args.provider,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=args.num_speakers,
            threshold=args.cluster_threshold,
        ),
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
    )
    if not config.validate():
        raise RuntimeError("sherpa-onnx rejected the diarization configuration")

    model_load_start = time.perf_counter()
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    model_load_s = time.perf_counter() - model_load_start
    if diarizer.sample_rate != sample_rate:
        raise RuntimeError(
            f"model expects {diarizer.sample_rate} Hz, input is {sample_rate} Hz"
        )

    progress = {"processed_chunks": 0, "total_chunks": 0, "callbacks": 0}

    def progress_callback(processed_chunks: int, total_chunks: int) -> int:
        progress.update({
            "processed_chunks": int(processed_chunks),
            "total_chunks": int(total_chunks),
            "callbacks": progress["callbacks"] + 1,
        })
        return 0

    diarize_start = time.perf_counter()
    raw_result = diarizer.process(samples, callback=progress_callback)
    diarize_s = time.perf_counter() - diarize_start
    segments = [
        {
            "start_s": float(item.start),
            "end_s": float(item.end),
            "speaker": f"speaker_{int(item.speaker):02d}",
        }
        for item in raw_result.sort_by_start_time()
    ]

    stop_sampler.set()
    sampler.join(timeout=max(1.0, sample_interval_s * 5))
    try:
        memory_samples.append({
            "elapsed_s": time.perf_counter() - SCRIPT_START,
            "rss_bytes": process.memory_info().rss,
        })
    except Exception as exc:
        sampler_errors.append(f"{type(exc).__name__}: {exc}")

    speaker_labels = sorted({item["speaker"] for item in segments})
    finite_bounds = all(
        math.isfinite(item["start_s"]) and math.isfinite(item["end_s"])
        for item in segments
    )
    valid_bounds = finite_bounds and all(
        0 <= item["start_s"] <= item["end_s"] <= duration_s + 0.05
        for item in segments
    )
    ordered = all(
        left["start_s"] <= right["start_s"]
        for left, right in zip(segments, segments[1:])
    )
    sample_gaps = [
        right["elapsed_s"] - left["elapsed_s"]
        for left, right in zip(memory_samples, memory_samples[1:])
    ]
    output = {"segments": segments}
    end_to_end_s = time.perf_counter() - SCRIPT_START
    artifact = {
        "schema_version": 1,
        "runner": "sherpa-onnx-offline-speaker-diarization",
        "input": {
            "path": str(audio_path),
            "sha256": sha256_file(audio_path),
            "sample_rate_hz": sample_rate,
            "channels": 1,
            "duration_s": duration_s,
        },
        "model": {
            "segmentation_path": str(segmentation_path),
            "segmentation_sha256": sha256_file(segmentation_path),
            "embedding_path": str(embedding_path),
            "embedding_sha256": sha256_file(embedding_path),
        },
        "configuration": {
            "known_num_speakers": (
                args.num_speakers if args.num_speakers > 0 else None
            ),
            "cluster_threshold": args.cluster_threshold,
            "min_duration_on_s": args.min_duration_on,
            "min_duration_off_s": args.min_duration_off,
            "num_threads_per_model": args.threads,
            "requested_provider": args.provider,
            "provider_verification_limit": (
                "sherpa-onnx accepts the provider name but this runner cannot prove "
                "which individual ONNX operations executed on that provider"
            ),
        },
        "runtime": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "packages": {
                "sherpa-onnx": package_version("sherpa-onnx"),
                "sherpa-onnx-core": package_version("sherpa-onnx-core"),
                "numpy": package_version("numpy"),
                "soundfile": package_version("soundfile"),
                "psutil": package_version("psutil"),
            },
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "timing": {
            "audio_load_s": audio_load_s,
            "model_load_s": model_load_s,
            "diarize_s": diarize_s,
            "script_wall_s": end_to_end_s,
            "diarize_rtf": diarize_s / duration_s,
            "script_rtf": end_to_end_s / duration_s,
            "scope": (
                "script wall starts before Python imports but excludes interpreter "
                "startup; diarize wall covers only OfflineSpeakerDiarization.process"
            ),
        },
        "memory": {
            "metric": "process RSS sampled during audio load, model load, and inference",
            "sample_interval_ms": args.memory_sample_ms,
            "sample_count": len(memory_samples),
            "peak_rss_bytes": max(
                (int(item["rss_bytes"]) for item in memory_samples), default=None
            ),
            "first_rss_bytes": (
                int(memory_samples[0]["rss_bytes"]) if memory_samples else None
            ),
            "last_rss_bytes": (
                int(memory_samples[-1]["rss_bytes"]) if memory_samples else None
            ),
            "max_sample_gap_s": max(sample_gaps, default=None),
            "sampler_errors": sampler_errors,
            "epistemic_limit": (
                "RSS is a process-level deployment proxy, not proof of peak physical "
                "memory on a 16 GB machine"
            ),
        },
        "progress": progress,
        "stability": {
            "segment_count": len(segments),
            "speaker_count": len(speaker_labels),
            "speaker_labels": speaker_labels,
            "finite_bounds": finite_bounds,
            "valid_bounds_with_50ms_tolerance": valid_bounds,
            "nondecreasing_start_times": ordered,
            "last_end_s": max((item["end_s"] for item in segments), default=None),
        },
        "output": output,
        "normalized_output_sha256": sha256_json(output),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "diarize_s": diarize_s,
        "diarize_rtf": diarize_s / duration_s,
        "peak_rss_bytes": artifact["memory"]["peak_rss_bytes"],
        "segment_count": len(segments),
        "speaker_count": len(speaker_labels),
    }))


if __name__ == "__main__":
    main()
