#!/usr/bin/env python3
"""Profile a prebuilt FluidAudio offline diarization CLI invocation.

Build and model provisioning are intentionally separate from a measured run.
This wrapper records fresh-process wall/RSS evidence and converts FluidAudio's
camelCase JSON into the benchmark's common ``output.segments`` contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a prebuilt FluidAudio offline diarization run"
    )
    parser.add_argument("--binary", required=True, help="Prebuilt fluidaudiocli")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True, help="Normalized evidence JSON")
    parser.add_argument("--raw-output", required=True, help="Raw FluidAudio JSON")
    parser.add_argument("--fluid-version", required=True)
    parser.add_argument("--fluid-commit", required=True)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--step-ratio", type=float, default=0.2)
    parser.add_argument("--min-segment-duration", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Request regular overlapping output instead of exclusive output",
    )
    parser.add_argument("--memory-sample-ms", type=float, default=100.0)
    parser.add_argument(
        "--model-dir",
        help="Optional model cache directory to hash; it is not passed to the CLI",
    )
    return parser.parse_args()


def model_inventory(model_dir: Path | None) -> dict[str, Any] | None:
    if model_dir is None:
        return None
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    files = []
    for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        files.append({
            "path": str(path.relative_to(model_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "path": str(model_dir),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def process_rss_bytes(pid: int) -> int | None:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return int(result.stdout.strip().splitlines()[0]) * 1024


def main() -> None:
    args = parse_args()
    binary = Path(args.binary).resolve()
    audio = Path(args.audio).resolve()
    output_path = Path(args.output).resolve()
    raw_output_path = Path(args.raw_output).resolve()
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else None
    for path in (binary, audio):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.num_speakers < 1:
        raise ValueError("--num-speakers must be positive for the interview protocol")
    if not 0 < args.step_ratio <= 1:
        raise ValueError("--step-ratio must be in (0, 1]")
    if args.memory_sample_ms <= 0:
        raise ValueError("--memory-sample-ms must be positive")
    if raw_output_path == output_path:
        raise ValueError("--raw-output and --output must differ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "process",
        str(audio),
        "--mode",
        "offline",
        "--output",
        str(raw_output_path),
        "--num-speakers",
        str(args.num_speakers),
        "--threshold",
        str(args.threshold),
        "--step-ratio",
        str(args.step_ratio),
        "--min-segment-duration",
        str(args.min_segment_duration),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.allow_overlap:
        command.append("--overlapping-segments")

    wall_start = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    samples: list[dict[str, int | float]] = []
    sample_errors: list[str] = []
    interval_s = args.memory_sample_ms / 1000.0
    while process.poll() is None:
        try:
            rss = process_rss_bytes(process.pid)
            if rss is not None:
                samples.append({
                    "elapsed_s": time.perf_counter() - wall_start,
                    "rss_bytes": rss,
                })
        except Exception as exc:
            sample_errors.append(f"{type(exc).__name__}: {exc}")
        time.sleep(interval_s)
    stdout, stderr = process.communicate()
    wall_s = time.perf_counter() - wall_start
    if process.returncode != 0:
        raise RuntimeError(
            f"FluidAudio exited {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    if not raw_output_path.is_file():
        raise RuntimeError("FluidAudio succeeded without writing its JSON output")

    raw = json.loads(raw_output_path.read_text())
    segments = [
        {
            "start_s": float(item["startTimeSeconds"]),
            "end_s": float(item["endTimeSeconds"]),
            "speaker": str(item["speakerId"]),
        }
        for item in raw.get("segments", [])
    ]
    duration_s = float(raw["durationSeconds"])
    finite_bounds = all(
        math.isfinite(item["start_s"]) and math.isfinite(item["end_s"])
        for item in segments
    )
    valid_bounds = finite_bounds and all(
        0 <= item["start_s"] <= item["end_s"] <= duration_s + 0.1
        for item in segments
    )
    nondecreasing = all(
        left["start_s"] <= right["start_s"]
        for left, right in zip(segments, segments[1:])
    )
    sample_gaps = [
        right["elapsed_s"] - left["elapsed_s"]
        for left, right in zip(samples, samples[1:])
    ]
    normalized_output = {"segments": segments}
    artifact = {
        "schema_version": 1,
        "runner": "FluidAudio-offline-diarization-profile",
        "input": {
            "path": str(audio),
            "bytes": audio.stat().st_size,
            "sha256": sha256_file(audio),
            "duration_s_reported_by_fluidaudio": duration_s,
        },
        "model": {
            "fluid_version": args.fluid_version,
            "fluid_commit": args.fluid_commit,
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "model_inventory": model_inventory(model_dir),
        },
        "configuration": {
            "mode": "offline",
            "known_num_speakers": args.num_speakers,
            "clustering_threshold": args.threshold,
            "segmentation_step_ratio": args.step_ratio,
            "min_segment_duration_s": args.min_segment_duration,
            "embedding_batch_size": args.batch_size,
            "overlapping_segments_requested": args.allow_overlap,
            "command": command,
        },
        "runtime": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "timing": {
            "fresh_process_wall_s": wall_s,
            "fresh_process_rtf": wall_s / duration_s,
            "fluidaudio_processing_s": float(raw["processingTimeSeconds"]),
            "fluidaudio_rtfx": float(raw["realTimeFactor"]),
            "fluidaudio_stage_timings": raw.get("timings"),
            "scope": (
                "fresh-process wall includes binary startup, cached model load/compile, "
                "audio preparation, diarization, and JSON writing; it excludes this "
                "Python wrapper startup and must be measured after a separate provisioning run"
            ),
        },
        "memory": {
            "metric": "target CLI process RSS sampled via ps",
            "sample_interval_ms": args.memory_sample_ms,
            "sample_count": len(samples),
            "peak_rss_bytes": max(
                (int(item["rss_bytes"]) for item in samples), default=None
            ),
            "first_rss_bytes": int(samples[0]["rss_bytes"]) if samples else None,
            "last_rss_bytes": int(samples[-1]["rss_bytes"]) if samples else None,
            "max_sample_gap_s": max(sample_gaps, default=None),
            "sampler_errors": sample_errors,
            "epistemic_limit": (
                "RSS omits memory held by system Core ML services and is not physical "
                "16 GB compatibility proof"
            ),
        },
        "raw_result": {
            "path": str(raw_output_path),
            "sha256": sha256_file(raw_output_path),
            "reported_speaker_count": raw.get("speakerCount"),
            "console_stdout": stdout,
            "console_stderr": stderr,
        },
        "stability": {
            "segment_count": len(segments),
            "speaker_labels": sorted({item["speaker"] for item in segments}),
            "finite_bounds": finite_bounds,
            "valid_bounds_with_100ms_tolerance": valid_bounds,
            "nondecreasing_start_times": nondecreasing,
            "last_end_s": max((item["end_s"] for item in segments), default=None),
        },
        "output": normalized_output,
        "normalized_output_sha256": sha256_json(normalized_output),
    }
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "fresh_process_wall_s": wall_s,
        "fresh_process_rtf": wall_s / duration_s,
        "peak_rss_bytes": artifact["memory"]["peak_rss_bytes"],
        "segment_count": len(segments),
        "speaker_count": len(artifact["stability"]["speaker_labels"]),
    }))


if __name__ == "__main__":
    main()
