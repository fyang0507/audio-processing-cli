#!/usr/bin/env python3
"""Measure FluidAudio then resource-aware Qwen turn ASR as fresh subprocesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cached FluidAudio diarization followed by resource-aware "
            "Qwen3-ASR turn transcription and record external sequential wall"
        )
    )
    parser.add_argument("--fluid-binary", required=True)
    parser.add_argument("--fluid-model-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--diarization-output", required=True)
    parser.add_argument("--diarization-raw-output", required=True)
    parser.add_argument("--asr-output", required=True)
    parser.add_argument("--output", required=True, help="Pipeline evidence JSON")
    parser.add_argument("--fluid-version", default="0.15.5")
    parser.add_argument(
        "--fluid-commit",
        default="19600a485baa4998812e4654b70d2bab8f2c9949",
    )
    parser.add_argument("--language", default="Cantonese")
    parser.add_argument("--max-tokens", type=int, default=16_384)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    fluid_runner = here / "run_fluidaudio_diarization.py"
    asr_runner = here / "run_turn_attributed_mlx_asr.py"
    fluid_binary = Path(args.fluid_binary).expanduser().resolve()
    fluid_model_dir = Path(args.fluid_model_dir).expanduser().resolve()
    audio = Path(args.audio).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    diarization_output = Path(args.diarization_output).expanduser().resolve()
    diarization_raw_output = Path(args.diarization_raw_output).expanduser().resolve()
    asr_output = Path(args.asr_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path, label in (
        (fluid_runner, "FluidAudio runner"),
        (asr_runner, "turn-ASR runner"),
        (fluid_binary, "FluidAudio binary"),
        (audio, "audio"),
    ):
        require_file(path, label)
    for path, label in (
        (fluid_model_dir, "FluidAudio model directory"),
        (model_path, "Qwen model snapshot"),
    ):
        require_dir(path, label)
    generated_paths = (
        diarization_output, diarization_raw_output, asr_output, output
    )
    if len(set(generated_paths)) != len(generated_paths):
        raise SystemExit("all output paths must be distinct")
    for path in generated_paths:
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")

    fluid_command = [
        sys.executable,
        str(fluid_runner),
        "--binary", str(fluid_binary),
        "--audio", str(audio),
        "--output", str(diarization_output),
        "--raw-output", str(diarization_raw_output),
        "--fluid-version", args.fluid_version,
        "--fluid-commit", args.fluid_commit,
        "--num-speakers", "2",
        "--threshold", "0.6",
        "--step-ratio", "0.1",
        "--min-segment-duration", "0",
        "--batch-size", "32",
        "--allow-overlap",
        "--model-dir", str(fluid_model_dir),
    ]
    asr_command = [
        sys.executable,
        str(asr_runner),
        "--model-path", str(model_path),
        "--audio", str(audio),
        "--diarization-run", str(diarization_output),
        "--output", str(asr_output),
        "--language", args.language,
        "--batch-size", "1",
        "--max-tokens", str(args.max_tokens),
    ]

    pipeline_start = time.perf_counter()
    fluid_start = time.perf_counter()
    fluid_process = subprocess.run(
        fluid_command, capture_output=True, text=True, check=False
    )
    fluid_wall_s = time.perf_counter() - fluid_start
    asr_process: subprocess.CompletedProcess[str] | None = None
    asr_wall_s: float | None = None
    if fluid_process.returncode == 0 and diarization_output.is_file():
        asr_start = time.perf_counter()
        asr_process = subprocess.run(
            asr_command, capture_output=True, text=True, check=False
        )
        asr_wall_s = time.perf_counter() - asr_start
    pipeline_wall_s = time.perf_counter() - pipeline_start

    fluid_artifact = (
        load_json(diarization_output) if diarization_output.is_file() else None
    )
    asr_artifact = load_json(asr_output) if asr_output.is_file() else None
    status = (
        "ok"
        if fluid_process.returncode == 0
        and asr_process is not None
        and asr_process.returncode == 0
        and asr_artifact is not None
        and asr_artifact.get("status") == "ok"
        else "error"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "python": sys.executable,
        },
        "input": {
            "audio_path": str(audio),
            "audio_sha256": sha256(audio),
            "fluid_binary": str(fluid_binary),
            "fluid_binary_sha256": sha256(fluid_binary),
            "fluid_model_dir": str(fluid_model_dir),
            "qwen_model_path": str(model_path),
        },
        "configuration": {
            "execution": "strictly sequential fresh subprocesses",
            "fluid": {
                "version": args.fluid_version,
                "commit": args.fluid_commit,
                "known_speakers": 2,
                "threshold": 0.6,
                "step_ratio": 0.1,
                "minimum_segment_s": 0.0,
                "embedding_batch_size": 32,
                "regular_overlap_output": True,
            },
            "asr": {
                "language": args.language,
                "batch_size": 1,
                "global_max_tokens": args.max_tokens,
                "cache_cleared_between_turns": True,
            },
        },
        "commands": {
            "fluid": fluid_command,
            "asr": asr_command,
        },
        "timing": {
            "external_pipeline_wall_s": pipeline_wall_s,
            "external_fluid_wrapper_subprocess_wall_s": fluid_wall_s,
            "external_asr_subprocess_wall_s": asr_wall_s,
            "scope": (
                "External pipeline wall begins immediately before launching the "
                "FluidAudio Python wrapper and ends after the Qwen Python process "
                "exits. It includes both interpreter startups, child runner work, "
                "the fresh FluidAudio CLI process, and artifact writes."
            ),
        },
        "resource_semantics": (
            "Stages run sequentially, never concurrently. FluidAudio CLI RSS omits "
            "Core ML service memory. Qwen process RSS and MLX active/cache counters "
            "overlap. Per-stage peaks are reported separately and must not be added "
            "or interpreted as physical-16-GB proof."
        ),
        "fluid": {
            "returncode": fluid_process.returncode,
            "stdout": fluid_process.stdout,
            "stderr": fluid_process.stderr,
            "artifact": str(diarization_output),
            "artifact_sha256": (
                sha256(diarization_output) if diarization_output.is_file() else None
            ),
            "raw_artifact": str(diarization_raw_output),
            "raw_artifact_sha256": (
                sha256(diarization_raw_output)
                if diarization_raw_output.is_file() else None
            ),
            "fresh_cli_wall_s": (
                fluid_artifact.get("timing", {}).get("fresh_process_wall_s")
                if fluid_artifact else None
            ),
            "peak_sampled_cli_rss_bytes": (
                fluid_artifact.get("memory", {}).get("peak_rss_bytes")
                if fluid_artifact else None
            ),
            "segments": (
                len(fluid_artifact.get("output", {}).get("segments", []))
                if fluid_artifact else None
            ),
        },
        "asr": {
            "returncode": asr_process.returncode if asr_process else None,
            "stdout": asr_process.stdout if asr_process else None,
            "stderr": asr_process.stderr if asr_process else None,
            "artifact": str(asr_output),
            "artifact_sha256": sha256(asr_output) if asr_output.is_file() else None,
            "status": asr_artifact.get("status") if asr_artifact else None,
            "fresh_runner_wall_s": (
                asr_artifact.get("timing", {}).get("fresh_runner_wall_s")
                if asr_artifact else None
            ),
            "service_job_after_model_load_s": (
                asr_artifact.get("timing", {}).get(
                    "service_job_after_model_load_s"
                ) if asr_artifact else None
            ),
            "peak_sampled_process_rss_bytes": (
                asr_artifact.get("memory", {}).get("peak_sampled_rss_bytes")
                if asr_artifact else None
            ),
            "peak_sampled_mlx_active_plus_cache_bytes": (
                asr_artifact.get("memory", {}).get(
                    "peak_sampled_mlx_active_plus_cache_bytes"
                ) if asr_artifact else None
            ),
            "turn_plan_sha256": (
                asr_artifact.get("turn_plan", {}).get("plan_sha256")
                if asr_artifact else None
            ),
            "output_segments_sha256": (
                asr_artifact.get("output", {}).get("segments_sha256")
                if asr_artifact else None
            ),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "external_pipeline_wall_s": pipeline_wall_s,
        "fluid_returncode": fluid_process.returncode,
        "asr_returncode": asr_process.returncode if asr_process else None,
        "fluid_segments": result["fluid"]["segments"],
        "asr_output_segments_sha256": result["asr"]["output_segments_sha256"],
    }))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
