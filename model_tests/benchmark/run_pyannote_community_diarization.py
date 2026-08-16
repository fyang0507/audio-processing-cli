#!/usr/bin/env python3
"""Profile direct pyannote Community-1 diarization under the common contract.

This intentionally calls the public ``pyannote.audio`` pipeline directly: it
does not import FluidAudio or invoke any Core ML conversion.  A child Python
process owns model loading and inference so its sampled RSS and wall time have
the same fresh-process scope as the FluidAudio CLI runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any


MODEL_ID = "pyannote/speaker-diarization-community-1"


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


def duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile direct pyannote Community-1 diarization"
    )
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True, help="Normalized evidence JSON")
    parser.add_argument("--raw-output", required=True, help="Direct pipeline JSON")
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--model-revision",
        default="main",
        help="Hugging Face revision to load; record an immutable commit for a final run",
    )
    parser.add_argument("--cache-dir", help="Optional Hugging Face model cache directory")
    parser.add_argument("--python", default=sys.executable, help="Python with pyannote.audio installed")
    parser.add_argument("--memory-sample-ms", type=float, default=100.0)
    parser.add_argument(
        "--regular-output",
        action="store_true",
        help="Use regular (overlap-permitting) output rather than exclusive output",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def worker(args: argparse.Namespace) -> None:
    """Load and run the public pipeline, writing only raw normalized output."""
    import torch
    from pyannote.audio import Pipeline

    audio = Path(args.audio).resolve()
    # pyannote.audio 4 delegates path decoding to TorchCodec.  Its published
    # macOS wheel currently cannot locate Homebrew FFmpeg dylibs in this
    # environment.  The benchmark inputs are canonical PCM16 WAV, so decoding
    # them here into the exact sample tensor avoids that unrelated runtime
    # failure and avoids any resampling or channel conversion.
    with wave.open(str(audio), "rb") as handle:
        if handle.getcomptype() != "NONE" or handle.getsampwidth() != 2:
            raise ValueError("direct pyannote runner requires uncompressed PCM16 WAV input")
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        pcm = handle.readframes(frames)
    waveform = torch.frombuffer(bytearray(pcm), dtype=torch.int16).reshape(
        frames, channels
    ).transpose(0, 1).to(torch.float32).div_(32768.0)
    pipeline = Pipeline.from_pretrained(
        MODEL_ID,
        revision=args.model_revision,
        token=True,
        cache_dir=args.cache_dir,
    )
    pipeline.to(torch.device(args.device))
    result = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=args.num_speakers,
    )
    annotation = (
        result.speaker_diarization
        if args.regular_output
        else result.exclusive_speaker_diarization
    )
    segments = [
        {
            "start_s": float(segment.start),
            "end_s": float(segment.end),
            "speaker": str(speaker),
        }
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]
    raw = {
        "model_id": MODEL_ID,
        "model_revision_requested": args.model_revision,
        "pyannote_audio_version": importlib.metadata.version("pyannote.audio"),
        "torch_version": torch.__version__,
        "device": args.device,
        "num_speakers": args.num_speakers,
        "output_kind": "regular" if args.regular_output else "exclusive",
        "audio_loader": "stdlib wave PCM16 -> torch waveform (no resampling)",
        "segments": segments,
    }
    Path(args.raw_output).write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args)
        return

    audio = Path(args.audio).resolve()
    output_path = Path(args.output).resolve()
    raw_output_path = Path(args.raw_output).resolve()
    # Do not resolve this path: virtual-environment ``bin/python`` is normally
    # a symlink to a base interpreter, and resolving it would discard the venv
    # prefix (and therefore pyannote's installed packages).
    python = Path(args.python).expanduser().absolute()
    for path in (audio, python):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.num_speakers < 1:
        raise ValueError("--num-speakers must be positive for the interview protocol")
    if args.memory_sample_ms <= 0:
        raise ValueError("--memory-sample-ms must be positive")
    if raw_output_path == output_path:
        raise ValueError("--raw-output and --output must differ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), str(Path(__file__).resolve()), "--worker",
        "--audio", str(audio), "--raw-output", str(raw_output_path),
        "--output", str(output_path), "--num-speakers", str(args.num_speakers),
        "--device", args.device,
        "--model-revision", args.model_revision,
    ]
    if args.cache_dir:
        command.extend(["--cache-dir", args.cache_dir])
    if args.regular_output:
        command.append("--regular-output")

    wall_start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples: list[dict[str, int | float]] = []
    sample_errors: list[str] = []
    interval_s = args.memory_sample_ms / 1000.0
    while process.poll() is None:
        try:
            rss = process_rss_bytes(process.pid)
            if rss is not None:
                samples.append({"elapsed_s": time.perf_counter() - wall_start, "rss_bytes": rss})
        except Exception as exc:  # pragma: no cover - depends on host ps behavior
            sample_errors.append(f"{type(exc).__name__}: {exc}")
        time.sleep(interval_s)
    stdout, stderr = process.communicate()
    wall_s = time.perf_counter() - wall_start
    if process.returncode != 0:
        raise RuntimeError(
            f"direct pyannote worker exited {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    if not raw_output_path.is_file():
        raise RuntimeError("direct pyannote worker succeeded without writing JSON output")

    raw = json.loads(raw_output_path.read_text())
    segments = raw["segments"]
    clip_duration_s = duration_s(audio)
    finite_bounds = all(
        math.isfinite(item["start_s"]) and math.isfinite(item["end_s"])
        for item in segments
    )
    valid_bounds = finite_bounds and all(
        0 <= item["start_s"] <= item["end_s"] <= clip_duration_s + 0.1
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
        "runner": "direct-pyannote-community-1-profile",
        "input": {
            "path": str(audio), "bytes": audio.stat().st_size,
            "sha256": sha256_file(audio), "duration_s": clip_duration_s,
        },
        "model": {
            "model_id": raw["model_id"],
            "model_revision_requested": raw["model_revision_requested"],
            "pyannote_audio_version": raw["pyannote_audio_version"],
            "torch_version": raw["torch_version"],
            "execution_device": raw["device"],
            "audio_loader": raw["audio_loader"],
        },
        "configuration": {
            "known_num_speakers": args.num_speakers,
            "output_kind": raw["output_kind"],
            "public_pipeline_direct": True,
            "command": command,
        },
        "runtime": {
            "machine": platform.machine(), "platform": platform.platform(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "timing": {
            "fresh_process_wall_s": wall_s,
            "fresh_process_rtf": wall_s / clip_duration_s,
            "scope": (
                "fresh worker process includes Python import, cached model load, "
                "audio preparation, direct pyannote inference, and JSON writing; "
                "provision the gated model in a separate process before timing"
            ),
        },
        "memory": {
            "metric": "direct pyannote worker RSS sampled via ps",
            "sample_interval_ms": args.memory_sample_ms,
            "sample_count": len(samples),
            "peak_rss_bytes": max((int(item["rss_bytes"]) for item in samples), default=None),
            "first_rss_bytes": int(samples[0]["rss_bytes"]) if samples else None,
            "last_rss_bytes": int(samples[-1]["rss_bytes"]) if samples else None,
            "max_sample_gap_s": max(sample_gaps, default=None),
            "sampler_errors": sample_errors,
            "epistemic_limit": "RSS is process memory, not physical-Mac compatibility proof.",
        },
        "raw_result": {"path": str(raw_output_path), "sha256": sha256_file(raw_output_path), "console_stdout": stdout, "console_stderr": stderr},
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
        "fresh_process_rtf": wall_s / clip_duration_s,
        "peak_rss_bytes": artifact["memory"]["peak_rss_bytes"],
        "segment_count": len(segments),
        "speaker_count": len(artifact["stability"]["speaker_labels"]),
    }))


if __name__ == "__main__":
    main()
