#!/usr/bin/env python3
"""Instrumented, fresh-process VibeVoice-ASR benchmark runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

SCRIPT_START = time.perf_counter()

import numpy as np
import psutil
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--mps-limit-gib", type=float)
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--full-prompt-logits",
        action="store_true",
        help="Disable generation-time logits_to_keep for a controlled baseline",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_name,sample_rate,channels", "-of", "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"path": str(path), "commit": None,
                                "tracked_dirty": None}
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain",
             "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL,
        )
        metadata["tracked_dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def source_file_metadata(path: Path, checkout: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256(path) if path.is_file() else None,
        "tracked_diff_sha256": None,
    }
    try:
        relative = path.resolve().relative_to(checkout.resolve())
        diff = subprocess.check_output(
            ["git", "-C", str(checkout), "diff", "--binary", "--", str(relative)],
            stderr=subprocess.DEVNULL,
        )
        metadata["tracked_diff_sha256"] = sha256_bytes(diff) if diff else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return metadata


def huggingface_revision(path: Path) -> dict[str, Any]:
    revisions: set[str] = set()
    cache_root = path / ".cache/huggingface/download"
    if cache_root.is_dir():
        for metadata_path in cache_root.rglob("*.metadata"):
            try:
                first_line = metadata_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[0].strip()
            except (OSError, IndexError):
                continue
            if len(first_line) == 40 and all(
                character in "0123456789abcdefABCDEF" for character in first_line
            ):
                revisions.add(first_line.lower())
    return {"path": str(path), "revisions": sorted(revisions)}


def rss_bytes() -> int:
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            # Audio decoders may exit between child enumeration and inspection.
            continue
    return total


def system_memory() -> dict[str, int | float]:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "system_available_bytes": int(virtual.available),
        "system_memory_percent": float(virtual.percent),
        "system_swap_used_bytes": int(swap.used),
        "system_swap_percent": float(swap.percent),
    }


def mps_memory(enabled: bool) -> dict[str, int]:
    if not enabled or not torch.backends.mps.is_available():
        return {}
    return {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "recommended_max_bytes": torch.mps.recommended_max_memory(),
    }


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    vibe_checkout = repo / "model_tests/vibevoice/VibeVoice"
    sys.path.insert(0, str(vibe_checkout))
    from vibevoice.modular.modeling_vibevoice_asr import (  # noqa: PLC0415
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.processor.vibevoice_asr_processor import (  # noqa: PLC0415
        VibeVoiceASRProcessor,
    )

    model_path = Path(args.model_path).resolve()
    audio_path = Path(args.audio).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "mps" and torch.backends.mps.is_available():
            torch.mps.manual_seed(args.seed)

    if args.device == "mps" and args.mps_limit_gib is not None:
        recommended = torch.mps.recommended_max_memory()
        fraction = args.mps_limit_gib * 1024**3 / recommended
        torch.mps.set_per_process_memory_fraction(fraction)

    samples: list[dict[str, Any]] = []
    sampler_errors: list[dict[str, str]] = []
    system_at_start = system_memory()
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.is_set():
            try:
                samples.append({
                    "elapsed_s": time.perf_counter() - process_start,
                    "rss_bytes": rss_bytes(),
                    **system_memory(),
                    **mps_memory(args.device == "mps"),
                })
            except Exception as exc:  # telemetry must not abort model inference
                sampler_errors.append({
                    "type": type(exc).__name__, "message": str(exc),
                })
            stop.wait(args.sample_interval)

    process_start = time.perf_counter()
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    status = "error"
    error: dict[str, str] | None = None
    raw_text = ""
    segments: list[dict[str, Any]] = []
    timing: dict[str, float] = {}
    first_parameter_dtype: str | None = None
    first_parameter_device: str | None = None
    input_token_count: int | None = None
    generated_token_count: int | None = None
    eos_observed: bool | None = None
    try:
        t0 = time.perf_counter()
        processor = VibeVoiceASRProcessor.from_pretrained(
            str(model_path), language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            str(model_path), dtype=dtype, attn_implementation=args.attention,
            trust_remote_code=True,
        ).to(args.device).eval()
        if args.full_prompt_logits:
            model._supports_logits_to_keep = lambda: False
        first_parameter = next(model.parameters())
        first_parameter_dtype = str(first_parameter.dtype).removeprefix("torch.")
        first_parameter_device = str(first_parameter.device)
        if args.device == "mps":
            torch.mps.synchronize()
        timing["load_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        inputs = processor(
            audio=[str(audio_path)], sampling_rate=None, return_tensors="pt",
            padding=True, add_generation_prompt=True,
        )
        inputs = {
            key: value.to(args.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        input_token_count = int(inputs["input_ids"].shape[1])
        if args.device == "mps":
            torch.mps.synchronize()
        timing["preprocess_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                pad_token_id=processor.pad_id,
                eos_token_id=processor.tokenizer.eos_token_id, do_sample=False,
            )
        if args.device == "mps":
            torch.mps.synchronize()
        timing["generate_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        eos = (generated == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        eos_observed = bool(len(eos))
        if len(eos):
            generated = generated[: eos[0] + 1]
        generated_token_count = int(generated.numel())
        raw_text = processor.decode(generated, skip_special_tokens=True)
        segments = processor.post_process_transcription(raw_text)
        timing["postprocess_s"] = time.perf_counter() - t0
        status = "ok"
    except Exception as exc:  # preserve failure evidence in the result artifact
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        stop.set()
        sampler.join()
        timing["end_to_end_s"] = time.perf_counter() - process_start
        timing["script_wall_s"] = time.perf_counter() - SCRIPT_START

    system_at_end = system_memory()

    audio = ffprobe(audio_path)
    duration = float(audio["format"]["duration"])
    parsed_segments = segments if isinstance(segments, list) else []
    normalized = json.dumps(parsed_segments, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"))
    bounds: list[tuple[float, float]] = []
    for item in parsed_segments:
        if not isinstance(item, dict):
            continue
        start_value = item.get("start_s", item.get("start_time"))
        end_value = item.get("end_s", item.get("end_time"))
        try:
            start = float(start_value)
            end = float(end_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end):
            bounds.append((start, end))
    starts = [item[0] for item in bounds]
    ends = [item[1] for item in bounds]
    bounds_valid = bool(parsed_segments) and len(bounds) == len(parsed_segments) and all(
        0 <= start <= end <= duration + 0.1
        and (index == 0 or start >= starts[index - 1])
        and (index == 0 or end >= ends[index - 1])
        for index, (start, end) in enumerate(bounds)
    )
    peak_swap_used = max(
        (int(item["system_swap_used_bytes"]) for item in samples),
        default=int(system_at_start["system_swap_used_bytes"]),
    )
    sample_gaps = [
        current["elapsed_s"] - previous["elapsed_s"]
        for previous, current in zip(samples, samples[1:])
    ]
    result = {
        "schema_version": 1,
        "status": status,
        "error": error,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "packages": package_versions([
                "torch", "transformers", "numpy", "psutil", "soundfile",
            ]),
        },
        "source": {
            "code": git_metadata(vibe_checkout),
            "modeling_source": source_file_metadata(
                vibe_checkout / "vibevoice/modular/modeling_vibevoice_asr.py",
                vibe_checkout,
            ),
            "tracked_patch": source_file_metadata(
                repo / "model_tests/benchmark/patches/vibevoice-logits-to-keep.patch",
                repo,
            ),
            "model": huggingface_revision(model_path),
        },
        "configuration": {
            "model_path": str(model_path), "device": args.device,
            "dtype": args.dtype, "attention": args.attention,
            "max_new_tokens": args.max_new_tokens,
            "mps_limit_gib": args.mps_limit_gib,
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "seed": args.seed,
            "full_prompt_logits": args.full_prompt_logits,
            "first_parameter_dtype": first_parameter_dtype,
            "first_parameter_device": first_parameter_device,
        },
        "tokens": {
            "input": input_token_count,
            "generated_including_eos": generated_token_count,
            "eos_observed": eos_observed,
            "hit_max_new_tokens": (
                generated_token_count == args.max_new_tokens and not eos_observed
                if generated_token_count is not None and eos_observed is not None
                else None
            ),
        },
        "audio": {
            "path": str(audio_path), "sha256": sha256(audio_path),
            "duration_s": duration, "probe": audio,
        },
        "timing": {
            **timing,
            "rtf_end_to_end": timing["end_to_end_s"] / duration,
            "rtf_script_wall": timing["script_wall_s"] / duration,
        },
        "memory": {
            "system_counters_scope": "host-wide; not process-attributable",
            "system_at_start": system_at_start,
            "system_at_end": system_at_end,
            "peak_sampled_rss_bytes": max((x["rss_bytes"] for x in samples), default=0),
            "peak_sampled_mps_current_bytes": max(
                (x.get("current_allocated_bytes", 0) for x in samples), default=0
            ),
            "peak_sampled_mps_driver_bytes": max(
                (x.get("driver_allocated_bytes", 0) for x in samples), default=0
            ),
            "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "minimum_system_available_bytes": min(
                (int(x["system_available_bytes"]) for x in samples), default=0
            ),
            "peak_host_system_swap_used_bytes": peak_swap_used,
            "peak_host_system_swap_delta_from_start_bytes": max(
                0, peak_swap_used - int(system_at_start["system_swap_used_bytes"])
            ),
            "sample_count": len(samples),
            "requested_sample_interval_s": args.sample_interval,
            "maximum_observed_sample_gap_s": max(sample_gaps, default=None),
            "sampler_errors": sampler_errors,
            "samples": samples,
        },
        "stability": {
            "output_parse_valid": (
                status == "ok" and isinstance(segments, list)
                and bool(parsed_segments)
                and all(isinstance(item, dict) for item in parsed_segments)
            ),
            "segment_bounds_finite_in_range_and_monotonic": (
                bounds_valid if parsed_segments else None
            ),
            "segment_count": len(parsed_segments),
            "first_segment_start_s": starts[0] if starts else None,
            "last_segment_end_s": ends[-1] if ends else None,
            "last_segment_end_ratio": ends[-1] / duration
            if ends and duration else None,
            "speaker_labels": sorted({
                str(item.get("speaker_id", item.get("speaker")))
                for item in parsed_segments
                if item.get("speaker_id", item.get("speaker")) is not None
            }),
        },
        "output": {
            "raw_text": raw_text, "segments": parsed_segments,
            "normalized_segments_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        },
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": status, "output": str(output_path), "timing": result["timing"],
        "memory": {key: value for key, value in result["memory"].items()
                   if key != "samples"},
        "segments_sha256": result["output"]["normalized_segments_sha256"],
        "error": error,
    }, ensure_ascii=False))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
