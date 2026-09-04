#!/usr/bin/env python3
"""Transcribe anonymous FluidAudio turns with one persistent Qwen3-ASR worker.

This is an integration benchmark, not a speaker-identification system. It uses
FluidAudio's anonymous labels exactly as emitted, abstains from overlapping
speech instead of transcribing it twice, and treats all turn bounds as upstream
diarization bounds rather than ASR or forced-alignment timestamps.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import resource
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from ._turn_attributed_mlx_asr_plan import *  # noqa: F403
    from ._turn_attributed_mlx_asr_runtime import *  # noqa: F403
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _turn_attributed_mlx_asr_plan import *  # noqa: F403
    from _turn_attributed_mlx_asr_runtime import *  # noqa: F403


def main() -> int:
    args = parse_args()
    process_start = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    model_path = Path(args.model_path).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    diarization_path = Path(args.diarization_run).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    for path in (audio_path, diarization_path):
        if not path.is_file():
            raise SystemExit(f"input not found: {path}")
    if not args.plan_only and not model_path.is_dir():
        raise SystemExit(f"local model snapshot not found: {model_path}")
    if not args.plan_only and not list(model_path.glob("*.safetensors")):
        raise SystemExit(f"model weights not found: {model_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"

    rss_reader = RssReader()
    samples: list[dict[str, Any]] = []
    sample_errors: list[str] = []
    phase = {"name": "startup"}
    stop_sampling = threading.Event()
    mx_holder: dict[str, Any] = {}

    def sample_resources() -> None:
        while not stop_sampling.is_set():
            try:
                sample: dict[str, Any] = {
                    "elapsed_s": time.perf_counter() - process_start,
                    "phase": phase["name"],
                    "rss_bytes": rss_reader.read(),
                }
                mx = mx_holder.get("module")
                if mx is not None:
                    sample.update({
                        "mlx_active_bytes": int(mx.get_active_memory()),
                        "mlx_cache_bytes": int(mx.get_cache_memory()),
                        "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                    })
                samples.append(sample)
            except Exception as exc:  # pragma: no cover - telemetry must not kill job
                sample_errors.append(f"{type(exc).__name__}: {exc}")
            stop_sampling.wait(args.sample_interval)

    sampler_thread = threading.Thread(target=sample_resources, daemon=True)
    sampler_thread.start()
    swap_start = mac_swap_snapshot()
    status = "error"
    error: dict[str, Any] | None = None
    timing: dict[str, float] = {}
    explicit_mlx_peaks: dict[str, int] = {}
    output_segments: list[dict[str, Any]] = []
    model: Any = None
    plan: dict[str, Any] | None = None
    audio: Any = None
    diarization: dict[str, Any] | None = None
    api_probe: dict[str, Any] | None = None
    qwen_result: dict[str, Any] | None = None
    source_probe: dict[str, Any] | None = None
    prepared_audio_hash: str | None = None
    model_source: Path | None = None
    model_parameter_bytes: int | None = None
    job_start: float | None = None

    try:
        phase["name"] = "import"
        t0 = time.perf_counter()
        import numpy as np  # noqa: PLC0415
        from mlx_audio.audio_io import read as audio_read  # noqa: PLC0415
        timing["import_s"] = time.perf_counter() - t0

        if not args.plan_only:
            import mlx.core as mx  # noqa: PLC0415
            from mlx.utils import tree_flatten  # noqa: PLC0415
            from mlx_audio.stt.utils import load_model  # noqa: PLC0415
            mx_holder["module"] = mx
            if not mx.metal.is_available():
                raise RuntimeError("MLX Metal device is unavailable")
            phase["name"] = "model_load"
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            model = load_model(model_path, lazy=False, strict=False)
            mx.eval(model.parameters())
            mx.synchronize()
            timing["model_load_s"] = time.perf_counter() - t0
            explicit_mlx_peaks["model_load"] = int(mx.get_peak_memory())
            flattened = tree_flatten(model.parameters())
            model_parameter_bytes = sum(int(value.nbytes) for _, value in flattened)
            model_source = Path(inspect.getfile(type(model))).resolve()
            method = model._generate_chunks_batched
            signature = inspect.signature(method)
            required = {
                "chunks", "max_tokens", "sampler", "language", "system_prompt",
                "batch_size", "verbose",
            }
            if not required.issubset(signature.parameters):
                raise RuntimeError(
                    "installed mlx-audio private batched API does not match runner"
                )
            api_probe = {
                "model_generate_signature": str(inspect.signature(model.generate)),
                "private_batched_method": "_generate_chunks_batched",
                "private_batched_signature": str(signature),
                "source_path": str(model_source),
                "source_sha256": sha256(model_source),
                "reason": (
                    "The public generate() accepts one waveform and batches only its "
                    "internally split chunks. This runner calls the inspected private "
                    "batched method to pass already bounded diarizer turns while keeping "
                    "one loaded model. The source hash and signature make that coupling explicit."
                ),
            }

        job_start = time.perf_counter()
        phase["name"] = "audio_and_turn_preparation"
        t0 = time.perf_counter()
        source_probe = ffprobe(audio_path)
        raw_audio, source_rate = audio_read(audio_path, always_2d=True, dtype="float32")
        if source_rate != SAMPLE_RATE or raw_audio.shape[1] != 1:
            raise ValueError(
                f"canonical input must be mono 16 kHz; got {source_rate} Hz, "
                f"{raw_audio.shape[1]} channels"
            )
        audio = np.ascontiguousarray(raw_audio[:, 0], dtype=np.float32)
        if args.duration_limit is not None:
            audio = audio[: min(len(audio), round(args.duration_limit * SAMPLE_RATE))]
        if len(audio) == 0:
            raise ValueError("empty audio after duration limit")
        prepared_audio_hash = array_sha256(audio)
        diarization = json.loads(diarization_path.read_text())
        raw_segments = diarization.get("output", {}).get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("diarization artifact lacks output.segments")
        plan = build_plan(
            raw_segments,
            total_samples=len(audio),
            raw_fragment_min_samples=round(
                args.raw_fragment_min_seconds * SAMPLE_RATE
            ),
            merge_gap_samples=round(args.merge_silence_max_seconds * SAMPLE_RATE),
            asr_turn_min_samples=round(args.asr_turn_min_seconds * SAMPLE_RATE),
        )
        timing["audio_and_turn_preparation_s"] = time.perf_counter() - t0

        if args.plan_only:
            status = "plan_only"
        else:
            phase["name"] = "inference"
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

            accepted: list[Turn] = plan["accepted"]
            # Duration bucketing is deterministic and reduces zero-padding within a
            # batch. Results are restored to chronological turn order afterward.
            inference_order = sorted(
                range(len(accepted)),
                key=lambda index: (
                    accepted[index].end - accepted[index].start, index
                ),
            )
            chunks = [
                (audio[accepted[index].start:accepted[index].end], 0.0)
                for index in inference_order
            ]
            texts: list[str] = []
            generated: list[int] = []
            prompts: list[int] = []
            processed: list[bool] = []
            remaining_tokens = args.max_tokens
            batch_count = 0
            cache_clear_observations: list[dict[str, Any]] = []
            for batch_start in range(0, len(chunks), args.batch_size):
                if remaining_tokens <= 0:
                    break
                group = chunks[batch_start:batch_start + args.batch_size]
                group_texts, group_generated, group_prompts, group_processed = (
                    model._generate_chunks_batched(
                        group,
                        max_tokens=remaining_tokens,
                        sampler=make_sampler(temp=0.0),
                        language=args.language,
                        system_prompt=None,
                        batch_size=args.batch_size,
                        verbose=False,
                    )
                )
                if not all(isinstance(value, bool) for value in group_processed):
                    raise TypeError("private batched API processed result is not bool list")
                mx.synchronize()
                before_clear = {
                    "batch_index": batch_count,
                    "batch_size": len(group),
                    "before_clear_active_bytes": int(mx.get_active_memory()),
                    "before_clear_cache_bytes": int(mx.get_cache_memory()),
                }
                samples.append({
                    "elapsed_s": time.perf_counter() - process_start,
                    "phase": "inference_before_batch_cache_clear",
                    "sample_origin": "synchronous_runner_observation",
                    "batch_index": batch_count,
                    "rss_bytes": rss_reader.read(),
                    "mlx_active_bytes": before_clear["before_clear_active_bytes"],
                    "mlx_cache_bytes": before_clear["before_clear_cache_bytes"],
                    "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                })
                texts.extend(group_texts)
                generated.extend(group_generated)
                prompts.extend(group_prompts)
                processed.extend(group_processed)
                remaining_tokens -= sum(group_generated)
                mx.clear_cache()
                mx.synchronize()
                before_clear.update({
                    "after_clear_active_bytes": int(mx.get_active_memory()),
                    "after_clear_cache_bytes": int(mx.get_cache_memory()),
                })
                cache_clear_observations.append(before_clear)
                samples.append({
                    "elapsed_s": time.perf_counter() - process_start,
                    "phase": "inference_after_batch_cache_clear",
                    "sample_origin": "synchronous_runner_observation",
                    "batch_index": batch_count,
                    "rss_bytes": rss_reader.read(),
                    "mlx_active_bytes": before_clear["after_clear_active_bytes"],
                    "mlx_cache_bytes": before_clear["after_clear_cache_bytes"],
                    "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                })
                batch_count += 1
            if len(processed) < len(chunks):
                missing = len(chunks) - len(processed)
                texts.extend([""] * missing)
                generated.extend([0] * missing)
                prompts.extend([0] * missing)
                processed.extend([False] * missing)
            timing["inference_s"] = time.perf_counter() - t0
            explicit_mlx_peaks["inference"] = int(mx.get_peak_memory())
            restored: dict[int, tuple[str, int, int, bool]] = {
                chronological_index: (text, gen, prompt, was_processed)
                for chronological_index, text, gen, prompt, was_processed in zip(
                    inference_order, texts, generated, prompts, processed
                )
            }
            unprocessed = []
            for turn_index, turn in enumerate(accepted):
                text, gen, prompt, was_processed = restored[turn_index]
                if not was_processed:
                    unprocessed.append(turn_record(turn, turn_index))
                    continue
                output_segments.append({
                    **turn_record(turn, turn_index),
                    "text": text,
                    "language": args.language,
                    "prompt_tokens": prompt,
                    "generation_tokens": gen,
                    "timestamp_source": "FluidAudio anonymous diarization turn",
                })
            qwen_result = {
                "input_turns": len(accepted),
                "processed_turns": sum(bool(value) for value in processed),
                "unprocessed_turns": unprocessed,
                "prompt_tokens": sum(prompts),
                "generation_tokens": sum(generated),
                "global_generation_budget_tokens": args.max_tokens,
                "generation_budget_remaining_tokens": args.max_tokens - sum(generated),
                "inference_order": "ascending duration_samples then chronological index",
                "output_restored_to_chronological_order": True,
                "inference_batch_count": batch_count,
                "cache_cleared_between_batches": True,
                "cache_clear_observations": cache_clear_observations,
            }
            status = "ok" if not unprocessed else "partial_generation_budget"
        timing["service_job_after_model_load_s"] = time.perf_counter() - job_start
        timing["fresh_runner_wall_s"] = time.perf_counter() - process_start
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": phase["name"],
            "traceback": traceback.format_exc(),
        }
        timing.setdefault("fresh_runner_wall_s", time.perf_counter() - process_start)
    finally:
        phase["name"] = "complete"
        stop_sampling.set()
        sampler_thread.join(timeout=max(1.0, args.sample_interval * 2))
        final_sample: dict[str, Any] = {
            "elapsed_s": time.perf_counter() - process_start,
            "phase": "complete",
            "rss_bytes": rss_reader.read(),
        }
        mx = mx_holder.get("module")
        if mx is not None:
            final_sample.update({
                "mlx_active_bytes": int(mx.get_active_memory()),
                "mlx_cache_bytes": int(mx.get_cache_memory()),
                "mlx_peak_active_bytes": int(mx.get_peak_memory()),
            })
        samples.append(final_sample)

    swap_end = mac_swap_snapshot()
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    duration_s = len(audio) / SAMPLE_RATE if audio is not None else None
    if duration_s:
        for name in ("inference_s", "service_job_after_model_load_s", "fresh_runner_wall_s"):
            if name in timing:
                timing[f"rtf_{name.removesuffix('_s')}"] = timing[name] / duration_s
    timing["cpu_user_s"] = usage_end.ru_utime - usage_start.ru_utime
    timing["cpu_system_s"] = usage_end.ru_stime - usage_start.ru_stime

    serialized_plan, abstentions = serialize_plan(
        plan, diarization, qwen_result
    )

    peak_active_plus_cache = max((
        int(item.get("mlx_active_bytes", 0)) + int(item.get("mlx_cache_bytes", 0))
        for item in samples
    ), default=None)
    result = {
        "schema_version": 1,
        "status": status,
        "error": error,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "epistemic_limits": [
            "FluidAudio speaker labels are anonymous clusters, not identities or roles.",
            "The participant oracle label belongs only in a separate score artifact.",
            "Output bounds come from diarization turns, not ASR or forced alignment.",
            "Overlapping speech is abstained rather than duplicated or arbitrarily assigned.",
            "Predeclared 250/300/500 ms policies are engineering thresholds, not tuned quality evidence.",
            "Sampled RSS and MLX allocator counters overlap and must not be summed.",
        ],
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "runtime": {
            "packages": package_versions([
                "mlx", "mlx-metal", "mlx-audio", "mlx-lm", "numpy", "miniaudio"
            ]),
            "offline_environment": {
                name: os.environ.get(name) for name in (
                    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"
                )
            },
        },
        "input": {
            "audio_path": str(audio_path),
            "audio_sha256": sha256(audio_path),
            "ffprobe": source_probe,
            "prepared_prefix_float32_sha256": prepared_audio_hash,
            "duration_s": duration_s,
            "duration_limit_requested_s": args.duration_limit,
            "diarization_path": str(diarization_path),
            "diarization_sha256": sha256(diarization_path),
            "diarization_output_segments_sha256": (
                stable_json_sha256(diarization["output"]["segments"])
                if diarization is not None else None
            ),
        },
        "model": {
            "path": str(model_path),
            "snapshot_revision": snapshot_revision(model_path),
            "config_sha256": (
                sha256(model_path / "config.json")
                if (model_path / "config.json").is_file() else None
            ),
            "weight_files": ([{
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            } for path in sorted(model_path.glob("*.safetensors"))]
                if model_path.is_dir() else []),
            "loaded_parameter_bytes": model_parameter_bytes,
            "api_probe": api_probe,
        },
        "configuration": {
            "language": args.language,
            "batch_size": args.batch_size,
            "batch_order": "ascending duration_samples then chronological index",
            "temperature": 0.0,
            "max_tokens_global": args.max_tokens,
            "sample_rate_hz": SAMPLE_RATE,
            "raw_fragment_min_seconds": args.raw_fragment_min_seconds,
            "merge_silence_max_seconds": args.merge_silence_max_seconds,
            "asr_turn_min_seconds": args.asr_turn_min_seconds,
            "threshold_provenance": (
                "predeclared engineering policy before the 3-minute smoke; not "
                "selected against transcript or diarization quality scores"
            ),
            "merge_guard": (
                "same anonymous label across a gap containing no kept speaker, no "
                "overlap, and no filtered-fragment abstention; never across another speaker"
            ),
        },
        "turn_plan": serialized_plan,
        "abstentions": abstentions,
        "qwen": qwen_result,
        "timing": timing,
        "memory": {
            "rss_source": rss_reader.source,
            "sample_interval_s": args.sample_interval,
            "sample_count": len(samples),
            "sample_errors": sample_errors,
            "peak_sampled_rss_bytes": max((
                int(item["rss_bytes"]) for item in samples
            ), default=None),
            "ru_maxrss_bytes": normalized_ru_maxrss(usage_end),
            "peak_sampled_mlx_active_bytes": max((
                int(item.get("mlx_active_bytes", 0)) for item in samples
            ), default=None),
            "peak_sampled_mlx_active_plus_cache_bytes": peak_active_plus_cache,
            "explicit_mlx_peak_active_bytes_by_phase": explicit_mlx_peaks,
            "host_swap_start": swap_start,
            "host_swap_end": swap_end,
            "host_swap_used_delta_bytes": (
                swap_end["used_bytes"] - swap_start["used_bytes"]
                if swap_start and swap_end else None
            ),
            "samples": samples,
        },
        "output": {
            "text": " ".join(item["text"] for item in output_segments),
            "segments": output_segments,
            "segment_count": len(output_segments),
            "anonymous_speakers": sorted({
                item["speaker"] for item in output_segments
            }),
            "segments_sha256": stable_json_sha256(output_segments),
            "last_end_s": max((
                float(item["end_s"]) for item in output_segments
            ), default=None),
        },
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "duration_s": duration_s,
        "accepted_turns": serialized_plan["accepted_turn_count"] if serialized_plan else None,
        "output_segments": len(output_segments),
        "service_job_s": timing.get("service_job_after_model_load_s"),
        "fresh_runner_s": timing.get("fresh_runner_wall_s"),
        "peak_rss_bytes": result["memory"]["peak_sampled_rss_bytes"],
        "peak_mlx_active_plus_cache_bytes": peak_active_plus_cache,
        "error": error,
    }, ensure_ascii=False))
    return 0 if status in {"ok", "plan_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
