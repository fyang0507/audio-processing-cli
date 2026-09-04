#!/usr/bin/env python3
"""Fresh-process, offline MLX-ASR benchmark runner.

The runner deliberately requires an existing local model snapshot. It never
passes a Hub model ID to mlx-audio, so a benchmark cannot silently become a
multi-gigabyte download.
"""

from __future__ import annotations

import hashlib
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
    from . import _mlx_asr_benchmark_support as _support
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _mlx_asr_benchmark_support as _support


def main() -> int:
    args = _support.parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    audio_path = Path(args.audio).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_dir():
        raise SystemExit(f"local model snapshot not found: {model_path}")
    if not audio_path.is_file():
        raise SystemExit(f"audio not found: {audio_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"model config not found: {config_path}")
    if not list(model_path.glob("*.safetensors")):
        raise SystemExit(f"model weights not found: {model_path}")

    config = json.loads(config_path.read_text())
    detected_family = _support.detect_family(config)
    family = detected_family if args.family == "auto" else args.family
    if family != detected_family:
        raise SystemExit(f"--family {family} disagrees with model config ({detected_family})")

    # Enforce offline behavior before importing mlx-audio/Transformers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    process_start = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    rss_reader = _support.RssReader()
    phase = {"name": "startup"}
    samples: list[dict[str, Any]] = []
    sampler_errors: list[dict[str, str]] = []
    stop = threading.Event()
    mx_holder: dict[str, Any] = {}
    explicit_mlx_phase_peaks: dict[str, int] = {}
    swap_start = _support.mac_swap_snapshot()

    def sample_memory() -> None:
        while True:
            sample: dict[str, Any] = {
                "elapsed_s": time.perf_counter() - process_start,
                "phase": phase["name"],
                "rss_bytes": rss_reader.read(),
            }
            mx = mx_holder.get("module")
            if mx is not None:
                try:
                    active = int(mx.get_active_memory())
                    cache = int(mx.get_cache_memory())
                    sample.update(
                        {
                            "mlx_active_bytes": active,
                            "mlx_cache_bytes": cache,
                            "mlx_active_plus_cache_bytes": active + cache,
                            "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                        }
                    )
                except Exception as exc:  # telemetry must not abort inference
                    sampler_errors.append(
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
            samples.append(sample)
            if stop.wait(args.sample_interval):
                break

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    timing: dict[str, float] = {}
    status = "error"
    error: dict[str, str] | None = None
    model: Any = None
    prepared_audio: Any = None
    transcription: Any = None
    model_parameter_bytes: int | None = None
    model_parameter_dtype_counts: dict[str, int] = {}
    model_source_file: Path | None = None
    source_probe: dict[str, Any] | None = None
    audio_duration_s: float | None = None
    prepared_audio_sha256: str | None = None

    try:
        phase["name"] = "import"
        t0 = time.perf_counter()
        import mlx.core as mx  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from mlx.utils import tree_flatten  # noqa: PLC0415
        from mlx_audio.stt.utils import load_audio, load_model  # noqa: PLC0415

        timing["import_s"] = time.perf_counter() - t0
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
        explicit_mlx_phase_peaks["model_load"] = int(mx.get_peak_memory())
        flattened = tree_flatten(model.parameters())
        model_parameter_bytes = sum(int(value.nbytes) for _, value in flattened)
        for _, value in flattened:
            dtype = str(value.dtype)
            model_parameter_dtype_counts[dtype] = model_parameter_dtype_counts.get(dtype, 0) + int(
                value.nbytes
            )
        model_source_file = Path(inspect.getfile(type(model))).resolve()

        phase["name"] = "audio_preprocess"
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        source_probe = _support.ffprobe(audio_path)
        loaded_audio = load_audio(str(audio_path), sr=16000)
        mx.eval(loaded_audio)
        mx.synchronize()
        prepared_audio = np.asarray(loaded_audio, dtype=np.float32)
        if prepared_audio.ndim != 1:
            prepared_audio = prepared_audio.reshape(-1)
        prepared_audio = np.ascontiguousarray(prepared_audio)
        audio_duration_s = len(prepared_audio) / 16000
        prepared_audio_sha256 = _support.array_sha256(prepared_audio)
        del loaded_audio
        mx.clear_cache()
        timing["audio_preprocess_s"] = time.perf_counter() - t0
        explicit_mlx_phase_peaks["audio_preprocess"] = int(mx.get_peak_memory())

        phase["name"] = "inference"
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        if family == "qwen3-asr":
            qwen_language = None if args.language.casefold() == "auto" else args.language
            transcription = model.generate(
                prepared_audio,
                language=qwen_language,
                chunk_duration=args.qwen_chunk_seconds,
                batch_size=args.qwen_batch_size,
                max_tokens=args.max_tokens,
                temperature=0.0,
                verbose=False,
            )
        else:
            transcription = model.generate(
                prepared_audio,
                language=args.language,
                task="transcribe",
                verbose=None,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                condition_on_previous_text=True,
                return_timestamps=True,
                word_timestamps=args.whisper_word_timestamps,
            )
        mx.synchronize()
        timing["inference_s"] = time.perf_counter() - t0
        explicit_mlx_phase_peaks["inference"] = int(mx.get_peak_memory())
        timing["fresh_process_wall_s"] = time.perf_counter() - process_start
        timing["service_job_wall_after_model_load_s"] = (
            timing["audio_preprocess_s"] + timing["inference_s"]
        )
        status = "ok"
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": phase["name"],
            "traceback": traceback.format_exc(),
        }
        timing.setdefault("fresh_process_wall_s", time.perf_counter() - process_start)
    finally:
        phase["name"] = "complete"
        mx = mx_holder.get("module")
        final_sample: dict[str, Any] = {
            "elapsed_s": time.perf_counter() - process_start,
            "phase": phase["name"],
            "rss_bytes": rss_reader.read(),
        }
        if mx is not None:
            try:
                active = int(mx.get_active_memory())
                cache = int(mx.get_cache_memory())
                final_sample.update(
                    {
                        "mlx_active_bytes": active,
                        "mlx_cache_bytes": cache,
                        "mlx_active_plus_cache_bytes": active + cache,
                        "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                    }
                )
            except Exception as exc:
                sampler_errors.append(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        samples.append(final_sample)
        stop.set()
        sampler.join()

    text = str(getattr(transcription, "text", "")) if transcription else ""
    raw_segments = getattr(transcription, "segments", None) if transcription else None
    segments = _support.normalize_segments(raw_segments)
    output_hash = _support.stable_json_sha256(segments)
    last_end_s = max((item["end_s"] for item in segments), default=None)
    timestamp_semantics = (
        "model_chunk_bounds_not_speech_timestamps"
        if family == "qwen3-asr"
        else (
            "native_word_and_segment_timestamps"
            if args.whisper_word_timestamps
            else "native_segment_timestamps"
        )
    )

    peak_by_phase: dict[str, dict[str, int]] = {}
    for sample in samples:
        name = str(sample["phase"])
        values = peak_by_phase.setdefault(name, {})
        for key in (
            "rss_bytes",
            "mlx_active_bytes",
            "mlx_cache_bytes",
            "mlx_active_plus_cache_bytes",
            "mlx_peak_active_bytes",
        ):
            if key in sample:
                values[key] = max(values.get(key, 0), int(sample[key]))

    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    if audio_duration_s and audio_duration_s > 0:
        if "inference_s" in timing:
            timing["rtf_inference"] = timing["inference_s"] / audio_duration_s
        if "service_job_wall_after_model_load_s" in timing:
            timing["rtf_service_job_after_model_load"] = (
                timing["service_job_wall_after_model_load_s"] / audio_duration_s
            )
        timing["rtf_fresh_process"] = timing["fresh_process_wall_s"] / audio_duration_s
        timing["projected_30m_service_job_s_at_observed_rtf"] = (
            timing.get("rtf_service_job_after_model_load", 0.0) * 1800
        )
    timing["cpu_user_s"] = usage_end.ru_utime - usage_start.ru_utime
    timing["cpu_system_s"] = usage_end.ru_stime - usage_start.ru_stime

    weight_files = []
    for path in sorted(model_path.glob("*.safetensors")):
        weight_files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _support.sha256(path),
            }
        )
    source_files = []
    for path in [
        Path(__file__).resolve(),
        model_source_file,
        Path(inspect.getfile(sys.modules["mlx_audio.stt.utils"])).resolve()
        if "mlx_audio.stt.utils" in sys.modules
        else None,
    ]:
        if path is not None and path.is_file():
            source_files.append({"path": str(path), "sha256": _support.sha256(path)})

    swap_end = _support.mac_swap_snapshot()
    swap_delta = None
    if swap_start is not None and swap_end is not None:
        swap_delta = swap_end["used_bytes"] - swap_start["used_bytes"]
    result = {
        "schema_version": 1,
        "status": status,
        "error": error,
        "epistemic_limits": [
            "This is one fresh-process run on one machine; it is not a latency distribution.",
            "MLX active/cache allocation and process RSS are overlapping unified-memory proxies and must not be summed.",
            "Qwen chunk bounds are processing-container intervals, not speech or word timestamps.",
            "Transcript edit distance measures orthographic agreement, not semantic equivalence or behavioral-analysis validity.",
        ],
        "runner": {
            "path": str(Path(__file__).resolve()),
            "argv": sys.argv,
            "source_files": source_files,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "physical_memory_bytes": _support.physical_memory_bytes(),
            "mlx_device": (mx_holder["module"].device_info() if "module" in mx_holder else None),
        },
        "runtime": {
            "packages": _support.package_versions(
                [
                    "mlx",
                    "mlx-metal",
                    "mlx-audio",
                    "numpy",
                    "scipy",
                    "transformers",
                    "tokenizers",
                    "huggingface-hub",
                ]
            ),
            "ffmpeg": _support.command_version(["ffmpeg", "-version"]),
            "ffprobe": _support.command_version(["ffprobe", "-version"]),
            "offline_environment": {
                name: os.environ.get(name)
                for name in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                )
            },
        },
        "model": {
            "path": str(model_path),
            "snapshot_revision": _support.snapshot_revision(model_path),
            "family": family,
            "config_sha256": _support.sha256(config_path),
            "quantization": config.get("quantization", config.get("quantization_config")),
            "parameter_bytes": model_parameter_bytes,
            "parameter_bytes_by_dtype": model_parameter_dtype_counts,
            "weight_files": weight_files,
        },
        "configuration": {
            "language": args.language,
            "qwen_language_argument": (
                None
                if family == "qwen3-asr" and args.language.casefold() == "auto"
                else args.language
                if family == "qwen3-asr"
                else None
            ),
            "qwen_chunk_seconds": (args.qwen_chunk_seconds if family == "qwen3-asr" else None),
            "qwen_batch_size": (args.qwen_batch_size if family == "qwen3-asr" else None),
            "max_tokens": args.max_tokens if family == "qwen3-asr" else None,
            "whisper_condition_on_previous_text": (True if family == "whisper" else None),
            "whisper_temperature_schedule": (
                [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if family == "whisper" else None
            ),
            "whisper_fallback_thresholds": (
                {
                    "compression_ratio": 2.4,
                    "log_probability": -1.0,
                    "no_speech_probability": 0.6,
                }
                if family == "whisper"
                else None
            ),
            "whisper_word_timestamps": (
                args.whisper_word_timestamps if family == "whisper" else None
            ),
            "target_rtf": args.target_rtf,
            "target_30m_wall_s": args.target_rtf * 1800,
            "sample_interval_s": args.sample_interval,
        },
        "audio": {
            "path": str(audio_path),
            "sha256": _support.sha256(audio_path),
            "probe": source_probe,
            "prepared_sample_rate": 16000,
            "prepared_channels": 1,
            "prepared_dtype": "float32",
            "prepared_duration_s": audio_duration_s,
            "prepared_audio_sha256": prepared_audio_sha256,
        },
        "timing": timing,
        "memory": {
            "rss_scope": "current_process_only",
            "rss_source": rss_reader.source,
            "mlx_metric_semantics": (
                "active plus cache is the sampled MLX allocator footprint proxy; "
                "all MLX counters overlap process RSS and omit non-MLX allocations"
            ),
            "peak_sampled_rss_bytes": max((int(item["rss_bytes"]) for item in samples), default=0),
            "ru_maxrss_bytes": _support.normalized_ru_maxrss(usage_end),
            "peak_sampled_mlx_active_bytes": max(
                (int(item.get("mlx_active_bytes", 0)) for item in samples),
                default=0,
            ),
            "peak_sampled_mlx_cache_bytes": max(
                (int(item.get("mlx_cache_bytes", 0)) for item in samples),
                default=0,
            ),
            "peak_sampled_mlx_active_plus_cache_bytes": max(
                (int(item.get("mlx_active_plus_cache_bytes", 0)) for item in samples),
                default=0,
            ),
            "explicit_mlx_peak_active_bytes_by_phase": explicit_mlx_phase_peaks,
            "peak_by_phase": peak_by_phase,
            "host_swap_start": swap_start,
            "host_swap_end": swap_end,
            "host_swap_used_delta_bytes": swap_delta,
            "sample_count": len(samples),
            "sampler_errors": sampler_errors,
            "samples": samples,
        },
        "output": {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "segments": segments,
            "normalized_segments_sha256": output_hash,
            "segment_count": len(segments),
            "timestamp_semantics": timestamp_semantics,
            "timestamps_monotonic": _support.monotonic_segments(segments),
            "last_segment_end_s": last_end_s,
            "last_segment_end_ratio": (
                last_end_s / audio_duration_s
                if last_end_s is not None and audio_duration_s
                else None
            ),
            "language": getattr(transcription, "language", None) if transcription else None,
            "prompt_tokens": getattr(transcription, "prompt_tokens", None)
            if transcription
            else None,
            "generation_tokens": getattr(transcription, "generation_tokens", None)
            if transcription
            else None,
        },
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "family": family,
                "model": model_path.name,
                "duration_s": audio_duration_s,
                "inference_s": timing.get("inference_s"),
                "service_job_s": timing.get("service_job_wall_after_model_load_s"),
                "fresh_process_s": timing.get("fresh_process_wall_s"),
                "rtf_service_job": timing.get("rtf_service_job_after_model_load"),
                "peak_rss_bytes": result["memory"]["peak_sampled_rss_bytes"],
                "peak_mlx_active_bytes": result["memory"]["peak_sampled_mlx_active_bytes"],
                "peak_mlx_active_plus_cache_bytes": result["memory"][
                    "peak_sampled_mlx_active_plus_cache_bytes"
                ],
                "segments": len(segments),
                "last_end_s": last_end_s,
            }
        )
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
