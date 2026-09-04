#!/usr/bin/env python3
"""Transcribe anonymous FluidAudio turns with one persistent Qwen3-ASR worker.

This is an integration benchmark, not a speaker-identification system. It uses
FluidAudio's anonymous labels exactly as emitted, abstains from overlapping
speech instead of transcribing it twice, and treats all turn bounds as upstream
diarization bounds rather than ASR or forced-alignment timestamps.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

if __package__:
    from .turn_attributed_mlx_asr import inference as _inference
    from .turn_attributed_mlx_asr import plan as _plan
    from .turn_attributed_mlx_asr import report as _report
    from .turn_attributed_mlx_asr import runtime as _runtime
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from turn_attributed_mlx_asr import inference as _inference
    from turn_attributed_mlx_asr import plan as _plan
    from turn_attributed_mlx_asr import report as _report
    from turn_attributed_mlx_asr import runtime as _runtime


def main() -> int:
    args = _runtime.parse_args()
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

    rss_reader = _runtime.RssReader()
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
                    sample.update(
                        {
                            "mlx_active_bytes": int(mx.get_active_memory()),
                            "mlx_cache_bytes": int(mx.get_cache_memory()),
                            "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                        }
                    )
                samples.append(sample)
            except Exception as exc:  # pragma: no cover - telemetry must not kill job
                sample_errors.append(f"{type(exc).__name__}: {exc}")
            stop_sampling.wait(args.sample_interval)

    sampler_thread = threading.Thread(target=sample_resources, daemon=True)
    sampler_thread.start()
    swap_start = _runtime.mac_swap_snapshot()
    status = "error"
    error: dict[str, Any] | None = None
    timing: dict[str, float] = {}
    explicit_mlx_peaks: dict[str, int] = {}
    output_segments: list[dict[str, Any]] = []
    plan: dict[str, Any] | None = None
    audio: Any = None
    diarization: dict[str, Any] | None = None
    api_probe: dict[str, Any] | None = None
    qwen_result: dict[str, Any] | None = None
    source_probe: dict[str, Any] | None = None
    prepared_audio_hash: str | None = None
    model_parameter_bytes: int | None = None
    job_start: float | None = None

    try:
        phase["name"] = "import"
        t0 = time.perf_counter()
        import numpy as np  # noqa: PLC0415
        from mlx_audio.audio_io import read as audio_read  # noqa: PLC0415

        timing["import_s"] = time.perf_counter() - t0

        if not args.plan_only:
            phase["name"] = "model_load"
            loaded_model = _inference.load_model(
                model_path,
                mlx_holder=mx_holder,
                timing=timing,
                explicit_mlx_peaks=explicit_mlx_peaks,
            )
            model_parameter_bytes = loaded_model.parameter_bytes
            api_probe = loaded_model.api_probe
        else:
            loaded_model = None

        job_start = time.perf_counter()
        phase["name"] = "audio_and_turn_preparation"
        t0 = time.perf_counter()
        source_probe = _runtime.ffprobe(audio_path)
        raw_audio, source_rate = audio_read(audio_path, always_2d=True, dtype="float32")
        if source_rate != _plan.SAMPLE_RATE or raw_audio.shape[1] != 1:
            raise ValueError(
                f"canonical input must be mono 16 kHz; got {source_rate} Hz, "
                f"{raw_audio.shape[1]} channels"
            )
        audio = np.ascontiguousarray(raw_audio[:, 0], dtype=np.float32)
        if args.duration_limit is not None:
            audio = audio[: min(len(audio), round(args.duration_limit * _plan.SAMPLE_RATE))]
        if len(audio) == 0:
            raise ValueError("empty audio after duration limit")
        prepared_audio_hash = _runtime.array_sha256(audio)
        diarization = json.loads(diarization_path.read_text())
        raw_segments = diarization.get("output", {}).get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("diarization artifact lacks output.segments")
        plan = _plan.build_plan(
            raw_segments,
            total_samples=len(audio),
            raw_fragment_min_samples=round(args.raw_fragment_min_seconds * _plan.SAMPLE_RATE),
            merge_gap_samples=round(args.merge_silence_max_seconds * _plan.SAMPLE_RATE),
            asr_turn_min_samples=round(args.asr_turn_min_seconds * _plan.SAMPLE_RATE),
        )
        timing["audio_and_turn_preparation_s"] = time.perf_counter() - t0

        if args.plan_only:
            status = "plan_only"
        else:
            assert loaded_model is not None
            phase["name"] = "inference"
            accepted: list[_plan.Turn] = plan["accepted"]
            inference_result = _inference.run_batches(
                loaded_model,
                accepted,
                audio,
                language=args.language,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                process_start=process_start,
                rss_reader=rss_reader,
                samples=samples,
                timing=timing,
                explicit_mlx_peaks=explicit_mlx_peaks,
            )
            output_segments = inference_result.segments
            qwen_result = inference_result.qwen
            status = inference_result.status
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
            final_sample.update(
                {
                    "mlx_active_bytes": int(mx.get_active_memory()),
                    "mlx_cache_bytes": int(mx.get_cache_memory()),
                    "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                }
            )
        samples.append(final_sample)

    swap_end = _runtime.mac_swap_snapshot()
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    duration_s = len(audio) / _plan.SAMPLE_RATE if audio is not None else None
    if duration_s:
        for name in ("inference_s", "service_job_after_model_load_s", "fresh_runner_wall_s"):
            if name in timing:
                timing[f"rtf_{name.removesuffix('_s')}"] = timing[name] / duration_s
    timing["cpu_user_s"] = usage_end.ru_utime - usage_start.ru_utime
    timing["cpu_system_s"] = usage_end.ru_stime - usage_start.ru_stime

    serialized_plan, abstentions = _plan.serialize_plan(plan, diarization, qwen_result)
    artifact = _report.build_artifact(
        args=args,
        runner_path=Path(__file__),
        status=status,
        error=error,
        model_path=model_path,
        audio_path=audio_path,
        diarization_path=diarization_path,
        source_probe=source_probe,
        prepared_audio_hash=prepared_audio_hash,
        duration_s=duration_s,
        diarization=diarization,
        model_parameter_bytes=model_parameter_bytes,
        api_probe=api_probe,
        serialized_plan=serialized_plan,
        abstentions=abstentions,
        qwen_result=qwen_result,
        timing=timing,
        rss_reader=rss_reader,
        samples=samples,
        sample_errors=sample_errors,
        usage_end=usage_end,
        explicit_mlx_peaks=explicit_mlx_peaks,
        swap_start=swap_start,
        swap_end=swap_end,
        output_segments=output_segments,
    )
    return _report.publish(output_path, artifact)


if __name__ == "__main__":
    raise SystemExit(main())
