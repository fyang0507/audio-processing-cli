"""Artifact assembly and stdout summary for the turn-attributed MLX benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
from pathlib import Path
from typing import Any

from . import plan as plan_module
from . import runtime


def _source_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": runtime.sha256(resolved)}


def runner_provenance(runner_path: Path) -> dict[str, Any]:
    helper_root = Path(__file__).resolve().parent
    sources = [
        {"name": name, **_source_record(path)}
        for name, path in [
            ("run_turn_attributed_mlx_asr.py", runner_path),
            *(
                (f"turn_attributed_mlx_asr/{path.name}", path)
                for path in sorted(helper_root.glob("*.py"))
            ),
        ]
    ]
    entrypoint = sources[0]
    return {
        "path": entrypoint["path"],
        "sha256": entrypoint["sha256"],
        "argv": sys.argv,
        "source_files": sources,
        "source_set_sha256": runtime.stable_json_sha256(
            [{"name": item["name"], "sha256": item["sha256"]} for item in sources]
        ),
    }


def build_artifact(
    *,
    args: argparse.Namespace,
    runner_path: Path,
    status: str,
    error: dict[str, Any] | None,
    model_path: Path,
    audio_path: Path,
    diarization_path: Path,
    source_probe: dict[str, Any] | None,
    prepared_audio_hash: str | None,
    duration_s: float | None,
    diarization: dict[str, Any] | None,
    model_parameter_bytes: int | None,
    api_probe: dict[str, Any] | None,
    serialized_plan: dict[str, Any] | None,
    abstentions: list[dict[str, Any]],
    qwen_result: dict[str, Any] | None,
    timing: dict[str, float],
    rss_reader: runtime.RssReader,
    samples: list[dict[str, Any]],
    sample_errors: list[str],
    usage_end: resource.struct_rusage,
    explicit_mlx_peaks: dict[str, int],
    swap_start: dict[str, int] | None,
    swap_end: dict[str, int] | None,
    output_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the complete evidence artifact without writing or summarizing it."""

    peak_active_plus_cache = max(
        (
            int(item.get("mlx_active_bytes", 0)) + int(item.get("mlx_cache_bytes", 0))
            for item in samples
        ),
        default=None,
    )
    return {
        "schema_version": 1,
        "status": status,
        "error": error,
        "runner": runner_provenance(runner_path),
        "epistemic_limits": [
            "FluidAudio speaker labels are anonymous clusters, not identities or roles.",
            "The participant oracle label belongs only in a separate score artifact.",
            "Output bounds come from diarization turns, not ASR or forced alignment.",
            "Overlapping speech is abstained rather than duplicated or arbitrarily assigned.",
            (
                "Predeclared 250/300/500 ms policies are engineering thresholds, "
                "not tuned quality evidence."
            ),
            "Sampled RSS and MLX allocator counters overlap and must not be summed.",
        ],
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "runtime": {
            "packages": runtime.package_versions(
                ["mlx", "mlx-metal", "mlx-audio", "mlx-lm", "numpy", "miniaudio"]
            ),
            "offline_environment": {
                name: os.environ.get(name)
                for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
            },
        },
        "input": {
            "audio_path": str(audio_path),
            "audio_sha256": runtime.sha256(audio_path),
            "ffprobe": source_probe,
            "prepared_prefix_float32_sha256": prepared_audio_hash,
            "duration_s": duration_s,
            "duration_limit_requested_s": args.duration_limit,
            "diarization_path": str(diarization_path),
            "diarization_sha256": runtime.sha256(diarization_path),
            "diarization_output_segments_sha256": (
                runtime.stable_json_sha256(diarization["output"]["segments"])
                if diarization is not None
                else None
            ),
        },
        "model": {
            "path": str(model_path),
            "snapshot_revision": runtime.snapshot_revision(model_path),
            "config_sha256": (
                runtime.sha256(model_path / "config.json")
                if (model_path / "config.json").is_file()
                else None
            ),
            "weight_files": (
                [
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": runtime.sha256(path),
                    }
                    for path in sorted(model_path.glob("*.safetensors"))
                ]
                if model_path.is_dir()
                else []
            ),
            "loaded_parameter_bytes": model_parameter_bytes,
            "api_probe": api_probe,
        },
        "configuration": {
            "language": args.language,
            "batch_size": args.batch_size,
            "batch_order": "ascending duration_samples then chronological index",
            "temperature": 0.0,
            "max_tokens_global": args.max_tokens,
            "sample_rate_hz": plan_module.SAMPLE_RATE,
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
            "peak_sampled_rss_bytes": max(
                (int(item["rss_bytes"]) for item in samples), default=None
            ),
            "ru_maxrss_bytes": runtime.normalized_ru_maxrss(usage_end),
            "peak_sampled_mlx_active_bytes": max(
                (int(item.get("mlx_active_bytes", 0)) for item in samples), default=None
            ),
            "peak_sampled_mlx_active_plus_cache_bytes": peak_active_plus_cache,
            "explicit_mlx_peak_active_bytes_by_phase": explicit_mlx_peaks,
            "host_swap_start": swap_start,
            "host_swap_end": swap_end,
            "host_swap_used_delta_bytes": (
                swap_end["used_bytes"] - swap_start["used_bytes"]
                if swap_start and swap_end
                else None
            ),
            "samples": samples,
        },
        "output": {
            "text": " ".join(item["text"] for item in output_segments),
            "segments": output_segments,
            "segment_count": len(output_segments),
            "anonymous_speakers": sorted({item["speaker"] for item in output_segments}),
            "segments_sha256": runtime.stable_json_sha256(output_segments),
            "last_end_s": max((float(item["end_s"]) for item in output_segments), default=None),
        },
    }


def publish(output_path: Path, artifact: dict[str, Any]) -> int:
    """Write the artifact and print the compact operator-facing summary."""

    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    serialized_plan = artifact["turn_plan"]
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "duration_s": artifact["input"]["duration_s"],
                "accepted_turns": (
                    serialized_plan["accepted_turn_count"] if serialized_plan else None
                ),
                "output_segments": artifact["output"]["segment_count"],
                "service_job_s": artifact["timing"].get("service_job_after_model_load_s"),
                "fresh_runner_s": artifact["timing"].get("fresh_runner_wall_s"),
                "peak_rss_bytes": artifact["memory"]["peak_sampled_rss_bytes"],
                "peak_mlx_active_plus_cache_bytes": artifact["memory"][
                    "peak_sampled_mlx_active_plus_cache_bytes"
                ],
                "error": artifact["error"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if artifact["status"] in {"ok", "plan_only"} else 1
