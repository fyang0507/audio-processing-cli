"""Encoding, durable publication, and post-codec verification."""

from __future__ import annotations

import os
from pathlib import Path

from ..dsp import regional_measurements
from ..media import (
    decode_audio,
    encode_output,
    measure_loudness,
    media_summary,
    probe_media,
    temporary_output_path,
    write_float_wav,
)
from ..profiles import Profile
from .models import LoudnessRun, PipelineError, PreparedRun, StageRun
from .reporting import _resolved_operations_hash, _round_loudness


def publish_output(
    profile: Profile,
    skipped_stages: set[str],
    prepared: PreparedRun,
    staged: StageRun,
    loudness: LoudnessRun,
    temp_dir: Path,
    report: dict[str, object],
) -> dict[str, object]:
    """Encode once, correct codec overshoot, verify, and atomically publish."""
    source = prepared.source
    output = prepared.output
    source_info = prepared.source_info
    probe = prepared.probe
    final_wav = loudness.final_wav
    program_operation = loudness.program_operation
    analysis = prepared.analysis
    source_stage_report = staged.source_stage_report
    abstained_source_region_ids = staged.abstained_source_region_ids
    stages = staged.stages
    resolved_adjustments = staged.resolved_adjustments
    stage = "program-loudness"
    assert output is not None
    with temporary_output_path(output) as encoded_temp:
        encode_output(
            source,
            final_wav,
            encoded_temp,
            original_sha256=str(source_info["sha256"]),
            has_video=bool(probe.get("has_video")),
        )
        after_probe = probe_media(encoded_temp)
        after_program = measure_loudness(
            encoded_temp,
            target_lufs=profile.target_lufs,
            target_lra=profile.target_lra_lu,
            target_true_peak=profile.target_true_peak_dbtp,
        )
        true_peak_limit = (
            profile.target_true_peak_dbtp
            if stage not in skipped_stages
            else -0.1
        )
        codec_peak_correction_db = 0.0
        codec_safe_wav = temp_dir / "codec-safe.wav"
        for _ in range(2):
            if (
                after_program["input_tp"] <= true_peak_limit
                or stage in skipped_stages
                or not profile.stage_enabled(stage)
            ):
                break
            correction = true_peak_limit - after_program["input_tp"] - 0.1
            codec_peak_correction_db += correction
            safe_audio, safe_sample_rate = decode_audio(final_wav)
            safe_audio = safe_audio * (
                10.0 ** (codec_peak_correction_db / 20.0)
            )
            write_float_wav(codec_safe_wav, safe_audio, safe_sample_rate)
            encode_output(
                source,
                codec_safe_wav,
                encoded_temp,
                original_sha256=str(source_info["sha256"]),
                has_video=bool(probe.get("has_video")),
            )
            after_probe = probe_media(encoded_temp)
            after_program = measure_loudness(
                encoded_temp,
                target_lufs=profile.target_lufs,
                target_lra=profile.target_lra_lu,
                target_true_peak=profile.target_true_peak_dbtp,
            )
        if program_operation is not None:
            program_operation["codec_peak_correction_db"] = round(
                codec_peak_correction_db, 6
            )
        after_audio, after_sample_rate = decode_audio(encoded_temp)
        after_regional = regional_measurements(
            after_audio, after_sample_rate, analysis
        )
        if source_stage_report["status"] == "applied":
            source_operations = source_stage_report.get("operations", [])
            assert isinstance(source_operations, list)
            source_operation_by_region = {
                str(item["region_id"]): item
                for item in source_operations
                if isinstance(item, dict) and "region_id" in item
            }
            final_region_evaluations: list[dict[str, object]] = []
            unbounded_failures: list[str] = []
            for measured_region in after_regional["machine_regions"]:
                region_id = str(measured_region["region_id"])
                difference = float(measured_region["difference_from_speech_db"])
                if region_id in abstained_source_region_ids:
                    final_region_evaluations.append(
                        {
                            "region_id": region_id,
                            "difference_from_speech_db": round(difference, 3),
                            "status": "abstained_overlap",
                        }
                    )
                    continue
                inside = (
                    profile.machine_relative_minimum_lu - 0.25
                    <= difference
                    <= profile.machine_relative_maximum_lu + 0.25
                )
                item = source_operation_by_region.get(region_id, {})
                resolved_gain = float(item.get("resolved_gain_db", 0.0))
                bounded = (
                    abs(resolved_gain - profile.machine_max_boost_db)
                    <= 0.05
                    or abs(
                        resolved_gain + profile.machine_max_attenuation_db
                    )
                    <= 0.05
                )
                status = (
                    "inside_target"
                    if inside
                    else "bounded_outside_target"
                    if bounded
                    else "outside_target"
                )
                if status == "outside_target":
                    unbounded_failures.append(region_id)
                final_region_evaluations.append(
                    {
                        "region_id": region_id,
                        "difference_from_speech_db": round(difference, 3),
                        "status": status,
                    }
                )
            source_stage_report["final_region_evaluations"] = (
                final_region_evaluations
            )
            if unbounded_failures:
                raise PipelineError(
                    "Source-balance verification failed for unbounded regions: "
                    + ", ".join(unbounded_failures)
                )
        output_info = media_summary(encoded_temp, after_probe)
        duration_delta_ms = 1000.0 * (
            float(output_info["duration_seconds"])
            - float(source_info["duration_seconds"])
        )
        timeline_ok = abs(duration_delta_ms) <= 50.0
        loudness_ok = (
            stage in skipped_stages
            or not profile.stage_enabled(stage)
            or abs(after_program["input_i"] - profile.target_lufs) <= 0.6
        )
        peak_ok = after_program["input_tp"] <= true_peak_limit
        if not timeline_ok:
            raise PipelineError(
                f"Timeline verification failed: output duration changed by {duration_delta_ms:.1f} ms"
            )
        if not loudness_ok:
            raise PipelineError(
                "Program loudness verification failed: "
                f"measured {after_program['input_i']:.2f} LUFS, target {profile.target_lufs:.2f} LUFS"
            )
        if not peak_ok:
            raise PipelineError(
                "True-peak verification failed: "
                f"measured {after_program['input_tp']:.2f} dBTP, limit {true_peak_limit:.2f} dBTP"
            )
        os.replace(encoded_temp, output)

    # Re-probe the final path so its path and hash describe the durable artifact.
    durable_probe = probe_media(output)
    durable_info = media_summary(output, durable_probe)
    report["output"] = durable_info
    report["measurements"]["after"] = {
        "program": _round_loudness(after_program),
        "regional": after_regional,
        "duration_delta_ms": round(duration_delta_ms, 3),
    }
    report["final_peak_validation"] = {
        "status": "pass",
        "measured_true_peak_dbtp": round(after_program["input_tp"], 3),
        "limit_true_peak_dbtp": round(true_peak_limit, 3),
    }
    report["timeline_preserved"] = timeline_ok
    report["dry_run"] = False
    report["rendered"] = True
    report["resolved_operations_sha256"] = _resolved_operations_hash(
        profile,
        stages,
        resolved_adjustments,
        source_info["sha256"],
    )
    return report
