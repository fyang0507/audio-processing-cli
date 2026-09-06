"""Program loudness normalization and predicted-output verification."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..dsp import SignalAnalysis, regional_measurements
from ..media import (
    decode_audio,
    measure_loudness,
    render_loudness_normalized,
    write_float_wav,
)
from ..profiles import Profile
from .models import LoudnessRun, PipelineError, PreparedRun, StageRun
from .reporting import _round_loudness
from .stages import _disabled_stage, _skipped_stage, _stage_result


def normalize_loudness(
    profile: Profile,
    skipped_stages: set[str],
    prepared: PreparedRun,
    staged: StageRun,
    temp_dir: Path,
    *,
    apply_region_corrections: Callable[..., np.ndarray],
) -> LoudnessRun:
    """Normalize program loudness and converge source-relative regional balance."""
    current = staged.current
    sample_rate = prepared.sample_rate
    analysis: SignalAnalysis = prepared.analysis
    stages = staged.stages
    source_stage_report = staged.source_stage_report
    abstained_source_region_ids = staged.abstained_source_region_ids
    pre_loudness_wav = temp_dir / "pre-loudness.wav"
    final_wav = temp_dir / "final.wav"
    write_float_wav(pre_loudness_wav, current, sample_rate)
    render_true_peak_target = profile.target_true_peak_dbtp - profile.codec_true_peak_headroom_db
    pre_program = measure_loudness(
        pre_loudness_wav,
        target_lufs=profile.target_lufs,
        target_lra=profile.target_lra_lu,
        target_true_peak=render_true_peak_target,
    )

    stage = "program-loudness"
    needs_normalization = (
        abs(pre_program["input_i"] - profile.target_lufs) > 0.5
        or pre_program["input_tp"] > profile.target_true_peak_dbtp
    )
    normalization_resolution: dict[str, object] | None = None
    program_operation: dict[str, object] | None = None
    if stage in skipped_stages:
        stages.append(_skipped_stage(stage, profile))
        if pre_program["input_tp"] > -0.1:
            raise PipelineError(
                "Final peak validation failed: program-loudness was skipped and the "
                f"predicted true peak is {pre_program['input_tp']:.2f} dBTP"
            )
        shutil.copyfile(pre_loudness_wav, final_wav)
    elif not profile.stage_enabled(stage):
        stages.append(_disabled_stage(stage, profile))
        shutil.copyfile(pre_loudness_wav, final_wav)
    elif needs_normalization:
        program_operation = {
            "type": "ebu-r128-linear-gain-and-true-peak-limiter",
            "target_lufs": profile.target_lufs,
            "target_lra_lu": profile.target_lra_lu,
            "target_true_peak_dbtp": profile.target_true_peak_dbtp,
            "render_true_peak_dbtp": render_true_peak_target,
            "codec_headroom_db": profile.codec_true_peak_headroom_db,
            "measured_input": _round_loudness(pre_program),
        }
        stages.append(
            _stage_result(
                stage,
                profile,
                {
                    "status": "applied",
                    "reason": "program_outside_profile_target",
                    "operations": [program_operation],
                    "component_evaluations": [
                        {
                            "component": "loudness-range",
                            "status": (
                                "no_op"
                                if pre_program["input_lra"] <= profile.target_lra_lu
                                else "abstained"
                            ),
                            "reason": (
                                "inside_target"
                                if pre_program["input_lra"] <= profile.target_lra_lu
                                else "dynamic_lra_control_would_change_relative_region_balance"
                            ),
                            "measured_lra_lu": round(pre_program["input_lra"], 3),
                            "measured_at": "before_initial_program_loudness",
                            "target_maximum_lra_lu": profile.target_lra_lu,
                        }
                    ],
                },
            )
        )
        normalization_resolution = render_loudness_normalized(
            pre_loudness_wav,
            final_wav,
            target_lufs=profile.target_lufs,
            target_lra=profile.target_lra_lu,
            target_true_peak=render_true_peak_target,
            measurement=pre_program,
            sample_rate=sample_rate,
        )
        program_operation["resolution"] = normalization_resolution
    else:
        stages.append(
            _stage_result(
                stage,
                profile,
                {
                    "status": "no_op",
                    "reason": "program_inside_target",
                    "operations": [],
                },
            )
        )
        shutil.copyfile(pre_loudness_wav, final_wav)

    simulated_program = measure_loudness(
        final_wav,
        target_lufs=profile.target_lufs,
        target_lra=profile.target_lra_lu,
        target_true_peak=profile.target_true_peak_dbtp,
    )
    simulated_audio, simulated_sample_rate = decode_audio(final_wav)
    simulated_regional = regional_measurements(simulated_audio, simulated_sample_rate, analysis)
    source_verification_passes: list[dict[str, object]] = []
    if (
        source_stage_report["status"] == "applied"
        and needs_normalization
        and program_operation is not None
    ):
        for verification_pass in range(1, 4):
            operations = source_stage_report.get("operations", [])
            assert isinstance(operations, list)
            operation_by_region = {
                str(item["region_id"]): item
                for item in operations
                if isinstance(item, dict) and "region_id" in item
            }
            corrections: dict[str, float] = {}
            for measured_region in simulated_regional["machine_regions"]:
                region_id = str(measured_region["region_id"])
                if region_id in abstained_source_region_ids:
                    continue
                difference = float(measured_region["difference_from_speech_db"])
                if (
                    profile.machine_relative_minimum_lu
                    <= difference
                    <= profile.machine_relative_maximum_lu
                ):
                    continue
                item = operation_by_region.get(region_id)
                if item is None:
                    region = next(
                        region
                        for region in analysis.machine_regions
                        if region.region_id == region_id
                    )
                    item = {
                        "type": "regional-full-band-gain",
                        "region_id": region_id,
                        "scope": {
                            "time": {
                                "start": round(region.start, 6),
                                "end": round(region.end, 6),
                            },
                            "frequency": "all",
                        },
                        "target_difference_db": profile.machine_relative_target_lu,
                        "resolved_gain_db": 0.0,
                        "boundary_fade_ms": profile.region_fade_ms,
                    }
                    operations.append(item)
                    operation_by_region[region_id] = item
                previous_gain = float(item.get("resolved_gain_db", 0.0))
                desired_delta = profile.machine_relative_target_lu - difference
                resolved_total = float(
                    np.clip(
                        previous_gain + desired_delta,
                        -profile.machine_max_attenuation_db,
                        profile.machine_max_boost_db,
                    )
                )
                delta = resolved_total - previous_gain
                if abs(delta) < 0.05:
                    continue
                corrections[region_id] = delta
                item["resolved_gain_db"] = round(resolved_total, 3)
                item["downstream_compensation_db"] = round(
                    float(item.get("downstream_compensation_db", 0.0)) + delta,
                    3,
                )
            if not corrections:
                break
            current = apply_region_corrections(
                current,
                sample_rate,
                analysis,
                corrections,
                profile.region_fade_ms,
            )
            write_float_wav(pre_loudness_wav, current, sample_rate)
            pre_program = measure_loudness(
                pre_loudness_wav,
                target_lufs=profile.target_lufs,
                target_lra=profile.target_lra_lu,
                target_true_peak=render_true_peak_target,
            )
            normalization_resolution = render_loudness_normalized(
                pre_loudness_wav,
                final_wav,
                target_lufs=profile.target_lufs,
                target_lra=profile.target_lra_lu,
                target_true_peak=render_true_peak_target,
                measurement=pre_program,
                sample_rate=sample_rate,
            )
            program_operation["resolution"] = normalization_resolution
            program_operation["measured_input_after_source_compensation"] = _round_loudness(
                pre_program
            )
            simulated_program = measure_loudness(
                final_wav,
                target_lufs=profile.target_lufs,
                target_lra=profile.target_lra_lu,
                target_true_peak=profile.target_true_peak_dbtp,
            )
            simulated_audio, simulated_sample_rate = decode_audio(final_wav)
            simulated_regional = regional_measurements(
                simulated_audio, simulated_sample_rate, analysis
            )
            source_verification_passes.append(
                {
                    "pass": verification_pass,
                    "corrections_db": {
                        key: round(value, 3) for key, value in sorted(corrections.items())
                    },
                    "resulting_regions": simulated_regional["machine_regions"],
                }
            )
        source_stage_report["downstream_verification_passes"] = source_verification_passes
    simulated_peak_limit = -0.1 if stage in skipped_stages else profile.target_true_peak_dbtp
    if simulated_program["input_tp"] > simulated_peak_limit:
        raise PipelineError(
            "Predicted true-peak verification failed: "
            f"measured {simulated_program['input_tp']:.2f} dBTP, "
            f"limit {simulated_peak_limit:.2f} dBTP"
        )

    return LoudnessRun(
        final_wav=final_wav,
        pre_program=pre_program,
        simulated_program=simulated_program,
        simulated_regional=simulated_regional,
        program_operation=program_operation,
        simulated_peak_limit=simulated_peak_limit,
    )
