"""Deterministic enhancement stages before program loudness normalization."""

from __future__ import annotations

from ..adjustments import GainAdjustment
from ..dsp import (
    apply_channel_balance,
    apply_environment_cleanup,
    apply_frequency_adjustments,
    apply_fullband_adjustments,
    apply_source_balance,
    apply_voice_enhancement,
)
from ..profiles import Profile
from .models import PreparedRun, StageRun


def _stage_result(
    stage: str,
    profile: Profile,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "name": stage,
        "profile_rule": f"{profile.name}.{stage}",
        **result,
    }


def _skipped_stage(stage: str, profile: Profile) -> dict[str, object]:
    return _stage_result(
        stage,
        profile,
        {"status": "skipped", "reason": "explicitly_skipped", "operations": []},
    )


def _disabled_stage(stage: str, profile: Profile) -> dict[str, object]:
    return _stage_result(
        stage,
        profile,
        {"status": "no_op", "reason": "disabled_by_profile", "operations": []},
    )


def process_stages(
    profile: Profile,
    skipped_stages: set[str],
    adjustments: list[GainAdjustment],
    prepared: PreparedRun,
) -> StageRun:
    """Apply every pre-loudness stage in the declared processing order."""
    current = prepared.audio
    sample_rate = prepared.sample_rate
    analysis = prepared.analysis
    stages: list[dict[str, object]] = []
    resolved_adjustments: list[dict[str, object]] = []

    stage = "channel-balance"
    if stage in skipped_stages:
        stages.append(_skipped_stage(stage, profile))
    elif not profile.stage_enabled(stage):
        stages.append(_disabled_stage(stage, profile))
    else:
        current, result = apply_channel_balance(current, profile, analysis)
        stages.append(_stage_result(stage, profile, result))

    stage = "environment-denoise"
    if stage in skipped_stages:
        stages.append(_skipped_stage(stage, profile))
    elif not profile.stage_enabled(stage):
        stages.append(_disabled_stage(stage, profile))
    else:
        current, result = apply_environment_cleanup(current, sample_rate, profile, analysis)
        stages.append(_stage_result(stage, profile, result))

    current, frequency_adjustments = apply_frequency_adjustments(
        current,
        sample_rate,
        adjustments,
        analysis.duration_seconds,
        profile.region_fade_ms,
    )
    resolved_adjustments.extend(frequency_adjustments)

    stage = "voice-enhance"
    if stage in skipped_stages:
        stages.append(_skipped_stage(stage, profile))
    elif not profile.stage_enabled(stage):
        stages.append(_disabled_stage(stage, profile))
    else:
        current, result = apply_voice_enhancement(current, sample_rate, profile, analysis)
        stages.append(_stage_result(stage, profile, result))

    stage = "source-balance"
    if stage in skipped_stages:
        source_stage_report = _skipped_stage(stage, profile)
        stages.append(source_stage_report)
    elif not profile.stage_enabled(stage):
        source_stage_report = _disabled_stage(stage, profile)
        stages.append(source_stage_report)
    else:
        current, result = apply_source_balance(current, sample_rate, profile, analysis)
        source_stage_report = _stage_result(stage, profile, result)
        stages.append(source_stage_report)
    raw_abstained_regions = source_stage_report.get("abstained_regions", [])
    abstained_source_region_ids = (
        {str(region_id) for region_id in raw_abstained_regions}
        if isinstance(raw_abstained_regions, list)
        else set()
    )

    current, fullband_adjustments = apply_fullband_adjustments(
        current,
        sample_rate,
        adjustments,
        analysis.duration_seconds,
        profile.region_fade_ms,
    )
    resolved_adjustments.extend(fullband_adjustments)
    resolved_adjustments.sort(key=lambda item: str(item["adjustment_id"]))
    return StageRun(
        current=current,
        stages=stages,
        resolved_adjustments=resolved_adjustments,
        source_stage_report=source_stage_report,
        abstained_source_region_ids=abstained_source_region_ids,
    )
