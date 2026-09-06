"""Report unresolved component decisions and fixed-source regional targets."""

from __future__ import annotations

import math

from ..profiles import Profile
from .models import StageRun

REGION_BASIS = {
    "timeline": "source",
    "detection": "original_source",
    "regional_measurements": "fixed_source_regions",
    "region_ids": "report_local",
}


def actual_program(measurement: dict[str, float]) -> dict[str, float]:
    """Expose measured input_* values; retain legacy raw diagnostics separately.

    FFmpeg loudnorm's output_* describes its hypothetical normalization pass,
    even when invoked only to measure an encoded file. Nonfinite/unavailable
    actual measurements stay absent, rather than inventing a numeric value.
    """
    names = {
        "input_i": "integrated_loudness_lufs",
        "input_lra": "loudness_range_lu",
        "input_tp": "true_peak_dbtp",
    }
    return {
        name: round(measurement[key], 3)
        for key, name in names.items()
        if key in measurement and math.isfinite(measurement[key])
    }


def source_region_evaluations(
    profile: Profile, staged: StageRun, regional: dict[str, object]
) -> list[dict[str, object]]:
    """Compare the same source intervals after processing, without re-detection."""
    stage = staged.source_stage_report
    if not (
        stage["status"] == "applied"
        or stage.get("reason")
        in {"all_machine_regions_inside_target", "speech_and_machine_audio_overlap"}
    ):
        return []
    operations = {str(o["region_id"]): o for o in stage.get("operations", []) if "region_id" in o}
    evaluations = []
    for measured in regional["machine_regions"]:
        region_id = str(measured["region_id"])
        difference = float(measured["difference_from_speech_db"])
        if region_id in staged.abstained_source_region_ids:
            status = "abstained_overlap"
        elif (
            profile.machine_relative_minimum_lu - 0.25
            <= difference
            <= profile.machine_relative_maximum_lu + 0.25
        ):
            status = "inside_target"
        else:
            gain = float(operations.get(region_id, {}).get("resolved_gain_db", 0))
            bounded = (
                abs(gain - profile.machine_max_boost_db) <= 0.05
                or abs(gain + profile.machine_max_attenuation_db) <= 0.05
            )
            status = "bounded_outside_target" if bounded else "outside_target"
        evaluations.append(
            {
                "region_id": region_id,
                "difference_from_speech_db": round(difference, 3),
                "status": status,
            }
        )
    return evaluations


def unresolved_outcomes(
    profile: Profile,
    staged: StageRun,
    program: dict[str, float],
    regional: dict[str, object],
    *,
    measured_at: str,
) -> list[dict[str, object]]:
    """An operation being applied does not assert that every goal was achieved.

    This ledger records explicit abstentions and unmet measured targets. It is
    not an assertion about perceptual quality or undetected signal conditions.
    """
    unresolved: list[dict[str, object]] = []
    for stage in staged.stages:
        name = str(stage["name"])
        details = []
        for component in stage.get("component_evaluations", []):
            if component["component"] == "loudness-range":
                # Use the final measured LRA below, not the pre-normalization value.
                continue
            if component["status"] in {
                "abstained",
                "failed",
                "bounded_outside_target",
                "outside_target",
            }:
                details.append({"stage": name, **component})
        for region_id in stage.get("abstained_regions", []):
            details.append(
                {
                    "stage": name,
                    "region_id": region_id,
                    "status": "abstained_overlap",
                    "reason": "speech_and_machine_audio_overlap",
                }
            )
        if not details and stage["status"] in {"abstained", "failed"}:
            details.append({"stage": name, "status": stage["status"], "reason": stage["reason"]})
        unresolved.extend(details)
        if (
            name == "program-loudness"
            and stage["status"] not in {"skipped", "failed"}
            and profile.program_loudness_enabled
        ):
            lra = program.get("input_lra")
            if lra is not None and math.isfinite(lra) and lra > profile.target_lra_lu:
                unresolved.append(
                    {
                        "stage": name,
                        "component": "loudness-range",
                        "status": "abstained",
                        "reason": "dynamic_lra_control_would_change_relative_region_balance",
                        "measured_lra_lu": round(lra, 3),
                        "target_maximum_lra_lu": profile.target_lra_lu,
                        "measured_at": measured_at,
                    }
                )
    for evaluation in source_region_evaluations(profile, staged, regional):
        if evaluation["status"] in {"bounded_outside_target", "outside_target"}:
            unresolved.append(
                {
                    "stage": "source-balance",
                    **evaluation,
                    "reason": "correction_bound_reached"
                    if evaluation["status"] == "bounded_outside_target"
                    else "region_outside_target",
                    "measured_at": measured_at,
                    "target_difference_db": {
                        "minimum": profile.machine_relative_minimum_lu,
                        "maximum": profile.machine_relative_maximum_lu,
                    },
                }
            )
    return unresolved
