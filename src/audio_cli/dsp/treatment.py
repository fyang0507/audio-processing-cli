"""Channel and time-scoped gain treatments."""

from __future__ import annotations

import numpy as np

from ..adjustments import GainAdjustment
from ..profiles import Profile
from .levels import rms_dbfs
from .regions import SignalAnalysis, _speech_samples, _time_scope, smooth_time_mask


def apply_channel_balance(
    audio: np.ndarray,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[np.ndarray, dict[str, object]]:
    if audio.shape[1] == 1:
        return audio, {"status": "no_op", "reason": "mono_source", "operations": []}
    difference = analysis.channel_difference_db
    if abs(difference) <= profile.channel_no_op_db:
        return audio, {
            "status": "no_op",
            "reason": "channel_difference_below_threshold",
            "operations": [],
        }
    if (
        analysis.channel_correlation is None
        or analysis.channel_correlation < profile.channel_correlation_minimum
    ):
        return audio, {
            "status": "abstained",
            "reason": "stereo_difference_may_be_intentional",
            "operations": [],
            "observed_correlation": analysis.channel_correlation,
        }
    half = float(
        np.clip(
            difference / 2.0,
            -profile.channel_max_correction_db,
            profile.channel_max_correction_db,
        )
    )
    gains_db = [-half, half]
    gains = np.power(10.0, np.asarray(gains_db, dtype=np.float64) / 20.0)
    output = audio * gains[None, :]
    return output.astype(np.float32), {
        "status": "applied",
        "reason": "correlated_channels_with_level_mismatch",
        "operations": [
            {
                "type": "linked-channel-gain",
                "left_gain_db": round(gains_db[0], 3),
                "right_gain_db": round(gains_db[1], 3),
                "maximum_correction_db": profile.channel_max_correction_db,
            }
        ],
    }


def _blend(original: np.ndarray, processed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shaped = mask[:, None]
    return (original * (1.0 - shaped) + processed * shaped).astype(np.float32)


def apply_source_balance(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[np.ndarray, dict[str, object]]:
    if not analysis.speech_regions:
        return audio, {
            "status": "abstained",
            "reason": "no_speech_reference",
            "operations": [],
        }
    if not analysis.machine_regions:
        return audio, {
            "status": "no_op",
            "reason": "no_non_speech_program_regions",
            "operations": [],
        }
    speech = _speech_samples(audio, analysis.speech_regions, sample_rate)
    speech_reference = rms_dbfs(speech)
    output = audio.copy()
    operations: list[dict[str, object]] = []
    inside_target: list[str] = []
    abstained: list[str] = []
    for region in analysis.machine_regions:
        start = max(0, round(region.start * sample_rate))
        end = min(output.shape[0], round(region.end * sample_rate))
        if region.overlaps_speech:
            abstained.append(region.region_id)
            continue
        measured = rms_dbfs(output[start:end])
        difference = measured - speech_reference
        if (
            profile.machine_relative_minimum_lu
            <= difference
            <= profile.machine_relative_maximum_lu
        ):
            inside_target.append(region.region_id)
            continue
        requested = profile.machine_relative_target_lu - difference
        resolved_gain = float(
            np.clip(
                requested,
                -profile.machine_max_attenuation_db,
                profile.machine_max_boost_db,
            )
        )
        mask = smooth_time_mask(
            output.shape[0],
            [(region.start, region.end)],
            sample_rate,
            profile.region_fade_ms,
        )
        gained = output * (10.0 ** (resolved_gain / 20.0))
        output = _blend(output, gained, mask)
        operations.append(
            {
                "type": "regional-full-band-gain",
                "region_id": region.region_id,
                "scope": _time_scope(region.start, region.end),
                "reference_speech_rms_dbfs": round(speech_reference, 3),
                "measured_before_rms_dbfs": round(measured, 3),
                "observed_difference_db": round(difference, 3),
                "target_difference_db": profile.machine_relative_target_lu,
                "requested_gain_db": round(requested, 3),
                "resolved_gain_db": round(resolved_gain, 3),
                "boundary_fade_ms": profile.region_fade_ms,
            }
        )
    if operations:
        status, reason = (
            "applied",
            "non_overlapping_regions_balanced_to_speech_reference",
        )
    elif abstained:
        status, reason = "abstained", "speech_and_machine_audio_overlap"
    else:
        status, reason = "no_op", "all_machine_regions_inside_target"
    return output, {
        "status": status,
        "reason": reason,
        "operations": operations,
        "inside_target_regions": inside_target,
        "abstained_regions": abstained,
    }


def apply_fullband_adjustments(
    audio: np.ndarray,
    sample_rate: int,
    adjustments: list[GainAdjustment],
    duration: float,
    fade_ms: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = audio.copy()
    resolved: list[dict[str, object]] = []
    for adjustment in adjustments:
        if not adjustment.is_full_band:
            continue
        end = adjustment.resolved_end(duration)
        mask = smooth_time_mask(
            output.shape[0], [(adjustment.start, end)], sample_rate, fade_ms
        )
        gained = output * (10.0 ** (adjustment.gain_db / 20.0))
        output = _blend(output, gained, mask)
        item = adjustment.as_dict(duration)
        item["resolved_transition"] = {"boundary_fade_ms": fade_ms}
        resolved.append(item)
    return output, resolved


def apply_machine_region_corrections(
    audio: np.ndarray,
    sample_rate: int,
    analysis: SignalAnalysis,
    corrections_db: dict[str, float],
    fade_ms: int,
) -> np.ndarray:
    output = audio.copy()
    by_id = {region.region_id: region for region in analysis.machine_regions}
    for region_id, gain_db in corrections_db.items():
        region = by_id[region_id]
        mask = smooth_time_mask(
            output.shape[0],
            [(region.start, region.end)],
            sample_rate,
            fade_ms,
        )
        gained = output * (10.0 ** (gain_db / 20.0))
        output = _blend(output, gained, mask)
    return output


__all__ = [
    "apply_channel_balance",
    "apply_fullband_adjustments",
    "apply_machine_region_corrections",
    "apply_source_balance",
]
