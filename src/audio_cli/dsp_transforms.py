"""Deterministic signal transforms applied by the enhancement pipeline."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .adjustments import GainAdjustment
from .dsp_levels import EPSILON, peak_dbfs, rms_dbfs
from .dsp_regions import (
    SignalAnalysis,
    _hard_region_mask,
    _speech_samples,
    _time_scope,
    resolve_speech_treatment_intervals,
    smooth_time_mask,
)
from .profiles import Profile


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


def _notch(
    audio: np.ndarray, frequency: float, sample_rate: int, quality: float = 35.0
) -> np.ndarray:
    b, a = signal.iirnotch(frequency, quality, fs=sample_rate)
    return signal.lfilter(b, a, audio, axis=0).astype(np.float32)


def apply_environment_cleanup(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[np.ndarray, dict[str, object]]:
    if not analysis.speech_regions:
        return audio, {
            "status": "abstained",
            "reason": "no_speech_detected",
            "operations": [],
        }
    intervals, resolved_transition = resolve_speech_treatment_intervals(
        audio, sample_rate, profile, analysis
    )
    mask = smooth_time_mask(
        audio.shape[0],
        intervals,
        sample_rate,
        profile.region_fade_ms,
        transition_placement=profile.speech_transition_placement,
    )
    processed = audio.astype(np.float64)
    operations: list[dict[str, object]] = []
    if (
        analysis.subbass_power_ratio >= profile.subbass_ratio_threshold
        or abs(analysis.dc_offset) > 1e-4
    ):
        sos = signal.butter(
            2, profile.highpass_hz, btype="highpass", fs=sample_rate, output="sos"
        )
        processed = signal.sosfilt(sos, processed, axis=0)
        operations.append(
            {
                "type": "minimum-phase-highpass",
                "cutoff_hz": profile.highpass_hz,
                "order": 2,
                "affected_scope": "speech_regions",
            }
        )
    if analysis.hum_excess_db >= profile.hum_excess_db_threshold:
        for frequency in (60.0, 120.0, 180.0):
            processed = _notch(processed, frequency, sample_rate)
        operations.append(
            {
                "type": "minimum-phase-dehum",
                "frequencies_hz": [60.0, 120.0, 180.0],
                "quality_factor": 35.0,
                "affected_scope": "speech_regions",
            }
        )
    broadband: dict[str, object]
    if analysis.speech_rms_dbfs < -45.0:
        broadband = {
            "component": "broadband-denoise",
            "status": "abstained",
            "reason": "input_speech_too_quiet_for_reliable_noise_estimate",
        }
    elif analysis.noise_floor_dbfs <= analysis.speech_rms_dbfs - 20.0:
        broadband = {
            "component": "broadband-denoise",
            "status": "no_op",
            "reason": "stationary_noise_below_threshold",
        }
    else:
        broadband = {
            "component": "broadband-denoise",
            "status": "abstained",
            "reason": "conservative_v1_has_no_reliable_stationary_noise_profile",
        }
    output = (
        _blend(audio, np.asarray(processed, dtype=np.float32), mask)
        if operations
        else audio
    )
    if operations:
        status, reason = "applied", "eligible_environmental_cleanup_resolved"
    elif broadband["status"] == "abstained":
        status, reason = "abstained", str(broadband["reason"])
    else:
        status, reason = "no_op", "environmental_components_below_threshold"
    return output, {
        "status": status,
        "reason": reason,
        "operations": operations,
        "resolved_transition": resolved_transition if operations else None,
        "component_evaluations": [broadband],
    }


def _peaking_coefficients(
    gain_db: float,
    center_hz: float,
    quality: float,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * math.pi * center_hz / sample_rate
    alpha = math.sin(omega) / (2.0 * quality)
    cos_omega = math.cos(omega)
    b = np.asarray(
        [1 + alpha * amplitude, -2 * cos_omega, 1 - alpha * amplitude],
        dtype=np.float64,
    )
    a = np.asarray(
        [1 + alpha / amplitude, -2 * cos_omega, 1 - alpha / amplitude],
        dtype=np.float64,
    )
    return b / a[0], a / a[0]


def apply_frequency_adjustments(
    audio: np.ndarray,
    sample_rate: int,
    adjustments: list[GainAdjustment],
    duration: float,
    fade_ms: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = audio.copy()
    resolved: list[dict[str, object]] = []
    for adjustment in adjustments:
        if adjustment.is_full_band:
            continue
        assert adjustment.frequency_low_hz is not None
        assert adjustment.frequency_high_hz is not None
        center = math.sqrt(adjustment.frequency_low_hz * adjustment.frequency_high_hz)
        bandwidth = adjustment.frequency_high_hz - adjustment.frequency_low_hz
        quality = max(0.25, center / bandwidth)
        b, a = _peaking_coefficients(adjustment.gain_db, center, quality, sample_rate)
        filtered = signal.lfilter(b, a, output, axis=0).astype(np.float32)
        end = adjustment.resolved_end(duration)
        mask = smooth_time_mask(
            output.shape[0], [(adjustment.start, end)], sample_rate, fade_ms
        )
        output = _blend(output, filtered, mask)
        item = adjustment.as_dict(duration)
        item["resolved_filter"] = {
            "kind": "minimum-phase-peaking-biquad",
            "center_hz": round(center, 3),
            "quality_factor": round(quality, 6),
            "boundary_fade_ms": fade_ms,
        }
        resolved.append(item)
    return output, resolved


def _compress(
    audio: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    ratio: float,
) -> tuple[np.ndarray, float]:
    frame = max(1, round(sample_rate * 0.02))
    count = math.ceil(audio.shape[0] / frame)
    pad = count * frame - audio.shape[0]
    padded = np.pad(audio, ((0, pad), (0, 0)))
    levels = 20.0 * np.log10(
        np.sqrt(
            np.mean(
                np.square(padded.reshape(count, frame, audio.shape[1])), axis=(1, 2)
            )
            + EPSILON
        )
    )
    over = np.maximum(levels - threshold_dbfs, 0.0)
    reduction_db = -(over - over / ratio)
    centers = np.minimum(np.arange(count) * frame + frame // 2, audio.shape[0] - 1)
    sample_positions = np.arange(audio.shape[0])
    interpolated = np.interp(
        sample_positions,
        centers,
        reduction_db,
        left=reduction_db[0],
        right=reduction_db[-1],
    )
    gains = np.power(10.0, interpolated / 20.0)
    return (audio * gains[:, None]).astype(np.float32), float(np.min(reduction_db))


def apply_voice_enhancement(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[np.ndarray, dict[str, object]]:
    if not analysis.speech_regions:
        return audio, {
            "status": "abstained",
            "reason": "no_speech_detected",
            "operations": [],
        }
    intervals, resolved_transition = resolve_speech_treatment_intervals(
        audio, sample_rate, profile, analysis
    )
    mask = smooth_time_mask(
        audio.shape[0],
        intervals,
        sample_rate,
        profile.region_fade_ms,
        transition_placement=profile.speech_transition_placement,
    )
    b, a = _peaking_coefficients(
        profile.voice_presence_gain_db, 3000.0, 0.9, sample_rate
    )
    tonal = signal.lfilter(b, a, audio, axis=0).astype(np.float32)
    speech_mask = _hard_region_mask(
        audio.shape[0], analysis.speech_regions, sample_rate
    )
    measured_before = rms_dbfs(tonal[speech_mask])
    desired_gain = profile.voice_target_rms_dbfs - measured_before
    gain_db = float(
        np.clip(
            desired_gain, -profile.voice_max_attenuation_db, profile.voice_max_gain_db
        )
    )
    leveled = tonal * (10.0 ** (gain_db / 20.0))
    compressed, maximum_reduction = _compress(
        leveled,
        sample_rate,
        profile.compressor_threshold_dbfs,
        profile.compressor_ratio,
    )
    output = _blend(audio, compressed, mask)
    measured_after = rms_dbfs(output[speech_mask])
    return output, {
        "status": "applied",
        "reason": "speech_regions_received_bounded_tonal_and_level_correction",
        "operations": [
            {
                "type": "speech-presence-eq",
                "center_hz": 3000.0,
                "gain_db": profile.voice_presence_gain_db,
                "quality_factor": 0.9,
                "filter": "minimum-phase-peaking-biquad",
            },
            {
                "type": "speech-leveling",
                "measured_before_rms_dbfs": round(measured_before, 3),
                "target_rms_dbfs": profile.voice_target_rms_dbfs,
                "resolved_gain_db": round(gain_db, 3),
                "maximum_boost_db": profile.voice_max_gain_db,
                "measured_after_rms_dbfs": round(measured_after, 3),
            },
            {
                "type": "speech-compression",
                "threshold_dbfs": profile.compressor_threshold_dbfs,
                "ratio": profile.compressor_ratio,
                "maximum_gain_reduction_db": round(maximum_reduction, 3),
            },
        ],
        "affected_regions": [
            f"speech_{index:03d}" for index in range(1, len(intervals) + 1)
        ],
        "resolved_transition": resolved_transition,
    }


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


def regional_measurements(
    audio: np.ndarray,
    sample_rate: int,
    analysis: SignalAnalysis,
) -> dict[str, object]:
    speech_level = rms_dbfs(
        _speech_samples(audio, analysis.speech_regions, sample_rate)
    )
    machines: list[dict[str, object]] = []
    for region in analysis.machine_regions:
        start = max(0, round(region.start * sample_rate))
        end = min(audio.shape[0], round(region.end * sample_rate))
        measured = rms_dbfs(audio[start:end])
        machines.append(
            {
                "region_id": region.region_id,
                "measured_rms_dbfs": round(measured, 3),
                "difference_from_speech_db": round(measured - speech_level, 3),
            }
        )
    return {
        "speech_program_rms_dbfs": round(speech_level, 3),
        "machine_regions": machines,
        "sample_peak_dbfs": round(peak_dbfs(audio), 3),
    }
