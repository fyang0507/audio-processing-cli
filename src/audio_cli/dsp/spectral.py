"""Speech-scoped cleanup and frequency-domain adjustments."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from ..adjustments import GainAdjustment
from ..profiles import Profile
from .regions import (
    SignalAnalysis,
    resolve_speech_treatment_intervals,
    smooth_time_mask,
)
from .treatment import _blend


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


__all__ = ["apply_environment_cleanup", "apply_frequency_adjustments"]
