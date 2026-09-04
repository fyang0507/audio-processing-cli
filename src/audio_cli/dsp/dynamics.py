"""Speech-scoped leveling, tonal shaping, and compression."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from ..profiles import Profile
from .levels import EPSILON, rms_dbfs
from .regions import (
    SignalAnalysis,
    _hard_region_mask,
    resolve_speech_treatment_intervals,
    smooth_time_mask,
)
from .spectral import _peaking_coefficients
from .treatment import _blend


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


__all__ = ["apply_voice_enhancement"]
