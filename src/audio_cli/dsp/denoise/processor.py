"""Bounded broadband suppression; pure arrays and fixed source-region evidence."""

from __future__ import annotations

import numpy as np
from scipy.signal import windows

from ...profiles import Profile
from ..regions import SignalAnalysis
from .estimation import estimate_noise
from .filtering import filter_noise


def apply_broadband_denoise(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
    intervals: list[tuple[float, float]],
) -> tuple[np.ndarray, dict[str, object], dict[str, object] | None]:
    """Resolve evidence and a channel-linked filter; caller applies speech scoping."""
    component: dict[str, object] = {"component": "broadband-denoise"}
    # Multiples of four permit exact quarter-window hops at arbitrary rates.
    frame = max(16, 4 * round(sample_rate * 0.032 / 4))
    if len(audio) < frame:
        return audio, {**component, "status": "abstained", "reason": "input_too_short"}, None
    window = windows.hann(frame, sym=False)
    estimate, refusal = estimate_noise(audio, sample_rate, analysis, intervals, window)
    if estimate is None:
        return audio, {**component, **refusal}, None
    output, maximum_applied = filter_noise(
        audio, estimate.power, window, profile.broadband_max_reduction_db
    )
    if maximum_applied < 0.05 or np.array_equal(output, audio):
        return (
            audio,
            {**component, "status": "no_op", "reason": "noise_reduction_below_threshold"},
            None,
        )
    operation: dict[str, object] = {
        "type": "bounded-spectral-denoise",
        "algorithm": "linked-spectral-subtraction-v1",
        "affected_scope": "speech_regions",
        "frame_samples": frame,
        "hop_samples": frame // 4,
        "window": "periodic-hann",
        "maximum_reduction_db": profile.broadband_max_reduction_db,
        "maximum_spectral_reduction_db": round(maximum_applied, 6),
        "channel_link": "maximum_signal_to_noise_ratio",
        "power_smoothing_frames": 3,
        "gain_smoothing_frames": 5,
        "gain_smoothing_bins": 3,
        **estimate.details,
    }
    return (
        output,
        {**component, "status": "applied", "reason": "stationary_noise_profile_resolved"},
        operation,
    )


__all__ = ["apply_broadband_denoise"]
