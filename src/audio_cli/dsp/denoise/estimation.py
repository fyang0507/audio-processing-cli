"""Conservative noise evidence from guarded, stationary non-program intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..regions import SignalAnalysis
from .reference import reference_consistency


@dataclass(frozen=True)
class NoiseEstimate:
    power: np.ndarray
    details: dict[str, object]


def estimate_noise(
    audio: np.ndarray,
    sample_rate: int,
    analysis: SignalAnalysis,
    intervals: list[tuple[float, float]],
    window: np.ndarray,
) -> tuple[NoiseEstimate | None, dict[str, object]]:
    """Require >=250 ms contiguous evidence, away from speech and salient program.

    VAD is not proof of noise. Reject changing or strongly tonal candidates, and
    require speech/noise contrast independently of absolute recording gain.
    The estimate is deliberately not inferred from speech-frame minima.
    """
    frame = window.size
    hop = frame // 4
    starts = np.arange(0, max(0, len(audio) - frame + 1), hop)
    eligible = np.ones(starts.size, dtype=bool)
    guard = round(0.12 * sample_rate)
    excluded = [*intervals, *((r.start, r.end) for r in analysis.machine_regions)]
    for start, end in excluded:
        eligible &= (starts + frame <= round(start * sample_rate) - guard) | (
            starts >= round(end * sample_rate) + guard
        )
    indices = np.flatnonzero(eligible)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    accepted = [r for r in runs if r.size and (r.size - 1) * hop + frame >= 0.25 * sample_rate]
    refusal: dict[str, object] = {"status": "abstained", "reason": "no_reliable_noise_only_region"}
    if not accepted:
        return None, refusal
    candidates = np.concatenate(accepted)
    # Uniform, deterministic sampling bounds spectral-estimation memory on long media.
    chosen = candidates[np.linspace(0, candidates.size - 1, min(256, candidates.size), dtype=int)]
    frames = audio[starts[chosen, None] + np.arange(frame)[None, :]].astype(np.float64)
    levels = np.mean(frames**2, axis=(1, 2))
    if float(np.max(levels)) < 1e-20:
        return None, {"status": "no_op", "reason": "noise_reference_is_silent"}
    floor = max(float(np.max(levels)) * 1e-12, 1e-30)
    level_db = 10 * np.log10(np.maximum(levels, floor))
    spread = float(np.percentile(level_db, 90) - np.percentile(level_db, 10))
    if spread > 6.0:
        return None, {"status": "abstained", "reason": "noise_reference_is_nonstationary"}
    reason, consistency = reference_consistency(audio, starts, accepted, window, sample_rate)
    if reason is not None:
        return None, {"status": "abstained", "reason": reason, **consistency}
    spectra = np.fft.rfft(frames * window[None, :, None], axis=1)
    power = np.median(np.abs(spectra) ** 2, axis=0) / np.log(2.0)
    power = uniform_filter1d(power, size=3, axis=0, mode="nearest")
    frequencies = np.fft.rfftfreq(frame, 1 / sample_rate)
    band = (frequencies >= 200) & (frequencies <= min(8000, sample_rate / 2))
    # A loud noise-only channel must not hide tonal material on the other side.
    # Ignore silent channels, but evaluate every channel carrying reference energy.
    active = np.mean(frames**2, axis=(0, 1)) > floor
    band_power = power[band][:, active]
    if not band_power.size or float(np.min(np.mean(band_power, axis=0))) <= 1e-30:
        return None, {"status": "abstained", "reason": "noise_reference_has_no_broadband_energy"}
    spectral_floor = np.maximum(np.max(band_power, axis=0) * 1e-12, 1e-30)
    flatness = float(
        np.min(
            np.exp(np.mean(np.log(np.maximum(band_power, spectral_floor)), axis=0))
            / np.mean(band_power, axis=0)
        )
    )
    if flatness < 0.15:
        return None, {"status": "abstained", "reason": "noise_reference_is_tonal"}
    speech_starts = []
    for region in analysis.speech_regions:
        selected = starts[
            (starts >= region.start * sample_rate) & (starts + frame <= region.end * sample_rate)
        ]
        if selected.size:
            speech_starts.append(selected)
    if not speech_starts:
        return None, {"status": "abstained", "reason": "insufficient_speech_for_noise_comparison"}
    speech_candidates = np.concatenate(speech_starts)
    speech_chosen = speech_candidates[
        np.linspace(0, speech_candidates.size - 1, min(256, speech_candidates.size), dtype=int)
    ]
    speech_frames = audio[speech_chosen[:, None] + np.arange(frame)[None, :]].astype(np.float64)
    contrast = 10 * np.log10(max(float(np.mean(speech_frames**2)), floor) / np.median(levels))
    if contrast < 3.0:
        return None, {"status": "abstained", "reason": "insufficient_speech_noise_contrast"}
    if contrast > 20.0:
        return None, {"status": "no_op", "reason": "stationary_noise_below_threshold"}
    return NoiseEstimate(
        power,
        {
            **consistency,
            "noise_reference_frame_count": int(chosen.size),
            "noise_reference_seconds": round(
                sum((r.size - 1) * hop + frame for r in accepted) / sample_rate, 6
            ),
            "noise_reference_rms_dbfs": round(float(10 * np.log10(np.median(levels))), 3),
            "speech_noise_contrast_db": round(float(contrast), 3),
            "noise_reference_level_spread_db": round(spread, 3),
            "noise_reference_spectral_flatness": round(flatness, 6),
            "noise_reference_guard_ms": 120,
            "noise_reference_minimum_contiguous_ms": 250,
        },
    ), {}
