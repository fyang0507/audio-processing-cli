"""Validate reference spectra over time before selecting a stationary model."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.ndimage import uniform_filter1d


def reference_consistency(
    audio: np.ndarray,
    starts: np.ndarray,
    runs: list[np.ndarray],
    window: np.ndarray,
    sample_rate: int,
) -> tuple[str | None, dict[str, object]]:
    """Inspect every eligible run in 250-500 ms blocks, using bounded memory.

    RMS alone cannot distinguish equal-level spectral regimes or channel swaps.
    Compare absolute octave-band power per channel across blocks, including DC
    and frequencies above 8 kHz when present. A fourfold (6 dB) power variation
    rejects the stationary model. Flatness must hold within each block/channel,
    never merely after pooling incompatible references. This is an evidence gate,
    not proof that an unseen background during speech is stationary too.
    """
    frame = window.size
    hop = frame // 4
    maximum_frames = max(1, int((0.5 * sample_rate - frame) // hop) + 1)
    frequencies = np.fft.rfftfreq(frame, 1 / sample_rate)
    edges = [0.0, 200.0]
    while edges[-1] < sample_rate / 2:
        edges.append(edges[-1] * 2)
    edges[-1] = float("inf")
    bands = [(frequencies >= low) & (frequencies < high) for low, high in pairwise(edges)]
    bands = [band for band in bands if np.any(band)]
    flatness_band = (frequencies >= 200) & (frequencies <= min(8000, sample_rate / 2))
    minimum_power = np.full((len(bands), audio.shape[1]), np.inf)
    maximum_power = np.zeros_like(minimum_power)
    minimum_level = np.full(audio.shape[1], np.inf)
    maximum_level = np.zeros(audio.shape[1])
    maximum_level_spread = 0.0
    minimum_flatness = 1.0
    block_count = 0
    for run in runs:
        count = (run.size + maximum_frames - 1) // maximum_frames
        for block in np.array_split(run, count):
            frames = audio[starts[block, None] + np.arange(frame)[None, :]].astype(np.float64)
            levels = np.mean(frames**2, axis=1)
            mean_level = np.mean(levels, axis=0)
            minimum_level = np.minimum(minimum_level, mean_level)
            maximum_level = np.maximum(maximum_level, mean_level)
            level_db = 10 * np.log10(np.maximum(levels, 1e-30))
            spread = np.percentile(level_db, 90, axis=0) - np.percentile(level_db, 10, axis=0)
            maximum_level_spread = max(maximum_level_spread, float(np.max(spread)))
            spectra = np.fft.rfft(frames * window[None, :, None], axis=1)
            power = np.mean(np.abs(spectra) ** 2, axis=0)
            band_power = np.stack([np.mean(power[band], axis=0) for band in bands])
            minimum_power = np.minimum(minimum_power, band_power)
            maximum_power = np.maximum(maximum_power, band_power)
            # Truly silent channels do not invalidate an otherwise useful reference.
            # A channel that goes silent only in some blocks still fails the power check.
            active = mean_level > 1e-30
            smooth = uniform_filter1d(power, size=3, axis=0, mode="nearest")
            selected = smooth[flatness_band][:, active]
            if np.any(active):
                flatness = 0.0
                if selected.size and np.all(np.mean(selected, axis=0) > 1e-30):
                    floor = np.maximum(np.max(selected, axis=0) * 1e-12, 1e-30)
                    flatness = float(
                        np.min(
                            np.exp(np.mean(np.log(np.maximum(selected, floor)), axis=0))
                            / np.mean(selected, axis=0)
                        )
                    )
                minimum_flatness = min(minimum_flatness, flatness)
            block_count += 1
    spectral_spread = float(
        np.max(10 * np.log10(np.maximum(maximum_power, 1e-30) / np.maximum(minimum_power, 1e-30)))
    )
    maximum_level_spread = max(
        maximum_level_spread,
        float(
            np.max(
                10 * np.log10(np.maximum(maximum_level, 1e-30) / np.maximum(minimum_level, 1e-30))
            )
        ),
    )
    details: dict[str, object] = {
        "noise_reference_block_count": block_count,
        "noise_reference_maximum_band_spread_db": round(spectral_spread, 3),
        "noise_reference_maximum_channel_level_spread_db": round(maximum_level_spread, 3),
        "noise_reference_minimum_block_flatness": round(minimum_flatness, 6),
    }
    reason = None
    if max(spectral_spread, maximum_level_spread) > 6.0:
        reason = "noise_reference_is_nonstationary"
    elif minimum_flatness < 0.15:
        reason = "noise_reference_is_tonal"
    return reason, details
