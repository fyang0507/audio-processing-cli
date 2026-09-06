"""Channel-linked bounded spectral subtraction with normalized overlap-add."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d


def filter_noise(
    audio: np.ndarray,
    noise_power: np.ndarray,
    window: np.ndarray,
    maximum_reduction_db: float,
) -> tuple[np.ndarray, float]:
    """Apply real shared gains, retaining each channel's complex spectral phase.

    G = sqrt(max(0, 1 - N/P)), floored at the profile's attenuation bound.
    Use the largest P/N across channels to protect speech present on either side.
    Local maxima protect harmonics/onsets; short averages reduce musical noise.
    Hann analysis/synthesis windows and sum-of-squared-window normalization follow
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html .
    A four-frame halo makes smoothing independent of the 256-frame work batches.
    """
    frame = window.size
    hop = frame // 4
    half = frame // 2
    tail = (-len(audio)) % hop
    padded = np.pad(audio.astype(np.float64), ((half, half + tail), (0, 0)), mode="reflect")
    starts = np.arange(0, len(padded) - frame + 1, hop)
    output = np.zeros_like(padded)
    weight = np.zeros(len(padded))
    floor = 10 ** (-maximum_reduction_db / 20)
    power_floor = max(float(np.max(noise_power)) * 1e-12, 1e-30)
    maximum_applied = 0.0
    for first in range(0, starts.size, 256):
        last = min(first + 256, starts.size)
        lo, hi = max(0, first - 4), min(starts.size, last + 4)
        frames = padded[starts[lo:hi, None] + np.arange(frame)[None, :]]
        spectra = np.fft.rfft(frames * window[None, :, None], axis=1)
        observed = uniform_filter1d(np.abs(spectra) ** 2, size=3, axis=0, mode="nearest")
        ratio = np.max(observed / np.maximum(noise_power, power_floor)[None, :, :], axis=2)
        gain = np.sqrt(np.maximum(0, 1 - 1 / np.maximum(ratio, 1e-12)))
        gain = np.clip(gain, floor, 1)
        gain = maximum_filter1d(gain, size=3, axis=1, mode="nearest")
        gain = maximum_filter1d(gain, size=3, axis=0, mode="nearest")
        gain = uniform_filter1d(gain, size=5, axis=0, mode="nearest")
        gain = uniform_filter1d(gain, size=3, axis=1, mode="nearest")
        selected = slice(first - lo, last - lo)
        gain = gain[selected]
        maximum_applied = max(maximum_applied, float(-20 * np.log10(np.min(gain))))
        restored = np.fft.irfft(spectra[selected] * gain[:, :, None], n=frame, axis=1)
        restored *= window[None, :, None]
        for offset, start in enumerate(starts[first:last]):
            output[start : start + frame] += restored[offset]
            weight[start : start + frame] += window**2
    valid = slice(half, half + len(audio))
    return (output[valid] / weight[valid, None]).astype(np.float32), maximum_applied
