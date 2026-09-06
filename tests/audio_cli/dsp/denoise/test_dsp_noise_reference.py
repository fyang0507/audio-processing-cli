"""Stationarity requires consistent spectra and channels before reference pooling."""

import numpy as np
import pytest
from scipy import signal

from audio_cli.dsp import analyze_signal, apply_environment_cleanup
from audio_cli.dsp.denoise import apply_broadband_denoise
from audio_cli.profiles import PROFILES
from audio_cli.vad_contract import SpeechRegion

RATE = 16_000
PROFILE = PROFILES["product-demo"]
SPEECH = [SpeechRegion(1, 3, 0.9, 1)]


def reference_fixture(kind, change_time=2):
    """Equal-RMS regimes, including the independent reviewer's seeded reproduction."""
    time = np.arange(4 * RATE) / RATE
    white = np.random.default_rng(612).normal(size=time.size)
    low = signal.sosfilt(signal.butter(4, 1200, fs=RATE, output="sos"), white)
    high = signal.sosfilt(signal.butter(4, 3500, btype="highpass", fs=RATE, output="sos"), white)
    if kind in {"broadband_change", "stationary_low", "stationary_high", "stationary_stereo"}:
        low += 0.3 * white
        high += 0.3 * white
    low *= 0.012 / np.std(low)
    high *= 0.012 / np.std(high)
    clean = 0.05 * np.sin(2 * np.pi * 220 * time) * ((time >= 1) & (time < 3))
    if kind == "channel_swap":
        noise = white * (0.012 / np.std(white))
        noise = np.column_stack(
            (
                noise * np.where(time < change_time, 1, 0.03),
                noise * np.where(time < change_time, 0.03, 1),
            )
        )
    elif kind == "stationary_stereo":
        noise = np.column_stack((low, high))
    elif kind in {"stationary_low", "stationary_high"}:
        noise = (low if kind == "stationary_low" else high)[:, None]
    else:
        noise = np.where(time < change_time, low, high)[:, None]
    return clean.astype(np.float32)[:, None], noise.astype(np.float32)


@pytest.mark.parametrize("kind", ["spectral_change", "broadband_change", "channel_swap"])
@pytest.mark.parametrize("change_time", [2.0, 0.4])
@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_incompatible_equal_level_references_are_rejected_before_pooling(kind, change_time, scale):
    clean, noise = reference_fixture(kind, change_time)
    audio = (clean + noise) * scale
    analysis = analyze_signal(audio, RATE, SPEECH, PROFILE)
    assert not analysis.machine_regions
    output, component, operation = apply_broadband_denoise(audio, RATE, PROFILE, analysis, [(1, 3)])
    # The old estimator's level-only check accepts all these references. Returning
    # the canonical samples is the protection, not merely a renamed component.
    np.testing.assert_array_equal(output, audio)
    assert component["status"] == "abstained"
    assert component["reason"] == "noise_reference_is_nonstationary"
    assert operation is None


def test_environment_cleanup_does_not_apply_broadband_to_the_reviewer_reproduction():
    clean, noise = reference_fixture("spectral_change")
    audio = clean + noise
    analysis = analyze_signal(audio, RATE, SPEECH, PROFILE)
    output, stage = apply_environment_cleanup(audio, RATE, PROFILE, analysis)
    assert output.shape == audio.shape
    assert stage["component_evaluations"][0]["status"] == "abstained"
    assert stage["component_evaluations"][0]["reason"] == "noise_reference_is_nonstationary"
    assert not any(o["type"] == "bounded-spectral-denoise" for o in stage["operations"])


def test_two_individually_usable_broadband_regimes_cannot_be_pooled():
    clean, noise = reference_fixture("broadband_change")
    audio = clean + noise
    analysis = analyze_signal(audio, RATE, SPEECH, PROFILE)
    for exclusions in ([(1, 4)], [(0, 3)]):
        output, component, operation = apply_broadband_denoise(
            audio, RATE, PROFILE, analysis, exclusions
        )
        assert not np.array_equal(output, audio)
        assert component["status"] == "applied"
        assert operation["noise_reference_minimum_block_flatness"] > 0.15
    output, component, operation = apply_broadband_denoise(audio, RATE, PROFILE, analysis, [(1, 3)])
    np.testing.assert_array_equal(output, audio)
    assert component["reason"] == "noise_reference_is_nonstationary"
    assert operation is None


@pytest.mark.parametrize("kind", ["stationary_low", "stationary_high", "stationary_stereo"])
@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_stationary_colored_noise_remains_eligible_with_speech_preservation(kind, scale):
    clean, noise = reference_fixture(kind)
    audio = (clean + noise) * scale
    analysis = analyze_signal(audio, RATE, SPEECH, PROFILE)
    output, component, operation = apply_broadband_denoise(audio, RATE, PROFILE, analysis, [(1, 3)])
    speech = slice(RATE, 3 * RATE)
    reference = clean[speech] * scale
    before_error = np.mean((audio[speech].astype(float) - reference) ** 2)
    after_error = np.mean((output[speech].astype(float) - reference) ** 2)
    assert 10 * np.log10(before_error / after_error) > 1.0
    projection = np.sum(output[speech] * reference, axis=0) / np.sum(reference**2)
    assert np.max(np.abs(20 * np.log10(projection))) < 1.0
    assert component["status"] == "applied"
    assert operation["maximum_reduction_db"] == 6
    assert output.shape == audio.shape
