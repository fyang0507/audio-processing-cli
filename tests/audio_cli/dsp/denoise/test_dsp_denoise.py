"""Signal evidence for bounded denoising; no models or user-media inference."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import signal

from audio_cli.dsp import analyze_signal, apply_environment_cleanup
from audio_cli.dsp.denoise import apply_broadband_denoise
from audio_cli.dsp.denoise.filtering import filter_noise
from audio_cli.profiles import PROFILES
from audio_cli.vad_contract import SpeechRegion

RATE = 16_000
PROFILE = PROFILES["product-demo"]


def speech_fixture(rate=RATE):
    """Voiced harmonics, syllable envelopes, and a band-limited unvoiced burst."""
    time = np.arange(4 * rate) / rate
    voiced = sum(
        np.sin(2 * np.pi * f * time) / (i + 1) for i, f in enumerate((220, 440, 660, 1100, 2200))
    )
    envelope = np.zeros_like(time)
    for start, end in ((1.0, 1.7), (2.15, 2.8)):
        inside = (time >= start) & (time < end)
        envelope[inside] = np.sin(np.pi * (time[inside] - start) / (end - start)) ** 0.5
    clean = 0.065 * voiced * envelope
    rng = np.random.default_rng(33)
    unvoiced = signal.sosfilt(
        signal.butter(3, [1800, 5000], btype="bandpass", fs=rate, output="sos"),
        rng.normal(size=time.size),
    )
    burst = (time >= 2.8) & (time < 3)
    clean[burst] += 0.055 * unvoiced[burst] * np.sin(np.pi * (time[burst] - 2.8) / 0.2)
    noise = rng.normal(0, 0.012, size=time.size)
    return clean.astype(np.float32)[:, None], noise.astype(np.float32)[:, None]


def run_broadband(audio, rate=RATE, speech=None):
    speech = speech if speech is not None else [SpeechRegion(1.0, 3.0, 0.9, 1.0)]
    analysis = analyze_signal(audio, rate, speech, PROFILE)
    return apply_broadband_denoise(
        audio, rate, PROFILE, analysis, [(r.start, r.end) for r in speech]
    )


def db_ratio(a, b):
    return 10 * np.log10(
        np.mean(np.square(a.astype(np.float64))) / np.mean(np.square(b.astype(np.float64)))
    )


@pytest.mark.parametrize("rate", [16_000, 44_100, 48_000])
def test_noise_attenuation_speech_preservation_and_timeline(rate):
    clean, noise = speech_fixture(rate)
    mixed = clean + noise
    snapshot = mixed.copy()
    output, component, operation = run_broadband(mixed, rate)
    speech = slice(round(1.05 * rate), round(2.95 * rate))
    pause = slice(round(1.8 * rate), round(2.0 * rate))
    # At least half the broadband noise power is removed, within a 6 dB bound.
    assert db_ratio(mixed[pause], output[pause]) > 3.0
    assert db_ratio(mixed[speech] - clean[speech], output[speech] - clean[speech]) > 2.0
    voiced = slice(round(1.15 * rate), round(1.6 * rate))
    projection = np.sum(output[voiced] * clean[voiced]) / np.sum(clean[voiced] ** 2)
    assert abs(20 * np.log10(projection)) < 1.0
    fricative = slice(round(2.83 * rate), round(2.97 * rate))
    assert abs(db_ratio(output[fricative], clean[fricative])) < 1.5
    assert output.shape == mixed.shape and output.dtype == np.float32
    np.testing.assert_array_equal(mixed, snapshot)
    assert (
        np.argmax(signal.correlate(output[voiced, 0], clean[voiced, 0], mode="full"))
        == len(clean[voiced]) - 1
    )
    repeated, _, _ = run_broadband(mixed, rate)
    np.testing.assert_array_equal(output, repeated)
    assert component["status"] == "applied"
    assert operation["type"] == "bounded-spectral-denoise"


def test_very_quiet_speech_is_preserved_when_relative_evidence_is_reliable():
    clean, noise = speech_fixture()
    loud, _, _ = run_broadband(clean + noise)
    quiet, component, _ = run_broadband((clean + noise) * 0.001)
    np.testing.assert_allclose(quiet / 0.001, loud, atol=3e-8)
    assert component["status"] == "applied"
    assert (
        db_ratio(noise[RATE : 3 * RATE], quiet[RATE : 3 * RATE] / 0.001 - clean[RATE : 3 * RATE])
        > 2
    )


@pytest.mark.parametrize("other_gain", [0.5, -0.75, -1.0, 0.0])
def test_linked_gain_preserves_channel_ratio_polarity_and_silent_channels(other_gain):
    clean, noise = speech_fixture()
    mixed = np.column_stack(((clean + noise)[:, 0], (clean + noise)[:, 0] * other_gain))
    output, _, _ = run_broadband(mixed)
    np.testing.assert_allclose(output[:, 1], other_gain * output[:, 0], atol=3e-8)
    assert db_ratio(mixed[29000:31000], output[29000:31000]) > 3.0


def test_speech_on_either_stereo_side_protects_the_same_linked_mask():
    clean, noise = speech_fixture()
    second_noise = np.random.default_rng(37).normal(0, 0.012, noise.shape)
    mixed = np.column_stack((noise[:, 0], (clean + second_noise)[:, 0])).astype(np.float32)
    output, _, _ = run_broadband(mixed)
    swapped, _, _ = run_broadband(mixed[:, ::-1])
    np.testing.assert_array_equal(output, swapped[:, ::-1])
    voiced = slice(19000, 25000)
    projection = np.sum(output[voiced, 1] * clean[voiced, 0]) / np.sum(clean[voiced] ** 2)
    assert abs(20 * np.log10(projection)) < 1.0


def test_unreliable_noise_evidence_never_uses_speech_as_noise():
    clean, noise = speech_fixture()
    mixed = clean + noise
    output, component, operation = run_broadband(mixed, speech=[SpeechRegion(0, 4, 0.9, 1)])
    np.testing.assert_array_equal(output, mixed)
    assert component["reason"] == "no_reliable_noise_only_region"
    assert operation is None


@pytest.mark.parametrize("kind", ["tonal", "changing", "no_contrast", "clean"])
def test_reference_refusals_leave_the_signal_untouched(kind):
    clean, noise = speech_fixture()
    if kind == "tonal":
        noise = (0.012 * np.sin(2 * np.pi * 800 * np.arange(len(clean)) / RATE))[:, None]
    elif kind == "changing":
        noise[: RATE // 2] *= 0.03
    elif kind == "no_contrast":
        clean *= 0.01
    else:
        noise *= 0.001
    mixed = (clean + noise).astype(np.float32)
    output, component, operation = run_broadband(mixed)
    np.testing.assert_array_equal(output, mixed)
    assert component["status"] in {"no_op", "abstained"}
    assert operation is None


@pytest.mark.parametrize("length", [0, 1, 80, 511, RATE * 4])
def test_silence_and_short_input_are_finite_and_unchanged(length):
    audio = np.zeros((length, 1), dtype=np.float32)
    # Analysis itself has no zero-length contract; use a valid empty-speech analysis.
    analysis = analyze_signal(np.zeros((RATE, 1), dtype=np.float32), RATE, [], PROFILE)
    output, component, operation = apply_broadband_denoise(audio, RATE, PROFILE, analysis, [])
    np.testing.assert_array_equal(output, audio)
    assert np.all(np.isfinite(output))
    assert component["status"] in {"abstained", "no_op"}
    assert operation is None


def test_overlap_add_reconstructs_endpoints_and_has_no_batch_seams():
    audio = np.random.default_rng(33).normal(0, 0.03, (RATE * 5 + 7, 2)).astype(np.float32)
    window = signal.windows.hann(512, sym=False)
    output, _ = filter_noise(audio, np.ones((257, 2)), window, 0.0)
    np.testing.assert_allclose(output, audio, atol=1e-9)
    bounded, _ = filter_noise(audio, np.ones((257, 2)) * 1e10, window, 6.0)
    np.testing.assert_allclose(bounded, audio * 10 ** (-6 / 20), atol=1e-8)


def test_environment_cleanup_is_scoped_and_has_no_abrupt_transition():
    clean, noise = speech_fixture()
    mixed = clean + noise
    analysis = analyze_signal(mixed, RATE, [SpeechRegion(1, 3, 0.9, 1)], PROFILE)
    # Isolate broadband processing from the separately tested minimum-phase filters.
    analysis = replace(analysis, subbass_power_ratio=0, hum_excess_db=0, dc_offset=0)
    output, stage = apply_environment_cleanup(mixed, RATE, PROFILE, analysis)
    np.testing.assert_array_equal(output[: RATE // 2], mixed[: RATE // 2])
    np.testing.assert_array_equal(output[round(3.5 * RATE) :], mixed[round(3.5 * RATE) :])
    assert db_ratio(noise[29000:31000], output[29000:31000]) > 3.0
    residual = (output - mixed)[:, 0]
    # No click: derivative of the removed noise remains below a modest multiple of
    # the known input noise RMS, including the two treatment transitions.
    assert np.max(np.abs(np.diff(residual))) < 0.05
    assert stage["component_evaluations"][0]["status"] == "applied"
    assert any(o["type"] == "bounded-spectral-denoise" for o in stage["operations"])


def test_applied_highpass_does_not_claim_applied_broadband():
    clean, noise = speech_fixture()
    mixed = clean + noise
    analysis = analyze_signal(mixed, RATE, [SpeechRegion(0, 4, 0.9, 1)], PROFILE)
    analysis = replace(analysis, subbass_power_ratio=0.1, hum_excess_db=0)
    output, stage = apply_environment_cleanup(mixed, RATE, PROFILE, analysis)
    assert not np.array_equal(output, mixed)
    assert stage["status"] == "applied"
    assert stage["component_evaluations"][0]["status"] == "abstained"
    assert [o["type"] for o in stage["operations"]] == ["minimum-phase-highpass"]


def test_adaptive_filter_is_independent_of_internal_batch_boundaries():
    clean, noise = speech_fixture()
    audio = np.column_stack(((clean + noise)[:, 0], (clean + noise)[:, 0] * -0.7))
    window = signal.windows.hann(512, sym=False)
    reference = np.fft.rfft(noise[:512, 0] * window)
    noise_power = np.abs(reference[:, None]) ** 2 * np.array([1, 0.49])
    output, _ = filter_noise(audio, noise_power, window, 6)
    # Shift 37 whole hops: physical frames align but land in different batches.
    offset = 37 * 128
    shifted, _ = filter_noise(audio[offset:], noise_power, window, 6)
    np.testing.assert_allclose(output[offset + 1024 : -1024], shifted[1024:-1024], atol=1e-8)


def test_tonal_reference_on_one_channel_is_not_hidden_by_noise_on_the_other():
    clean, noise = speech_fixture()
    time = np.arange(len(clean)) / RATE
    tone = 0.02 * np.sin(2 * np.pi * 880 * time)
    audio = np.column_stack(((clean + noise)[:, 0], tone + clean[:, 0])).astype(np.float32)
    output, component, operation = run_broadband(audio)
    np.testing.assert_array_equal(output, audio)
    assert component["reason"] == "noise_reference_is_tonal"
    assert operation is None
