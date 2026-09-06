"""Optional real-filter checks, after explicit native package provisioning."""

import os

import numpy as np
import pytest
from scipy import signal

from audio_cli.media.denoise import render_rnnoise

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("AUDIO_TEST_RNNOISE") != "1",
        reason="requires explicit rnnoise-voice provisioning and AUDIO_TEST_RNNOISE=1",
    ),
]


@pytest.fixture
def model():
    from audio_cli.packages import verified_artifact

    path, provenance = verified_artifact("rnnoise-voice")
    return path, provenance["sha256"]


@pytest.mark.parametrize("count", [1, 479, 480, 481, 997, 48001])
@pytest.mark.parametrize("channels", [1, 2])
def test_actual_dry_filter_preserves_samples_and_partial_tail(count, channels, model):
    audio = np.random.default_rng(82).uniform(-0.1, 0.1, (count, channels)).astype(np.float32)
    output = render_rnnoise(audio, 48000, *model, mix=0)
    assert output.shape == audio.shape
    np.testing.assert_allclose(output, audio, atol=1e-6, rtol=1e-5)


def test_actual_wet_filter_has_no_remaining_frame_delay_and_retains_channel_count(model):
    # A sustained harmonic probe exercises actual recurrent inference; it is not
    # linguistic speech or a basis for recognition/intelligibility claims.
    time = np.arange(96001) / 48000
    tone = np.zeros_like(time)
    for harmonic in range(1, 14):
        tone += 0.12 / harmonic * np.sin(2 * np.pi * harmonic * 173 * time)
    envelope = np.sin(np.pi * np.minimum(time / 0.1, 1) / 2) ** 2
    tone *= envelope
    audio = np.column_stack((tone, tone)).astype(np.float32)
    wet = render_rnnoise(audio, 48000, *model)
    assert wet.shape == audio.shape and np.isfinite(wet).all()
    assert np.max(np.abs(wet)) > 1e-4
    np.testing.assert_array_equal(wet[:, 0], wet[:, 1])
    assert not np.allclose(wet, audio, atol=1e-5)
    # Allow the model filter's own high-pass phase response. This is a check for
    # residual frame buffering, not sample-identical wet synthesis.
    correlations = signal.correlate(
        wet[4800:-4800, 0], audio[4800:-4800, 0], mode="full", method="fft"
    )
    center = len(wet[4800:-4800]) - 1
    assert abs(np.argmax(correlations[center - 20 : center + 21]) - 20) <= 4


def test_actual_model_suppresses_stationary_noise_beyond_its_highpass(model):
    audio = np.random.default_rng(44).normal(0, 0.02, (96001, 1)).astype(np.float32)
    wet = render_rnnoise(audio, 48000, *model)
    # Exclude recurrent warm-up. A bypass or the published high-pass alone cannot
    # meet this 3 dB broadband-noise requirement on this seeded white-noise fixture.
    ratio = np.mean(wet[24000:] ** 2) / np.mean(audio[24000:] ** 2)
    assert 10 * np.log10(max(ratio, 1e-30)) < -3


def test_actual_asymmetric_channels_match_their_mono_runs(model):
    audio = np.random.default_rng(18).normal(0, 0.02, (48001, 2)).astype(np.float32)
    time = np.arange(len(audio)) / 48000
    audio[:, 0] += (0.15 * np.sin(2 * np.pi * 270 * time)).astype(np.float32)
    stereo = render_rnnoise(audio, 48000, *model)
    for channel in (0, 1):
        mono = render_rnnoise(audio[:, channel : channel + 1], 48000, *model)
        np.testing.assert_array_equal(stereo[:, channel], mono[:, 0])


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_calibrated_guide_preserves_synthetic_voice_while_reducing_changing_noise(scale, model):
    from test_dsp_denoise import speech_fixture

    from audio_cli.dsp import analyze_signal
    from audio_cli.pipeline.denoise import DenoiserModel, apply_model_cleanup
    from audio_cli.profiles import PROFILES
    from audio_cli.vad_contract import SpeechRegion

    clean, noise = speech_fixture(48000)
    time = np.arange(len(clean)) / 48000
    clean = (clean * scale).astype(np.float32)
    mixed = clean + (noise * scale * (0.2 + 2.8 * time[:, None] / 4)).astype(np.float32)
    profile = PROFILES["product-demo"]
    analysis = analyze_signal(mixed, 48000, [SpeechRegion(1, 3, 0.9, 1)], profile)
    output, report = apply_model_cleanup(
        mixed,
        48000,
        profile,
        analysis,
        DenoiserModel(model[0], model[1], {"package": "rnnoise-voice"}),
    )
    component = report["component_evaluations"][0]
    speech = slice(50400, 141600)
    pause = slice(86400, 96000)
    voiced = slice(55200, 76800)
    error_ratio = np.mean((mixed[speech] - clean[speech]) ** 2) / np.mean(
        (output[speech] - clean[speech]) ** 2
    )
    assert 10 * np.log10(error_ratio) > 2
    assert 10 * np.log10(np.mean(mixed[pause] ** 2) / np.mean(output[pause] ** 2)) > 3
    projection = np.sum(output[voiced] * clean[voiced]) / np.sum(clean[voiced] ** 2)
    assert abs(20 * np.log10(projection)) < 1
    assert component["status"] == "applied"
