"""Model inference is continuous; application is limited to the speech treatment."""

from pathlib import Path

import numpy as np
import pytest

from audio_cli.dsp import analyze_signal, resolve_speech_treatment_intervals, smooth_time_mask
from audio_cli.media import MediaError
from audio_cli.pipeline.denoise import DenoiserModel, apply_model_cleanup
from audio_cli.profiles import PROFILES
from audio_cli.vad_contract import SpeechRegion

PROFILE = PROFILES["product-demo"]
MODEL = DenoiserModel(Path("managed-model.rnnn"), "local-digest", {"package": "rnnoise-voice"})
RATE = 48000


def fixture(channels=1):
    audio = np.random.default_rng(15).normal(0, 0.02, (RATE * 3, channels)).astype(np.float32)
    analysis = analyze_signal(audio, RATE, [SpeechRegion(1, 2, 0.9, 1)], PROFILE)
    return audio, analysis


@pytest.mark.parametrize("channels", [1, 2])
def test_continuous_inference_with_no_stationary_reference_requirement(channels, monkeypatch):
    audio, analysis = fixture(channels)
    before = audio.copy()
    calls = []

    def render(samples, rate, path, digest):
        calls.append(samples.copy())
        assert (rate, path, digest) == (RATE, MODEL.path, MODEL.sha256)
        return samples * 0.4

    # Isolate model blend from the independently tested high-pass/hum filters.
    monkeypatch.setattr(
        "audio_cli.pipeline.denoise.prepare_environment_filters", lambda *a: (a[0], [])
    )
    monkeypatch.setattr("audio_cli.pipeline.denoise.render_rnnoise", render)
    output, report = apply_model_cleanup(audio, RATE, PROFILE, analysis, MODEL)
    intervals, _ = resolve_speech_treatment_intervals(audio, RATE, PROFILE, analysis)
    mask = smooth_time_mask(
        len(audio),
        intervals,
        RATE,
        PROFILE.region_fade_ms,
        transition_placement=PROFILE.speech_transition_placement,
    )
    assert len(calls) == 1 and calls[0].shape == audio.shape
    np.testing.assert_array_equal(audio, before)
    np.testing.assert_array_equal(output[mask == 0], audio[mask == 0])
    np.testing.assert_allclose(output[mask == 1], audio[mask == 1] * 10 ** (-6 / 20), atol=1e-8)
    assert report["component_evaluations"][0]["status"] == "applied"
    operation = report["operations"][-1]
    assert operation["channel_link"] == "maximum_guide_to_input_magnitude_ratio"
    assert operation["noise_reduction"]["status"] == "abstained"
    assert operation["maximum_spectral_reduction_db"] <= 6


def test_no_speech_does_not_run_model(monkeypatch):
    audio = np.zeros((RATE, 1), np.float32)
    analysis = analyze_signal(audio, RATE, [], PROFILE)
    monkeypatch.setattr(
        "audio_cli.pipeline.denoise.render_rnnoise", lambda *a: pytest.fail("no speech")
    )
    output, report = apply_model_cleanup(audio, RATE, PROFILE, analysis, MODEL)
    assert output is audio
    assert report["reason"] == "no_speech_detected"


def test_model_failure_propagates_without_stationary_fallback(monkeypatch):
    audio, analysis = fixture()

    def fail(*args):
        raise MediaError("model failed")

    monkeypatch.setattr("audio_cli.pipeline.denoise.render_rnnoise", fail)
    with pytest.raises(MediaError, match="model failed"):
        apply_model_cleanup(audio, RATE, PROFILE, analysis, MODEL)
