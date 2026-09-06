"""The neural guide cannot override shared mask bounds or input spectral phase."""

import numpy as np
import pytest

from audio_cli.dsp import apply_guided_denoise
from audio_cli.profiles import PROFILES

PROFILE = PROFILES["product-demo"]
RATE = 48000


def fixture(channels=2):
    return np.random.default_rng(51).normal(0, 0.02, (RATE * 2 + 1, channels)).astype(np.float32)


def test_excessive_model_suppression_is_bounded_and_proportional_channels_survive():
    audio = fixture(1)
    audio = np.concatenate((audio, audio * 0.3), axis=1)
    output, decision, operation = apply_guided_denoise(audio, audio * 0.01, RATE, PROFILE)
    floor = 10 ** (-PROFILE.broadband_max_reduction_db / 20)
    assert decision["status"] == "applied"
    np.testing.assert_allclose(output, audio * floor, atol=1e-8)
    np.testing.assert_allclose(output[:, 1], output[:, 0] * 0.3, atol=1e-8)
    assert operation["maximum_spectral_reduction_db"] == PROFILE.broadband_max_reduction_db


def test_unattenuated_proposal_on_either_active_channel_protects_both():
    audio = fixture()
    guide = audio.copy()
    guide[:, 0] = 0
    output, decision, operation = apply_guided_denoise(audio, guide, RATE, PROFILE)
    assert decision["status"] == "no_op"
    assert operation is None
    np.testing.assert_array_equal(output, audio)


def test_silent_channel_does_not_veto_other_channel_and_stays_silent():
    audio = fixture()
    audio[:, 1] = 0
    output, decision, _operation = apply_guided_denoise(audio, audio * 0.2, RATE, PROFILE)
    assert decision["status"] == "applied"
    assert np.std(output[:, 0]) < np.std(audio[:, 0]) * 0.51
    np.testing.assert_array_equal(output[:, 1], 0)


@pytest.mark.parametrize("factor", [1, 2])
def test_identity_or_amplifying_guide_does_not_change_input(factor):
    audio = fixture()
    output, decision, operation = apply_guided_denoise(audio, audio * factor, RATE, PROFILE)
    assert decision["status"] == "no_op" and operation is None
    np.testing.assert_array_equal(output, audio)


@pytest.mark.parametrize("invalid", ["shape", "nan"])
def test_invalid_guide_refuses(invalid):
    audio = fixture()
    guide = audio[:-1] if invalid == "shape" else audio * np.nan
    with pytest.raises(ValueError, match="guide"):
        apply_guided_denoise(audio, guide, RATE, PROFILE)


def test_silent_guide_and_input_do_not_invent_applied_mask():
    audio = np.zeros((RATE, 1), np.float32)
    output, decision, operation = apply_guided_denoise(audio, audio, RATE, PROFILE)
    assert decision["status"] == "no_op" and operation is None
    np.testing.assert_array_equal(output, audio)
