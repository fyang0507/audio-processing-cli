"""Calibrate model inference without changing the signal being enhanced."""

import numpy as np
import pytest

from audio_cli.dsp import calibrate_guide_input
from audio_cli.vad_contract import SpeechRegion


def calibrate(audio, speech=None):
    return calibrate_guide_input(
        audio,
        1000,
        speech if speech is not None else [SpeechRegion(0, 1, 0.9, 1)],
        target_rms_dbfs=-24,
        peak_limit_dbfs=-3,
        maximum_gain_db=40,
    )


def test_one_shared_guide_gain_reaches_speech_reference_and_preserves_input():
    audio = np.full((1000, 2), [0.005, -0.01], np.float32)
    before = audio.copy()
    guide, gain, report = calibrate(audio)
    assert report["scope"] == "model_guide_only"
    assert report["resolved_speech_rms_dbfs"] == -24
    assert 10 * np.log10(np.mean(guide.astype(float) ** 2)) == pytest.approx(-24, abs=1e-5)
    np.testing.assert_allclose(guide / gain, audio, atol=1e-8)
    np.testing.assert_array_equal(audio, before)
    np.testing.assert_array_equal(guide[:, 1], -2 * guide[:, 0])


def test_loud_program_outside_speech_limits_guide_headroom():
    audio = np.full((2000, 1), 0.001, np.float32)
    audio[1500] = 0.5
    guide, _gain, report = calibrate(audio)
    assert report["input_speech_rms_dbfs"] == pytest.approx(-60)
    assert report["resolved_speech_rms_dbfs"] < -24
    assert np.max(np.abs(guide)) == pytest.approx(10 ** (-3 / 20))


def test_extremely_quiet_input_respects_calibration_gain_bound():
    audio = np.full((1000, 1), 1e-6, np.float32)
    guide, gain, report = calibrate(audio)
    assert gain == 100 and report["resolved_gain_db"] == 40
    np.testing.assert_allclose(guide, audio * 100)


@pytest.mark.parametrize("speech", [[], [SpeechRegion(0, 1, 0.9, 1)]])
def test_missing_or_silent_speech_has_no_invented_rms_measurement(speech):
    guide, gain, report = calibrate(np.zeros((1000, 1), np.float32), speech)
    assert guide is None and gain == 1
    assert report == {"status": "abstained", "reason": "no_speech_energy"}
