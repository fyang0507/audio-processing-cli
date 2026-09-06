"""The model process boundary must not shorten, shift, or corrupt samples."""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from audio_cli.media import MediaError
from audio_cli.media.denoise import render_rnnoise


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "quote':model.rnnn"
    data = b"test fixture, not a neural model"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("count", [1, 479, 480, 481, 997])
@pytest.mark.parametrize("channels", [1, 2])
def test_process_receives_padded_float_pcm_and_compensates_synthesis_delay(
    count, channels, model, monkeypatch
):
    audio = np.random.default_rng(7).uniform(-0.2, 0.2, (count, channels)).astype(np.float32)
    directories = []

    def child(command, *, cwd, input, capture_output, check):
        directories.append(cwd)
        assert Path(cwd, "model.rnnn").read_bytes() == model[0].read_bytes()
        assert "arnndn=m=model.rnnn:mix=1.0" in command
        assert "48000" in command
        padded = np.frombuffer(input, dtype="<f4").reshape((-1, channels))
        assert len(padded) % 480 == 0 and len(padded) >= count + 480
        np.testing.assert_array_equal(padded[:count], audio)
        np.testing.assert_array_equal(padded[count:], 0)
        # The independent child double models the declared 480-sample synthesis
        # delay. Real-filter evidence is recorded separately, using the pinned model.
        delayed = np.pad(padded, ((480, 0), (0, 0)))[: len(padded)]
        return subprocess.CompletedProcess(command, 0, delayed.astype("<f4").tobytes(), b"")

    monkeypatch.setattr("audio_cli.media.denoise.subprocess.run", child)
    output = render_rnnoise(audio, 48000, *model)
    np.testing.assert_array_equal(output, audio)
    assert not directories[0].exists()


@pytest.mark.parametrize("kind", ["short", "long", "nonfinite", "unaligned", "failure"])
def test_refuses_invalid_backend_output_without_fallback(kind, model, monkeypatch):
    def child(command, **kwargs):
        samples = np.frombuffer(kwargs["input"], dtype="<f4").copy()
        if kind == "failure":
            raise subprocess.CalledProcessError(1, command, stderr=b"unknown filter arnndn")
        if kind == "short":
            samples = samples[:-1]
        elif kind == "long":
            samples = np.append(samples, np.float32(0))
        elif kind == "nonfinite":
            samples[-1] = np.nan
        raw = samples.tobytes() + (b"x" if kind == "unaligned" else b"")
        return subprocess.CompletedProcess(command, 0, raw, b"")

    monkeypatch.setattr("audio_cli.media.denoise.subprocess.run", child)
    with pytest.raises(MediaError):
        render_rnnoise(np.zeros((1000, 1), np.float32), 48000, *model)


@pytest.mark.parametrize("kind", ["changed", "symlink", "fifo"])
def test_model_must_match_verified_bytes_and_be_regular(kind, model, tmp_path, monkeypatch):
    path, digest = model
    if kind == "changed":
        path.write_bytes(b"different bytes")
    else:
        replacement = tmp_path / "replacement"
        if kind == "symlink":
            replacement.symlink_to(path)
        else:
            import os

            os.mkfifo(replacement)
        path = replacement
    monkeypatch.setattr(
        "audio_cli.media.denoise.subprocess.run",
        lambda *a, **k: pytest.fail("model must be checked before launch"),
    )
    with pytest.raises(MediaError):
        render_rnnoise(np.zeros((1000, 1), np.float32), 48000, path, digest)


@pytest.mark.parametrize("rate,shape", [(16000, (10, 1)), (48000, (10, 3)), (48000, (10,))])
def test_no_implicit_resampling_or_downmix(rate, shape, model):
    with pytest.raises(MediaError, match="48 kHz mono or stereo"):
        render_rnnoise(np.zeros(shape, np.float32), rate, *model)
