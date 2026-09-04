from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from audio_cli.media import (
    EmptyPcmRangeError,
    MediaError,
    canonical_pcm_duration,
    clip_canonical_pcm,
    read_canonical_pcm16,
)


def _write_wav(path: Path, samples: tuple[int, ...], *, rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def test_canonical_pcm_read_and_clip_preserve_the_sample_grid(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "clip.wav"
    _write_wav(source, (-32_768, -16_384, 0, 16_384, 32_767))

    samples, rate = read_canonical_pcm16(source)
    assert rate == 16_000
    np.testing.assert_allclose(
        samples,
        np.array((-1.0, -0.5, 0.0, 0.5, 32_767 / 32_768), dtype=np.float32),
    )
    assert canonical_pcm_duration(source) == 0.000313

    clipped, bounds = clip_canonical_pcm(
        source,
        target,
        start=0.00001,
        end=0.00019,
    )

    assert clipped == target
    assert bounds == (1 / 16_000, 4 / 16_000)
    clipped_samples, _rate = read_canonical_pcm16(target)
    np.testing.assert_allclose(clipped_samples, samples[1:4])


def test_canonical_pcm_rejects_wrong_headers_and_empty_ranges(tmp_path: Path) -> None:
    wrong_rate = tmp_path / "wrong-rate.wav"
    _write_wav(wrong_rate, (0,), rate=8_000)

    with pytest.raises(ValueError, match="mono 16 kHz PCM16"):
        canonical_pcm_duration(wrong_rate)

    malformed = tmp_path / "malformed.wav"
    malformed.write_bytes(b"not a wave file")
    with pytest.raises(MediaError):
        read_canonical_pcm16(malformed)

    source = tmp_path / "source.wav"
    _write_wav(source, (0,))
    with pytest.raises(EmptyPcmRangeError, match="no complete sample"):
        clip_canonical_pcm(source, tmp_path / "empty.wav", start=0.0, end=0.0)
