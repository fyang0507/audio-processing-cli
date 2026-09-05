"""Canonical mono PCM16 WAV inspection and sample-aligned extraction."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from .errors import MediaError


class EmptyPcmRangeError(ValueError):
    """A valid second range that owns no sample on the canonical grid."""


def _canonical_header(handle: wave.Wave_read) -> tuple[int, int, int]:
    rate = handle.getframerate()
    channels = handle.getnchannels()
    width = handle.getsampwidth()
    if rate != 16_000 or channels != 1 or width != 2:
        raise ValueError("canonical decode is not mono 16 kHz PCM16")
    return rate, channels, width


def canonical_pcm_duration(path: Path) -> float:
    """Read the exact duration from a canonical WAV header."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate, _channels, _width = _canonical_header(handle)
            return round(handle.getnframes() / float(rate), 6)
    except wave.Error as exc:
        raise MediaError(str(exc)) from exc


def read_canonical_pcm16(path: Path) -> tuple[np.ndarray, int]:
    """Load canonical PCM16 samples as normalized float32 mono audio."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate, _channels, _width = _canonical_header(handle)
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    except wave.Error as exc:
        raise MediaError(str(exc)) from exc
    return samples.astype(np.float32) / 32768.0, rate


def clip_canonical_pcm(
    source: Path,
    target: Path,
    *,
    start: float,
    end: float,
) -> tuple[Path, tuple[float, float]]:
    """Write a canonical clip and return its exact source-timeline sample bounds."""
    try:
        with wave.open(str(source), "rb") as reader:
            rate, channels, width = _canonical_header(reader)
            # Select samples by their source timestamps: START <= timestamp < END.
            # The same ceiling rule on both sides lets adjacent runs meet at one
            # sample index without reprocessing a pre-range sample.
            frame_count = reader.getnframes()
            first = min(frame_count, max(0, math.ceil(start * rate)))
            last = min(frame_count, max(0, math.ceil(end * rate)))
            if first >= last:
                raise EmptyPcmRangeError(
                    "range selects no complete sample on the canonical 16 kHz timeline"
                )
            reader.setpos(first)
            frames = reader.readframes(last - first)
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(width)
            writer.setframerate(rate)
            writer.writeframes(frames)
    except wave.Error as exc:
        raise MediaError(str(exc)) from exc
    return target, (first / rate, last / rate)
