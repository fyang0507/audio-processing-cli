"""RNNoise FFmpeg mechanics, with a verified private model and sample alignment."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import numpy as np

from .errors import MediaError
from .files import temporary_directory

FRAME_SAMPLES = 480


def _model_bytes(path: Path, expected_sha256: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise MediaError("RNNoise model is not a regular file")
        # Bound the read even if the inode grows after stat; this model family is
        # a small text network, not an unbounded archive.
        data = source.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise MediaError("RNNoise model exceeds the 16 MiB process input limit")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise MediaError("RNNoise model changed after package verification; verify packages again")
    return data


def render_rnnoise(
    audio: np.ndarray,
    sample_rate: int,
    model: Path,
    model_sha256: str,
    *,
    mix: float = 1.0,
) -> np.ndarray:
    """Run continuous independent channel states, compensating one synthesis frame.

    The caller chooses wet/dry policy and treatment scope. The mix=0 path exists to
    verify the filter's analysis/synthesis timing independently of neural gains.
    No channel conversion, implicit resampling, model fetch, or alternate filter.
    """
    if sample_rate != 48_000 or audio.ndim != 2 or audio.shape[1] not in (1, 2):
        raise MediaError("RNNoise requires 48 kHz mono or stereo floating-point audio")
    if not len(audio) or not np.isfinite(audio).all() or not 0 <= mix <= 1:
        raise MediaError("RNNoise requires nonempty finite audio and a mix between zero and one")
    count, channels = audio.shape
    padded_count = ((count + FRAME_SAMPLES - 1) // FRAME_SAMPLES + 1) * FRAME_SAMPLES
    padded = np.pad(audio, ((0, padded_count - count), (0, 0))).astype("<f4")
    try:
        data = _model_bytes(model, model_sha256)
        with temporary_directory("audio-rnnoise-") as directory:
            # A fixed relative name avoids FFmpeg's separate filter-path escaping
            # rules even when the managed root contains quotes, colons, or spaces.
            (directory / "model.rnnn").write_bytes(data)
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-f",
                    "f32le",
                    "-ar",
                    "48000",
                    "-ac",
                    str(channels),
                    "-i",
                    "pipe:0",
                    "-af",
                    f"arnndn=m=model.rnnn:mix={mix}",
                    "-f",
                    "f32le",
                    "-c:a",
                    "pcm_f32le",
                    "pipe:1",
                ],
                cwd=directory,
                input=padded.tobytes(),
                capture_output=True,
                check=True,
            )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"").decode("utf-8", errors="replace")[-4000:]
        raise MediaError(f"RNNoise rendering failed: {detail or exc}") from exc
    expected_bytes = padded_count * channels * 4
    if len(result.stdout) != expected_bytes:
        raise MediaError("RNNoise returned an unexpected sample count; output was not published")
    decoded = np.frombuffer(result.stdout, dtype="<f4").reshape((-1, channels))
    if not np.isfinite(decoded).all():
        raise MediaError("RNNoise returned nonfinite samples; output was not published")
    return decoded[FRAME_SAMPLES : FRAME_SAMPLES + count].copy()
