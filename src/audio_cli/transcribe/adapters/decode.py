"""Canonical decode command construction.

The source is input-only.  Every backend sees the one temporary PCM WAV this command writes,
so all bounds remain on one 16 kHz sample timeline.
"""

from __future__ import annotations

from pathlib import Path


def command(source: Path, target: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
