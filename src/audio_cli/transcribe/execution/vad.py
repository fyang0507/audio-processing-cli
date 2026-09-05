"""Run transcription VAD against canonical mono PCM audio."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from audio_cli.media import read_canonical_pcm16
from audio_cli.vad import SileroOnnxVad

from ..adapters.silero import normalize_vad_regions
from . import runtime


def _detect_vad(
    path: Path, detector: Any | None, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], float, int]:
    """Run core VAD in a short frame so its PCM and session die before model stages."""
    started = time.perf_counter()
    samples, rate = read_canonical_pcm16(path)
    selected = detector or SileroOnnxVad()
    regions = selected.detect(samples, rate, **config)
    normalized = normalize_vad_regions(regions)
    duration = round(len(samples) / float(rate), 6)
    for index, region in enumerate(normalized):
        if region["end"] > duration:
            raise ValueError(f"Silero VAD region {index} exceeds canonical source duration")
    return normalized, round(time.perf_counter() - started, 6), runtime._self_peak_rss()
