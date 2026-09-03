"""Publish only the VAD fields the normalized schema owns."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def normalize_vad_regions(regions: Iterable[Mapping[str, Any] | object]) -> list[dict[str, float]]:
    result = []
    for region in regions:
        if isinstance(region, Mapping):
            start, end = region["start"], region["end"]
        else:
            start, end = region.start, region.end
        result.append({"start": round(float(start), 6), "end": round(float(end), 6)})
    return result
