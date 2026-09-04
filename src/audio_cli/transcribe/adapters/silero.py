"""Publish only the VAD fields the normalized schema owns."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def _finite_json_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Silero VAD {field} must be a JSON number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"Silero VAD {field} must be a finite JSON number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Silero VAD {field} must be a finite JSON number")
    return parsed


def normalize_vad_regions(regions: Iterable[Mapping[str, Any] | object]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    previous_end: float | None = None
    for index, region in enumerate(regions):
        try:
            if isinstance(region, Mapping):
                start_raw, end_raw = region["start"], region["end"]
            else:
                start_raw, end_raw = region.start, region.end
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"Silero VAD region {index} lacks start/end bounds") from exc
        start = _finite_json_number(start_raw, f"region {index} start")
        end = _finite_json_number(end_raw, f"region {index} end")
        if start < 0 or end <= start:
            raise ValueError(
                f"Silero VAD region {index} must have non-negative increasing bounds"
            )
        if previous_end is not None and start < previous_end:
            raise ValueError(
                f"Silero VAD region {index} overlaps or precedes the previous region"
            )
        normalized_start = round(start, 6)
        normalized_end = round(end, 6)
        if normalized_end <= normalized_start:
            raise ValueError(
                f"Silero VAD region {index} is empty after timestamp rounding"
            )
        if result and normalized_start < result[-1]["end"]:
            raise ValueError(
                f"Silero VAD region {index} overlaps after timestamp rounding"
            )
        result.append({"start": normalized_start, "end": normalized_end})
        previous_end = end
    return result
