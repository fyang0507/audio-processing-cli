"""Normalize per-segment forced-alignment output."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

ROUNDING_TOLERANCE_SECONDS = 0.000501


def normalize_aligned_words(
    raw: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    values = raw.get("segments", [])
    if not isinstance(values, list):
        raise TypeError("aligner stage result segments must be an array")
    by_id = {str(item["unit_id"]): item for item in segments}
    normalized: dict[str, list[dict[str, Any]]] = {}
    for item in values:
        if not isinstance(item, Mapping):
            raise TypeError("aligner stage segment must be an object")
        identifier = str(item.get("unit_id", ""))
        if identifier not in by_id or identifier in normalized:
            raise ValueError(f"aligner returned unknown or duplicate unit {identifier!r}")
        words = item.get("words")
        if words is None:
            continue
        if not isinstance(words, list):
            continue
        candidate = []
        try:
            for word in words:
                if not isinstance(word, Mapping) or not isinstance(word.get("text"), str):
                    raise TypeError(f"aligner returned an invalid word for {identifier!r}")
                start = float(word["start"])
                end = float(word["end"])
                unit_start = float(by_id[identifier].get("start", 0.0))
                unit_end = float(by_id[identifier].get("end", float("inf")))
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError(f"aligner returned invalid bounds for {identifier!r}")
                if start < unit_start and unit_start - start <= ROUNDING_TOLERANCE_SECONDS:
                    start = unit_start
                if end > unit_end and end - unit_end <= ROUNDING_TOLERANCE_SECONDS:
                    end = unit_end
                if start < unit_start or end < start or end > unit_end:
                    raise ValueError(f"aligner returned invalid bounds for {identifier!r}")
                candidate.append({"text": word["text"], "start": start, "end": end})
        except (KeyError, TypeError, ValueError):
            continue
        normalized[identifier] = candidate
    return normalized
