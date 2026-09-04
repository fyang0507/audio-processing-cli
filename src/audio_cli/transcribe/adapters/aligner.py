"""Normalize per-segment forced-alignment output."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

ROUNDING_TOLERANCE_SECONDS = 0.000501


def _plain(text: str) -> str:
    return "".join(
        character.casefold()
        for character in text
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _strict_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("alignment bounds must be JSON numbers")
    return float(value)


def normalize_aligned_words(
    raw: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if "segments" not in raw:
        raise ValueError("aligner stage result is missing segments")
    values = raw["segments"]
    if not isinstance(values, list):
        raise TypeError("aligner stage result segments must be an array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in segments:
        identifier = str(item["unit_id"])
        if identifier in by_id:
            raise ValueError(f"aligner received duplicate requested unit {identifier!r}")
        by_id[identifier] = item
    normalized: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            raise TypeError("aligner stage segment must be an object")
        identifier = str(item.get("unit_id", ""))
        if identifier not in by_id or identifier in seen_ids:
            raise ValueError(f"aligner returned unknown or duplicate unit {identifier!r}")
        seen_ids.add(identifier)
        if "words" not in item:
            raise ValueError(f"aligner result for unit {identifier!r} is missing words")
        words = item["words"]
        if words is None:
            continue
        if not isinstance(words, list):
            raise TypeError(f"aligner words for unit {identifier!r} must be an array or null")
        candidate = []
        try:
            unit_start = _strict_number(by_id[identifier].get("start", 0.0))
            raw_unit_end = by_id[identifier].get("end")
            unit_end = float("inf") if raw_unit_end is None else _strict_number(raw_unit_end)
            if (
                not math.isfinite(unit_start)
                or (raw_unit_end is not None and not math.isfinite(unit_end))
                or unit_end < unit_start
            ):
                raise ValueError(f"aligner received invalid unit bounds for {identifier!r}")
            previous_end = float("-inf")
            for word in words:
                if not isinstance(word, Mapping) or not isinstance(word.get("text"), str):
                    raise TypeError(f"aligner returned an invalid word for {identifier!r}")
                if not _plain(word["text"]):
                    raise ValueError(f"aligner returned an empty word token for {identifier!r}")
                start = _strict_number(word["start"])
                end = _strict_number(word["end"])
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError(f"aligner returned invalid bounds for {identifier!r}")
                if start < unit_start and unit_start - start <= ROUNDING_TOLERANCE_SECONDS:
                    start = unit_start
                if end > unit_end and end - unit_end <= ROUNDING_TOLERANCE_SECONDS:
                    end = unit_end
                if start < unit_start or end < start or end > unit_end:
                    raise ValueError(f"aligner returned invalid bounds for {identifier!r}")
                if start < previous_end:
                    raise ValueError(
                        f"aligner returned overlapping or reversed words for {identifier!r}"
                    )
                candidate.append({"text": word["text"], "start": start, "end": end})
                previous_end = end
        except (KeyError, OverflowError, TypeError, ValueError):
            continue
        normalized[identifier] = candidate
    missing = set(by_id) - seen_ids
    if missing:
        rendered = ", ".join(repr(identifier) for identifier in sorted(missing))
        raise ValueError(f"aligner result is missing requested units: {rendered}")
    return normalized
