"""Normalize per-segment forced-alignment output."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
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


@dataclass(frozen=True)
class AlignmentResult:
    """Validated words and evidence-based per-unit rejection classifications."""

    words: dict[str, list[dict[str, Any]]]
    rejections: dict[str, dict[str, Any]]
    corrections: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def normalize_aligned_words(
    raw: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Compatibility projection; workflows retain the full normalization evidence."""
    return normalize_alignment(raw, segments).words


def _overrun_ms(lower: float, upper: float) -> float:
    """Subtract serialized decimal values without binary cancellation at the limit."""
    value = float(max(Decimal(0), Decimal(str(lower)) - Decimal(str(upper))) * 1000)
    if not math.isfinite(value):
        raise ValueError("alignment overrun is too large to represent in milliseconds")
    return value


def _boundary_evidence(
    start: float,
    end: float,
    unit_start: float,
    unit_end: float,
    start_overrun_ms: float,
    end_overrun_ms: float,
    max_overrun_ms: float,
) -> dict[str, Any]:
    if not math.isfinite(unit_end):
        raise ValueError("alignment boundary evidence requires an explicit finite unit end")
    return {
        "original_bounds": [start, end],
        "unit_bounds": [unit_start, unit_end],
        "start_overrun_ms": start_overrun_ms,
        "end_overrun_ms": end_overrun_ms,
        "max_overrun_ms": max_overrun_ms,
    }


def normalize_alignment(
    raw: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    *,
    max_overrun_ms: float = 0.501,
) -> AlignmentResult:
    """Validate unit streams and record explicitly allowed boundary corrections."""
    try:
        limit_ms = _strict_number(max_overrun_ms)
    except (TypeError, OverflowError) as exc:
        raise ValueError("max_overrun_ms must be finite and at least 0.501") from exc
    rounding_ms = ROUNDING_TOLERANCE_SECONDS * 1000
    if not math.isfinite(limit_ms) or limit_ms < rounding_ms:
        raise ValueError("max_overrun_ms must be finite and at least 0.501")
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
    rejections: dict[str, dict[str, Any]] = {}
    corrections: dict[str, list[dict[str, Any]]] = {}
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
            rejections[identifier] = {"code": "provider_unavailable"}
            continue
        if not isinstance(words, list):
            raise TypeError(f"aligner words for unit {identifier!r} must be an array or null")
        candidate = []
        unit_corrections = []
        rejection: dict[str, Any] = {"code": "invalid_bounds"}
        try:
            unit_start = _strict_number(by_id[identifier].get("start", 0.0))
            raw_unit_end = by_id[identifier].get("end")
            unit_end = float("inf") if raw_unit_end is None else _strict_number(raw_unit_end)
            if (
                not math.isfinite(unit_start)
                or unit_start < 0
                or (raw_unit_end is not None and not math.isfinite(unit_end))
                or unit_end < unit_start
            ):
                raise ValueError(f"aligner received invalid unit bounds for {identifier!r}")
            previous_end = float("-inf")
            previous_raw_end = float("-inf")
            for word_index, word in enumerate(words):
                rejection = {"code": "invalid_token", "word_index": word_index}
                if not isinstance(word, Mapping) or not isinstance(word.get("text"), str):
                    raise TypeError(f"aligner returned an invalid word for {identifier!r}")
                if not _plain(word["text"]):
                    raise ValueError(f"aligner returned an empty word token for {identifier!r}")
                rejection["code"] = "invalid_bounds"
                start = _strict_number(word["start"])
                end = _strict_number(word["end"])
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError(f"aligner returned invalid bounds for {identifier!r}")
                if end < start:
                    raise ValueError(f"aligner returned reversed bounds for {identifier!r}")
                rejection["code"] = "word_order"
                if start < previous_raw_end:
                    raise ValueError(
                        f"aligner returned overlapping or reversed words for {identifier!r}"
                    )
                rejection["code"] = "invalid_bounds"
                applied_start, applied_end = max(start, unit_start), min(end, unit_end)
                if (
                    end < unit_start
                    or start > unit_end
                    or (end > start and applied_end <= applied_start)
                ):
                    raise ValueError(
                        f"aligner returned wholly outside or collapsing bounds for {identifier!r}"
                    )
                rejection["code"] = "word_order"
                if applied_start < previous_end:
                    raise ValueError(
                        f"aligner returned overlapping or reversed words for {identifier!r}"
                    )
                rejection["code"] = "invalid_bounds"
                start_overrun = _overrun_ms(unit_start, start)
                end_overrun = _overrun_ms(end, unit_end) if math.isfinite(unit_end) else 0.0
                overrun = max(start_overrun, end_overrun)
                if overrun > rounding_ms:
                    boundary = _boundary_evidence(
                        start,
                        end,
                        unit_start,
                        unit_end,
                        start_overrun,
                        end_overrun,
                        limit_ms,
                    )
                    if overrun > limit_ms:
                        rejection.update(code="out_of_unit_bounds", boundary=boundary)
                        raise ValueError(
                            f"aligner exceeded the configured boundary limit for {identifier!r}"
                        )
                    unit_corrections.append(
                        {
                            **boundary,
                            "applied_bounds": [applied_start, applied_end],
                            "word_index": word_index,
                        }
                    )
                previous_raw_end = end
                start, end = applied_start, applied_end
                candidate.append({"text": word["text"], "start": start, "end": end})
                previous_end = end
        except (KeyError, OverflowError, TypeError, ValueError):
            rejections[identifier] = rejection
            continue
        normalized[identifier] = candidate
        if unit_corrections:
            corrections[identifier] = unit_corrections
    missing = set(by_id) - seen_ids
    if missing:
        rendered = ", ".join(repr(identifier) for identifier in sorted(missing))
        raise ValueError(f"aligner result is missing requested units: {rendered}")
    return AlignmentResult(normalized, rejections, corrections)
