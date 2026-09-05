"""Exact turn reconciliation from FluidAudio's anonymous intervals.

The thresholds and partition algorithm are the measured configuration in
model_tests/benchmark/run_turn_attributed_mlx_asr.py.  This module removes raw Core ML
embeddings at the boundary and makes every source sample belong to exactly one category.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

SAMPLE_RATE = 16_000
RAW_FRAGMENT_MIN_SAMPLES = round(0.250 * SAMPLE_RATE)
MERGE_GAP_MAX_SAMPLES = round(0.300 * SAMPLE_RATE)
TURN_MIN_SAMPLES = round(0.500 * SAMPLE_RATE)


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    speakers: tuple[str, ...]
    fragment: bool

    @property
    def kind(self) -> str:
        if len(self.speakers) > 1:
            return "overlap"
        if len(self.speakers) == 1:
            return "single"
        if self.fragment:
            return "raw_fragment"
        return "gap"


@dataclass
class _Turn:
    start: int
    end: int
    speaker: str


@dataclass(frozen=True)
class DiarizationPlan:
    units: tuple[dict[str, Any], ...]
    turns: tuple[dict[str, Any], ...]
    overlaps: tuple[dict[str, Any], ...]
    short_turns: tuple[dict[str, Any], ...]
    raw_fragments: tuple[dict[str, Any], ...]


def _seconds(samples: int) -> float:
    return round(samples / SAMPLE_RATE, 6)


def _raw_segments(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    nested = payload.get("output")
    top_level = "segments" in payload
    nested_level = isinstance(nested, Mapping) and "segments" in nested
    if top_level == nested_level:
        raise ValueError("FluidAudio result must carry segments at exactly one accepted location")
    values = payload["segments"] if top_level else nested["segments"]
    if not isinstance(values, list):
        raise TypeError("FluidAudio result lacks a segments array")
    if any(not isinstance(item, Mapping) for item in values):
        raise TypeError("FluidAudio segments must be objects")
    return values


def _finite_json_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"FluidAudio {field} must be a JSON number")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"FluidAudio {field} must be a finite JSON number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"FluidAudio {field} must be a finite JSON number")
    return parsed


def _speaker_label(value: Any) -> str:
    # Recorded FluidAudio output uses string labels (for example ``"S1"``), while
    # the benchmark normalization also treats integer speaker ids as opaque labels.
    # Do not stringify arbitrary JSON values into plausible-looking identities.
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("FluidAudio speaker label must be a string or integer")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("FluidAudio speaker label must not be empty")
        return value
    return str(value)


def _one_alias(item: Mapping[str, Any], aliases: tuple[str, str], field: str) -> Any:
    present = [name for name in aliases if name in item]
    if len(present) != 1:
        raise ValueError(f"FluidAudio segment must carry exactly one {field} field")
    return item[present[0]]


def _normalize(item: Mapping[str, Any], total: int) -> dict[str, Any]:
    # FluidAudio 0.15.5 emits camelCase.  The snake-case variant is accepted for runner
    # artifacts and test doubles.  Nothing else, notably its 256-float embedding, crosses.
    start_raw = _one_alias(item, ("startTimeSeconds", "start_s"), "start")
    end_raw = _one_alias(item, ("endTimeSeconds", "end_s"), "end")
    speaker_raw = _one_alias(item, ("speakerId", "speaker"), "speaker")
    start_s = _finite_json_number(start_raw, "segment start")
    end_s = _finite_json_number(end_raw, "segment end")
    speaker = _speaker_label(speaker_raw)
    # Crop in seconds before converting to samples. Besides matching the existing
    # source crop, this keeps a finite but enormous backend value from overflowing
    # during multiplication and leaking an untyped exception.
    duration_s = total / SAMPLE_RATE
    start = round(max(0.0, min(duration_s, start_s)) * SAMPLE_RATE)
    end = round(max(0.0, min(duration_s, end_s)) * SAMPLE_RATE)
    if end <= start:
        raise ValueError("FluidAudio returned an empty interval after source crop")
    return {"start": start, "end": end, "speaker": speaker}


def _spans(kept: list[dict[str, Any]], filtered: list[dict[str, Any]], total: int) -> list[_Span]:
    speaker_events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    fragment_events: dict[int, list[int]] = defaultdict(list)
    for item in kept:
        speaker_events[item["start"]].append((item["speaker"], 1))
        speaker_events[item["end"]].append((item["speaker"], -1))
    for item in filtered:
        fragment_events[item["start"]].append(1)
        fragment_events[item["end"]].append(-1)
    boundaries = sorted({0, total, *speaker_events, *fragment_events})
    active: Counter[str] = Counter()
    fragment_active = 0
    result: list[_Span] = []
    for left, right in pairwise(boundaries):
        for speaker, delta in speaker_events.get(left, []):
            active[speaker] += delta
        for delta in fragment_events.get(left, []):
            fragment_active += delta
        if right <= left:
            continue
        speakers = tuple(sorted(speaker for speaker, count in active.items() if count > 0))
        span = _Span(
            left,
            right,
            speakers,
            fragment_active > 0 and not speakers,
        )
        if (
            result
            and result[-1].end == span.start
            and result[-1].speakers == span.speakers
            and result[-1].fragment == span.fragment
        ):
            previous = result[-1]
            result[-1] = _Span(previous.start, span.end, span.speakers, span.fragment)
        else:
            result.append(span)
    return result


def _turns(spans: list[_Span]) -> list[_Turn]:
    result: list[_Turn] = []
    index = 0
    while index < len(spans):
        span = spans[index]
        if span.kind != "single":
            index += 1
            continue
        turn = _Turn(span.start, span.end, span.speakers[0])
        cursor = index
        while cursor + 2 < len(spans):
            gap, following = spans[cursor + 1], spans[cursor + 2]
            if not (
                gap.kind == "gap"
                and gap.end - gap.start <= MERGE_GAP_MAX_SAMPLES
                and following.kind == "single"
                and following.speakers == (turn.speaker,)
            ):
                break
            turn.end = following.end
            cursor += 2
        result.append(turn)
        index = cursor + 1
    return result


def reconcile_turns(payload: Mapping[str, Any], *, duration_seconds: float) -> DiarizationPlan:
    duration = _finite_json_number(duration_seconds, "source duration")
    if duration < 0:
        raise ValueError("FluidAudio source duration must be non-negative")
    try:
        total = round(duration * SAMPLE_RATE)
    except OverflowError as exc:
        raise ValueError("FluidAudio source duration is too large") from exc
    normalized = [_normalize(item, total) for item in _raw_segments(payload)]
    kept = [item for item in normalized if item["end"] - item["start"] >= RAW_FRAGMENT_MIN_SAMPLES]
    filtered = [
        item for item in normalized if item["end"] - item["start"] < RAW_FRAGMENT_MIN_SAMPLES
    ]
    spans = _spans(kept, filtered, total)
    turns = _turns(spans)
    accepted = [item for item in turns if item.end - item.start >= TURN_MIN_SAMPLES]
    short = [item for item in turns if item.end - item.start < TURN_MIN_SAMPLES]

    units = tuple(
        {
            "unit_id": f"turn_{index}",
            "speaker": turn.speaker,
            "start": _seconds(turn.start),
            "end": _seconds(turn.end),
        }
        for index, turn in enumerate(accepted)
    )
    public_turns = tuple(
        {
            "turn_id": item["unit_id"],
            "speaker": item["speaker"],
            "start": item["start"],
            "end": item["end"],
        }
        for item in units
    )
    overlaps = tuple(
        {"start": _seconds(span.start), "end": _seconds(span.end)}
        for span in spans
        if span.kind == "overlap"
    )
    raw = tuple(
        {"start": _seconds(span.start), "end": _seconds(span.end)}
        for span in spans
        if span.kind == "raw_fragment"
    )
    shorts = tuple({"start": _seconds(turn.start), "end": _seconds(turn.end)} for turn in short)

    # The runner's partition invariant: accepted windows cannot overlap. The other span kinds
    # arise from the same atomic timeline, so a violation here indicates adapter drift.
    if any(left["end"] > right["start"] for left, right in pairwise(units)):
        raise ValueError("reconciled turn windows overlap")
    return DiarizationPlan(units, public_turns, overlaps, shorts, raw)
