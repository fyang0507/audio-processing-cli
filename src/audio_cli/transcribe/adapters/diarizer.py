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
    values = payload.get("segments")
    if values is None and isinstance(payload.get("output"), Mapping):
        values = payload["output"].get("segments")
    if not isinstance(values, list):
        raise TypeError("FluidAudio result lacks a segments array")
    return values


def _normalize(item: Mapping[str, Any], total: int) -> dict[str, Any]:
    # FluidAudio 0.15.5 emits camelCase.  The snake-case variant is accepted for runner
    # artifacts and test doubles.  Nothing else, notably its 256-float embedding, crosses.
    start_raw = item.get("startTimeSeconds", item.get("start_s"))
    end_raw = item.get("endTimeSeconds", item.get("end_s"))
    speaker_raw = item.get("speakerId", item.get("speaker"))
    start_s, end_s = float(start_raw), float(end_raw)
    if not math.isfinite(start_s) or not math.isfinite(end_s) or speaker_raw is None \
            or not str(speaker_raw):
        raise ValueError("FluidAudio returned invalid fields")
    start = max(0, min(total, round(start_s * SAMPLE_RATE)))
    end = max(0, min(total, round(end_s * SAMPLE_RATE)))
    if end <= start:
        raise ValueError("FluidAudio returned an empty interval after source crop")
    return {"start": start, "end": end, "speaker": str(speaker_raw)}


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
        speakers = tuple(sorted(
            speaker for speaker, count in active.items() if count > 0
        ))
        span = _Span(
            left,
            right,
            speakers,
            fragment_active > 0 and not speakers,
        )
        if result and result[-1].end == span.start \
                and result[-1].speakers == span.speakers \
                and result[-1].fragment == span.fragment:
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
    total = max(0, round(duration_seconds * SAMPLE_RATE))
    normalized = [_normalize(item, total) for item in _raw_segments(payload)]
    kept = [item for item in normalized if item["end"] - item["start"] >= RAW_FRAGMENT_MIN_SAMPLES]
    filtered = [item for item in normalized if item["end"] - item["start"] < RAW_FRAGMENT_MIN_SAMPLES]
    spans = _spans(kept, filtered, total)
    turns = _turns(spans)
    accepted = [item for item in turns if item.end - item.start >= TURN_MIN_SAMPLES]
    short = [item for item in turns if item.end - item.start < TURN_MIN_SAMPLES]

    units = tuple({
        "unit_id": f"turn_{index}",
        "speaker": turn.speaker,
        "start": _seconds(turn.start),
        "end": _seconds(turn.end),
    } for index, turn in enumerate(accepted))
    public_turns = tuple({
        "turn_id": item["unit_id"], "speaker": item["speaker"],
        "start": item["start"], "end": item["end"],
    } for item in units)
    overlaps = tuple({"start": _seconds(span.start), "end": _seconds(span.end)}
                     for span in spans if span.kind == "overlap")
    raw = tuple({"start": _seconds(span.start), "end": _seconds(span.end)}
                for span in spans if span.kind == "raw_fragment")
    shorts = tuple({"start": _seconds(turn.start), "end": _seconds(turn.end)}
                   for turn in short)

    # The runner's partition invariant: accepted windows cannot overlap. The other span kinds
    # arise from the same atomic timeline, so a violation here indicates adapter drift.
    if any(left["end"] > right["start"] for left, right in pairwise(units)):
        raise ValueError("reconciled turn windows overlap")
    return DiarizationPlan(units, public_turns, overlaps, shorts, raw)
