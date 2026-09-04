"""Deterministic turn planning and serialization for the MLX benchmark."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .runtime import stable_json_sha256

__all__ = [
    "SAMPLE_RATE",
    "AtomicSpan",
    "Turn",
    "build_atomic_timeline",
    "build_plan",
    "build_turns",
    "normalized_interval",
    "seconds",
    "serialize_plan",
    "span_record",
    "stable_json_sha256",
    "turn_record",
]

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AtomicSpan:
    start: int
    end: int
    active_speakers: tuple[str, ...]
    filtered_fragment_active: bool

    @property
    def kind(self) -> str:
        if len(self.active_speakers) > 1:
            return "overlap"
        if len(self.active_speakers) == 1:
            return "single_speaker"
        if self.filtered_fragment_active:
            return "raw_fragment_abstain"
        return "gap"


@dataclass
class Turn:
    start: int
    end: int
    speaker: str
    active_samples: int
    bridge_gap_samples: int = 0


def seconds(samples: int) -> float:
    return samples / SAMPLE_RATE


def normalized_interval(
    item: dict[str, Any], *, total_samples: int, index: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        start_s = float(item["start_s"])
        end_s = float(item["end_s"])
        speaker = str(item["speaker"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"source_index": index, "reason": f"invalid_fields:{exc}"}
    if not math.isfinite(start_s) or not math.isfinite(end_s):
        return None, {"source_index": index, "reason": "nonfinite_bounds"}
    start = max(0, min(total_samples, round(start_s * SAMPLE_RATE)))
    end = max(0, min(total_samples, round(end_s * SAMPLE_RATE)))
    if end <= start:
        return None, {
            "source_index": index,
            "reason": "empty_after_crop",
            "source_start_s": start_s,
            "source_end_s": end_s,
            "speaker": speaker,
        }
    return {
        "source_index": index,
        "start": start,
        "end": end,
        "speaker": speaker,
    }, None


def build_atomic_timeline(
    kept: list[dict[str, Any]],
    filtered: list[dict[str, Any]],
    total_samples: int,
) -> list[AtomicSpan]:
    speaker_events: dict[int, list[tuple[str, int]]] = defaultdict(list)
    filtered_events: dict[int, list[int]] = defaultdict(list)
    for item in kept:
        speaker_events[item["start"]].append((item["speaker"], 1))
        speaker_events[item["end"]].append((item["speaker"], -1))
    for item in filtered:
        filtered_events[item["start"]].append(1)
        filtered_events[item["end"]].append(-1)
    boundaries = sorted({0, total_samples, *speaker_events, *filtered_events})
    active: Counter[str] = Counter()
    filtered_active = 0
    spans: list[AtomicSpan] = []
    for left, right in pairwise(boundaries):
        for speaker, delta in speaker_events.get(left, []):
            active[speaker] += delta
        for delta in filtered_events.get(left, []):
            filtered_active += delta
        if right <= left:
            continue
        span = AtomicSpan(
            start=left,
            end=right,
            active_speakers=tuple(
                sorted(speaker for speaker, count in active.items() if count > 0)
            ),
            filtered_fragment_active=filtered_active > 0,
        )
        if (
            spans
            and spans[-1].end == span.start
            and spans[-1].active_speakers == span.active_speakers
            and spans[-1].filtered_fragment_active == span.filtered_fragment_active
        ):
            previous = spans[-1]
            spans[-1] = AtomicSpan(
                previous.start,
                span.end,
                previous.active_speakers,
                previous.filtered_fragment_active,
            )
        else:
            spans.append(span)
    return spans


def build_turns(spans: list[AtomicSpan], merge_gap_samples: int) -> list[Turn]:
    turns: list[Turn] = []
    index = 0
    while index < len(spans):
        span = spans[index]
        if span.kind != "single_speaker":
            index += 1
            continue
        speaker = span.active_speakers[0]
        turn = Turn(span.start, span.end, speaker, span.end - span.start)
        cursor = index
        while cursor + 2 < len(spans):
            gap = spans[cursor + 1]
            following = spans[cursor + 2]
            if not (
                gap.kind == "gap"
                and gap.end - gap.start <= merge_gap_samples
                and following.kind == "single_speaker"
                and following.active_speakers == (speaker,)
            ):
                break
            turn.end = following.end
            turn.bridge_gap_samples += gap.end - gap.start
            turn.active_samples += following.end - following.start
            cursor += 2
        turns.append(turn)
        index = cursor + 1
    return turns


def turn_record(turn: Turn, turn_index: int) -> dict[str, Any]:
    return {
        "turn_index": turn_index,
        "start_s": seconds(turn.start),
        "end_s": seconds(turn.end),
        "speaker": turn.speaker,
        "window_duration_s": seconds(turn.end - turn.start),
        "active_duration_s": seconds(turn.active_samples),
        "bridge_gap_duration_s": seconds(turn.bridge_gap_samples),
    }


def span_record(span: AtomicSpan, reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "start_s": seconds(span.start),
        "end_s": seconds(span.end),
        "duration_s": seconds(span.end - span.start),
        "anonymous_speakers": list(span.active_speakers),
    }


def build_plan(
    raw_segments: list[dict[str, Any]],
    *,
    total_samples: int,
    raw_fragment_min_samples: int,
    merge_gap_samples: int,
    asr_turn_min_samples: int,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, item in enumerate(raw_segments):
        record, failure = normalized_interval(item, total_samples=total_samples, index=index)
        if failure is not None:
            invalid.append(failure)
        elif record is not None:
            normalized.append(record)
    kept = [item for item in normalized if item["end"] - item["start"] >= raw_fragment_min_samples]
    filtered = [
        item for item in normalized if item["end"] - item["start"] < raw_fragment_min_samples
    ]
    spans = build_atomic_timeline(kept, filtered, total_samples)
    turns = build_turns(spans, merge_gap_samples)
    accepted = [turn for turn in turns if turn.end - turn.start >= asr_turn_min_samples]
    short_turns = [turn for turn in turns if turn.end - turn.start < asr_turn_min_samples]

    accepted_bounds = [(turn.start, turn.end) for turn in accepted]
    if any(left[1] > right[0] for left, right in pairwise(accepted_bounds)):
        raise RuntimeError("accepted turn windows overlap")
    accepted_window_samples = sum(end - start for start, end in accepted_bounds)
    accepted_active_samples = sum(turn.active_samples for turn in accepted)
    accepted_bridge_samples = sum(turn.bridge_gap_samples for turn in accepted)
    short_samples = sum(turn.active_samples for turn in short_turns)
    overlap_spans = [span for span in spans if span.kind == "overlap"]
    raw_fragment_spans = [span for span in spans if span.kind == "raw_fragment_abstain"]
    gap_spans = [span for span in spans if span.kind == "gap"]
    gap_samples = sum(span.end - span.start for span in gap_spans)
    # A short silence merged into an accepted turn is context, not unclaimed gap.
    unclaimed_gap_samples = gap_samples - accepted_bridge_samples
    if unclaimed_gap_samples < 0:
        raise RuntimeError("negative unclaimed gap coverage")
    disjoint_samples = (
        accepted_active_samples
        + accepted_bridge_samples
        + short_samples
        + sum(span.end - span.start for span in overlap_spans)
        + sum(span.end - span.start for span in raw_fragment_spans)
        + unclaimed_gap_samples
    )
    if disjoint_samples != total_samples:
        raise RuntimeError(f"coverage partition mismatch: {disjoint_samples} != {total_samples}")

    return {
        "normalized": normalized,
        "kept": kept,
        "filtered": filtered,
        "invalid": invalid,
        "spans": spans,
        "turns": turns,
        "accepted": accepted,
        "short_turns": short_turns,
        "overlap_spans": overlap_spans,
        "raw_fragment_spans": raw_fragment_spans,
        "coverage": {
            "total_s": seconds(total_samples),
            "accepted_transcription_window_s": seconds(accepted_window_samples),
            "accepted_single_speaker_active_s": seconds(accepted_active_samples),
            "accepted_bridge_silence_context_s": seconds(accepted_bridge_samples),
            "short_turn_abstain_s": seconds(short_samples),
            "overlap_abstain_s": seconds(sum(span.end - span.start for span in overlap_spans)),
            "raw_fragment_only_abstain_s": seconds(
                sum(span.end - span.start for span in raw_fragment_spans)
            ),
            "unclaimed_gap_s": seconds(unclaimed_gap_samples),
            "partition_sum_s": seconds(disjoint_samples),
            "accepted_active_fraction_of_file": accepted_active_samples / total_samples,
            "accepted_window_fraction_of_file": accepted_window_samples / total_samples,
        },
    }


def serialize_plan(
    plan: dict[str, Any] | None,
    diarization: dict[str, Any] | None,
    qwen_result: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    serialized_plan: dict[str, Any] | None = None
    abstentions: dict[str, Any] | None = None
    if plan is not None:
        accepted_records = [turn_record(turn, index) for index, turn in enumerate(plan["accepted"])]
        short_records = [
            {**turn_record(turn, index), "reason": "final_turn_below_asr_minimum"}
            for index, turn in enumerate(plan["short_turns"])
        ]
        overlap_records = [
            span_record(span, "multiple_anonymous_speakers_active")
            for span in plan["overlap_spans"]
        ]
        raw_fragment_records = [
            {
                "reason": "raw_diarizer_fragment_below_predeclared_minimum",
                "source_index": item["source_index"],
                "start_s": seconds(item["start"]),
                "end_s": seconds(item["end"]),
                "duration_s": seconds(item["end"] - item["start"]),
                "anonymous_speaker": item["speaker"],
            }
            for item in plan["filtered"]
        ]
        serialized_plan = {
            "raw_diarization_segments": len(diarization["output"]["segments"]),
            "normalized_in_prefix_segments": len(plan["normalized"]),
            "kept_raw_segments": len(plan["kept"]),
            "filtered_raw_fragments": len(plan["filtered"]),
            "invalid_or_outside_prefix_segments": len(plan["invalid"]),
            "atomic_spans": len(plan["spans"]),
            "candidate_turns_before_final_minimum": len(plan["turns"]),
            "accepted_turns": accepted_records,
            "accepted_turn_count": len(accepted_records),
            "anonymous_speaker_turn_counts": dict(
                Counter(record["speaker"] for record in accepted_records)
            ),
            "coverage": plan["coverage"],
            "plan_sha256": stable_json_sha256(accepted_records),
        }
        abstentions = {
            "policy": (
                "Overlap is never double-transcribed. Raw fragments and final "
                "short turns below the predeclared thresholds are retained here "
                "rather than silently assigned. Anonymous labels are not identities."
            ),
            "raw_fragment_records": raw_fragment_records,
            "raw_fragment_record_count": len(raw_fragment_records),
            "raw_fragment_only_atomic_spans": [
                span_record(span, "only_filtered_raw_fragments_active")
                for span in plan["raw_fragment_spans"]
            ],
            "overlap_records": overlap_records,
            "overlap_record_count": len(overlap_records),
            "short_turn_records": short_records,
            "short_turn_record_count": len(short_records),
            "invalid_records": plan["invalid"],
            "unprocessed_generation_budget_records": (
                qwen_result["unprocessed_turns"] if qwen_result else []
            ),
        }
    return serialized_plan, abstentions
