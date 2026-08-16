#!/usr/bin/env python3
"""Compare diarization across exact repeated-audio windows.

The comparison is anonymous-speaker aware and frame based, so equivalent
speaker activity is not rejected merely because adjacent intervals were merged
differently. Exact interval counts and active-set transition agreement remain
visible as separate stability diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def segments_from_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    raw = run.get("output", run).get("segments", [])
    return [
        {
            "start_s": float(item.get("start_s", item.get("startTimeSeconds"))),
            "end_s": float(item.get("end_s", item.get("endTimeSeconds"))),
            "speaker": str(
                item.get("speaker", item.get("speaker_id", item.get("speakerId")))
            ),
        }
        for item in raw
    ]


def window_segments(
    segments: list[dict[str, Any]], start_s: float, duration_s: float
) -> list[dict[str, Any]]:
    end_s = start_s + duration_s
    return [
        {
            "start_s": max(float(item["start_s"]), start_s) - start_s,
            "end_s": min(float(item["end_s"]), end_s) - start_s,
            "speaker": item["speaker"],
        }
        for item in segments
        if float(item["start_s"]) < end_s and float(item["end_s"]) > start_s
    ]


def frame_activity(
    segments: list[dict[str, Any]], duration_s: float, frame_s: float
) -> list[frozenset[str]]:
    frame_count = math.ceil(duration_s / frame_s)
    labels = sorted({item["speaker"] for item in segments})
    differences = {label: [0] * (frame_count + 1) for label in labels}
    for item in segments:
        # Frame i represents its center, (i + 0.5) * frame_s.
        first = max(0, math.ceil(item["start_s"] / frame_s - 0.5))
        stop = min(frame_count, math.ceil(item["end_s"] / frame_s - 0.5))
        if first >= stop:
            continue
        differences[item["speaker"]][first] += 1
        differences[item["speaker"]][stop] -= 1
    counts = {label: 0 for label in labels}
    activity: list[frozenset[str]] = []
    for index in range(frame_count):
        for label in labels:
            counts[label] += differences[label][index]
        activity.append(frozenset(label for label in labels if counts[label] > 0))
    return activity


def transition_boundaries(activity: list[frozenset[str]], frame_s: float) -> list[float]:
    return [
        index * frame_s
        for index in range(1, len(activity))
        if activity[index] != activity[index - 1]
    ]


def boundary_agreement(
    reference: list[float], hypothesis: list[float], tolerance_s: float
) -> dict[str, Any]:
    unmatched = set(range(len(reference)))
    errors: list[float] = []
    for boundary in hypothesis:
        candidates = [
            index
            for index in unmatched
            if abs(reference[index] - boundary) <= tolerance_s
        ]
        if not candidates:
            continue
        match = min(candidates, key=lambda index: abs(reference[index] - boundary))
        unmatched.remove(match)
        errors.append(abs(reference[match] - boundary))
    precision = len(errors) / len(hypothesis) if hypothesis else None
    recall = len(errors) / len(reference) if reference else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tolerance_ms": tolerance_s * 1000,
        "reference_count": len(reference),
        "hypothesis_count": len(hypothesis),
        "matched_count": len(errors),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "maximum_matched_error_ms": max(errors, default=0) * 1000,
    }


def partial_reference_diagnostics(
    reference_segments: list[dict[str, Any]],
    window_activity: list[list[frozenset[str]]],
    duration_s: float,
    frame_s: float,
) -> dict[str, Any]:
    reference_activity = frame_activity(reference_segments, duration_s, frame_s)
    reference_speech = [bool(frame) for frame in reference_activity]
    halves = []
    for index, activity in enumerate(window_activity):
        scores = {}
        for label in sorted(set().union(*activity)):
            hypothesis = [label in frame for frame in activity]
            tp = sum(ref and hyp for ref, hyp in zip(reference_speech, hypothesis))
            fp = sum(not ref and hyp for ref, hyp in zip(reference_speech, hypothesis))
            fn = sum(ref and not hyp for ref, hyp in zip(reference_speech, hypothesis))
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            )
            scores[label] = {
                "overlap_s": tp * frame_s,
                "hypothesis_interval_s": (tp + fp) * frame_s,
                "reference_interval_s": (tp + fn) * frame_s,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        oracle = max(scores, key=lambda label: scores[label]["f1"] or -1)
        halves.append({
            "window": index,
            "by_hypothesis_speaker": scores,
            "oracle_best_hypothesis_speaker": oracle,
            "oracle_best": scores[oracle],
        })
    return {
        "epistemic_limit": (
            "The SpiCE reference labels participant utterance intervals only. "
            "This does not score the interviewer, full diarization, or VAD."
        ),
        "halves": halves,
    }


def compare_windows(
    reference: list[frozenset[str]],
    hypothesis: list[frozenset[str]],
    frame_s: float,
) -> dict[str, Any]:
    ref_labels = sorted(set().union(*reference))
    hyp_labels = sorted(set().union(*hypothesis))
    if len(ref_labels) != len(hyp_labels):
        raise ValueError("repeat windows emitted different speaker counts")
    best: tuple[int, dict[str, str], list[frozenset[str]]] | None = None
    for targets in itertools.permutations(ref_labels):
        mapping = dict(zip(hyp_labels, targets))
        mapped = [frozenset(mapping[label] for label in frame) for frame in hypothesis]
        error = sum(len(left ^ right) for left, right in zip(reference, mapped))
        if best is None or error < best[0]:
            best = (error, mapping, mapped)
    assert best is not None
    error, mapping, mapped = best
    frame_count = len(reference)
    reference_speaker_frames = sum(len(frame) for frame in reference)
    exact = sum(left == right for left, right in zip(reference, mapped))
    speech = sum(bool(left) == bool(right) for left, right in zip(reference, mapped))
    voiced_union = sum(bool(left or right) for left, right in zip(reference, mapped))
    voiced_exact = sum(
        left == right and bool(left or right)
        for left, right in zip(reference, mapped)
    )
    return {
        "optimal_anonymous_speaker_mapping": mapping,
        "speaker_frame_error_rate": (
            error / reference_speaker_frames if reference_speaker_frames else None
        ),
        "speaker_frame_error_s": error * frame_s,
        "reference_speaker_s": reference_speaker_frames * frame_s,
        "exact_active_set_frame_fraction_all_audio": exact / frame_count,
        "exact_active_set_frame_fraction_voiced_union": (
            voiced_exact / voiced_union if voiced_union else None
        ),
        "speech_activity_frame_agreement": speech / frame_count,
        "mapped_hypothesis_activity": mapped,
    }


def interval_pair_counts(
    reference: list[dict[str, Any]],
    hypothesis: list[dict[str, Any]],
    mapping: dict[str, str],
    tolerances_s: list[float],
) -> dict[str, int]:
    mapped = [
        {**item, "speaker": mapping[item["speaker"]]} for item in hypothesis
    ]
    counts: dict[str, int] = {}
    for tolerance in tolerances_s:
        unmatched = set(range(len(mapped)))
        matched = 0
        for item in reference:
            candidates = [
                index
                for index in unmatched
                if mapped[index]["speaker"] == item["speaker"]
                and abs(mapped[index]["start_s"] - item["start_s"]) <= tolerance
                and abs(mapped[index]["end_s"] - item["end_s"]) <= tolerance
            ]
            if not candidates:
                continue
            match = min(
                candidates,
                key=lambda index: abs(mapped[index]["start_s"] - item["start_s"])
                + abs(mapped[index]["end_s"] - item["end_s"]),
            )
            unmatched.remove(match)
            matched += 1
        counts[f"within_{tolerance * 1000:g}ms"] = matched
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--period-seconds", type=float, required=True)
    parser.add_argument("--frame-ms", type=float, default=10.0)
    parser.add_argument("--boundary-tolerance-ms", type=float, default=50.0)
    parser.add_argument(
        "--partial-reference",
        help="Optional repeated-period participant-only reference manifest",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.period_seconds <= 0 or args.frame_ms <= 0:
        parser.error("period-seconds and frame-ms must be positive")

    run_path = Path(args.run).resolve()
    run = json.loads(run_path.read_text())
    segments = segments_from_run(run)
    frame_s = args.frame_ms / 1000
    window_items = [
        window_segments(segments, index * args.period_seconds, args.period_seconds)
        for index in range(2)
    ]
    activity = [
        frame_activity(items, args.period_seconds, frame_s) for items in window_items
    ]
    comparison = compare_windows(activity[0], activity[1], frame_s)
    mapped_hypothesis = comparison.pop("mapped_hypothesis_activity")
    interval_matches = interval_pair_counts(
        window_items[0],
        window_items[1],
        comparison["optimal_anonymous_speaker_mapping"],
        [0.01, 0.1],
    )
    boundary_score = boundary_agreement(
        transition_boundaries(activity[0], frame_s),
        transition_boundaries(mapped_hypothesis, frame_s),
        args.boundary_tolerance_ms / 1000,
    )
    output = {
        "schema_version": 1,
        "metric": "repeated-audio diarization structural agreement",
        "epistemic_limit": (
            "Both windows contain the same audio. Agreement is a deterministic "
            "duration-stability diagnostic, not independent diarization quality "
            "evidence or evidence about unseen one-hour recordings."
        ),
        "run": str(run_path),
        "run_sha256": sha256_file(run_path),
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "period_s": args.period_seconds,
        "frame_ms": args.frame_ms,
        "window_interval_counts": [len(items) for items in window_items],
        "same_label_intervals_matching_both_bounds": interval_matches,
        "interval_exactness_interpretation": (
            "The repeat windows are not interval-for-interval stable. Frame activity "
            "agreement must not be used to hide interval or boundary differences."
        ),
        "comparison": comparison,
        "active_set_transition_agreement": boundary_score,
    }
    if args.partial_reference:
        reference_path = Path(args.partial_reference).resolve()
        reference = json.loads(reference_path.read_text())
        utterances = reference.get("utterances")
        if not utterances:
            raise ValueError("partial reference must contain utterances")
        reference_segments = [
            {
                "start_s": float(item["start_s"]),
                "end_s": float(item["end_s"]),
                "speaker": str(item["speaker"]),
            }
            for item in utterances
        ]
        if len({item["speaker"] for item in reference_segments}) != 1:
            raise ValueError("partial reference must contain exactly one speaker")
        output["partial_reference"] = {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
            **partial_reference_diagnostics(
                reference_segments, activity, args.period_seconds, frame_s
            ),
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "speaker_frame_error_rate": comparison["speaker_frame_error_rate"],
        "exact_active_set_fraction": comparison[
            "exact_active_set_frame_fraction_all_audio"
        ],
        "transition_f1": boundary_score["f1"],
        "interval_counts": output["window_interval_counts"],
    }))


if __name__ == "__main__":
    main()
