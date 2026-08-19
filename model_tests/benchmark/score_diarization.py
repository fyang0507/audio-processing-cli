#!/usr/bin/env python3
"""Score speaker attribution and annotation-order speaker-change boundaries.

This reports mechanical agreement with ELAN annotations. It is not a speaker-
identification metric and does not validate interview-behavior inferences.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


def reference_segments(reference: dict[str, Any]) -> list[dict[str, Any]]:
    if "utterances" in reference:
        return [
            {
                "start_s": float(item["start_s"]),
                "end_s": float(item["end_s"]),
                "speaker": str(item["speaker"]),
            }
            for item in reference["utterances"]
        ]
    if "segments" not in reference:
        raise ValueError("unsupported diarization reference")
    return [
        {
            "start_s": float(item["start_ms"]) / 1000,
            "end_s": float(item["end_ms"]) / 1000,
            "speaker": str(item["speaker"]),
        }
        for item in reference["segments"]
    ]


def hypothesis_segments(run: dict[str, Any]) -> list[dict[str, Any]]:
    container = run.get("output", run)
    raw = container.get("segments")
    if raw is None:
        raw = container.get("result", {}).get("sentences", [])
    segments = []
    for item in raw:
        speaker = item.get("speaker_id", item.get("speaker", item.get("speakerId")))
        if speaker is None:
            continue
        start = item.get("start_s", item.get("start_time", item.get("startTimeSeconds")))
        end = item.get("end_s", item.get("end_time", item.get("endTimeSeconds")))
        if start is None or end is None:
            raise ValueError("speaker-labeled hypothesis segment lacks a time bound")
        segments.append({
            "start_s": float(start),
            "end_s": float(end),
            "speaker": str(speaker),
        })
    return segments


def active_speakers(segments: list[dict[str, Any]], time_s: float) -> set[str]:
    return {
        item["speaker"] for item in segments
        if item["start_s"] <= time_s < item["end_s"]
    }


def candidate_mappings(hyp_speakers: list[str], ref_speakers: list[str]):
    if not hyp_speakers:
        yield {}
        return
    targets: list[str | None] = [*ref_speakers]
    targets.extend([None] * max(0, len(hyp_speakers) - len(ref_speakers)))
    seen: set[tuple[str | None, ...]] = set()
    for assignment in itertools.permutations(targets, len(hyp_speakers)):
        if assignment in seen:
            continue
        seen.add(assignment)
        yield dict(zip(hyp_speakers, assignment))


def mapped_hypothesis(speakers: set[str], mapping: dict[str, str | None]) -> set[str]:
    return {mapped for speaker in speakers if (mapped := mapping.get(speaker)) is not None}


def frame_error(ref: set[str], hyp: set[str]) -> tuple[int, int, int]:
    missed = max(0, len(ref) - len(hyp))
    false_alarm = max(0, len(hyp) - len(ref))
    confusion = min(len(ref), len(hyp)) - len(ref & hyp)
    return missed, false_alarm, confusion


def score_frames(
    references: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    duration_s: float,
    frame_s: float,
    collar_s: float,
    include_overlap: bool,
) -> dict[str, Any]:
    boundaries = [
        boundary
        for item in references
        for boundary in (item["start_s"], item["end_s"])
    ]
    frames: list[tuple[set[str], set[str]]] = []
    total_frames = math.ceil(duration_s / frame_s)
    for index in range(total_frames):
        time_s = min(duration_s, (index + 0.5) * frame_s)
        ref = active_speakers(references, time_s)
        if not include_overlap and len(ref) > 1:
            continue
        if collar_s and any(abs(time_s - boundary) < collar_s for boundary in boundaries):
            continue
        frames.append((ref, active_speakers(hypotheses, time_s)))

    ref_speakers = sorted({item["speaker"] for item in references})
    hyp_speakers = sorted({item["speaker"] for item in hypotheses})
    best: tuple[int, dict[str, str | None], tuple[int, int, int]] | None = None
    for mapping in candidate_mappings(hyp_speakers, ref_speakers):
        counts = [0, 0, 0]
        for ref, hyp in frames:
            for offset, count in enumerate(frame_error(ref, mapped_hypothesis(hyp, mapping))):
                counts[offset] += count
        total_error = sum(counts)
        if best is None or total_error < best[0]:
            best = (total_error, mapping, tuple(counts))
    assert best is not None
    error, mapping, (missed, false_alarm, confusion) = best
    reference_speaker_frames = sum(len(ref) for ref, _ in frames)
    return {
        "frame_ms": round(frame_s * 1000, 6),
        "collar_ms": round(collar_s * 1000, 6),
        "overlap_scored": include_overlap,
        "evaluated_audio_s": len(frames) * frame_s,
        "evaluated_fraction": len(frames) / total_frames if total_frames else None,
        "reference_speaker_s": reference_speaker_frames * frame_s,
        "mapping": mapping,
        "missed_speech_s": missed * frame_s,
        "false_alarm_s": false_alarm * frame_s,
        "speaker_confusion_s": confusion * frame_s,
        "diarization_error_rate": error / reference_speaker_frames if reference_speaker_frames else None,
    }


def change_boundaries(segments: list[dict[str, Any]]) -> list[float]:
    ordered = sorted(segments, key=lambda item: (item["start_s"], item["end_s"]))
    boundaries: list[float] = []
    previous_speaker: str | None = None
    for item in ordered:
        if previous_speaker is not None and item["speaker"] != previous_speaker:
            boundaries.append(item["start_s"])
        previous_speaker = item["speaker"]
    return boundaries


def score_speaker_change_boundaries(
    references: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    tolerance_s: float,
) -> dict[str, Any]:
    ref = change_boundaries(references)
    hyp = change_boundaries(hypotheses)
    unmatched = set(range(len(ref)))
    errors: list[float] = []
    for boundary in hyp:
        candidates = [index for index in unmatched if abs(ref[index] - boundary) <= tolerance_s]
        if not candidates:
            continue
        match = min(candidates, key=lambda index: abs(ref[index] - boundary))
        unmatched.remove(match)
        errors.append(abs(ref[match] - boundary))
    matched = len(errors)
    precision = matched / len(hyp) if hyp else None
    recall = matched / len(ref) if ref else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall else None
    )
    return {
        "definition": "chronological annotation-order speaker changes",
        "tolerance_s": tolerance_s,
        "reference_boundaries": len(ref),
        "hypothesis_boundaries": len(hyp),
        "matched_boundaries": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_absolute_error_median_s": statistics.median(errors) if errors else None,
        "matched_absolute_error_p95_s": (
            sorted(errors)[math.ceil(0.95 * len(errors)) - 1] if errors else None
        ),
    }


def score_single_speaker_utterance_intervals(
    references: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    duration_s: float,
    frame_s: float,
) -> dict[str, Any]:
    ref_speakers = sorted({item["speaker"] for item in references})
    if len(ref_speakers) != 1:
        raise ValueError("single-speaker interval scoring requires one reference speaker")
    ref_speaker = ref_speakers[0]
    hypothesis_speakers = sorted({item["speaker"] for item in hypotheses})
    counts = {
        speaker: {"true_positive": 0, "false_positive": 0, "false_negative": 0}
        for speaker in hypothesis_speakers
    }
    for index in range(math.ceil(duration_s / frame_s)):
        time_s = min(duration_s, (index + 0.5) * frame_s)
        ref_active = ref_speaker in active_speakers(references, time_s)
        hyp_active = active_speakers(hypotheses, time_s)
        for speaker in hypothesis_speakers:
            if ref_active and speaker in hyp_active:
                counts[speaker]["true_positive"] += 1
            elif not ref_active and speaker in hyp_active:
                counts[speaker]["false_positive"] += 1
            elif ref_active and speaker not in hyp_active:
                counts[speaker]["false_negative"] += 1
    scores = {}
    for speaker, speaker_counts in counts.items():
        tp = speaker_counts["true_positive"]
        fp = speaker_counts["false_positive"]
        fn = speaker_counts["false_negative"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall else None
        )
        scores[speaker] = {
            "overlap_s": tp * frame_s,
            "hypothesis_utterance_interval_s": (tp + fp) * frame_s,
            "reference_utterance_interval_s": (tp + fn) * frame_s,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    best_speaker = max(
        scores, key=lambda speaker: scores[speaker]["f1"] or -1,
        default=None,
    )
    return {
        "reference_speaker": ref_speaker,
        "frame_ms": frame_s * 1000,
        "by_hypothesis_speaker": scores,
        "oracle_best_hypothesis_speaker": best_speaker,
        "oracle_best": scores.get(best_speaker) if best_speaker is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-ms", type=float, default=10.0)
    parser.add_argument("--collar-ms", type=float, default=250.0)
    parser.add_argument(
        "--speaker-change-tolerance-s", "--turn-tolerance-s",
        dest="speaker_change_tolerance_s", type=float, default=1.0,
        help="Tolerance for annotation-order speaker-change boundary matching",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference)
    run_path = Path(args.run)
    reference = json.loads(reference_path.read_text())
    run = json.loads(run_path.read_text())
    references = reference_segments(reference)
    hypotheses = hypothesis_segments(run)
    is_partial_reference = "utterances" in reference
    duration_s = (
        float(reference["source"]["duration_s"])
        if is_partial_reference
        else float(reference["clip"]["duration_ms"]) / 1000
    )
    frame_s = args.frame_ms / 1000
    collar_s = args.collar_ms / 1000
    result: dict[str, Any] = {
        "schema_version": 2,
        "metric": "speaker-time attribution and annotation-order speaker-change agreement",
        "epistemic_limit": (
            "These scores measure agreement with one CantoMap ELAN slice after an "
            "optimal anonymous-speaker mapping. They do not identify people, judge "
            "conversation quality, or validate behavioral analysis. CantoMap's ELAN "
            "utterance annotations are not independently adjudicated diarization "
            "ground truth; dense boundaries and overlaps make speaker-change matching "
            "approximate."
        ),
        "reference": str(reference_path.resolve()),
        "run": str(run_path.resolve()),
        "reference_speakers": sorted({item["speaker"] for item in references}),
        "hypothesis_speakers": sorted({item["speaker"] for item in hypotheses}),
        "reference_interval_count": len(references),
        "hypothesis_speaker_labeled_interval_count": len(hypotheses),
    }
    if is_partial_reference:
        result.update({
            "metric": "partial-reference participant utterance-interval overlap",
            "epistemic_limit": (
                "The corpus supplies hand-corrected participant utterance intervals "
                "only, not independently frame-adjudicated VAD. This oracle mapping "
                "diagnostic does not score interviewer intervals, full diarization, "
                "speaker identity, or behavioral-analysis validity."
            ),
            "participant_utterance_interval_overlap": score_single_speaker_utterance_intervals(
                references, hypotheses, duration_s, frame_s
            ),
        })
    else:
        result.update({
        "exclusive_no_collar": score_frames(
            references, hypotheses, duration_s, frame_s, 0.0, False
        ),
        "exclusive_with_collar": score_frames(
            references, hypotheses, duration_s, frame_s, collar_s, False
        ),
        "overlap_included_no_collar": score_frames(
            references, hypotheses, duration_s, frame_s, 0.0, True
        ),
        "speaker_change_boundaries": score_speaker_change_boundaries(
            references, hypotheses, args.speaker_change_tolerance_s
        ),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if is_partial_reference:
        print(json.dumps({
            "oracle_best_speaker": result["participant_utterance_interval_overlap"][
                "oracle_best_hypothesis_speaker"
            ],
            "oracle_best_f1": result["participant_utterance_interval_overlap"][
                "oracle_best"
            ]["f1"],
        }))
    else:
        print(json.dumps({
            "exclusive_der_250ms": result["exclusive_with_collar"][
                "diarization_error_rate"
            ],
            "speaker_change_boundary_f1": result["speaker_change_boundaries"]["f1"],
        }))


if __name__ == "__main__":
    main()
