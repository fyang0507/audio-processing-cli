#!/usr/bin/env python3
"""Compare repeated-audio windows as a structural long-form stability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def normalized_segments(run: dict[str, Any]) -> list[dict[str, Any]]:
    raw = run["output"].get("segments", [])
    segments = []
    for item in raw:
        segments.append({
            "start_s": float(item.get("start_s", item.get("start_time", 0))),
            "end_s": float(item.get("end_s", item.get("end_time", 0))),
            "text": item.get("text", ""),
            "speaker": item.get("speaker_id", item.get("speaker")),
        })
    return segments


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--period-seconds", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--timestamp-tolerance-ms", type=float, default=2.0,
        help="Tolerance for paired timestamps after repeat-window rebasing",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.period_seconds <= 0 or args.repeats < 2:
        parser.error("period-seconds must be positive and repeats must be at least 2")

    run_path = Path(args.run)
    run = json.loads(run_path.read_text())
    segments = normalized_segments(run)
    crossing = [
        item for item in segments
        if any(
            item["start_s"] < boundary < item["end_s"]
            for boundary in (
                args.period_seconds * index for index in range(1, args.repeats)
            )
        )
    ]
    windows = []
    for index in range(args.repeats):
        start = index * args.period_seconds
        end = start + args.period_seconds
        items = [
            {
                **item,
                "start_s": round(item["start_s"] - start, 6),
                "end_s": round(item["end_s"] - start, 6),
            }
            for item in segments
            if item["start_s"] >= start and item["end_s"] <= end
        ]
        serialized = canonical_json(items)
        text_serialized = canonical_json([item["text"] for item in items])
        windows.append({
            "index": index,
            "start_s": start,
            "end_s": end,
            "segment_count": len(items),
            "normalized_segments_sha256": hashlib.sha256(
                serialized.encode()
            ).hexdigest(),
            "text_sequence_sha256": hashlib.sha256(
                text_serialized.encode()
            ).hexdigest(),
        })
    first = windows[0]
    first_items = [
        {
            **item,
            "start_s": item["start_s"],
            "end_s": item["end_s"],
        }
        for item in segments
        if item["start_s"] >= 0 and item["end_s"] <= args.period_seconds
    ]
    paired_drift: list[dict[str, Any]] = []
    for index in range(1, args.repeats):
        start = index * args.period_seconds
        end = start + args.period_seconds
        items = [
            {
                **item,
                "start_s": item["start_s"] - start,
                "end_s": item["end_s"] - start,
            }
            for item in segments
            if item["start_s"] >= start and item["end_s"] <= end
        ]
        pairable = (
            len(items) == len(first_items)
            and all(
                left["text"] == right["text"]
                and left["speaker"] == right["speaker"]
                for left, right in zip(first_items, items)
            )
        )
        drifts = [
            abs(left[key] - right[key])
            for left, right in zip(first_items, items)
            for key in ("start_s", "end_s")
        ] if pairable else []
        paired_drift.append({
            "compared_to_window": 0,
            "window": index,
            "pairable_text_and_speaker_sequence": pairable,
            "paired_timestamp_count": len(drifts),
            "maximum_absolute_timestamp_drift_s": max(drifts, default=None),
            "median_absolute_timestamp_drift_s": statistics.median(drifts)
            if drifts else None,
            "p95_absolute_timestamp_drift_s": sorted(drifts)[
                math.ceil(0.95 * len(drifts)) - 1
            ] if drifts else None,
            "within_timestamp_tolerance": bool(drifts) and max(drifts) <= (
                args.timestamp_tolerance_ms / 1000
            ),
        })
    result = {
        "schema_version": 2,
        "metric": "repeated-fixture structural repeat consistency",
        "epistemic_limit": (
            "Equality across duplicated audio is a deterministic stability gate. It "
            "does not measure transcript accuracy or general long-form robustness."
        ),
        "run": str(run_path.resolve()),
        "period_s": args.period_seconds,
        "repeats": args.repeats,
        "segments_crossing_repeat_boundaries": len(crossing),
        "windows": windows,
        "timestamp_tolerance_ms": args.timestamp_tolerance_ms,
        "paired_window_drift": paired_drift,
        "all_normalized_segments_equal": all(
            item["normalized_segments_sha256"]
            == first["normalized_segments_sha256"] for item in windows
        ),
        "all_text_sequences_equal": all(
            item["text_sequence_sha256"] == first["text_sequence_sha256"]
            for item in windows
        ),
        "all_text_speaker_sequences_and_timestamps_within_tolerance": all(
            item["pairable_text_and_speaker_sequence"]
            and item["within_timestamp_tolerance"] for item in paired_drift
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "all_segments_equal": result["all_normalized_segments_equal"],
        "all_text_equal": result["all_text_sequences_equal"],
        "all_within_tolerance": result[
            "all_text_speaker_sequences_and_timestamps_within_tolerance"
        ],
        "maximum_timestamp_drift_s": max(
            (item["maximum_absolute_timestamp_drift_s"] for item in paired_drift
             if item["maximum_absolute_timestamp_drift_s"] is not None),
            default=None,
        ),
        "counts": [item["segment_count"] for item in windows],
    }))


if __name__ == "__main__":
    main()
