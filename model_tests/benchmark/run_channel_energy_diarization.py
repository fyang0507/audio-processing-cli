#!/usr/bin/env python3
"""Deterministic stereo channel-energy diarization diagnostic.

This is an oracle-labeled corpus diagnostic, not a production diarizer. It
never assigns a person or role to a channel. Reference labels are used only to
describe channel dominance after the threshold ladder has been declared; run
artifacts retain anonymous ``channel_left`` and ``channel_right`` labels for
the existing diarization scorer to map optimally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import wave
from array import array
from pathlib import Path
from typing import Any


PCM16_SCALE = 32768.0
DB_FLOOR = -120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Original stereo PCM WAV")
    parser.add_argument("--reference", required=True, help="Prepared CantoMap reference JSON")
    parser.add_argument("--output-dir", required=True, help="Generated run-artifact directory")
    parser.add_argument("--frame-ms", type=float, default=10.0)
    parser.add_argument("--activity-dbfs", type=float, default=-45.0)
    parser.add_argument(
        "--dominance-db",
        type=float,
        nargs="+",
        default=[0.0, 3.0, 6.0],
        help="Predeclared absolute L/R difference ladder in dB",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dbfs(sum_squares: int, sample_count: int) -> float:
    if sum_squares <= 0 or sample_count <= 0:
        return DB_FLOOR
    rms = math.sqrt(sum_squares / sample_count) / PCM16_SCALE
    return max(DB_FLOOR, 20.0 * math.log10(rms))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def active_reference_speakers(
    segments: list[dict[str, Any]], time_s: float
) -> set[str]:
    return {
        str(item["speaker"])
        for item in segments
        if float(item["start_ms"]) / 1000 <= time_s < float(item["end_ms"]) / 1000
    }


def merge_frame_labels(
    frame_labels: list[set[str]], frame_s: float, duration_s: float
) -> list[dict[str, Any]]:
    open_starts: dict[str, int] = {}
    segments: list[dict[str, Any]] = []
    all_labels = sorted(set().union(*frame_labels)) if frame_labels else []
    for index, labels in enumerate([*frame_labels, set()]):
        for label in all_labels:
            if label in labels and label not in open_starts:
                open_starts[label] = index
            elif label not in labels and label in open_starts:
                start_index = open_starts.pop(label)
                segments.append({
                    "start_s": round(start_index * frame_s, 6),
                    "end_s": round(min(duration_s, index * frame_s), 6),
                    "speaker": label,
                })
    return sorted(segments, key=lambda item: (item["start_s"], item["end_s"], item["speaker"]))


def main() -> int:
    args = parse_args()
    if args.frame_ms <= 0:
        raise ValueError("--frame-ms must be positive")
    if len(set(args.dominance_db)) != len(args.dominance_db):
        raise ValueError("--dominance-db values must be unique")
    if any(value < 0 for value in args.dominance_db):
        raise ValueError("--dominance-db values must be nonnegative")

    audio_path = Path(args.audio).resolve()
    reference_path = Path(args.reference).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads(reference_path.read_text())
    clip = reference["clip"]
    clip_start_s = float(clip["source_start_ms"]) / 1000
    clip_end_s = float(clip["source_end_ms"]) / 1000
    duration_s = clip_end_s - clip_start_s

    with wave.open(str(audio_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        if channels != 2 or sample_width != 2 or source.getcomptype() != "NONE":
            raise ValueError("source must be uncompressed stereo PCM16 WAV")
        frame_samples_float = sample_rate * args.frame_ms / 1000
        frame_samples = round(frame_samples_float)
        if not math.isclose(frame_samples, frame_samples_float, abs_tol=1e-9):
            raise ValueError("frame duration must resolve to an integral source-sample count")
        start_frame = round(clip_start_s * sample_rate)
        clip_frames = round(duration_s * sample_rate)
        source.setpos(start_frame)
        pcm_bytes = source.readframes(clip_frames)

    pcm = array("h")
    pcm.frombytes(pcm_bytes)
    if sys.byteorder != "little":
        pcm.byteswap()
    if len(pcm) != clip_frames * 2:
        raise ValueError("short PCM read")
    if clip_frames % frame_samples:
        raise ValueError("clip length is not an exact number of analysis frames")

    frame_count = clip_frames // frame_samples
    frames: list[dict[str, float]] = []
    total_left_sq = 0
    total_right_sq = 0
    total_cross = 0
    for frame_index in range(frame_count):
        begin = frame_index * frame_samples * 2
        end = begin + frame_samples * 2
        left_sq = 0
        right_sq = 0
        cross = 0
        for offset in range(begin, end, 2):
            left = pcm[offset]
            right = pcm[offset + 1]
            left_sq += left * left
            right_sq += right * right
            cross += left * right
        left_dbfs = dbfs(left_sq, frame_samples)
        right_dbfs = dbfs(right_sq, frame_samples)
        frames.append({
            "left_dbfs": left_dbfs,
            "right_dbfs": right_dbfs,
            "left_minus_right_db": left_dbfs - right_dbfs,
        })
        total_left_sq += left_sq
        total_right_sq += right_sq
        total_cross += cross

    frame_s = args.frame_ms / 1000
    reference_segments = reference["segments"]
    reference_speakers = sorted({str(item["speaker"]) for item in reference_segments})
    exclusive_differences: dict[str, list[float]] = {
        speaker: [] for speaker in reference_speakers
    }
    for index, frame in enumerate(frames):
        midpoint_s = (index + 0.5) * frame_s
        active = active_reference_speakers(reference_segments, midpoint_s)
        if len(active) == 1:
            exclusive_differences[next(iter(active))].append(
                frame["left_minus_right_db"]
            )

    oracle_dominance = {}
    for speaker, values in exclusive_differences.items():
        median = statistics.median(values) if values else None
        oracle_dominance[speaker] = {
            "exclusive_frames": len(values),
            "exclusive_duration_s": round(len(values) * frame_s, 6),
            "left_minus_right_db_median": median,
            "left_minus_right_db_p10": percentile(values, 0.10),
            "left_minus_right_db_p90": percentile(values, 0.90),
            "left_louder_fraction": (
                sum(value > 0 for value in values) / len(values) if values else None
            ),
            "right_louder_fraction": (
                sum(value < 0 for value in values) / len(values) if values else None
            ),
            "oracle_dominant_channel": (
                "channel_left" if median is not None and median > 0
                else "channel_right" if median is not None and median < 0
                else None
            ),
        }

    threshold_runs = []
    for dominance_db in args.dominance_db:
        labels_by_frame: list[set[str]] = []
        for frame in frames:
            maximum = max(frame["left_dbfs"], frame["right_dbfs"])
            difference = frame["left_minus_right_db"]
            if maximum < args.activity_dbfs:
                labels = set()
            elif difference > dominance_db:
                labels = {"channel_left"}
            elif difference < -dominance_db:
                labels = {"channel_right"}
            else:
                labels = {"channel_left", "channel_right"}
            labels_by_frame.append(labels)
        segments = merge_frame_labels(labels_by_frame, frame_s, duration_s)
        threshold_slug = f"{dominance_db:g}".replace(".", "p")
        run_path = output_dir / f"channel_energy_dominance_{threshold_slug}db.json"
        run = {
            "schema_version": 1,
            "runner": "deterministic-stereo-channel-energy-diagnostic",
            "epistemic_limit": (
                "Anonymous channel activity derived from one CantoMap stereo file. "
                "It is not speaker identification, a learned diarizer, or a "
                "generalizable production threshold."
            ),
            "input": {
                "path": str(audio_path),
                "sha256": sha256(audio_path),
                "clip_start_s": clip_start_s,
                "clip_end_s": clip_end_s,
                "duration_s": duration_s,
                "sample_rate_hz": sample_rate,
                "channels": channels,
            },
            "configuration": {
                "frame_ms": args.frame_ms,
                "activity_dbfs": args.activity_dbfs,
                "dominance_db": dominance_db,
                "rule": (
                    "Below activity gate: no label. Above gate and absolute L/R "
                    "difference greater than dominance threshold: louder anonymous "
                    "channel only. Otherwise: both anonymous channel labels."
                ),
            },
            "output": {"segments": segments},
        }
        run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        label_durations = {
            label: round(sum(
                item["end_s"] - item["start_s"]
                for item in segments if item["speaker"] == label
            ), 6)
            for label in ("channel_left", "channel_right")
        }
        threshold_runs.append({
            "dominance_db": dominance_db,
            "run_path": str(run_path),
            "run_sha256": sha256(run_path),
            "segment_count": len(segments),
            "label_durations_s": label_durations,
        })

    denominator = frame_count * frame_samples
    result = {
        "schema_version": 1,
        "diagnostic": "CantoMap stereo channel-energy activity",
        "epistemic_limit": (
            "Reference labels are used only for an oracle description of channel "
            "dominance. Channel names do not identify people or roles. Thresholds "
            "were predeclared and must not be selected as a production default from "
            "this same slice."
        ),
        "source": {
            "path": str(audio_path),
            "sha256": sha256(audio_path),
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "clip_start_s": clip_start_s,
            "clip_end_s": clip_end_s,
            "clip_duration_s": duration_s,
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "speaker_labels": reference_speakers,
            "segment_count": len(reference_segments),
            "label_role": "opaque corpus IDs; no person or interview-role inference",
        },
        "configuration": {
            "frame_ms": args.frame_ms,
            "activity_dbfs": args.activity_dbfs,
            "predeclared_dominance_db_ladder": args.dominance_db,
        },
        "clip_channel_measurements": {
            "left_rms_dbfs": dbfs(total_left_sq, denominator),
            "right_rms_dbfs": dbfs(total_right_sq, denominator),
            "uncentered_channel_correlation": (
                total_cross / math.sqrt(total_left_sq * total_right_sq)
                if total_left_sq and total_right_sq else None
            ),
        },
        "oracle_speaker_exclusive_channel_dominance": oracle_dominance,
        "threshold_runs": threshold_runs,
    }
    summary_path = output_dir / "channel_energy_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "summary": str(summary_path),
        "oracle_dominance": oracle_dominance,
        "threshold_runs": threshold_runs,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
