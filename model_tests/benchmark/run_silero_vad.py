#!/usr/bin/env python3
"""Run the pinned product-CLI Silero VAD on the frozen CantoMap slice.

The score is speech-activity agreement against the union of CantoMap's two
ELAN speaker tiers.  It is not an independently adjudicated VAD benchmark and
does not evaluate speaker attribution or dense conversational turns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from audio_cli.vad import MODEL_SHA256, MODEL_URL, MODEL_VERSION, SileroOnnxVad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Prepared mono PCM16 16 kHz WAV")
    parser.add_argument("--reference", required=True, help="Prepared CantoMap reference JSON")
    parser.add_argument("--output", required=True, help="Run artifact JSON")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--exit-threshold", type=float, default=0.35)
    parser.add_argument("--min-speech-ms", type=int, default=100)
    parser.add_argument("--min-silence-ms", type=int, default=300)
    parser.add_argument("--speech-pad-ms", type=int, default=120)
    parser.add_argument("--score-frame-ms", type=float, default=10.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pcm16_mono_16k(path: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != 16_000
            or source.getcomptype() != "NONE"
        ):
            raise ValueError("--audio must be mono PCM16 16 kHz WAV")
        sample_count = source.getnframes()
        raw = source.readframes(sample_count)
    if len(raw) != sample_count * 2:
        raise ValueError("short PCM read")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sample_count / 16_000


def rss_bytes() -> int | None:
    try:
        return int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        ).strip()) * 1024
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def interval_active(
    intervals: list[tuple[float, float]], time_s: float
) -> bool:
    return any(start <= time_s < end for start, end in intervals)


def score_activity(
    reference: dict[str, Any],
    hypothesis: list[tuple[float, float]],
    duration_s: float,
    frame_s: float,
) -> dict[str, float | int | None]:
    if frame_s <= 0:
        raise ValueError("--score-frame-ms must be positive")
    references = [
        (float(item["start_ms"]) / 1000, float(item["end_ms"]) / 1000)
        for item in reference["segments"]
    ]
    total_frames = math.ceil(duration_s / frame_s)
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for index in range(total_frames):
        midpoint_s = min(duration_s, (index + 0.5) * frame_s)
        ref_active = interval_active(references, midpoint_s)
        hyp_active = interval_active(hypothesis, midpoint_s)
        if ref_active and hyp_active:
            counts["true_positive"] += 1
        elif hyp_active:
            counts["false_positive"] += 1
        elif ref_active:
            counts["false_negative"] += 1
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "frame_ms": round(frame_s * 1000, 6),
        "total_frames": total_frames,
        "true_positive_s": tp * frame_s,
        "false_positive_s": fp * frame_s,
        "false_negative_s": fn * frame_s,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio).resolve()
    reference_path = Path(args.reference).resolve()
    output_path = Path(args.output).resolve()
    samples, duration_s = load_pcm16_mono_16k(audio_path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_duration_s = float(reference["clip"]["duration_ms"]) / 1000
    if not math.isclose(duration_s, expected_duration_s, abs_tol=1 / 16_000):
        raise ValueError("audio duration does not match CantoMap reference")

    load_start = time.perf_counter()
    detector = SileroOnnxVad()
    load_s = time.perf_counter() - load_start
    inference_start = time.perf_counter()
    regions = detector.detect(
        samples,
        16_000,
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )
    inference_s = time.perf_counter() - inference_start
    region_dicts = [region.as_dict() for region in regions]
    hypothesis = [(region.start, region.end) for region in regions]
    score = score_activity(
        reference,
        hypothesis,
        duration_s,
        args.score_frame_ms / 1000,
    )
    result = {
        "schema_version": 1,
        "runner": "audio-processing-cli SileroOnnxVad",
        "epistemic_limit": (
            "One 149.9-second CantoMap slice. Speech activity is scored against "
            "the union of 83 ELAN speaker utterance intervals, not independently "
            "adjudicated VAD ground truth. This does not evaluate speaker identity, "
            "diarization, overlap handling, or other languages/dialects."
        ),
        "model": {
            "name": "Silero VAD",
            "version": MODEL_VERSION,
            "source": MODEL_URL,
            "expected_sha256": MODEL_SHA256,
            "resolved_path": str(detector.model_path),
            "resolved_sha256": sha256(detector.model_path),
        },
        "input": {
            "path": str(audio_path),
            "sha256": sha256(audio_path),
            "duration_s": duration_s,
            "sample_rate_hz": 16_000,
            "channels": 1,
        },
        "reference": {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "scope": "union of the two CantoMap ELAN speaker tiers",
            "interval_count": len(reference["segments"]),
        },
        "configuration": {
            "threshold": args.threshold,
            "exit_threshold": args.exit_threshold,
            "min_speech_ms": args.min_speech_ms,
            "min_silence_ms": args.min_silence_ms,
            "speech_pad_ms": args.speech_pad_ms,
        },
        "timing": {"model_load_s": load_s, "inference_s": inference_s},
        "memory": {"rss_after_inference_bytes": rss_bytes()},
        "output": {"region_count": len(region_dicts), "regions": region_dicts},
        "speech_activity_agreement": score,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "regions": len(region_dicts),
        "load_s": load_s,
        "inference_s": inference_s,
        "precision": score["precision"],
        "recall": score["recall"],
        "f1": score["f1"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
