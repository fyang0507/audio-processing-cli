#!/usr/bin/env python3
"""Does the MLX forced aligner reproduce the recorded torch aligner's output?

`mlx-audio==0.4.5` — the version the Qwen stacks are already pinned to — ships
`qwen3_forced_aligner`. If it aligns equivalently to `qwen-asr`'s torch
`Qwen3ForcedAligner`, the `word_timestamps` add-on stops needing a PyTorch environment and
the fast long-form path runs entirely in `mlx`. That is an environment-layout question, so
it is answered by rerunning the exact recorded case rather than by reading model cards.

The recorded case is `model_tests/forced_aligner/test_multispeaker_pipeline.py`, whose output
`result_hybrid_multispeaker.json` is what the spec documents cite for the aligner: 19
segments, 17 with word streams and 2 `[Environmental Sounds]` segments with none. This probe
mirrors that script's inputs exactly — same wav, same VibeVoice segment list, same
CJK-detection rule for the language argument, same offset-and-round arithmetic — and changes
only the implementation.

Two differences from the recorded run are deliberate and neither is a bug:

- The torch run used **fp32 on CPU**; this one uses the **8-bit MLX** checkpoint on Metal.
  Timing is therefore not a like-for-like speed comparison, and a token whose bound differs
  may differ because of quantization rather than because of the implementation. The probe
  reports the disagreement; it does not attribute it.
- The torch run loaded a local `local_dir` snapshot of `Qwen/Qwen3-ForcedAligner-0.6B`; this
  one loads `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`, a different repository. Both
  revisions are recorded below.

Run it in a provisioned `mlx` environment:

    /path/to/envs/mlx/bin/python model_tests/benchmark/run_mlx_forced_aligner_probe.py \
      --output model_tests/benchmark_runs/mlx_forced_aligner_probe.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WAV = REPO / "model_tests/forced_aligner/sample2_multispeaker.wav"
VIBEVOICE_SEGMENTS = REPO / "model_tests/vibevoice/VibeVoice/result_vibevoice_multispeaker.json"
RECORDED_TORCH = REPO / "model_tests/forced_aligner/result_hybrid_multispeaker.json"

ALIGNER_REPO = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
ALIGNER_REVISION = "0e1a68e91d815300c7c9754b2a7639378b23db15"
TORCH_ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
TORCH_ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"

# The recorded script's language rule, copied rather than improved: a probe that changes the
# input alongside the implementation cannot attribute a difference to either.
CJK_RE = re.compile(r"[一-鿿]")


def strip_punctuation(text: str) -> str:
    without = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", "", without).lower()


def peak_memory_bytes() -> int | None:
    import mlx.core as mx

    for getter in ("get_peak_memory", "metal"):
        target = getattr(mx, getter, None)
        if target is None:
            continue
        function = target if callable(target) else getattr(target, "get_peak_memory", None)
        if callable(function):
            try:
                return int(function())
            except Exception:  # pragma: no cover - telemetry, never load-bearing
                return None
    return None


def align_all(model, audio, sample_rate: int, segments: list[dict]) -> tuple[list[dict], float]:
    """Mirror test_multispeaker_pipeline.py's loop against the MLX model."""
    import numpy as np

    output: list[dict] = []
    total = 0.0
    for segment in segments:
        speaker, text = segment["Speaker"], segment["Content"]
        start_s, end_s = segment["Start"], segment["End"]

        if speaker == "N/A" or not text.strip():
            output.append({"speaker": speaker, "start": start_s, "end": end_s,
                           "text": text, "words": None})
            continue

        language = "Chinese" if CJK_RE.search(text) else "English"
        clip = np.asarray(audio[int(start_s * sample_rate):int(end_s * sample_rate)],
                          dtype=np.float32)

        started = time.time()
        try:
            result = model.generate(audio=clip, text=text, language=language)
            elapsed = time.time() - started
            total += elapsed
            words = [
                {"text": item.text,
                 "start": round(item.start_time + start_s, 3),
                 "end": round(item.end_time + start_s, 3)}
                for item in result
            ]
            error = None
        except Exception as exc:  # a failure to align is a result, not a crash
            elapsed = time.time() - started
            words, error = None, f"{type(exc).__name__}: {exc}"

        entry = {"speaker": speaker, "start": start_s, "end": end_s, "text": text,
                 "words": words, "align_seconds": round(elapsed, 3)}
        if error:
            entry["error"] = error
        output.append(entry)
    return output, total


def compare(mlx_segments: list[dict], torch_segments: list[dict]) -> dict:
    """Compare token sequences and bounds, segment by segment."""
    comparison = {
        "segment_count_matches": len(mlx_segments) == len(torch_segments),
        "mlx_segments": len(mlx_segments),
        "torch_segments": len(torch_segments),
        "segments_with_words": {
            "mlx": sum(1 for s in mlx_segments if s.get("words")),
            "torch": sum(1 for s in torch_segments if s.get("words")),
        },
        "token_sequences_identical": 0,
        "token_sequences_differing": 0,
        "token_count_mismatch": 0,
        "per_segment": [],
    }
    deltas_start: list[float] = []
    deltas_end: list[float] = []
    failures: list[dict] = []

    for index, (mine, theirs) in enumerate(zip(mlx_segments, torch_segments)):
        mine_words = mine.get("words") or []
        their_words = theirs.get("words") or []
        entry: dict = {
            "index": index,
            "speaker": mine.get("speaker"),
            "text": mine.get("text"),
            "mlx_tokens": len(mine_words),
            "torch_tokens": len(their_words),
        }
        if mine.get("error"):
            entry["mlx_error"] = mine["error"]
            failures.append({"index": index, "error": mine["error"]})

        mine_text = [w["text"] for w in mine_words]
        their_text = [w.get("text") for w in their_words]
        entry["token_texts_identical"] = mine_text == their_text
        if mine_text == their_text:
            comparison["token_sequences_identical"] += 1
        else:
            comparison["token_sequences_differing"] += 1
            entry["mlx_token_texts"] = mine_text
            entry["torch_token_texts"] = their_text

        if len(mine_words) != len(their_words):
            comparison["token_count_mismatch"] += 1
        else:
            segment_start = [abs(a["start"] - b["start"]) for a, b in zip(mine_words, their_words)
                             if isinstance(b.get("start"), (int, float))]
            segment_end = [abs(a["end"] - b["end"]) for a, b in zip(mine_words, their_words)
                           if isinstance(b.get("end"), (int, float))]
            deltas_start.extend(segment_start)
            deltas_end.extend(segment_end)
            if segment_start:
                entry["max_abs_start_delta_s"] = round(max(segment_start), 3)
                entry["max_abs_end_delta_s"] = round(max(segment_end), 3)

        # The punctuation-floor invariant, asserted on the MLX output rather than assumed.
        if mine_words:
            joined = strip_punctuation("".join(mine_text))
            entry["punctuation_invariant_holds"] = joined == strip_punctuation(mine["text"])
        comparison["per_segment"].append(entry)

    def summarize(values: list[float], label: str) -> dict:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return {
            "count": len(ordered),
            "mean_s": round(sum(ordered) / len(ordered), 4),
            "median_s": round(ordered[len(ordered) // 2], 4),
            "p95_s": round(ordered[index], 4),
            "max_s": round(ordered[-1], 4),
            "label": label,
        }

    comparison["bound_deltas"] = {
        "start": summarize(deltas_start, "abs(mlx.start - torch.start) over aligned tokens"),
        "end": summarize(deltas_end, "abs(mlx.end - torch.end) over aligned tokens"),
        "note": (
            "Deltas are between two implementations at different precisions (8-bit MLX vs "
            "fp32 torch), not against labeled boundaries. Neither side is ground truth: "
            "boundary MAE against labels is unmeasured on every path in this repository."
        ),
    }
    comparison["mlx_failures"] = failures
    comparison["punctuation_invariant_violations"] = [
        entry["index"] for entry in comparison["per_segment"]
        if entry.get("punctuation_invariant_holds") is False
    ]
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import importlib.metadata as metadata

    from huggingface_hub import snapshot_download
    from mlx_audio.audio_io import read as audio_read
    from mlx_audio.stt import load as stt_load

    snapshot = snapshot_download(ALIGNER_REPO, revision=ALIGNER_REVISION,
                                 local_files_only=True)
    print(f"aligner snapshot: {snapshot}")

    load_started = time.time()
    model = stt_load(snapshot)
    load_seconds = time.time() - load_started
    print(f"loaded in {load_seconds:.2f}s")

    audio, sample_rate = audio_read(str(WAV), always_2d=True)
    audio = audio.mean(axis=1)
    duration = audio.shape[0] / sample_rate

    segments = json.loads(VIBEVOICE_SEGMENTS.read_text())["segments"]
    recorded = json.loads(RECORDED_TORCH.read_text())

    aligned, total_align = align_all(model, audio, sample_rate, segments)
    for entry in aligned:
        words = entry.get("words")
        marker = "-" if words is None else f"{len(words)} tokens"
        print(f"  [{entry['start']:6.2f}-{entry['end']:6.2f}] speaker={entry['speaker']} {marker}")

    document = {
        "probe": "mlx-forced-aligner-equivalence",
        "question": (
            "Does mlx-audio's qwen3_forced_aligner reproduce the recorded torch "
            "Qwen3ForcedAligner output closely enough to move the aligner into the mlx "
            "environment?"
        ),
        "runner": "model_tests/benchmark/run_mlx_forced_aligner_probe.py",
        "environment": {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "packages": {
                name: metadata.version(name)
                for name in ("mlx", "mlx-audio", "mlx-lm", "transformers", "huggingface-hub",
                             "numpy")
            },
            "torch_installed": False,
        },
        "model": {
            "repo": ALIGNER_REPO,
            "revision": ALIGNER_REVISION,
            "snapshot": snapshot,
            "quantization": "8-bit",
            "load_seconds": round(load_seconds, 3),
        },
        "compared_against": {
            "artifact": str(RECORDED_TORCH.relative_to(REPO)),
            "runner": "model_tests/forced_aligner/test_multispeaker_pipeline.py",
            "repo": TORCH_ALIGNER_REPO,
            "revision": TORCH_ALIGNER_REVISION,
            "precision": "float32 on cpu",
            "recorded_total_align_seconds": recorded.get("total_align_time_s"),
        },
        "input": {
            "audio": str(WAV.relative_to(REPO)),
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 2),
            "segment_source": str(VIBEVOICE_SEGMENTS.relative_to(REPO)),
            "segments": len(segments),
        },
        "observed": {
            "total_align_seconds": round(total_align, 3),
            "peak_memory_bytes": peak_memory_bytes(),
            "note": (
                "8-bit Metal against a recorded fp32 CPU run; treat the two total_align "
                "figures as different configurations, not as a speedup measurement."
            ),
        },
        "segments": aligned,
        "comparison": compare(aligned, recorded["segments"]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    summary = document["comparison"]
    print(f"\ntoken sequences identical: {summary['token_sequences_identical']}"
          f"/{len(summary['per_segment'])}")
    print(f"bound deltas (start): {summary['bound_deltas']['start']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
