#!/usr/bin/env python3
"""Does MLX VibeVoice-ASR reproduce the recorded torch run, and at what memory?

`mlx-audio==0.4.5` ships `vibevoice_asr`, and `mlx-community/VibeVoice-ASR-8bit` exists. If
that path transcribes equivalently, the `vibevoice` stack stops needing a PyTorch
environment, the tracked `logits_to_keep` patch, and the VibeVoice source checkout — three
provisioning surfaces removed at once.

The second question is memory, and it may matter more. The recorded torch run of this exact
fixture peaked at **20.84 GiB live MPS** at bfloat16, which is why the spec documents warn
that 16 GiB is not validated for this stack. An 8-bit MLX path is a different memory story,
and this probe measures it rather than assuming it.

The comparison is against a recorded artifact from `run_vibevoice.py`, so it is like-for-like
on input: the same audio file, verified by sha256 before anything loads. It is deliberately
*not* like-for-like on configuration — 8-bit MLX versus bfloat16 torch — and the
implementations differ, so a difference is a finding rather than a defect.

Two normalization details the comparison depends on:

- The torch path's `post_process_transcription` emits `start_time`/`end_time`/`speaker_id`/
  `text`; mlx-audio's `parse_transcription` emits `start`/`end`/`speaker_id`/`text`. The keys
  are renamed before hashing, so a hash mismatch means content differs rather than that two
  libraries chose different words for the same field.
- VibeVoice emits `Speaker: "N/A"` on non-speech segments. That is the absence of a label and
  must never become a speaker id, so the probe reports such segments separately.

Run it in a provisioned `mlx` environment:

    /path/to/envs/mlx/bin/python model_tests/benchmark/run_mlx_vibevoice_probe.py \
      --recorded model_tests/benchmark_runs/vibe_mps_bf16_logitskeep_cantomap150s.json \
      --output model_tests/benchmark_runs/mlx_vibevoice_probe_cantomap150s.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MODEL_REPO = "mlx-community/VibeVoice-ASR-8bit"
MODEL_REVISION = "725c72e54d6ef875472c27fbc50fab470a960940"
TORCH_MODEL_REPO = "microsoft/VibeVoice-ASR"
TORCH_MODEL_REVISION = "d0c9efdb8d614685062c04425d91e01b6f37d944"

# mlx-audio's key names, mapped onto the ones the recorded torch artifact uses.
KEY_MAP = {"start": "start_time", "end": "end_time", "speaker_id": "speaker_id",
           "text": "text"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_hash(segments: list[dict]) -> str:
    """The recorded runner's recipe, unchanged (run_vibevoice.py:297)."""
    body = json.dumps(segments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def rename_keys(segments: list[dict]) -> list[dict]:
    return [{KEY_MAP.get(key, key): value for key, value in segment.items()}
            for segment in segments]


def peak_memory_bytes() -> int | None:
    import mlx.core as mx

    getter = getattr(mx, "get_peak_memory", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:  # pragma: no cover - telemetry only
            return None
    return None


def compare(mine: list[dict], theirs: list[dict], duration: float) -> dict:
    result: dict = {
        "mlx_segments": len(mine),
        "torch_segments": len(theirs),
        "segment_count_matches": len(mine) == len(theirs),
        "normalized_sha256_matches": normalized_hash(mine) == normalized_hash(theirs),
        "mlx_normalized_sha256": normalized_hash(mine),
        "torch_normalized_sha256": normalized_hash(theirs),
    }

    texts_mine = [segment.get("text") for segment in mine]
    texts_theirs = [segment.get("text") for segment in theirs]
    result["text_sequences_identical"] = texts_mine == texts_theirs

    paired = list(zip(mine, theirs))
    result["segments_with_identical_text"] = sum(
        1 for a, b in paired if a.get("text") == b.get("text"))
    result["segments_with_identical_speaker"] = sum(
        1 for a, b in paired if a.get("speaker_id") == b.get("speaker_id"))

    starts = [abs(float(a["start_time"]) - float(b["start_time"])) for a, b in paired
              if a.get("start_time") is not None and b.get("start_time") is not None]
    ends = [abs(float(a["end_time"]) - float(b["end_time"])) for a, b in paired
            if a.get("end_time") is not None and b.get("end_time") is not None]

    def summarize(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return {"count": len(ordered), "mean_s": round(sum(ordered) / len(ordered), 4),
                "median_s": round(ordered[len(ordered) // 2], 4),
                "p95_s": round(ordered[index], 4), "max_s": round(ordered[-1], 4)}

    result["segment_bound_deltas"] = {
        "start": summarize(starts), "end": summarize(ends),
        "note": ("Pairwise by index, valid only while segment counts match; neither side is "
                 "ground truth."),
    }

    result["mlx_speaker_labels"] = sorted({str(segment.get("speaker_id"))
                                           for segment in mine
                                           if segment.get("speaker_id") is not None})
    result["torch_speaker_labels"] = sorted({str(segment.get("speaker_id"))
                                             for segment in theirs
                                             if segment.get("speaker_id") is not None})
    result["mlx_non_label_speaker_segments"] = [
        index for index, segment in enumerate(mine)
        if str(segment.get("speaker_id")) == "N/A"
    ]

    ends_mine = [float(s["end_time"]) for s in mine if s.get("end_time") is not None]
    result["coverage"] = {
        "mlx_last_segment_end_s": max(ends_mine) if ends_mine else None,
        "duration_s": duration,
        "mlx_last_end_ratio": round(max(ends_mine) / duration, 4) if ends_mine and duration
        else None,
        "note": ("A last end far below the duration means the decode stopped early; that is "
                 "the exit-4 case, not a quality result."),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded", required=True, type=Path,
                        help="recorded torch artifact from run_vibevoice.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="defaults to the recorded run's max_new_tokens")
    parser.add_argument("--model-repo", default=MODEL_REPO,
                        help="MLX VibeVoice repo; bf16 isolates implementation differences "
                             "from quantization ones")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    args = parser.parse_args()
    model_repo, model_revision = args.model_repo, args.model_revision

    import importlib.metadata as metadata

    from huggingface_hub import snapshot_download
    from mlx_audio.audio_io import read as audio_read
    from mlx_audio.stt import load as stt_load

    recorded = json.loads(args.recorded.read_text())
    audio_path = Path(recorded["audio"]["path"])
    expected_sha = recorded["audio"]["sha256"]
    duration = float(recorded["audio"]["duration_s"])
    max_tokens = args.max_tokens or int(recorded["configuration"]["max_new_tokens"])

    if not audio_path.is_file():
        print(f"recorded audio missing: {audio_path}")
        return 2
    actual_sha = sha256_file(audio_path)
    if actual_sha != expected_sha:
        print(f"audio sha256 mismatch: recorded {expected_sha}, found {actual_sha}")
        return 2
    print(f"audio verified: {audio_path.name} ({duration:.1f}s, sha256 {actual_sha[:12]})")

    snapshot = snapshot_download(model_repo, revision=model_revision, local_files_only=True)
    print(f"model snapshot: {snapshot}")

    load_started = time.perf_counter()
    model = stt_load(snapshot)
    load_seconds = time.perf_counter() - load_started
    print(f"loaded in {load_seconds:.2f}s")

    audio, sample_rate = audio_read(str(audio_path), always_2d=True)
    audio = audio.mean(axis=1)

    generate_started = time.perf_counter()
    output = model.generate(audio, sampling_rate=sample_rate, max_tokens=max_tokens,
                            temperature=0.0)
    generate_seconds = time.perf_counter() - generate_started
    print(f"generated in {generate_seconds:.2f}s")

    mine = rename_keys(output.segments or [])
    theirs = recorded["output"]["segments"]

    document = {
        "probe": "mlx-vibevoice-equivalence",
        "question": ("Does mlx-audio's vibevoice_asr reproduce the recorded torch VibeVoice "
                     "run, and what does the 8-bit MLX path peak at?"),
        "runner": "model_tests/benchmark/run_mlx_vibevoice_probe.py",
        "environment": {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "packages": {name: metadata.version(name) for name in
                         ("mlx", "mlx-audio", "mlx-lm", "transformers", "huggingface-hub",
                          "numpy")},
            "torch_installed": False,
        },
        "model": {"repo": model_repo, "revision": model_revision, "snapshot": snapshot,
                  "quantization": model_repo.rsplit("-", 1)[-1],
                  "load_seconds": round(load_seconds, 3)},
        "configuration": {"max_tokens": max_tokens, "temperature": 0.0,
                          "sampling_rate_passed": sample_rate,
                          "note": ("mlx-audio resamples to VibeVoice's 24 kHz internally; the "
                                   "fixture's own rate is declared rather than assumed.")},
        "compared_against": {
            "artifact": str(args.recorded.relative_to(REPO)) if args.recorded.is_absolute()
            else str(args.recorded),
            "runner": "model_tests/benchmark/run_vibevoice.py",
            "repo": TORCH_MODEL_REPO, "revision": TORCH_MODEL_REVISION,
            "configuration": recorded["configuration"],
            "timing": recorded["timing"],
            "memory": {key: value for key, value in recorded["memory"].items()
                       if key != "samples"},
            "patch": "vibevoice logits_to_keep (tracked); the MLX path applies no patch",
        },
        "input": {"audio": str(audio_path), "sha256": actual_sha,
                  "duration_seconds": duration, "sample_rate": sample_rate},
        "observed": {
            "load_seconds": round(load_seconds, 3),
            "generate_seconds": round(generate_seconds, 3),
            "rtf_generate": round(generate_seconds / duration, 4) if duration else None,
            "peak_mlx_memory_bytes": peak_memory_bytes(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "prompt_tokens": getattr(output, "prompt_tokens", None),
            "generation_tokens": getattr(output, "generation_tokens", None),
            "note": ("MLX and PyTorch memory counters have different scopes and must not be "
                     "summed or subtracted; compare each against its own recorded run."),
        },
        "output": {"segments": mine, "text_head": (output.text or "")[:400]},
        "comparison": compare(mine, theirs, duration),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")

    summary = document["comparison"]
    print(f"\nsegments: mlx {summary['mlx_segments']} vs torch {summary['torch_segments']}")
    print(f"identical text sequence: {summary['text_sequences_identical']}")
    print(f"normalized hash matches: {summary['normalized_sha256_matches']}")
    print(f"peak MLX memory: {document['observed']['peak_mlx_memory_bytes']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
