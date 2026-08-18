#!/usr/bin/env python3
"""Compact the MLX collapse probes into one tracked result.

The raw probe artifacts live under untracked `benchmark_runs/`, so a fresh clone would have
the conclusions and none of the evidence. This produces the tracked compact record: the
figures a reader needs, plus a sha256 of each raw artifact so a local copy can be matched
against the numbers quoted here.

It summarizes, and does not decide. Whether the `vibevoice` stack should move to MLX is a
product judgement about a different transcript; what belongs here is what the runs showed.

    python model_tests/benchmark/summarize_mlx_collapse_probes.py \
      --output model_tests/benchmark/results/2026-08-17-mlx-collapse-probes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "model_tests/benchmark_runs"

ALIGNER = RUNS / "mlx_forced_aligner_probe.json"
VIBEVOICE_8BIT = RUNS / "mlx_vibevoice_probe_cantomap150s.json"
VIBEVOICE_BF16 = RUNS / "mlx_vibevoice_probe_cantomap150s_bf16.json"
TORCH_VIBEVOICE = RUNS / "vibe_mps_bf16_logitskeep_cantomap150s.json"


def digest(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path.relative_to(REPO)), "present": False}
    return {
        "path": str(path.relative_to(REPO)),
        "present": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    aligner = json.loads(ALIGNER.read_text())
    eight = json.loads(VIBEVOICE_8BIT.read_text())
    bf16 = json.loads(VIBEVOICE_BF16.read_text())
    torch_run = json.loads(TORCH_VIBEVOICE.read_text())

    def texts(document: dict) -> list[str]:
        return [segment.get("text") for segment in document["output"]["segments"]]

    aligner_comparison = aligner["comparison"]
    document = {
        "probe_set": "mlx-collapse",
        "question": (
            "mlx-audio 0.4.5 ships MLX implementations of the forced aligner, FireRed's AED, "
            "and VibeVoice-ASR. Which of the PyTorch environments can that remove?"
        ),
        "runners": [
            "model_tests/benchmark/run_mlx_forced_aligner_probe.py",
            "model_tests/benchmark/run_mlx_vibevoice_probe.py",
        ],
        "environment": aligner["environment"],
        "verdicts": {
            "qwen3-forcedaligner": "equivalent",
            "vibevoice-asr-7b": "not_equivalent",
            "firered-asr2s": "blocked_no_mlx_punctuator",
        },
        "forced_aligner": {
            "verdict": "equivalent",
            "mlx": {"repo": aligner["model"]["repo"], "revision": aligner["model"]["revision"],
                    "quantization": "8-bit"},
            "torch": {"repo": aligner["compared_against"]["repo"],
                      "revision": aligner["compared_against"]["revision"],
                      "precision": aligner["compared_against"]["precision"]},
            "fixture": aligner["input"],
            "segments": aligner_comparison["segments_with_words"],
            "token_sequences_identical": aligner_comparison["token_sequences_identical"],
            "token_sequences_differing": aligner_comparison["token_sequences_differing"],
            "token_count_mismatches": aligner_comparison["token_count_mismatch"],
            "bound_deltas": aligner_comparison["bound_deltas"],
            "punctuation_invariant_violations":
                aligner_comparison["punctuation_invariant_violations"],
            "align_seconds": {"mlx_8bit": aligner["observed"]["total_align_seconds"],
                              "torch_fp32_cpu":
                                  aligner["compared_against"]["recorded_total_align_seconds"]},
            "reading": (
                "Every one of the 246 aligned tokens carries the same text on both paths, all "
                "17 word-bearing segments match token for token, both non-speech segments stay "
                "wordless, and the punctuation invariant holds. Bounds agree at a median of 0 "
                "with a tail: P95 80 ms, max 1.6 s, concentrated in long English filler-heavy "
                "segments. Neither path is scored against labels, so the tail is a difference "
                "rather than an error, and it is the cost of the move."
            ),
        },
        "vibevoice": {
            "verdict": "not_equivalent",
            "fixture": {"audio": eight["input"]["audio"], "sha256": eight["input"]["sha256"],
                        "duration_seconds": eight["input"]["duration_seconds"]},
            "segment_counts": {"mlx_8bit": eight["comparison"]["mlx_segments"],
                               "mlx_bf16": bf16["comparison"]["mlx_segments"],
                               "torch_bf16": eight["comparison"]["torch_segments"]},
            "mlx_precisions_agree_on_text": texts(eight) == texts(bf16),
            "segments_with_identical_text_vs_torch":
                eight["comparison"]["segments_with_identical_text"],
            "speaker_labels": {"mlx": eight["comparison"]["mlx_speaker_labels"],
                              "torch": eight["comparison"]["torch_speaker_labels"]},
            "coverage": eight["comparison"]["coverage"],
            "generate_seconds": {
                "mlx_8bit": eight["observed"]["generate_seconds"],
                "mlx_bf16": bf16["observed"]["generate_seconds"],
                "torch_bf16": torch_run["timing"]["generate_s"],
            },
            "peak_memory_bytes": {
                "mlx_8bit": eight["observed"]["peak_mlx_memory_bytes"],
                "mlx_bf16": bf16["observed"]["peak_mlx_memory_bytes"],
                "torch_bf16_mps_current": torch_run["memory"]["peak_sampled_mps_current_bytes"],
                "note": ("MLX and PyTorch MPS counters have different scopes. Each figure is "
                         "comparable to others from the same counter and must not be "
                         "differenced across them."),
            },
            "differences_observed": [
                ("47 MLX segments against 49 torch segments on identical audio, with full "
                 "coverage on both sides (last end == duration), so this is different "
                 "segmentation rather than a truncated decode."),
                ("Orthography shifts to traditional forms in places: 大樹/係/邊/間 where torch "
                 "emitted 大树/系/边/间."),
                "At least one lexical difference: torch '蓝印车站' against MLX '男人車站'.",
                ("Non-speech event tags disagree on the same interval: torch '[Silence]' "
                 "against MLX '[Human Sounds]'."),
            ],
            "reading": (
                "8-bit and bf16 MLX produce identical text on all 47 segments, so the "
                "divergence from torch is the implementation and not quantization. MLX 8-bit "
                "is the only attractive MLX configuration: 12.4 GB peak and 56 s where torch "
                "peaked at 20.8 GB and took 80 s, while MLX bf16 costs 20.7 GB and 275 s for "
                "the same text. That makes the memory case real — the spec's warning that "
                "16 GiB is unvalidated is about the 20 GB torch path — but adopting MLX "
                "replaces every recorded VibeVoice figure, so it is a re-measurement decision "
                "rather than a lock change."
            ),
            "speaker_absence": (
                "Both paths withhold a speaker on non-speech segments rather than inventing "
                "one: torch's post-processing emits speaker_id null, and mlx-audio omits the "
                "key entirely. The raw VibeVoice 'Speaker: \"N/A\"' the adapter floor names is "
                "visible before post-processing, in "
                "model_tests/vibevoice/VibeVoice/result_vibevoice_multispeaker.json."
            ),
        },
        "firered": {
            "verdict": "blocked_no_mlx_punctuator",
            "reading": (
                "mlx-audio's fireredasr2 module is the AED only — Conformer encoder, "
                "transformer decoder, beam search — and mlx-audio ships no punctuation "
                "restoration model anywhere (every 'punc' hit in the package is TTS text "
                "normalization). mlx-community publishes FireRedASR2-AED-mlx and nothing for "
                "FireRedPunc, FireRedLID, or FireRedVAD. Punctuated text is a floor, not a "
                "capability, so a FireRed stack without FireRedPunc is not a conforming "
                "backend: its PyTorch environment survives. Moving only the AED would split "
                "one stack across two environments, which is worse than not moving it."
            ),
            "third_party_conversions_not_evaluated": [
                "42ailab/FireRedPunc-ONNX", "aufklarer/FireRedLID-887M-MLX-8bit",
                "tardigrade-doc/FireRedVAD_onnx", "illitan/FireRedVAD-CoreML",
            ],
            "note": (
                "An ONNX FireRedPunc would run in `core`, which already carries onnxruntime "
                "for Silero VAD. It is recorded as an option, not a plan: none of these "
                "conversions has been run here, and their equivalence to the recorded "
                "FireRedPunc output is unmeasured."
            ),
        },
        "mlx_audio_inventory": {
            "note": (
                "Read from the provisioned mlx environment, for the record rather than as a "
                "plan. Nothing below is measured here."
            ),
            "stt_models_relevant": ["qwen3_asr", "qwen3_forced_aligner", "fireredasr2",
                                    "vibevoice_asr"],
            "vad_models": ["silero_vad", "fsmn", "smart_turn", "sortformer"],
            "lid_models": ["ecapa_tdnn", "wav2vec2"],
            "worth_knowing": (
                "sortformer is a diarization model, so the `swift` environment has an MLX "
                "candidate too. FluidAudio was chosen on measured evidence in "
                "model_tests/benchmark/DIARIZATION.md; replacing it would need the same "
                "comparison rerun, and none of it is done."
            ),
        },
        "raw_artifacts": [digest(path) for path in
                          (ALIGNER, VIBEVOICE_8BIT, VIBEVOICE_BF16, TORCH_VIBEVOICE)],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")
    print(f"verdicts: {document['verdicts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
