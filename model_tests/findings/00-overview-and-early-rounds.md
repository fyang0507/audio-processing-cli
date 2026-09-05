# ASR model evaluation (Issue #2): verbatim transcript for downstream video editing

Tracks: [Issue #2 — Pretrained model applicability benchmark](https://github.com/fyang0507/audio-processing-cli/issues/2) (references [Issue #1](https://github.com/fyang0507/audio-processing-cli/issues/1) for product/architecture contract). This doc covers the **ASR** and **word alignment** sections of Issue #2's benchmark plan (sections 2 and 4) plus long-form, native-diarization, and selected dedicated-diarizer evidence. Dedicated VAD, broader diarizer, audio-event, denoising, and separation comparisons (sections 3, 5, and 8–10) remain open.

> **Reading note:** Rounds 1–3 are the exploratory history. The controlled
> continuation dated 2026-08-12 through 2026-08-13 supersedes their deployment, performance,
> dialect, diarization, and final-recommendation claims.

## Goal

Find an ASR approach that can produce a **verbatim transcript** suitable for downstream
video editing — specifically: diarization (who spoke), Mandarin/English code-switching
within a single sentence, Chinese dialect handling, and **filler-word preservation**
(um/uh/呃/嗯/like) so filler words can be precisely cut in an edit pass. This is the
"all-in-one video-editing audio pipeline" use case from the original project scope.
The continuation also evaluates fast long-form interview transcription.

Hardware used for testing: MacBook, Apple M4 Max, **64GB** unified memory, no CUDA
(Apple Silicon only) — note this is 4x Issue #2's stated **~16GB RAM** target, so a
peak-RAM finding below is a real constraint violation, not just a slow-but-fits number.
Environment tooling: `uv` for all Python venvs (see `.gitignore`d `.venv`/`venv` dirs
under each subfolder here).

Two test clips (repo root, not committed elsewhere):
- `autio-test-sample.m4a` — 27.8s, single speaker, Mandarin/English code-switch + Sichuanese dialect.
- `test-sample-multispeaker.m4a` — 139.3s, two speakers, code-switch + dialect + a deliberate filler-word-heavy monologue for verbatim testing.

## Models/tools tested

1. **FireRedTeam/FireRedASR2-AED**, via the full **FireRedASR2S** system (VAD+LID+ASR+Punc cascade) — `firered/FireRedASR2S/`
2. **microsoft/VibeVoice-ASR** (7B) — `vibevoice/VibeVoice/`
3. **microsoft/VibeVoice-ASR-BitNet** (quantized 1.5B-decoder edge variant), via **VibeASR.cpp** — `vibeasr_cpp/VibeASR.cpp/`
4. **Qwen/Qwen3-ForcedAligner-0.6B**, via the `qwen-asr` pip package — `forced_aligner/`
5. **mlx-community/Qwen3-ASR-0.6B-8bit**, **Qwen3-ASR-1.7B-8bit**, and
   **whisper-large-v3-turbo-asr-4bit**, via `mlx-audio`
6. **sherpa-onnx** pyannote segmentation with 3D-Speaker/TitaNet embeddings, and
   FluidAudio's **v0.15.5 offline VBx diarization CLI** backed by its
   `FluidInference/speaker-diarization-coreml` package, as dedicated diarizers.
   Here *FluidAudio* names the product/SDK and diarization pipeline—not one
   separately benchmarked model.

Still unmeasured from Issue #2's candidate list: CrisperWhisper 2.0, Whisper
large-v3-turbo 8-bit/whisper.cpp, pyannote Community-1, and NeMo Sortformer.
The measured Qwen and Whisper entries are the specific MLX quantizations named
above; no claim transfers to the untested variants.

## ⚠️ Legacy deployment measurements vs. Issue #2's ~16GB RAM budget

These measurements predate the reproducible harness. Raw `/usr/bin/time -l`
artifacts were not retained, and the timing scopes differ between systems. Short
input memory is **not** representative of long autoregressive context. The table
is retained as history; use the controlled continuation for decisions.

| System | Peak RSS | Disk (weights) | RTF (multi-speaker, CPU) | Fits ~16GB budget? |
|---|---|---|---|---|
| FireRedASR2S (VAD+LID+ASR+Punc) | **12.5 GB** | ~9.2 GB | 1.82 | Yes, but tight (~78% of budget) |
| VibeVoice-ASR 7B (float32) | **32.4 GB** | ~17 GB | 0.95 | **No — 2x over budget** |
| VibeVoice-ASR-BitNet (GGUF, quantized) | **7.7 GB** | ~1.7 GB | 0.27–0.29 | Yes, comfortable |
| Qwen3-ForcedAligner-0.6B | **6.2 GB** | ~1.8 GB | ~0.03 (alignment only) | Yes, comfortable |

The controlled round resolves the central uncertainty: VibeVoice 7B contains
8.674B BF16 parameters, a 16.157 GiB weight floor before runtime state. A strict
16 GiB MPS allocator cap OOMs during model load. BF16 CPU sampled 15.31 GiB on
28 seconds but took 163.0 seconds end to end and leaves no safe whole-system
headroom. BitNet fits but its observed hallucination and compression failures
disqualify it as a default verbatim fallback. No tested VibeVoice 7B
configuration is a validated 16 GB deployment.
Specifically, the tested strict PyTorch MPS allocator cap hard-fails stock BF16
loading and does not trigger automatic offload; this is not a physical 16 GB
Mac test.

## Round 1 — single-speaker sample, first pass

| | FireRedASR2-AED | VibeVoice-ASR |
|---|---|---|
| Output | `firered/FireRedASR2S/result_firered.json` | `vibevoice/VibeVoice/result_vibevoice_singlespeaker.json` |
| Granularity | word-level timestamps, sentence-level confidence and LID | segment-level only, no word-level |
| Diarization | none | yes (single speaker correctly tagged) |

Both handled the code-switching reasonably. VibeVoice got a semantically ambiguous
closing clause right ("这个模型能力它是怎么样的" — "how is this model's capability") where
FireRed mis-heard it as nonsensical "这个默写能力它是哪么样的" ("dictation ability"). FireRed
preserved authentic Sichuanese phrasing ("看哈") that VibeVoice normalized toward standard
Mandarin ("看一下"). One 28s clip is too small to call a WER winner either way.

## Round 2 — multi-speaker + filler-word sample, with CPU timing profile

Raw outputs: `firered/FireRedASR2S/result_firered_multispeaker.json`,
`vibevoice/VibeVoice/result_vibevoice_multispeaker.json`. Full console logs (all
stage-by-stage timing) under `logs/`.

### Timing (RTF = inference time ÷ audio duration; audio = 139.3s)

| | FireRedASR2S (CPU) | VibeVoice-ASR (MPS) | VibeVoice-ASR (CPU) |
|---|---|---|---|
| Load | 4.8s | ~14s | ~7s |
| Inference | 252.8s | 204.4s | **131.8s** |
| RTF | 1.82 | 1.47 | **0.95** |

This historical one-run result suggested CPU was faster than MPS. It was not a
valid device conclusion: the generation timer lacked explicit MPS
synchronization, no repeat distribution or fallback check was recorded, and
the stored segment times differ by 10 ms. The seeded, synchronized continuation
reverses the result. VibeVoice's acoustic tokenizer also samples a Gaussian
latent, so greedy decoding alone does not make unseeded runs deterministic.

FireRed's CPU cost breaks down as VAD 0.2s / **ASR 140.6s** / **LID 109.0s (43% of total)**
/ Punc 2.8s across 58 VAD-detected segments — LID alone is expensive because it reruns
an 868M-param model once per segment, completely unbatched. This is a fixable
inefficiency (batch segments through ASR/LID together), not an inherent ceiling, if
FireRed is ever revisited.

### Quality

- **Diarization**: FireRed has none — the two-speaker exchange comes out as one
  undifferentiated stream. VibeVoice emitted the expected two anonymous speaker
  labels and plausible alternation, but this clip has no frozen DER or
  speaker-change-boundary labels.
- **Code-switch/proper nouns**: VibeVoice correctly transcribed "Fortnite" throughout;
  FireRed heard "for for night"/"fornight."
- **Filler-word verbatim fidelity** (the core ask): both preserve fillers rather than
  silently dropping them, but the shape differs. FireRed's VAD-driven segmentation
  fragments the monologue into ~58 tiny chunks, so fillers often land as isolated
  one-word "sentences" (`Um.` / `Like.` / `呃。`) — some at low confidence (0.44–0.60) —
  and it mis-transcribed "filler words"→"feeler words" and "disfluency"→"the
  difference" (wrong meaning). VibeVoice keeps the monologue as a few long,
  naturally-punctuated sentences with fillers preserved as clearly comma-delimited
  inline tokens, e.g.:

  > "So, what this, um, this, uh, this test, uh, is about is actually just trying to,
  > you know, like, trying to see whether the model is able to detect filler words
  > and, uh, thus providing, you know, the verbatim, um, kind of records that we can
  > use for, you know, like, downstream editing."

  This reads as more directly usable for programmatic filler-stripping than FireRed's
  fragmented output.

**Net for round 2 (historical qualitative reading)**: VibeVoice looked stronger
on the two local clips, at the cost of a much bigger model and no word-level
timestamps. That observation is not an overall quality ranking. On the frozen
CantoMap slice below, FireRed is faster/lighter while VibeVoice has a small raw
transcript-agreement advantage and native speaker labels.

## Round 3 — closing VibeVoice's word-level-timestamp gap

For product-demo editing, VibeVoice's granularity deficit matters: it has
segment-level timestamps but no word-level timing. Deployment footprint,
unvalidated speaker attribution, and cross-configuration output variability are
additional constraints. Two candidates were investigated for granularity:

### 3a. VibeVoice-ASR-BitNet (edge/CPU-only variant) — rejected

Built from source (`cmake -B build && cmake --build build`, GGUF weights ~1.7GB) and run
on both clips.

- **CPU-only by design** — `n_gpu_layers = 0` is hardcoded, there is no GPU path at all
  (unlike the full model, which supports MPS/XPU/CUDA/CPU).
- Fast and light: RTF 0.27 (single-speaker) / 0.29 (multi-speaker) — 4–7x faster than the
  full 7B model — and only **7.7GB peak RAM**, comfortably inside the ~16GB budget.
- **No timestamps, no diarization, and no path to get them.** Confirmed from source
  (`utils/prompt_builder.h`): the `Start/End/Speaker/Content` JSON output format is
  explicitly documented in-code as calibrated for the 7B model only. Empirically,
  passing `--prompt-format json` to the BitNet model produced just 1 output token — it
  breaks down rather than producing structure.
- **Accuracy regression that disqualifies it for "verbatim"**: it hallucinated a
  person's name that was never spoken ("你好，我是梁少峰，Fortnite" vs. the correct "你好，
  我正在刷Fortnite" per both other models), translated "This is a test" into Chinese
  instead of transcribing the English verbatim, and appears to skip/compress a chunk of
  the filler-word monologue relative to the other two models.
- **Verdict**: best RAM/speed footprint of everything tested, but not trustworthy enough
  for a verbatim-transcript use case as tested. Worth revisiting only if independently
  validated on cleaner, non-disfluent speech — its failure modes above were observed
  specifically on repetitive/disfluent content, which may be a harder case for a
  distilled model than typical ASR eval sets.

### 3b. Qwen3-ForcedAligner-0.6B as a post-hoc word-aligner — provisional

Idea: keep VibeVoice-ASR's transcript (best diarization + filler fidelity), and run a
small, fast forced-aligner per diarized segment to recover word-level timestamps that
VibeVoice itself doesn't produce.

- Single forward pass per segment (not autoregressive generation) — very fast: 0.52s
  model load, **4.64s total alignment time for the entire 139.3s file** (RTF ≈ 0.03), and
  only **6.2GB peak RAM**.
- **Code-switching works out of the box**, despite no explicit documentation of this.
  Its tokenizer (`qwen_asr/inference/qwen3_forced_aligner.py`,
  `tokenize_space_lang`/`split_segment_with_chinese`) walks CJK-character boundaries
  even with **zero whitespace** between scripts, so `"我们要来test一下"` correctly becomes
  `我`/`们`/`要`/`来`/`test`/`一`/`下` with `test` kept as one alignable unit. The
  `language` parameter only changes tokenization behavior for Japanese/Korean — Chinese
  and English (and therefore code-switched combinations of the two) go through the same
  generic path.
- Runs on plain CPU (`device_map="cpu"`, `torch.float32`), no CUDA/flash-attn
  requirement. Small model (0.6B, ~1.8GB download).
- Verified every ASR-provided filler token (um/uh/like/呃) across the
  multi-speaker file received a monotonic word-level interval, e.g.:

  ```
  Speaker 0 [80.17-83.27]: 呃，detect the defluency。
      [80.81-81.21] 呃    <-- FILLER
      [81.29-81.85] detect
      [81.85-81.93] the
      [81.93-82.89] defluency
  ```

- Docs note reference text/audio should be ≤5 minutes per call — not an issue here since
  we align per VibeVoice-segment (all well under that).
- **Not yet measured**: boundary accuracy against ground truth (Issue #2 section 4 asks
  for start/end-boundary MAE and P95 error) — what's verified so far is qualitative
  (monotonic, mostly plausible durations), not a quantitative MAE number. The
  hybrid output also contains 9 zero-duration tokens out of 246, so “precise”
  cuts are not yet a validated claim.

Full hybrid output: `forced_aligner/result_hybrid_multispeaker.json`. Scripts:
`forced_aligner/test_single_speaker.py`, `forced_aligner/test_multispeaker_pipeline.py`.

## Issue #2 research-question checklist — what this round answered

From Issue #2 §2, **FireRedASR2-AED**:
- [x] Does its code-switching performance hold on real Mandarin-English technical speech? — Mostly yes; some proper-noun errors ("Fortnite"→"fornight").
- [x] How well does it preserve fillers/disfluencies? — Preserves them, but fragmented across ~58 tiny VAD segments, some at low confidence (0.44–0.60).
- [ ] Are native word timestamps accurate enough to skip a forced aligner? — Timestamps exist and look plausible qualitatively; no quantitative boundary-MAE measurement yet.
- [x] Peak RAM on Apple Silicon / CPU — historical full-stack observation:
  **12.5 GB**; reproducible LID-off 30-minute run: **9.12 GiB RSS**.
- [x] Real-time factor on representative Mac hardware — historical full stack
  with LID: **1.82**; reproducible LID-off/batch-4 30-minute E2E: **0.370**.
- [x] Installation complexity outside CUDA-first examples — Needed a mac-compatible torch build (pip.conf pointed to an unreachable internal mirror; worked around with `uv`, which hits public PyPI directly) and one missing dependency (`kaldi_native_fbank`) not declared in `pyproject.toml`.
- [x] Long-form structural stability — FireRed completed the real 30-minute
  SpiCE excerpt and an exact-repeat 60-minute stress fixture with monotonic
  output through the end; the repeated halves preserved exact text with at most
  1 ms boundary drift.

From Issue #2 §2, **VibeVoice / VibeASR BitNet** (tested both the full 7B and the BitNet variant, not just BitNet):
- [x] Verbatim/disfluency behavior — Full 7B: strong, preserves fillers cleanly. BitNet: preserves fillers but hallucinates content (invented a name) and appears to drop/compress content on long disfluent speech.
- [x] Word-timestamp availability/quality in practical local runtime — Full 7B: segment-level only. BitNet: none, and the structured-output prompt format breaks down (1-token output) since it's calibrated for the 7B model only.
- [x] Speaker metadata availability if exposed — Full 7B: yes, with the expected
  two labels on two conversation fixtures. CantoMap DER and speaker-change
  agreement are now measured below. BitNet: none (plain text only).
- [x] Code-switch quality — Full 7B: strong (correctly kept "Fortnite" in English). BitNet: regresses — translates some code-switched English into Chinese instead of transcribing verbatim.
- [x] CPU RTF and peak RAM — historical FP32 observation: RTF 0.95 and
  **32.4 GB**. The controlled continuation reports synchronized repeats and
  BF16 behavior. BitNet: RTF 0.27–0.29, **7.7 GB**.
- [x] Long-form structural stability — Full 7B completed one 30-minute pass;
  monolithic 60-minute VibeVoice and repeated long-form runs remain untested.
- [x] Whether the C++/GGML path materially simplifies packaging — Yes: `cmake` build (~2 min), single static binary (`asr_infer`), no Python/PyTorch runtime needed for inference itself.

From Issue #2 §4, **Qwen3-ForcedAligner**:
- [ ] English word alignment accuracy — Not measured against ground truth; qualitatively plausible.
- [ ] Chinese character/word alignment accuracy — Same as above.
- [x] Mixed Chinese-English utterance alignment — Works correctly; tokenizer splits CJK/Latin boundaries with zero whitespace required.
- [x] Filler alignment (um, 呃, etc.) — Every ASR-provided filler received an
  interval, but ground-truth boundary accuracy is not established.
- [x] Local memory / runtime — **6.2 GB** peak RAM, ~4.6s to align the full 139s file.

Not yet started from Issue #2: §3 VAD candidates (Silero VAD, FireRedVAD
standalone), dedicated §5 diarizers (pyannote Community-1, NeMo Sortformer), §6
speaker ID, §8 audio-event detection, §9 denoising, and §10 separation. The
reproducible harness and frozen CantoMap/SpiCE fixtures now start §14–15, but the
planned 20–30-clip corpus, multi-dialect breadth, and boundary MAE remain open.

## Historical answer (superseded): VibeVoice-ASR + Qwen3-ForcedAligner

This was a reasonable product-demo hypothesis from two local clips, but it
combined incomparable timing scopes and overstated unmeasured quality. It is
still the recommended **editing-specific** route on a memory-adequate machine;
it is not the universal pipeline or a 16 GB solution. See the use-case routing
in the continuation.
