# ASR model evaluation (Issue #2): verbatim transcript for downstream video editing

Tracks: [Issue #2 — Pretrained model applicability benchmark](https://github.com/fyang0507/audio-processing-cli/issues/2) (references [Issue #1](https://github.com/fyang0507/audio-processing-cli/issues/1) for product/architecture contract). This doc covers the **ASR** and **word alignment** sections of Issue #2's benchmark plan (sections 2 and 4) — VAD, diarization, audio-event detection, denoising, and separation (sections 3, 5–10) are not yet started.

## Goal

Find an ASR approach that can produce a **verbatim transcript** suitable for downstream
video editing — specifically: diarization (who spoke), Mandarin/English code-switching
within a single sentence, Chinese dialect handling, and **filler-word preservation**
(um/uh/呃/嗯/like) so filler words can be precisely cut in an edit pass. This is the
"all-in-one video-editing audio pipeline" use case from the original project scope
(the fast-long-form-transcription use case is a separate, still-open concern).

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

Not yet tested from Issue #2's candidate list: CrisperWhisper 2.0, Qwen3-ASR 0.6B/1.7B (the ASR model, distinct from the forced aligner), Whisper large-v3-turbo/whisper.cpp.

## ⚠️ Deployment metrics vs. Issue #2's ~16GB RAM budget

Measured with `/usr/bin/time -l` on the single-speaker clip (peak RAM is dominated by
model weights/runtime, not input length, so this is representative of the multi-speaker
clip's cost too):

| System | Peak RSS | Disk (weights) | RTF (multi-speaker, CPU) | Fits ~16GB budget? |
|---|---|---|---|---|
| FireRedASR2S (VAD+LID+ASR+Punc) | **12.5 GB** | ~9.2 GB | 1.82 | Yes, but tight (~78% of budget) |
| VibeVoice-ASR 7B (float32) | **32.4 GB** | ~17 GB | 0.95 | **No — 2x over budget** |
| VibeVoice-ASR-BitNet (GGUF, quantized) | **7.7 GB** | ~1.7 GB | 0.27–0.29 | Yes, comfortable |
| Qwen3-ForcedAligner-0.6B | **6.2 GB** | ~1.8 GB | ~0.03 (alignment only) | Yes, comfortable |

**This matters because the recommendation below (VibeVoice-ASR + Qwen3-ForcedAligner)
does not fit Issue #2's stated ~16GB constraint.** A CLI pipeline would run each stage
as its own process, so peak RAM at any moment is bottlenecked by whichever stage is
loaded — the aligner's small footprint doesn't offset VibeVoice's 32.4GB. On the 64GB
test machine this went unnoticed; on the actual ~16GB target it would almost certainly
OOM or swap heavily. This wasn't checked before because the earlier rounds tested for
*quality* fit, not deployment fit — flagging it now per Issue #2's own decision
principle ("choose models based on ... local deployment cost ... not generic benchmark
scores alone").

Practical implication: if ~16GB is a hard constraint, the two viable options from what's
been tested so far are **FireRedASR2S** (fits, tight) or **VibeVoice-ASR-BitNet +
Qwen3-ForcedAligner** (fits comfortably) — both with the accuracy tradeoffs documented
below. If ~16GB is a soft/aspirational target rather than a hard ceiling, VibeVoice-ASR
7B remains the quality leader. **This needs a decision from the issue owner** — not
resolved by this doc. One untested option that might resolve the tension: running
VibeVoice-ASR at bf16 instead of float32 (the reference demo script forces float32 on
CPU/MPS; bf16-on-CPU is untested here and might roughly halve the 32.4GB figure, but
possibly at a speed or numerical-stability cost).

## Round 1 — single-speaker sample, first pass

| | FireRedASR2-AED | VibeVoice-ASR |
|---|---|---|
| Output | `firered/FireRedASR2S/result_firered.json` | `vibevoice/VibeVoice/result_vibevoice_singlespeaker.json` |
| Granularity | word-level timestamps + confidence, sentence-level LID | segment-level only, no word-level |
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

Surprise finding: VibeVoice-ASR ran **faster on plain CPU than on MPS** (Apple GPU) —
output was byte-for-byte identical between devices (greedy decoding is deterministic),
so this is a pure speed finding, not a quality tradeoff. Don't assume MPS beats CPU for
this model without checking.

FireRed's CPU cost breaks down as VAD 0.2s / **ASR 140.6s** / **LID 109.0s (43% of total)**
/ Punc 2.8s across 58 VAD-detected segments — LID alone is expensive because it reruns
an 868M-param model once per segment, completely unbatched. This is a fixable
inefficiency (batch segments through ASR/LID together), not an inherent ceiling, if
FireRed is ever revisited.

### Quality

- **Diarization**: FireRed has none — the two-speaker exchange comes out as one
  undifferentiated stream. VibeVoice correctly split `Speaker 0`/`Speaker 1` at every
  turn boundary and tagged two silence/noise gaps as `[Environmental Sounds]`.
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

**Net for round 2**: VibeVoice-ASR led on every quality axis tested and was not slower
once compared CPU-to-CPU, at the cost of a much bigger model (~17GB vs FireRed's ~9GB
across 4 modules) and no built-in VAD/punctuation/word-level-timestamp modules — and, per
the RAM finding above, at nearly 3x FireRed's peak RAM footprint.

## Round 3 — closing VibeVoice's word-level-timestamp gap

VibeVoice's one real deficit vs. FireRed is granularity: segment-level timestamps only,
no word-level timing, which matters for precisely cutting a single filler word out of a
video. Two candidates investigated to close that gap:

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

### 3b. Qwen3-ForcedAligner-0.6B as a post-hoc word-aligner — accepted

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
- Verified every filler word (um/uh/like/呃) across the multi-speaker file got its own
  precise word-level start/end, e.g.:

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
  (monotonic, plausible durations), not a quantitative MAE number.

Full hybrid output: `forced_aligner/result_hybrid_multispeaker.json`. Scripts:
`forced_aligner/test_single_speaker.py`, `forced_aligner/test_multispeaker_pipeline.py`.

## Issue #2 research-question checklist — what this round answered

From Issue #2 §2, **FireRedASR2-AED**:
- [x] Does its code-switching performance hold on real Mandarin-English technical speech? — Mostly yes; some proper-noun errors ("Fortnite"→"fornight").
- [x] How well does it preserve fillers/disfluencies? — Preserves them, but fragmented across ~58 tiny VAD segments, some at low confidence (0.44–0.60).
- [ ] Are native word timestamps accurate enough to skip a forced aligner? — Timestamps exist and look plausible qualitatively; no quantitative boundary-MAE measurement yet.
- [x] Peak RAM on Apple Silicon / CPU — **12.5 GB** (measured).
- [x] Real-time factor on representative Mac hardware — **RTF 1.82** full pipeline (ASR-only stage ≈ 1.01).
- [x] Installation complexity outside CUDA-first examples — Needed a mac-compatible torch build (pip.conf pointed to an unreachable internal mirror; worked around with `uv`, which hits public PyPI directly) and one missing dependency (`kaldi_native_fbank`) not declared in `pyproject.toml`.
- [ ] Long-form stability — Only tested to 139s (via internal VAD chunking); not tested at 30–60 min.

From Issue #2 §2, **VibeVoice / VibeASR BitNet** (tested both the full 7B and the BitNet variant, not just BitNet):
- [x] Verbatim/disfluency behavior — Full 7B: strong, preserves fillers cleanly. BitNet: preserves fillers but hallucinates content (invented a name) and appears to drop/compress content on long disfluent speech.
- [x] Word-timestamp availability/quality in practical local runtime — Full 7B: segment-level only. BitNet: none, and the structured-output prompt format breaks down (1-token output) since it's calibrated for the 7B model only.
- [x] Speaker metadata availability if exposed — Full 7B: yes, correct 2-speaker diarization. BitNet: none (plain text only).
- [x] Code-switch quality — Full 7B: strong (correctly kept "Fortnite" in English). BitNet: regresses — translates some code-switched English into Chinese instead of transcribing verbatim.
- [x] CPU RTF and peak RAM — Full 7B: RTF 0.95, **32.4 GB** peak RAM (2x over the ~16GB budget). BitNet: RTF 0.27–0.29, **7.7 GB** peak RAM (fits comfortably).
- [x] Whether the C++/GGML path materially simplifies packaging — Yes: `cmake` build (~2 min), single static binary (`asr_infer`), no Python/PyTorch runtime needed for inference itself.

From Issue #2 §4, **Qwen3-ForcedAligner**:
- [ ] English word alignment accuracy — Not measured against ground truth; qualitatively plausible.
- [ ] Chinese character/word alignment accuracy — Same as above.
- [x] Mixed Chinese-English utterance alignment — Works correctly; tokenizer splits CJK/Latin boundaries with zero whitespace required.
- [x] Filler alignment (um, 呃, etc.) — Every filler in the test file got its own precise start/end.
- [x] Local memory / runtime — **6.2 GB** peak RAM, ~4.6s to align the full 139s file.

Not yet started from Issue #2: §3 VAD candidates (Silero VAD, FireRedVAD standalone), §5 diarization candidates (pyannote Community-1, NeMo Sortformer — VibeVoice's diarization was observed as a side effect, not benchmarked as a dedicated diarizer), §6 speaker ID, §8 audio-event detection, §9 denoising, §10 separation, §14 formal benchmark dataset (20-30 clips), §15 full metrics tables (WER/CER, DER, boundary MAE), §18 most deliverables.

## Answer: VibeVoice-ASR + Qwen3-ForcedAligner (quality-optimal; RAM-budget caveat above)

Pipeline: **VibeVoice-ASR** (diarization + verbatim filler-preserving transcript,
segment-level timestamps) → **Qwen3-ForcedAligner**, run once per VibeVoice segment
(word-level timestamps within that segment, offset back into absolute time).

Combined cost on the 139.3s multi-speaker file: 131.8s (VibeVoice, CPU) + 4.6s (aligner)
≈ 136.5s ⇒ **RTF ≈ 0.98, essentially real-time**, for output that now has diarization
*and* word-level timing *and* verbatim filler preservation — strictly better on every
quality axis tested than either FireRed or VibeVoice-ASR alone, and better than
VibeVoice-ASR-BitNet's speed-for-accuracy tradeoff.

This beats FireRedASR2S's one advantage (word-level timing) at negligible extra compute
cost (~3.5s of aligner time per minute of audio), while avoiding FireRed's missing
diarization and its more fragmented/error-prone filler handling. **The catch is peak
RAM**: this pipeline needs ~32.4GB (bottlenecked by VibeVoice-ASR alone) on a project
that targets ~16GB. If that constraint is hard, **FireRedASR2S** (12.5GB, tight fit) is
the fallback quality-first pick, or **VibeVoice-ASR-BitNet + Qwen3-ForcedAligner**
(7.7GB, comfortable fit) if BitNet's verbatim-trust issues turn out to be tolerable or
fixable (e.g. with hotword/context conditioning, or narrower use on cleaner speech).

## Open follow-ups (not yet done)

- **Decision needed from issue owner**: is the ~16GB RAM target a hard ceiling or an
  aspirational default? This determines whether VibeVoice-ASR 7B is viable at all.
- Try VibeVoice-ASR at bf16 on CPU (untested) — might roughly halve the 32.4GB figure.
- Human review of the hybrid transcript (`forced_aligner/result_hybrid_multispeaker.json`)
  against the actual audio — spot-check the forced-aligner's word boundaries and
  VibeVoice's transcript for anything the automated comparison in this doc missed.
- A "timestamp-less verbatim audited" transcript variant — i.e. a cleaned, human-checked
  text-only version of the verbatim transcript (fillers kept, errors fixed), independent
  of the timing metadata, likely derived from the same VibeVoice pass.
- Quantitative boundary-MAE measurement for both FireRed's native timestamps and
  Qwen3-ForcedAligner's output, per Issue #2 §4's metrics — needs ground-truth-labeled
  clips, which don't exist yet (Issue #2 §14).
- Everything here is based on two short clips (28s, 139s). No conclusions should be
  treated as a statistically meaningful WER/CER benchmark — this is a capability/fit
  comparison, not an accuracy benchmark. Issue #2 §14's 20-30 clip benchmark set doesn't
  exist yet.
- BitNet was only tested on these two clips; its hallucination behavior on
  cleaner/longer non-disfluent speech is untested and could look different.
- CrisperWhisper 2.0, Qwen3-ASR, and Whisper large-v3-turbo (Issue #2's other ASR
  candidates) haven't been tested at all yet.
