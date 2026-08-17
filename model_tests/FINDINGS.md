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

## Continuation — deployment, Cantonese conversation, and long-form interview evidence (2026-08-12)

Rounds 1–3 remain an exploratory history. For deployment, performance,
dialect, diarization, and the pipeline recommendation, this section supersedes
their conclusions. Unless a row says otherwise, performance is one fresh
process run. Transcript-agreement metrics do not establish semantic accuracy,
speaker identity, conversational quality, or the validity of behavioral or
hiring assessments.

### Reproducibility contract

The current harness lives under `model_tests/benchmark/` and is designed to
record the source audio hash/probe, runtime and checkpoint revisions, requested
device/dtype and the first parameter's actual device/dtype, input/output token
counts, synchronized stage timers, resource samples, normalized output hash,
and failure state. The VibeVoice artifacts used
here predate that final evidence contract, so runner/source/stability and system
memory fields that were not embedded are explicitly marked as recovered or
unrecoverable in `benchmark/results/2026-08-12-evidence.json`; that sidecar also
hashes every central local run and score artifact. Large corpora and raw run
artifacts remain ignored; frozen manifests, scorers, patches, the compact
aggregate `benchmark/results/2026-08-12.json`, and its evidence sidecar are
tracked.

The relevant source/checkpoint snapshots are:

- VibeVoice code `94da20d9…`, checkpoint `d0c9efdb…`, PyTorch 2.13.0,
  Transformers 4.57.6;
- FireRedASR2S code `4e7d9aaf…`, ASR `2304afed…`, VAD `7990aacc…`,
  punctuation `e448fd96…`, with LID disabled for the production-shaped tests;
- Apple M4 Max, 16 CPU / 40 GPU cores, 64 GiB unified memory, macOS 26.5.2.

The quality denominator is explicit: “native-script mixed-token error rate”
treats each Han character and alphanumeric span (internal apostrophe/hyphen
allowed; punctuation separates spans) as a token. It excludes
whole-segment control labels such as `[Silence]`, SpiCE language-origin tags,
and silenced `xxx` placeholders. CantoMap's prepared reference additionally
removes pause markers, ampersand annotations, and `xxx`; 21 of 83 annotation
segments become empty, leaving 62 text-bearing segments / 381 tokens. It is
deliberately orthography-sensitive;
Traditional→Simplified conversion and valid Cantonese particle spellings count
as substitutions.

### Evidence snapshot

| Test | VibeVoice 7B | FireRedASR2S | Supported conclusion |
|---|---|---|---|
| 27.8s seeded repeats | CPU FP32 `model.generate()` median **21.04s**; MPS FP32 **7.91s**; MPS BF16 **4.65s** (`n=3` each) | — | Synchronized MPS FP32 generate-call wall time is **2.66×** lower than CPU FP32; MPS BF16 is another observed **1.70×** lower. Model loading can erase the FP32 advantage for one short job window. |
| 16 GiB MPS cap | **OOM during model load** at 16.04 GiB allocated; an 18 GiB cap passed the 27.8s clip | — | The tested cap hard-fails stock BF16 load and does not auto-offload. A short 18 GiB pass is not physical-machine validation. |
| CantoMap Cantonese, 149.9s | E2E 83.41s, RTF **0.556**, 171/381 errors = **44.88%** | E2E 69.57s, RTF **0.464**, 9.16 GiB RSS, 182/381 = **47.77%** | Eleven edits on one conversation cannot rank dialect quality. FireRed is faster/lighter; VibeVoice supplies speaker labels. |
| SpiCE Cantonese interview, 30m | Runner wall 853.83s, RTF **0.474**, 20.28 GiB peak live MPS allocation | Runner wall 665.26s, RTF **0.370**, 9.12 GiB peak RSS | Both complete long form faster than real time on this host. FireRed's runner time is ~22% lower and it is much lighter. |
| Exact-repeat SpiCE stress, 60m | Not run monolithically; the 30m output-token density made truncation likely under the tested ceiling | Runner wall **1,344.58s (22.41m)**, RTF **0.373**, 9.15 GiB peak RSS | FireRed sustained **2.68× realtime**. Overall peak remained load-dominated, while inference-phase RSS rose 1.19 GiB (18%) versus 30m. The halves had identical text sequences and at most 1 ms timestamp drift; repeated audio is stability, not accuracy evidence. |

Fast and modular alternatives, using the same SpiCE source where applicable:

| Component / gate | Measured result | Decision / limit |
|---|---|---|
| BitNet ASR, participant mic, 60s | 19.17s wall, RTF **0.3195**, 13.81 GiB RSS; scoped 143/63 rejection diagnostic with zero Han output | **Reject.** Thirty minutes was not run; projected wall is 9.39–9.59 minutes, already beyond target, with no timestamps or diarization. |
| Qwen3-ASR 0.6B 8-bit, participant mic, 30m / duplicate 60m | **31.62s / 60.90s** fresh runner wall; 2.22 / 3.05 GiB RSS; 30m mixed-token error **54.46%** | **Provisional speed-first ASR.** The score includes possible interviewer bleed; its 180s bounds are processing containers, not speech timestamps. |
| Qwen3-ASR 1.7B 8-bit, participant mic, 30m | **54.56s** fresh runner wall, RTF **0.0303**, 3.57 GiB RSS; participant-only-reference error **74.09%** | The single-stream score is bleed-confounded: it emitted 849 more insertions than 0.6B. The identical-turn comparison below supersedes this row for model selection. |
| Whisper large-v3-turbo 4-bit, participant mic, 3m | 11.17s wall, RTF **0.0620**, 0.82 GiB RSS; native words but **95.58%** error | **Reject** this checkpoint/configuration for Cantonese; no 30m run. |
| FluidAudio dedicated diarizer, canonical mix, 30m / duplicate 60m | **14.74s / 31.38s** wall; 0.55 / 0.79 GiB target-process RSS; 30m oracle participant-interval F1 **95.4%** | **Provisional segmentation front end.** Partial participant-only reference; Core ML service memory omitted; dense CantoMap speaker changes failed. |
| FluidAudio → resource-aware Qwen 0.6B, canonical mix, 30m | **45.67s external sequential wall**, 195/195 accepted turns; Qwen stage 1.66 GiB RSS / 3.20 GiB MLX proxy; oracle-participant error **52.64%** | **Fastest measured end-to-end route / footprint fallback.** Reproduced exact intervals, turns, and transcript; 91.42% single-speaker activity accepted. The same-turn 1.7B row supersedes it for transcript quality. |
| Same FluidAudio turns → Qwen 1.7B, canonical mix, 30m | **53.77s ASR runner wall**, 195/195 identical input turns; 3.02 GiB RSS / 4.86 GiB MLX proxy; oracle-participant error **33.56%** | **Preferred interview worker.** It removes 758 edits versus 0.6B on the same turns, mostly 700 substitutions. Full sequential wall is not separately measured; the observed stages total about 69 seconds. |

Stage timing is synchronized and reproducible within each runner, but the two
runners do not include exactly the same Python/module-import boundary. At 30
minutes that few-second difference does not explain the 188.6-second gap, but
the 22% figure remains configuration-specific rather than a universal model
speed ratio. Cross-run load time is also noisy because filesystem/allocator
cache state is not controlled; generation medians and long runner walls are
the useful signals.

### Long-form interview result

The fixture is a real 30-minute conversational portion from [SpiCE](https://doi.org/10.5683/SP2/MJOXP3),
not concatenated short local clips. It contains a bilingual Cantonese interview
recorded in stereo. The frozen transcript has 153 hand-corrected participant
utterances and 3,972 mixed-script tokens. The interviewer is audible but is not
transcribed by the corpus, which creates an important evaluation boundary.

VibeVoice completed one monolithic pass with 13,562 prompt tokens and 11,345
generated tokens. It returned 184 nonempty segments from 0.00 through 1,800.00
seconds with monotonic bounds. Both anonymous speaker labels occur throughout
the recording. Against hand-corrected participant utterance intervals, an
**oracle-selected** mapping of `Speaker 1` gives 95.1% precision, 98.8% recall,
and 96.9% F1 (1,053.51s overlap / 1,107.74s hypothesis / 1,066.11s reference at
a 10 ms frame step). This strongly supports the mapping for this file; it is
not speaker identification, independently adjudicated VAD, or full DER. The
184 total segments comprise 174 speaker-labeled and 10 control segments.

On the oracle participant stream, VibeVoice's normalized mixed-token error is 45.29%;
FireRed's participant-microphone error is 49.02%. VibeVoice's first/middle/last
three-minute windows are 42.86% / 45.13% / 49.53%; FireRed's are 45.97% /
51.07% / 53.74%. Windows include whole segments overlapping each 180-second
range, not sample-exact crops. This hints at harder content or drift near the
end, but three windows from one speaker do not isolate the cause. Many common
substitutions are convention equivalents: `係→系`, `呢→咧`, `um→嗯`,
`個→个`, `呀→啊`, and Traditional→Simplified characters. Both outputs visibly
retain Cantonese forms, English switches, and filled pauses; the current labels
do not support formal semantic, code-switch, or filler-recall claims. As a
diagnostic, the reference contains 119 `um` spans and VibeVoice's participant
stream contains 119 `嗯` spans, with 101 paired as `um→嗯` in the edit
alignment; 151 Latin spans match exactly. Those counts mix transcription
conventions, fragments, Jyutping, and English, so they must not be reported as
recall.

The recording also demonstrates a cheaper, conditional channel-first route:
the participant and interviewer use dedicated microphones with audible bleed.
On a participant interval the other channel was about 18.9 dB lower; on an
interviewer interval it was about 13.7 dB lower. FireRed VAD rejected the bleed
in this file and transcribed the participant microphone
through 1,796.44 seconds with 413 VAD regions, 571 sentences, and 3,833 native
word intervals. Use capture metadata for channel→role, validate dominance and
cross-talk, and deduplicate bleed before merging. This file did not require a
neural diarizer with FireRed; that is not a general isolated-channel guarantee.

For the hour boundary, an exact two-copy FireRed fixture completed in 1,344.58
seconds (22.41 minutes), RTF 0.3735, with 9.15 GiB peak process RSS. The peak
occurred during model load; sampled inference RSS peaked at 7.80 GiB, so memory
growth did not overtake the 9.15 GiB load peak. Inference-phase RSS did rise
from 6.61 GiB at 30 minutes to 7.80 GiB at 60 minutes (+1.19 GiB / 18.0%), so a
longer-duration plateau is not established. It produced 826 VAD regions, 1,142
sentences, and 7,666 words through 3,596.44 seconds. After rebasing, each half
had 571 sentences with an identical text sequence; the speaker field is null in
both because FireRed has no diarization. The first half also exactly matches the
standalone 30-minute run's text and timestamps. Exact structured
hashes differed because a small subset of timestamps shifted by 1 ms; every
paired boundary stayed within the frozen 2 ms tolerance. This validates one
deterministic duration/stability path, not independent transcript quality or a
distribution of hour-long interviews.

VibeVoice's resource behavior is the blocker. The 30-minute patched BF16 run
peaked at **20.28 GiB of live MPS tensor allocation**. Metal driver allocation
reached 73.09 GiB and plateaued early; that counter includes allocator caches
and MPSGraph allocations and is not resident physical memory, so it must not be
added to RSS. An operator observed roughly 40 GiB of host-wide swap, but there
was no baseline/delta or concurrent-load trace and macOS swap is not
process-attributable; this is anecdotal pressure evidence only. The runner did
not sample swap, physical footprint, thermal state, or GPU utilization. The
result proves completion on this 64 GiB host, not operational safety on a
smaller machine.

If this recording's output-token density held, 60 minutes would need roughly
22.7K generated tokens, exceeding this run's 16,384 generated-token ceiling. A
monolithic 60-minute VibeVoice run was therefore not attempted: it would add
further memory pressure while using a cap likely to truncate. Operational
hour-long support should use bounded chunks,
overlap/reconciliation, and a higher per-chunk safety margin—not a larger
monolithic context.

### Fast-ASR candidate gate

**Reject VibeVoice-ASR-BitNet for this interview task.** In the controlled
60-second sampled run it took 19.17 seconds (RTF 0.3195), peaked at 13.81 GiB
process RSS / 12.90 GiB macOS physical footprint, and emitted no Han characters.
The scoped rejection diagnostic covers only three fully contained participant
utterances: 143 edits / 63 reference tokens (226.98%), with 59 Han reference
tokens and zero in the hypothesis. That is not a full-minute corpus score, but
the Vietnamese-like output, lack of timestamps/diarization, and resource margin
are enough to reject this configuration. **No 30-minute BitNet run was made.**
At the measured 60-second rate, 30 minutes is only *projected* at 9.59 minutes
for repeated fresh jobs or 9.39 minutes with optimistic model-load amortization;
the measured VAE stage alone projects to 7.98 minutes. A separate 180-second
greedy failure is diagnostic, not the basis of that projection.

**Qwen3-ASR 0.6B 8-bit is the measured speed-first ASR candidate.** On the real
30-minute participant microphone, synchronized inference took 25.30 seconds
and the whole fresh runner took 31.62 seconds (RTF 0.01757), with 2.22 GiB peak
process RSS, a 4.28 GiB peak sampled MLX active-plus-cache proxy, and zero
observed swap growth. The process was fresh, but persistent Metal/kernel,
filesystem, and OS caches had been warmed by earlier MLX jobs. Its full-stream
mixed-token error is 2,163 / 3,972 = 54.46%. Unlike VibeVoice's oracle-filtered
speaker score, Qwen's single stream can include audible interviewer bleed, so
that number is not a clean model ranking.

The duplicated 60-minute Qwen stress pass is also measured: 50.68 seconds of
synchronized inference, 60.90 seconds runner wall (RTF 0.01692), 3.05 GiB peak
RSS, 4.39 GiB MLX active-plus-cache proxy, and no observed swap growth. All 20
low-energy containers reached 3,600.00 seconds while using 10,169 of the 16,384
generation-token budget. The library does not expose a finish reason; completing
every container with 6,215 tokens left establishes that the global cap was not
hit, and EOS termination is inferred from the pinned library loop. Eight
interior container pairs with repeatable bounds had identical text after
rebasing the second copy; join-adjacent containers were excluded because the
global low-energy splitter placed a boundary at 1,802.23 seconds. Repeated audio
tests duration/resource stability, not independent quality.

Qwen's reported 176–184-second bounds are **processing containers, not speech,
word, or speaker timestamps**. Its speed therefore becomes useful only behind
channel/VAD/diarization intervals whose absolute bounds are retained. That
production-shaped integration is now measured below; the monolithic result is
retained as the ASR-only duration/resource baseline.

**Qwen3-ASR 1.7B 8-bit also clears the speed gate; its original monolithic score
could not decide whether the extra cost was justified.** Its actual 30-minute
participant-microphone run took 54.56 seconds
(RTF 0.0303), peaked at 3.57 GiB RSS / 5.58 GiB sampled MLX
active-plus-cache, and returned ten processing containers through 1,800
seconds. Against the participant-only reference it scored 2,943 / 3,972 =
74.09%, versus 54.46% for 0.6B. That difference is dominated by 849 additional
insertions: the larger model transcribed substantially more audible interviewer
bleed while the reference labels only the participant. It is therefore not a
clean intrinsic quality loss. The fair diarizer-attributed comparison below
resolves model selection in 1.7B's favor.

**Reject the tested Whisper large-v3-turbo 4-bit MLX checkpoint for Cantonese.**
With its standard temperature-fallback configuration and native word
timestamps, the three-minute runner took 11.17 seconds (RTF 0.0620) and peaked
at 0.82 GiB RSS / 2.84 GiB MLX active-plus-cache, but error was 368 / 385 =
95.58% with severe Mandarin-normalization and mixed-script corruption. The
standard fallback removed a repetition loop from a discarded greedy-only
diagnostic; it did not recover usable text. It was not advanced to 30 minutes.
Whisper 8-bit was pruned without inference, so no result is claimed for it.

### Conversational Cantonese result

The frozen CantoMap slice is 149.9 seconds of two-speaker Hong Kong Cantonese
MapTask conversation: 83 ELAN segments, 75 dense annotation-order speaker
changes, and 4.13 seconds of cross-speaker overlap. [CantoMap](https://github.com/gwinterstein/CantoMap)
is useful here because it is conversation-native and is not listed in
FireRed's published dialect table.

VibeVoice's primary exclusive, zero-collar **ELAN-agreement DER** is **38.81%**
over 145.76 scored seconds (97.24% of the clip): 32.15 seconds of
false alarm from stretching speech labels across annotated gaps dominates 1.90
seconds of miss and 2.88 seconds of confusion. The 250 ms-collar DER drops to
11.42%, but dense boundaries leave only 57.8% of the clip evaluated, so that is
a sensitivity result, not the headline. Approximate speaker-change boundary
precision is 95.1%, recall 52.0%, and F1 67.2% at a one-second tolerance; the
predicted changes are usually close when emitted, but the model undersegments
the dense ELAN changes (39 matched / 41 predicted / 75 reference). These are
annotation-order speaker changes, not validated conversational turns. VibeVoice
emitted 49 total segments: 46 speaker-labeled and 3 control segments. FireRed
exposes no speakers.

The transcript difference is small: VibeVoice requires 171 edits and FireRed
182 on 381 reference tokens. The denominator covers 62 text-bearing ELAN
segments; 21 of 83 annotation segments become empty under CantoMap's
corpus-marker normalization. This is **not** evidence that either model “wins
Cantonese,” much less Chinese dialects generally. For breadth, the staged set
in `benchmark/DATASETS.md` adds MagicHub Sichuan, Shanghai Wu, Guangzhou Yue,
Changsha Xiang, Nanchang Gan, and Zhengzhou regional Mandarin conversations.
Several MagicHub sets appear in FireRed's own reported table, so they are
calibration material rather than a blind head-to-head. A go/no-go decision
still needs a small consented, post-release held-out set from the actual target
population and separate reporting by variety.

### Dedicated-diarizer result

**What “FluidAudio” denotes in these results.** The experiment invoked the
prebuilt `fluidaudiocli` from FluidAudio tag `v0.15.5`, commit
`19600a485baa4998812e4654b70d2bab8f2c9949`, in offline mode with a known
two-speaker prior. That CLI ran FluidAudio's offline VBx diarization pipeline
using the provisioned `FluidInference/speaker-diarization-coreml` package.
Accordingly, the figures below evaluate that specific Core ML-backed pipeline,
configuration, and model-package inventory—not a generic model called
“FluidAudio” and not FluidAudio's product capabilities as a whole.

Neither lightweight diarizer solved the dense CantoMap conversation. Across
the four sherpa-onnx CPU configurations, the best short-run row was INT8
segmentation plus the Chinese 3D-Speaker embedding: 8.81 seconds script wall,
0.38 GiB peak process RSS, 44.99% exclusive zero-collar ELAN-agreement DER, and
only 9 / 75 speaker-change boundaries matched at one second (12.0% recall).
The TitaNet/INT8 row collapsed to one output speaker. FluidAudio's
quality-oriented regular configuration took 1.26 seconds and 0.31 GiB sampled
target-process RSS, but its zero-collar DER was 47.52% and it matched only
3 / 75 changes (4.0% recall); the faster configuration matched none. RSS omits
memory held by system Core ML services. These results are **dense-change
failures**, not evidence that either tool can supply reliable conversational
turns from this CantoMap slice.

On the canonical 30-minute SpiCE downmix, selected sherpa took 133.86 seconds
(RTF 0.0744) and 0.73 GiB RSS. Its oracle-selected speaker overlapped the
participant utterance intervals with 60.7% precision, 99.1% recall, and 75.3%
F1. FluidAudio took 14.74 seconds (RTF 0.00819) and 0.55 GiB sampled
target-process RSS; its oracle-selected speaker reached 96.7% precision, 94.2%
recall, and 95.4% F1. Those labels cover only 153 hand-corrected participant
utterance intervals—not interviewer intervals or independently adjudicated
frame VAD—so the figures are partial-reference overlap diagnostics, not DER,
speaker identity, or behavioral-analysis validation. FluidAudio's exact-copy
60-minute resource stress took 31.38 seconds (RTF 0.00872) and 0.79 GiB sampled
target-process RSS through 3,599.85 seconds, but its two halves differed in
interval segmentation; that repeated-audio run supports duration/footprint,
not boundary stability or quality.

The original CantoMap source also exposes a useful capture lesson: it is 44.1
kHz stereo, while the frozen ASR/diarization fixture is a 16 kHz mono downmix.
On oracle-exclusive frames, corpus tier `F` was usually right-dominant and tier
`G` usually left-dominant, but an untuned raw channel-energy baseline produced
1,412–3,422 predicted changes against 75 references and 74.83%–140.03%
zero-collar error. Preserve source channels and role metadata, but do not treat
instantaneous channel dominance as diarization. A causal channel-aware test
still needs predeclared VAD, hysteresis/minimum-duration, and bleed/ambiguity
policies evaluated on held-out conversations.

### Direct pyannote offboarding test

Direct offboarding is feasible: `pyannote.audio` 4.0.7 loaded
`pyannote/speaker-diarization-community-1` directly at immutable Hub revision
`3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`, with the same known-two-speaker
prior and regular (overlap-permitting) output as the selected FluidAudio quality
run. This is a direct PyTorch/MPS pipeline, not FluidAudio or its Core ML/VBx
implementation. The current macOS TorchCodec wheel could not locate any of its
FFmpeg dylibs when passed an audio path. The tracked runner therefore decodes
only the frozen uncompressed PCM16 WAV fixtures with Python's standard library
and passes their exact sample tensor to pyannote—no resampling or channel
conversion. That workaround is a real integration cost, not a model-quality
claim.

On CantoMap, Community-1 improved the limited dense-change agreement evidence:
**19.20%** exclusive 250-ms-collar error, **46.36%** overlap-included
zero-collar error, and **42.00%** one-second speaker-change F1, compared with
FluidAudio's **22.31%**, **48.05%**, and **5.50%**. It cost **10.80 s** wall
(RTF 0.0721) and **1.30 GiB** peak target-process RSS, compared with
FluidAudio's **1.26 s** (RTF 0.00839) and **0.31 GiB** sampled CLI RSS. The
gain is confined to this one non-adjudicated CantoMap slice; it does not make
either system reliable for backchannels or overlap analysis.

On the same 30-minute canonical SpiCE downmix, Community-1 took **43.18 s**
(RTF 0.0240), used **1.51 GiB** peak process RSS, produced 307 regular
intervals, and reached **97.26% precision, 92.22% recall, 94.67% F1** on the
oracle-selected participant label. FluidAudio took **14.74 s**, used **0.55
GiB** sampled CLI RSS, produced 589 intervals, and reached **96.65% / 94.22% /
95.42%**. This remains participant-only interval overlap, not full DER; the
direct route was both slower/heavier and 0.75 F1 points lower on the available
long-form diagnostic. A CPU-only Community-1 compatibility run on CantoMap
was still more expensive (**81.79 s**, **4.27 GiB**), while MPS yielded the
same exclusive segments.

**Decision:** do not migrate the production diarizer off FluidAudio yet.
Community-1 is a legitimate modular fallback and should remain behind the
same `output.segments` adapter contract, but it is not a simplification on this
Mac: it adds gated model access, a large PyTorch runtime, and the decoder
workaround while losing the current speed/footprint lead. Conversely, this does
not justify adopting FluidAudio as a general platform: keep the pinned
standalone Silero ONNX VAD and evaluate every additional capability separately.
If vendor removal becomes a hard requirement, promote the direct MPS route only
after rerunning sample-exact reconciliation plus Qwen on its 307 intervals and
testing an independent, fully two-speaker-labeled interview set.

### Integrated speed-first interview route

The proposed fast path is no longer just a sum of component timings. One
external orchestrator ran fresh subprocesses strictly sequentially on the
canonical 30-minute mix: FluidAudio quality diarization, deterministic
turn reconciliation, then one persistent Qwen3-ASR 0.6B 8-bit worker. External
wall was **45.67 seconds (RTF 0.0254)** including both interpreter startups,
the fresh FluidAudio CLI, model load, inference, and artifact writes. The two
subprocess walls were 15.39 and 30.28 seconds. This rerun reproduced all 589
FluidAudio intervals, the exact turn plan, and all 195 ASR segment dictionaries.

The frozen reconciler accepts 1,645.62 seconds of anonymous single-speaker
activity (91.42% of the file), retains 2.24 seconds of bridge silence, and
explicitly abstains on 3.70 seconds of overlap, 6.55 seconds of final short
turns, and 14.77 seconds containing only filtered raw fragments. Another
127.11 seconds is unclaimed gap; the sample-level categories sum to 1,800
seconds. Bounds and anonymous labels come from FluidAudio, not Qwen or speaker
identity inference.

Cache-bounded batch-one Qwen took 29.54 seconds inside the sequential rerun and
peaked at 1.66 GiB sampled process RSS / 3.20 GiB sampled MLX
active-plus-cache. FluidAudio peaked at 0.55 GiB CLI RSS in its preceding stage,
but this omits Core ML service memory. The stage metrics overlap internally and
must not be added. Clearing the reusable MLX cache after each turn reduced the
ASR allocator proxy from 18.22 GiB to 3.20 GiB; all 195 outputs, prompt tokens,
and generated tokens remained exactly equal to the cache-retaining batch-one
run. This is strong allocator evidence, not a physical 16 GB machine test.

The corpus labels only the participant. With the participant stream selected
by the independent FluidAudio interval-overlap oracle, turn-attributed Qwen
scores 2,091 / 3,972 = **52.64%** orthography-sensitive mixed-token error. The
full two-speaker output scores 103.07% only because interviewer text is treated
as insertion against the participant-only reference; it is not a quality
metric. The result closes the local speed plus coarse anonymous-speaker-turn
integration gate. It does not validate dense backchannels, speaker identity,
word timestamps, overlap transcription, or interview behavioral constructs.

On the **exact same 195 turns**, Qwen 1.7B completed in 53.77 seconds of fresh
runner wall (52.15 seconds synchronized inference), with 3.02 GiB RSS, a 4.86
GiB MLX active-plus-cache proxy, and zero observed swap growth. Its
oracle-participant stream scored **1,333 / 3,972 = 33.56%**. Compared with
0.6B's 52.64%, hypothesis length stayed nearly equal (3,758 vs 3,746 tokens)
while the larger model removed 758 edits: 35 deletions, 23 insertions, and 700
substitutions. Because the audio windows, anonymous label, prompt-token count,
turn policy, runtime, and scoring reference are held constant, this is the
cleanest local ASR model comparison in the interview study. It makes 1.7B the
preferred interview worker when quality matters; 0.6B remains the faster,
smaller tier. A fresh sequential 1.7B orchestration was not rerun, so its
roughly 69-second diarizer-plus-ASR component sum is not an end-to-end
measurement.

### Qwen3-ASR capability probe (2026-08-16)

A fresh-process MLX probe tested both cached 8-bit checkpoints on the same
139.284-second Mandarin-English, Sichuanese-bearing, two-speaker filler clip.
The runner passed `language=None` (`--language auto`) rather than relying on an
upstream capability declaration. The compact record is
[`benchmark/results/2026-08-16-qwen-capabilities.json`](benchmark/results/2026-08-16-qwen-capabilities.json).

Both checkpoints returned Han and Latin text in the same segment: 0.6B emitted
120 Han characters and 129 Latin word tokens; 1.7B emitted 121 and 125. That
verifies mixed-script retention through this adapter, not code-switch accuracy.
The one unlabelled dialect probe was mixed: 1.7B retained the observed
Sichuanese `耍啥子`, while 0.6B rendered `刷啥子`; it cannot validate the
claimed dialect inventory or a population-level error rate.

Automatic language output exists, but it is not a code-switch locator or a
safe routing signal from this probe. On the same audio, 0.6B returned one
`Chinese` label and 1.7B returned one `English` label for the entire processing
container. Both returned `speaker=null`, one 0.0–139.284-second container bound,
and no speech, word, or character times. Therefore Qwen remains a transcript
worker behind external VAD/channel/diarization bounds; add ForcedAligner only
when word or character timing is required. No native filler-specific or
verbatim mode was observed.

### VibeVoice resource and acceleration conclusions

1. **The tested 16 GiB MPS allocator limit hard-fails stock BF16 model load.**
   `torch.mps.set_per_process_memory_fraction` produced OOM at 16.04 GiB and
   did not automatically offload or merely slow down. The checkpoint has 8.674B
   BF16 parameters = 16.157 GiB before KV cache, activations, acoustic encoder
   state, or the OS. CPU BF16 reduces sampled process RSS to 15.31 GiB on 28
   seconds but takes 163.0 seconds (RTF 5.87). This is not a physical 16 GB Mac
   test and does not rule out an explicitly quantized/offloaded implementation.
2. **The old CPU-over-GPU conclusion is retired.** With explicit
   `torch.mps.synchronize()`, `PYTORCH_ENABLE_MPS_FALLBACK=0`, a fixed seed, and
   three fresh processes per configuration, synchronized `model.generate()`
   wall time was 2.66× lower for MPS FP32 than CPU FP32 on this host. That timer
   covers speech encoding, prompt prefill, and autoregressive decode—not an
   isolated GPU kernel. MPS BF16 was another observed 1.70× lower than MPS FP32,
   but emitted 96 rather than 97 tokens, so this is a configuration ratio rather
   than a pure dtype-kernel gain. Disabling unsupported-op fallback does not prove
   full GPU residency or saturation. Output hashes are stable within each
   device/dtype configuration but differ across configurations, so
   “byte-identical across CPU/GPU” is false.
3. **Use a persistent worker.** For the 28-second clip, median runner job-window
   latency is 25.18s on CPU FP32 and 27.21s on MPS FP32 despite the generation
   ratio; loading dominates. MPS BF16 cuts it to 12.42s. The job window starts
   after Python/module imports, so it is not complete CLI cold-start latency. A
   service should load once and queue jobs instead of spawning a model per clip.
4. **Plain answer on `logits_to_keep`: keep the patch. It removes avoidable
   work, but this benchmark did not measure a speed or peak-memory improvement.**
   Upstream projects the entire audio prompt through a 152,064-token
   language-model head even though generation consumes only the last position.
   The tracked patch preserves training behavior and lets Transformers request
   one logit position. Seeded 28-second and 150-second A/B pairs have equal
   normalized outputs and neutral timing (4.665s vs 4.734s; 75.31s vs 76.34s).
   At the 30-minute BF16 prompt, the logical full-prompt output tensor would be
   about 3.84 GiB if materialized, but lazy execution and the 0.2-second sampler
   did not establish that allocation or resolve a peak delta. Treat this as a
   correctness-gated scaling cleanup, not a claimed optimization result.
5. **Chunking does not solve the weight floor.** It bounds prompt/KV growth,
   output-token caps, retries, and memory pressure, but stock BF16 7B cannot fit
   under the tested strict 16 GiB MPS allocator cap.
   A true 16 GB route needs a smaller trustworthy model, quantization that
   preserves verbatim behavior, explicit CPU/disk offload, or a remote worker.

### Recommended processing pipelines

The backend contract should be capability-driven rather than one fixed chain:

```text
capture/probe
  -> preserve dedicated role-mic channels when available
  -> bounded audio jobs + conservative VAD
  -> ASR backend selected by use case and memory
  -> normalize raw text, times, anonymous speakers, confidence/provenance
  -> add only missing capabilities (role mapping, diarization, word alignment)
  -> descriptive features / evidence-linked analysis
  -> human review
```

| Use case | Recommended v1 route | Why / limit |
|---|---|---|
| **Product-demo editing** | Persistent **VibeVoice MPS BF16 + `logits_to_keep`**, then **Qwen3-ForcedAligner per selected segment** only when edit-grade word boundaries are requested | Best qualitative verbatim/structure result from the local demo tests. Always cut/rerender from the original media. This route is provisional until boundary MAE/P95 is labeled, and it is not a 16 GB pipeline. |
| **Interview analysis, dedicated participant/interviewer microphones** | Preserve channels → validate dominance/bleed → transcribe each channel with **Qwen 1.7B** for quality/speed, **Qwen 0.6B** for minimum footprint, or **FireRed LID-off/batch-4** for native word times → deduplicate cross-talk → merge by retained time → map channel→role from capture metadata | Prefer this route whenever capture permits it; channel metadata is cheaper and more trustworthy than inferring identity. Only the participant mic was ASR-scored, so interviewer-channel transcription, deduplication, and merged latency/quality remain integration gates. |
| **Interview analysis, mono/mixed, quality-speed default** | **FluidAudio quality diarization/VAD → sample-exact turn reconciliation with overlap abstention → persistent Qwen3-ASR 1.7B 8-bit, batch one with cache clearing → reattach turn bounds → role confirmation** | On the same 195 turns, ASR took **53.77s** and oracle-participant error was **33.56%**, with 3.02 GiB RSS / 4.86 GiB MLX proxy. The observed Fluid plus ASR stages total about 69s, but this exact 1.7B chain was not timed by the end-to-end orchestrator. |
| **Interview analysis, mono/mixed, minimum footprint/latency** | Same FluidAudio route → **Qwen3-ASR 0.6B 8-bit**, batch one with cache clearing | **45.67s measured sequential wall on 30m**, 195/195 turns, 91.42% single-speaker activity coverage, and 52.64% oracle-participant error; Qwen stage 1.66 GiB RSS / 3.20 GiB MLX proxy. Prefer only when its quality tradeoff is acceptable. |
| **Interview analysis, mono/mixed, timestamp/quality balance** | **FireRed LID-off/batch-4 + FluidAudio intervals**, normalized into one word/speaker schema | FireRed is the more conservative Cantonese ASR/timestamp route: native word intervals, 49.02% participant-mic error, and 9.12 GiB RSS. It misses the under-five-minute 30m target at 665.26s, and the combined pipeline has not been run or memory-profiled. |
| **Chinese dialect conversation** | Use **FireRed** as the current balanced multilingual/dialect default; use **VibeVoice 7B + aligner** when product-editing structure matters and memory is available; use **Qwen 1.7B + FluidAudio** only as the measured Cantonese interview route | Evidence is Cantonese-only and cannot rank Chinese dialects generally. FireRed and VibeVoice differ by only 11/381 CantoMap edits. VibeVoice, sherpa, and FluidAudio all missed many of 75 dense annotation-order changes, so none is a validated dense-turn diarizer. |

For hour-long interviews, use bounded jobs with stable absolute offsets,
idempotent retries, and explicit merge/deduplication. Qwen's measured 180-second
low-energy policy reached one hour, but it is not itself a timestamp source;
the production speed route should feed it attributed speech turns from channels
or diarization. VibeVoice should use bounded overlapping jobs rather than a
monolithic hour. Mixed recordings need global speaker reconciliation; never
assume `Speaker 0` remains the same person across independent generations.

Post-interview analysis needs speaker intervals and a reliable
candidate/interviewer role map, not word-level alignment by default. After role
and interval validation, the pipeline can report **non-evaluative observations**:
attributed speaking-time duration/share, distributions of response-gap and
speaker-segment durations, timestamp-linked question/response pairs, overlap or
unassigned intervals, and verbatim evidence snippets. These describe the
recording; they do not label an overlap as an interruption, score a response,
or infer a trait. The Qwen-plus-Fluid route now validates anonymous
interval-to-ASR integration, but both it and the channel-first route still need
non-oracle role mapping and two-sided human labels before even those descriptors
are a product default. ASR agreement and diarization do **not**
validate personality, competence, deception, or hiring recommendations. Any
evaluative construct needs separate consent, frozen human labels/rubric,
fairness analysis, and human oversight.

### Corrections (2026-08-17)

Found while specifying the `transcribe` command against the recorded artifacts
rather than against these summaries. Both corrections supersede the earlier
statements wherever they appear here, in `DECISION_REPORT.md`, and in
`EXPERIMENT_RESULTS.md`.

- **FireRed emits no per-word confidence.** Round 1's granularity row said
  "word-level timestamps + confidence". Every word object is exactly
  `{start_ms, end_ms, text}` (`fireredasr2s/fireredasr2system.py:181-184`),
  confirmed over 12,370 word tokens across all five recorded FireRed artifacts.
  Confidence exists at *sentence* level as `asr_confidence`, non-null on every
  sentence and ranging 0.158–0.997 on the recorded runs. The capability was part of
  the case for FireRed as an audit route; that specific argument does not hold, while
  its dialect-form, region-LID, native-word-time, and resource arguments are
  unaffected.
- **FireRedPunc emits punctuated sentences, not marks with bounds.** It returns
  `punc_sentences` — punctuated sentence strings with sentence bounds
  (`fireredpunc/punc.py:109-119`) — while `words` is built from the pre-punctuation
  AED timestamps. Two consequences that were not recorded before: the punctuation
  stage also **recases** text, because `RuleBaedTxtFix.fix` lowercases its input and
  then re-capitalizes sentence starts and standalone `i`
  (`fireredpunc/punc.py:349-382`), so sentence text and word text differ in case by
  construction — 234 characters across the recorded artifacts. And the sentence text
  is therefore not reconstructible from the word stream: the invariant that does hold
  is that stripping punctuation and whitespace from a sentence's text yields exactly
  the concatenation of its word texts, compared case-insensitively. Verified on all
  five FireRed artifacts and on all 17 aligned segments of
  `forced_aligner/result_hybrid_multispeaker.json`.

Two smaller attributions corrected in the same pass, both from
`results/2026-08-13-turn-attributed-fast-asr.json`: the MLX cache is cleared after
every *batch* rather than every turn (the two coincide only at `batch_size: 1`), and
every figure in that record's controlled A/B was produced with the language hint
`"Cantonese"` rather than on the no-hint path.

### What remains unproven

- No physical 16 GB machine test or trustworthy quantized VibeVoice 7B path.
  Process/allocator proxies do not prove whole-system safety; FluidAudio RSS
  omits Core ML service memory.
- The 45.67-second Qwen-plus-Fluid result is a measured sequential end-to-end
  systems path, but it still lacks physical-16-GB validation, two-sided
  transcript/activity labels, overlap transcription, and verified
  candidate/interviewer role mapping. Its Core ML memory scope is incomplete.
- No completed two-channel interview pipeline: the interviewer microphone,
  channel-dominance/bleed gates, cross-talk deduplication, and merged transcript
  have not been benchmarked.
- No monolithic 60-minute VibeVoice run or repeated independent long-form
  distribution. The 60-minute FireRed, Qwen, and FluidAudio fixtures duplicate
  the same 30 minutes and provide only duration/resource/repeat evidence.
- No population-level dialect conclusion. CantoMap is one speaker pair/slice;
  SpiCE is one participant/session; non-Cantonese dialect breadth is a staged plan.
- No shared Traditional/Simplified/particle equivalence set, semantic review,
  genuine English-span labels, or formal filler/repair recall.
- No ground-truth forced-aligner boundary MAE/P95. Monotonic timestamps are a
  structural gate, not proof of edit precision.
- CantoMap's collar sensitivity and SpiCE's participant-only transcript prevent
  a simple “diarization accuracy” headline. Report the denominator and label
  scope with every speaker metric.
- No reliable dense conversational-turn diarizer from the tested set: VibeVoice
  matched 39/75 CantoMap changes, the best sherpa row 9/75, and the selected
  FluidAudio quality row 3/75 at one second. These are annotation-order changes,
  not adjudicated conversational turns.
- CrisperWhisper, Whisper 8-bit/whisper.cpp, pyannote
  Community-1, and Sortformer remain unmeasured; Issue #2's dedicated VAD and
  audio-event tracks also remain incomplete.
