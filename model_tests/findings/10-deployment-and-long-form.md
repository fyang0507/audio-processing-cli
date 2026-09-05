
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
