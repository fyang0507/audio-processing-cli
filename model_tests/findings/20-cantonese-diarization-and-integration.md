
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
