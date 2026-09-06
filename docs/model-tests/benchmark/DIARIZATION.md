# Long-form interview diarization study

## Recommendation

For an offline, two-person interview on Apple Silicon, use **FluidAudio 0.15.5
offline diarization with the known two-speaker prior and its quality preset**
(`stepRatio=0.1`, `minSegmentDuration=0`). Keep regular overlap-permitting
intervals as the source record; derive an exclusive view only when assigning ASR
words.

On the canonical mixed-channel 30-minute Cantonese interview, it completed in
**14.74 seconds** with **587,481,088 bytes (560.3 MiB) peak sampled target-process
RSS**. The oracle-selected participant label reached **96.65% precision, 94.22%
recall, and 95.42% F1** against the available participant utterance intervals.
The interviewer is not labeled, so this is useful role-separation evidence, not
full DER.

The recommendation is deliberately use-case bounded:

- **Long-turn, two-role interviews:** FluidAudio is the primary local route. It
  is far inside the requested five-minute budget and leaves ASR as the dominant
  latency.
- **Dense, short-turn conversation:** still open. On the fully labeled
  CantoMap slice, FluidAudio's selected preset assigned only 1.34 speaker-seconds
  to its minority label and matched 3 of 75 annotated speaker changes within one
  second. Do not generalize the strong long-interview result to rapid backchannels
  or interruption analysis.
- **Portable CPU fallback:** sherpa-onnx is public, token-free, and small, but
  the selected configuration took 133.86 seconds on 30 minutes and one output
  label absorbed 1,740 of 1,768 labeled seconds. It is not the preferred
  interview separator on this evidence.
- **Direct pyannote offboarding candidate:** Community-1 is technically viable
  with a known-two-speaker prior and PyTorch MPS, but not the default migration.
  Its matched regular-output CantoMap run improved 250-ms-collar agreement
  error/change F1 to **19.20% / 42.00%** versus FluidAudio's **22.31% / 5.50%**;
  its SpiCE participant-interval F1 was slightly lower (**94.67%** versus
  **95.42%**) while wall/RSS rose to **43.18 s / 1.51 GiB** versus **14.74 s /
  560 MiB**. Direct pyannote also needs gated Hub access and a PCM-WAV loading
  workaround on this Mac because TorchCodec cannot load its FFmpeg dylibs.
  Keep it as an adapter-compatible fallback, not an immediate simplification.
- **Clean dedicated channels:** first validate channel dominance and bleed. If
  roles remain separable, per-channel ASR can be more reliable than downmix
  diarization; use diarization as an audit/deduplication stage.

All local measurements were on a 64 GB M4 Max. Sampled process RSS is encouraging
for a 16 GB target, but it excludes memory held by system Core ML services and is
not a physical-16-GB compatibility result.

Machine-readable measurements, artifact hashes, and the exact model/fixture
provenance are in `results/2026-08-12-diarization.json`.

## What was measured

### Fixtures and labels

The 149.9-second CantoMap gate has two anonymous ELAN speaker tiers, 83
utterance-alignment intervals, 75 chronological annotation-order speaker
changes, and cross-speaker overlap. It supports speaker-time agreement after an
optimal anonymous-label mapping, including overlap at zero collar and exclusive
views at zero and 250 ms collars. The ELAN annotations are not independently
adjudicated diarization ground truth, so these values are reported as
"error/agreement," not benchmark DER.

The canonical SpiCE input is a reproducible 30-minute mono downmix of the
participant and interviewer dedicated microphone channels. Both source channels
contain bleed; neither speaker is isolated in the downmix. SpiCE supplies 153
hand-corrected **participant** utterance intervals but no interviewer tier. It
supports wall/RSS/output checks and an oracle participant-interval diagnostic,
not all-speaker DER or role identity. See
`manifests/spice_vf19a_cantonese_interview30m_canonical_mix.json`.

The one-hour fixture repeats that exact 30-minute PCM twice. The decoded original,
first-half, and second-half PCM all have SHA-256
`465cae97476254ae563420fc11e503b4cc2a7e09b687b5a7b12b158590384c9d`.
It is valid only for duration/resource and repeat-consistency diagnostics.

### Timing and memory

- FluidAudio measurements are external fresh-process wall after a separate
  download/compile provisioning run. They include cached model load/compile,
  file preparation, diarization, and JSON output.
- FluidAudio's reported stage timers overlap and are internally unsuitable for
  summation: on the long runs its reported `totalProcessingSeconds` exceeded
  externally observed wall. External fresh-process wall is canonical.
- FluidAudio RSS samples the CLI process with `ps` about every 100 ms. Core ML
  services may hold additional memory elsewhere.
- sherpa wall begins before Python imports but excludes interpreter startup; its
  RSS sampler observes the Python process. Its long native call caused one
  33-second sampling gap, so 747.5 MiB is a lower-bound sampled peak.
- No competing ASR/model benchmark ran during these measured inference calls.

## Local results

### Fully labeled CantoMap gate

All runs used a known two-speaker prior. Error values use 10 ms frames and an
optimal anonymous-speaker mapping. Change F1 uses a one-second tolerance.

| Implementation/configuration | Wall | Peak RSS | Emitted speakers | Overlap error, 0 collar | Exclusive error, 250 ms collar | Change F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sherpa FP32 + Chinese 3D-Speaker | 7.65 s | 397.0 MiB | 2 | 47.78% | 21.07% | 12.35% |
| **sherpa INT8 + Chinese 3D-Speaker** | 8.81 s | 384.7 MiB | 2 | **45.73%** | **20.64%** | **21.43%** |
| sherpa FP32 + English TitaNet-S | 4.42 s | 379.4 MiB | 2 | 47.86% | 21.07% | 10.13% |
| sherpa INT8 + English TitaNet-S | 5.64 s | 384.4 MiB | **1** | 47.08% | 20.64% | n/a |
| FluidAudio fast (`0.2`, `1 s`) | 0.76 s | 301.9 MiB | 2 | 48.72% | 25.71% | 0% |
| **FluidAudio quality (`0.1`, `0 s`)** | **1.26 s** | **312.7 MiB** | 2 | 48.05% | 22.31% | 5.50% |

The 250 ms collar evaluates only 86.64 of 149.9 seconds on this dense slice; it
must not be read without that denominator and the zero-collar result. The sherpa
INT8/Chinese configuration earned the sherpa long-form run through the best
overlap agreement and change coverage. FluidAudio's quality preset earned its
long run only relative to its fast preset; neither Fluid preset passed a
dense-turn quality gate.

### Canonical 30-minute interview

| Implementation | Wall / speed | Peak RSS | Output | Participant-only oracle diagnostic |
| --- | --- | ---: | --- | --- |
| sherpa INT8 + Chinese 3D-Speaker | 133.86 s; 13.45x real time | 747.5 MiB | 86 intervals; 28.15 s vs 1,740.08 s by label | P 60.70%, R 99.08%, F1 75.28% |
| **FluidAudio quality** | **14.74 s; 122.09x real time** | **560.3 MiB** | 589 intervals; 1,039.07 s vs 636.66 s by label | **P 96.65%, R 94.22%, F1 95.42%** |

FluidAudio was **9.08x faster** than the selected sherpa run on this fixture and
used less sampled process RSS. The balanced interval counts (296/293) are not
the quality claim; the participant-only overlap measurement is. Conversely,
sherpa's high participant recall does not compensate for its low precision and
near-whole-file dominant label.

### Exact-repeat 60-minute stress

FluidAudio quality completed the duplicated hour in **31.38 seconds**
(114.72x real time), at **851,427,328 bytes (812.0 MiB)** peak sampled target RSS.
It emitted 1,137 intervals across two labels and reached 3,599.847 seconds.

The duplicated halves are **not interval-for-interval stable**, despite identical
PCM:

- 588 intervals in half one versus 549 in half two;
- only 124 same-label intervals match both bounds within 10 ms and 440 within
  100 ms;
- active-set transition F1 is 81.42% at a 50 ms tolerance;
- coarse 10 ms-frame active-speaker sets agree on 99.36% of all audio, and the
  participant-only oracle F1 is 95.43% versus 95.41% across halves.

This shows low duration/resource cost and similar coarse role activity, but
context-dependent fine segmentation. It must not be described as exact repeat
stability or as an independent one-hour quality result.

## Candidate and deployment review

The review uses official project documentation/model cards current on
2026-08-12. Published speeds and DERs below are not compared numerically to our
local results because hardware, labels, collar, and overlap conventions differ.

| Candidate | Access, offline, license | Apple Silicon and overlap | Disposition |
| --- | --- | --- | --- |
| **FluidAudio offline VBx** | Apache-2.0 SDK; public ungated `FluidInference/speaker-diarization-coreml` model is CC-BY-4.0; offline after staging | Native macOS/iOS Core ML, `.all` compute units except CPU-only filterbank; known/min/max speaker controls; regular overlap or exclusive output | **Primary local interview route.** Locally measured above. |
| sherpa-onnx pyannote segmentation + embeddings | Public GitHub assets, no account; Apache-2.0 runtime; packaged segmentation MIT; tested embedding upstreams Apache-2.0; offline after download | Official macOS/arm64 support; locally measured CPU provider; overlap-preserving output | **Portable fallback**, but weak long-form separation here. |
| pyannote Community-1 | CC-BY-4.0, local after download; first access requires accepting Hugging Face conditions and a token | CPU default and CUDA documented; no official MPS performance claim; regular overlap-aware plus exclusive diarization | **Unmeasured gated quality follow-up.** FluidAudio is a Core ML conversion/implementation, not evidence for Python runtime speed parity. |
| NVIDIA Streaming Sortformer 4spk v2.1 | NVIDIA Open Model License; NeMo/Hugging Face loading path uses a token; streaming hours-long audio | Official results use RTX 6000 Ada; no official MPS path found; at most four speakers; frame probabilities can retain overlap; primarily English | **CUDA/server option**, not the Mac baseline. |
| NVIDIA offline Sortformer 4spk v1 | CC-BY-NC-4.0; NeMo/token | Official card uses RTX A6000 48 GB, limits input to roughly 12 minutes, overlap-aware, primarily English | **Reject for 30–60 minute local use.** |

Primary sources:

- FluidAudio repository and offline documentation:
  <https://github.com/FluidInference/FluidAudio>
- FluidAudio Core ML model card:
  <https://huggingface.co/FluidInference/speaker-diarization-coreml>
- FluidAudio published benchmark conventions:
  <https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md>
- sherpa-onnx diarization documentation and models:
  <https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html> and
  <https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html>
- pyannote Community-1 model card:
  <https://huggingface.co/pyannote/speaker-diarization-community-1>
- NVIDIA Streaming Sortformer v2.1 and offline v1 model cards:
  <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1> and
  <https://huggingface.co/nvidia/diar_sortformer_4spk-v1>

FluidAudio's official VoxConverse table reports 122.06x for its fast preset and
64.75x for its quality preset on the hardware identified by that document as an
M4 Pro with 48 GB. NVIDIA reports RTF 0.002 at 30.4-second latency and 0.093 at
1.04-second latency on an RTX 6000 Ada. These are source claims, not local or
cross-candidate measurements.

## Reproduction

### FluidAudio

The measured source was tag `v0.15.5`, commit
`19600a485baa4998812e4654b70d2bab8f2c9949`, built with Swift 6.3.3:

```bash
git clone https://github.com/FluidInference/FluidAudio.git /tmp/FluidAudio
git -C /tmp/FluidAudio checkout 19600a485baa4998812e4654b70d2bab8f2c9949
swift build --package-path /tmp/FluidAudio -c release --product fluidaudiocli
```

Provision once outside measurement, then invoke the tracked wrapper in a fresh
process. The public model repository's observed `main` SHA at provisioning was
`1ed7a662fdc7109e36d822db793ee6eebdaf8594`; the Fluid downloader resolves
`main` rather than pinning a revision, so the compiled 23-file inventory and
hash in the compact result are the settled local identity.

```bash
python3 model_tests/benchmark/run_fluidaudio_diarization.py \
  --binary /tmp/FluidAudio/.build/release/fluidaudiocli \
  --audio model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_canonical_mix_16k.wav \
  --output model_tests/benchmark_runs/fluidaudio_quality_spice30m.json \
  --raw-output model_tests/benchmark_runs/fluidaudio_quality_spice30m.raw.json \
  --fluid-version 0.15.5 \
  --fluid-commit 19600a485baa4998812e4654b70d2bab8f2c9949 \
  --num-speakers 2 --threshold 0.6 \
  --step-ratio 0.1 --min-segment-duration 0 --batch-size 32 \
  --allow-overlap \
  --model-dir "$HOME/Library/Application Support/FluidAudio/Models/speaker-diarization"
```

### sherpa-onnx

Use sherpa-onnx 1.13.5, soundfile 0.14.0, and psutil 7.2.2. The four exact
model SHA-256 values, URLs, and licenses are frozen in
`manifests/sherpa_onnx_diarization_models.json`.

```bash
python3 model_tests/benchmark/run_sherpa_diarization.py \
  --audio model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_canonical_mix_16k.wav \
  --segmentation-model model_tests/benchmark_data/diarization/sherpa_onnx/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx \
  --embedding-model model_tests/benchmark_data/diarization/sherpa_onnx/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
  --num-speakers 2 --threads 4 --provider cpu \
  --output model_tests/benchmark_runs/sherpa_int8_3dspeaker_spice30m.json
```

Use `score_diarization.py` for the full CantoMap or partial SpiCE label contract,
and `score_repeat_diarization.py` for the anonymous-speaker-aware repeated-hour
diagnostic.

## Pipeline integration and remaining gates

1. Preserve original channels. Produce one canonical 16 kHz mono analysis mix
   only when the diarizer requires it.
2. Run fast ASR and FluidAudio as independent jobs. Benchmark them isolated
   first; a shared-memory Mac still needs a concurrent contention measurement.
3. Preserve regular overlapping diarization. Assign each timestamped ASR word
   to the speaker with maximum interval overlap; abstain on gaps and ambiguous
   ties. If ASR has no word times, either ASR speaker-homogeneous diarizer spans
   or add forced alignment—do not treat ASR processing chunks as speech times.
4. Map anonymous labels to candidate/interviewer only through dedicated-channel
   evidence, an enrollment sample, or human confirmation.
5. Limit downstream analysis to auditable descriptive signals such as talk time,
   turn latency, and interruption candidates with uncertainty retained. Speaker
   labels do not validate personality, competence, deception, or hiring-suitability
   inference.

Before production: run the exact pipeline on a physical 16 GB Mac; acquire
multiple independently checked, fully two-role-labeled interviews (including
short turns and overlap); test automatic speaker count if interviews are not
always two-person; compare regular versus exclusive ASR reconciliation; and
measure isolated versus concurrent end-to-end wall time.
