# Issue #2 benchmark harness

This directory turns the exploratory ASR notes in `../FINDINGS.md` into a
repeatable benchmark. Large model weights, downloaded corpora, source audio,
and generated run artifacts remain local; manifests and small aggregate
results belong here.

## Evidence contract

Every benchmark result must record:

- source-audio identity, license, SHA-256, duration, channels, and sample rate;
- model checkpoint revision and runtime/package versions;
- requested device/dtype, actual first-parameter device/dtype, attention
  implementation, and relevant environment variables;
- input-token and generated-token counts;
- fresh-process end-to-end wall time and model-generation time separately;
- process RSS, explicitly host-wide system swap/available-memory counters, and
  MPS allocator telemetry when applicable; host counters are not
  process-attributable and physical footprint remains a follow-up where exposed;
- exit status, parse validity, transcript coverage, and a hash of normalized
  structured output;
- the denominator and label source for every quality metric.

The harness does **not** turn deterministic checks into quality claims. Output
parsing, monotonic timestamps, and transcript coverage are stability gates.
CER, speaker-attributed CER, DER, speaker-change-boundary F1, dialect-token recall,
filler recall, and alignment error require frozen human/reference labels.

## Benchmark funnel

1. Calibrate on the existing 28-second and 139-second local fixtures: VibeVoice
   CPU/MPS and float32/bfloat16, followed by FireRed with optional LID and batching.
2. Test memory-limit behavior on the existing 112-second product demo and
   139-second conversation. MPS allocator limits are allocator experiments,
   not proof of compatibility with a physical 16 GB machine.
3. Run 30-minute and 60-minute duration stress. Concatenated local recordings
   measure stability, speed, and resource growth only—not interview quality or
   diarization accuracy.
4. Run conversation-native dialect evaluation with frozen references.
5. Validate the selected configuration on a physical 16 GB Apple Silicon Mac
   before declaring the deployment target supported.

See `DATASETS.md` for the dialect set, `CANTOMAP.md` for the frozen MapTask
slice, and `SPICE.md` for the 30-minute interview and repeated one-hour stress
fixtures.

The separate local interview diarization study is in `DIARIZATION.md`, with
machine-readable evidence in `results/2026-08-12-diarization.json`. It measures
FluidAudio and sherpa-onnx independently from ASR, keeps CantoMap full-label
agreement separate from SpiCE's participant-only diagnostic, and records the
exact-repeat hour's fine-boundary instability.

The compact aggregate from the 2026-08-12 controlled continuation is tracked in
`results/2026-08-12.json`. Its compact evidence sidecar,
`results/2026-08-12-evidence.json`, hashes the exact local run and score
artifacts and marks pre-contract fields that could not be recovered. Full
memory timelines and transcripts stay under ignored `benchmark_runs/`.

The separate BitNet long-form rejection profile is in `BITNET_LONGFORM.md`,
with machine-readable evidence in
`results/2026-08-12-bitnet-longform.json`. It distinguishes measured 60- and
180-second prefixes from the explicit 30-minute projection; no 30-minute
BitNet run is represented as measured evidence.

The MLX speed-first interview continuation is in `MLX_FAST_ASR.md`, with exact
provenance in `results/2026-08-12-mlx-fast-asr.json`. It includes actual
30-minute Qwen3-ASR 0.6B and 1.7B runs and a rejected three-minute Whisper
turbo 4-bit calibration; the unmeasured Whisper 8-bit candidate remains
explicitly unranked.

The Qwen runtime-capability probe is in
`results/2026-08-16-qwen-capabilities.json`. It verifies automatic language
output, mixed-script retention, and the absence of speaker/speech-time fields
for both locally cached Qwen checkpoints on one mixed-language clip.

The concrete FluidAudio-to-Qwen integration is in
`TURN_ATTRIBUTED_FAST_ASR.md`, with compact evidence in
`results/2026-08-13-turn-attributed-fast-asr.json`. It defines sample-exact
overlap abstention and anonymous-turn reconciliation, profiles the actual
30-minute pipeline workers, and shows why batch one plus an MLX cache clear
between turns is the measured resource-aware configuration. Its controlled
same-turn A/B prefers Qwen 1.7B for transcript quality and speed while retaining
0.6B as the smallest and fastest option.

## Runners

Run VibeVoice with the virtual environment in its local source checkout. Keep
CPU fallback disabled when the question is GPU acceleration, seed the acoustic
tokenizer, and use the tracked `logits_to_keep` patch:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 \
  model_tests/vibevoice/VibeVoice/.venv/bin/python \
  model_tests/benchmark/run_vibevoice.py \
  --model-path model_tests/vibevoice/VibeVoice/pretrained_models/VibeVoice-ASR \
  --audio <input-media> \
  --output <result.json> \
  --device mps \
  --dtype bfloat16 \
  --seed 1234
```

See `run_vibevoice.py --help` for attention, output-token, allocator-limit, and
controlled full-prompt-logit baseline options.

Run FireRed with its own virtual environment. The runner converts any input
audio or video to a temporary 16 kHz mono PCM WAV, loads only the requested
modules, and writes raw FireRed output plus scorer-compatible normalized
segments to one evidence artifact:

```bash
model_tests/firered/FireRedASR2S/.venv/bin/python \
  model_tests/benchmark/run_firered.py \
  --audio <input-media> \
  --output <result.json> \
  --lid off \
  --asr-batch-size 4 \
  --punc-batch-size 4
```

Use `--lid on` when dialect/language labels are part of the required output.
Keep LID off when benchmarking whether its extra model load and per-batch work
are necessary. Each invocation must be a fresh process; do not compare results
produced by repeatedly loading configurations in one Python process.

Run an already-cached Qwen3-ASR or Whisper MLX snapshot entirely offline with:

```bash
<mlx-python> model_tests/benchmark/run_mlx_asr.py \
  --model-path <existing-local-snapshot> \
  --audio <input-media> \
  --output <result.json> \
  --language Cantonese \
  --qwen-chunk-seconds 180 \
  --qwen-batch-size 1
```

The runner rejects Hub IDs and requires the weights to exist locally. Qwen
output intervals are processing-container bounds, not speech timestamps;
speech-turn timing must come from upstream VAD/diarization or a forced aligner.
Use `--language auto` to pass no Qwen language hint and record the model's
single detected label; this is not localized code-switch metadata.
Whisper can emit native word timestamps with `--whisper-word-timestamps`.

For an already produced regular-overlap FluidAudio artifact, transcribe
anonymous, non-overlapping speaker turns with one persistent Qwen worker:

```bash
<mlx-python> model_tests/benchmark/run_turn_attributed_mlx_asr.py \
  --model-path <existing-qwen3-asr-0.6b-8bit-snapshot> \
  --audio <canonical-16khz-mono-mix.wav> \
  --diarization-run <fluidaudio-result.json> \
  --output <turn-attributed-asr-result.json> \
  --language Cantonese --batch-size 1
```

The runner never maps an anonymous label to a person or role. It records
overlap and short-fragment abstentions separately, verifies an exact
sample-level coverage partition, and clears reusable MLX cache between batches.
It relies on a source-hashed private `mlx-audio` batch method and fails closed
if the inspected signature changes.

Measure both cached stages as strictly sequential fresh subprocesses with
`run_interview_pipeline.py`. It records exact child commands, external pipeline
wall, artifact hashes, and per-stage resource semantics. Provide new output
paths; the orchestrator refuses to overwrite an existing artifact.

Profile a prebuilt FluidAudio offline diarizer after a separate model
provisioning run with `run_fluidaudio_diarization.py`; profile the public CPU
sherpa-onnx baseline with `run_sherpa_diarization.py`. Both normalize output to
`output.segments` for `score_diarization.py`. For an exactly repeated
diarization stress fixture, use `score_repeat_diarization.py`, which compares
anonymous-speaker activity as well as interval and transition stability.

Both runner formats expose `output.segments`, so a frozen reference can be
scored with:

```bash
python model_tests/benchmark/score_asr.py \
  --reference <reference.json> \
  --run <result.json> \
  --output <score.json>

python model_tests/benchmark/score_diarization.py \
  --reference <cantomap-reference.json> \
  --run <vibevoice-result.json> \
  --output <diarization-score.json>
```

The ASR scorer reports orthography-sensitive native-script mixed-token error:
one token per Han character or alphanumeric span with internal apostrophes or
hyphens; punctuation separates tokens. SpiCE language-origin tags and its
silenced `xxx` placeholders are excluded. Whole-segment control labels such as
`[Silence]` are also excluded by default; use `--include-control-segments` only
for a diagnostic. It measures transcript agreement, not semantic or dialect
quality. When a corpus reference covers only one
speaker, `--hypothesis-speaker <label>` can score that anonymous output speaker;
the chosen label is an oracle mapping and must be reported as such. The
diarization scorer reports speaker-time error after optimal anonymous-speaker
mapping and approximate annotation-order speaker-change boundary F1. Its
primary score excludes overlaps and includes both zero-collar
and 250 ms-collar variants; it does not identify people or validate behavioral
analysis. For participant-only SpiCE references it reports explicitly
oracle-selected overlap precision and recall against hand-corrected utterance
intervals instead of a full DER or an independently adjudicated VAD score.

For a duration stress fixture made by exact repetition, use
`score_repeat_stability.py --period-seconds <source-duration>` to compare each
window after timestamp rebasing. It reports exact hashes plus paired text,
speaker, and timestamp consistency (2 ms default tolerance). This is a
deterministic structural gate only; duplicated audio is never valid independent
quality evidence.
