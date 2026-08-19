# VibeVoice-ASR-BitNet long-form profile

This profile answers one narrow question: can the current CPU-only
`VibeASR.cpp` runtime transcribe a 30-minute interview in less than five
minutes on the benchmark host? On the tested Cantonese participant-microphone
fixture, **no**. The bounded 60-second configuration already takes 19.17
seconds, and its visible transcript is not usable Cantonese.

The 30-minute result below is a projection, not a measured run. A full
30-minute burn was deliberately stopped because the measured VAE work alone
already rules out the latency target for the same sequential implementation,
while the sampled output fails the basic language/script gate.

## Host, runtime, and target

- Apple M4 Max, 12 performance + 4 efficiency CPU cores, 64 GiB unified
  memory; macOS 26.5.2.
- `VibeASR.cpp` commit `5cbce71c65911a7e10639ac13b6ab6929e4c8f9e`,
  Release build, CPU-only (`n_gpu_layers = 0`), Apple clang 21.0.0.
- I8_S VAE plus I2_S/Q6_K LM, 1,695,957,664 bytes of GGUF weights.
- Target: 300 seconds for 1,800 seconds of audio, or RTF `< 0.1667`.

The machine has more than 16 GiB. The process-level memory measurements below
therefore describe this run; they do not prove that the same configuration is
healthy on a physical 16 GiB machine.

## Measured results

| Run | Decode | Fresh wall / RTF | Internal stages | Peak memory | Output gate |
|---|---|---:|---|---:|---|
| Frozen 60 s prefix, `-t 8 -c 4096 --max-tokens 2048` | Default sampler: top-k 40, top-p 0.9, temperature 0.7, fixed runtime seed 42 | **19.17 s / 0.3195** | VAE 15.955 s; prefill 0.999 s; decode 1.810 s / 190 tokens; internal total 19.082 s | **13.812 GiB** max RSS; 12.897 GiB macOS peak footprint | EOG before cap, but 0 Han codepoints; visible output is Vietnamese-like hallucination |
| Frozen 180 s prefix, `-t 8 --greedy` | Greedy diagnostic | **160.33 s / 0.8907** | VAE 54.388 s; prefill 3.733 s; decode 100.542 s / 7,782 tokens; internal total 159.334 s | **34.831 GiB** max RSS; 37.200 GiB macOS peak footprint | EOG before cap, but 0 Han codepoints and severe repeated-filler degeneration |

The greedy run is a failure characterization, not the speed estimate. Its
autoregressive degeneration makes it unsuitable for production and explains
why a second thread-count sweep was not useful. Output repeat stability was
not measured: the sampled run is seeded in source, but only one run was made.

For a narrowly scoped transcript-agreement diagnostic, the first-minute sample
was compared with the three frozen participant utterances fully contained in
0--60 seconds (18.333--55.333 s). The reference has 63 mixed tokens, including
59 Han characters; the hypothesis has 143 normalized tokens and no Han
characters. Edit distance is 143/63, or **226.98%**. This is not a general
Cantonese accuracy estimate: the reference excludes interviewer bleed, covers
only three complete utterances, and the hypothesis includes the whole minute.
It is sufficient only as a rejection gate for this output.

## Why the five-minute target is missed

The measured 60-second fresh-process result yields two explicit projections:

- naive sequential fresh processes: `19.17 s * 30 = 575.1 s`, or **9.585 min**;
- optimistic persistent-server estimate, subtracting 0.3149 s of measured
  model loading from all but the first chunk: **563.328 s / 9.389 min**.

The second number is an estimate, not a server measurement. It excludes
protocol, file enumeration, merge, and any overlap needed to protect boundary
words. More decisively, the measured acoustic + semantic VAE work is 15.9553
seconds per minute. Repeating that work for 30 sequential chunks is 478.659
seconds, or **7.978 minutes before LM prefill or decoding**. Therefore model
load amortization cannot reach five minutes in this implementation.

Parallel chunk workers were not tested. They could reduce elapsed time on a
large-memory host, but two copies of a process that peaked at 13.812 GiB would
not be a 16 GiB design, and CPU contention means linear speedup cannot be
assumed.

## Why monolithic 30-minute input is not the production shape

At 24 kHz with the runtime's fixed 3,200-sample compression ratio, 30 minutes
creates 13,500 speech-pad tokens. The fixed prompt adds about 48 tokens, so the
default 16,384-token context leaves only about 2,836 positions for output. The
current GGUF tokenizer counts 8,563 tokens for the frozen raw participant
reference. A monolithic default-context run therefore cannot contain that
reference-shaped prompt and transcript.

A larger context would also increase KV-cache memory. The 180-second default
16,384-context diagnostic already peaked at 34.831 GiB RSS. The bounded
60-second, 4,096-context configuration is the only tested chunk size below 16
GiB process RSS, and even that has little physical-machine headroom.

The runtime's persistent `asr_stream_server` is the correct *shape* for testing
sequential chunks because it loads the models once and clears the KV cache per
file. This rejected configuration can be reproduced as follows; it was not run
over all 30 chunks because the one-chunk latency and output gates had already
failed:

```bash
# Run from the repository root. benchmark_data and benchmark_runs are ignored.
mkdir -p model_tests/benchmark_data/spice/bitnet-60s-chunks
ffmpeg -nostdin -hide_banner -loglevel error -y \
  -i model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_participant_24k.wav \
  -f segment -segment_time 60 -reset_timestamps 1 -c:a pcm_s16le \
  model_tests/benchmark_data/spice/bitnet-60s-chunks/chunk-%03d.wav

cd model_tests/vibeasr_cpp/VibeASR.cpp
{
  for chunk in ../../benchmark_data/spice/bitnet-60s-chunks/chunk-*.wav; do
    printf '%s\n' "$PWD/$chunk"
  done
  printf 'EXIT\n'
} | ./build/bin/asr_stream_server \
  --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
  --lm-model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf \
  -t 8 -c 4096 --max-tokens 2048 --no-token-stream
```

Non-overlapping chunks can lose boundary words. Overlap reconciliation is not
implemented here because BitNet emits only plain text, with neither word nor
segment timestamps. Adding overlap would increase the already-failing runtime.

## Exact primary run

The first-minute fixture is an exact PCM copy of the first 60 seconds of the
frozen 180-second prefix:

```bash
ffmpeg -nostdin -hide_banner -loglevel error -y \
  -i model_tests/benchmark_data/spice/VF19A_Cantonese_interview3m_participant_24k.wav \
  -t 60 -c:a pcm_s16le \
  model_tests/benchmark_data/spice/VF19A_Cantonese_interview60s_participant_24k.wav

cd model_tests/vibeasr_cpp/VibeASR.cpp
/usr/bin/time -l ./build/bin/asr_infer \
  --vae-model models/vibeasr/vibeasr-vae-encoder-i8_s.gguf \
  --lm-model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf \
  --audio ../../benchmark_data/spice/VF19A_Cantonese_interview60s_participant_24k.wav \
  -t 8 -c 4096 --max-tokens 2048 \
  > ../../benchmark_runs/bitnet_spice60s_t8_ctx4096_sample.stdout.txt \
  2> ../../benchmark_runs/bitnet_spice60s_t8_ctx4096_sample.stderr.log
```

Exact hashes, byte counts, stage timings, score denominator, and projection
formulae are frozen in `results/2026-08-12-bitnet-longform.json`. Raw logs and
transcripts remain in ignored `benchmark_runs/` and are identified by SHA-256
in that sidecar.

## Capability conclusion

This runtime supplies neither timestamps nor diarization in its usable plain
text mode. Asking the BitNet model for the 7B-oriented JSON structure is not a
validated way to add them. On this Cantonese fixture it also fails the
transcript-language gate. Consequently it is not the open long-form interview
solution, even though its quantized weights are small. Diarization must be
studied as an independent component attached to a different, validated ASR.
