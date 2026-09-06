# MLX fast-ASR continuation

This bounded funnel tested already-cached MLX checkpoints against the same
SpiCE Cantonese participant-microphone interview used by the FireRed and
VibeVoice continuation. It did not download model weights. Exact hashes and
unrounded values are in `results/2026-08-12-mlx-fast-asr.json`.

## Outcome

| Candidate | Audio | Fresh runner wall | Runner RTF | Peak RSS | MLX active + cache proxy | Mixed-token error | Timestamp result | Decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Qwen3-ASR 0.6B 8-bit | 3 min | 24.55 s | 0.1364 | 1.41 GiB | 3.92 GiB | 49.87% | one 180 s processing container | advance |
| Qwen3-ASR 0.6B 8-bit | 30 min | 31.62 s | 0.01757 | 2.22 GiB | 4.28 GiB | 54.46% | ten 176–184 s processing containers | provisional speed-first route |
| Qwen3-ASR 0.6B 8-bit | 60 min duplicate stress fixture | 60.90 s | 0.01692 | 3.05 GiB | 4.39 GiB | not rescored | twenty 176–184 s processing containers | duration/resource gate passed |
| Qwen3-ASR 1.7B 8-bit | 3 min | 9.21 s | 0.05115 | 2.76 GiB | 5.24 GiB | 81.30% | one 180 s processing container | advance on speed and structural usability |
| Qwen3-ASR 1.7B 8-bit | 30 min | 54.56 s | 0.03031 | 3.57 GiB | 5.58 GiB | 74.09% | ten 176–184 s processing containers | monolithic score is bleed-confounded; superseded by same-turn A/B |
| Whisper large-v3-turbo 4-bit | 3 min | 11.17 s | 0.06204 | 0.82 GiB | 2.84 GiB | 95.58% | native segment and word timestamps | reject |

All rows are single fresh-process observations on an M4 Max with 64 GiB. RSS
and MLX allocator values overlap and must not be added. Active plus cache is an
allocator-footprint proxy, not proof that a physical 16 GiB target is safe.
The 30- and 60-minute Qwen runs had zero observed host swap growth.

The Qwen 30-minute run is measured, not projected: synchronized inference took
25.30 seconds and the whole runner took 31.62 seconds. However, it followed an
earlier MLX job. Persistent Metal/kernel, filesystem, and OS caches were not
cleared, so this is a fresh process with warmed persistent compilation/cache
state—not a controlled cold start. The first-ever three-minute Qwen job spent
19.24 seconds in audio preparation; the later 30-minute job spent 4.77 seconds.
That inversion is cache-state evidence, not favorable duration scaling.

The measured 60-minute duplicate-fixture pass took 50.68 seconds for
synchronized inference and 60.90 seconds for the fresh runner. It used 10,169
of the 16,384-token global generation budget and returned all 20 expected
low-energy containers through exactly 3,600.00 seconds. The library does not
return a finish reason, but its sequential loop only advances after a chunk
stops at EOS or exhausts the remaining global budget; all containers completed
with 6,215 tokens left, so the cap was not reached and EOS termination is the
source-grounded inference. This repeated audio is a duration/resource test, not
an independent quality sample.

The global low-energy splitter moved a join-adjacent cut to 1,802.23 seconds,
so naive whole-half transcript hashes are not comparable. For the eight
interior containers whose bounds do repeat, rebasing the second copy by 1,800
seconds gives equal bounds (maximum floating-point difference below
4e-13 seconds) and exactly equal text. The join-adjacent containers are excluded
from that stability statement rather than counted as model disagreement.

The 1.7B continuation is also measured, not projected. Its 30-minute fresh
runner took 54.56 seconds (33.0x real time), covered all ten containers through
1,800.00 seconds, and observed no host-swap growth. Relative to the warmed
0.6B 30-minute observation it was 1.73x slower, used 1.61x sampled RSS
(1.35 GiB more), and used 1.30x MLX active-plus-cache proxy (1.30 GiB more).
Both remain far inside the
five-minute service target on this host. The 1.7B run was not advanced to the
duplicated hour because the available participant-only reference did not show
a clear quality win; the existing 0.6B hour already closes the shared runtime's
duration/resource gate.

A later controlled test supplies the attributed-turn decision gate that this
monolithic funnel lacked. On the exact same 195 FluidAudio turns, batch-one
cache-cleared 1.7B took 53.77 seconds and scored **33.56%** oracle-participant
MER versus 29.90 seconds and **52.64%** for 0.6B. That result, documented in
`TURN_ATTRIBUTED_FAST_ASR.md`, supersedes the monolithic disposition: prefer
1.7B for interview transcript quality and speed; retain 0.6B as the smallest
and fastest option.

## Quality and timestamp limits

The 49.87% three-minute and 54.46% full-interview values are
orthography-sensitive mixed Han-character/Latin-token error rates. The
reference covers only the participant, while interviewer speech remains audible
on the participant microphone. Qwen emits one undiarized stream and sometimes
transcribes that bleed, so these are not apples-to-apples with VibeVoice's
oracle-filtered participant speaker. They also do not measure semantic
equivalence or behavioral-analysis validity.

That scope mismatch is especially visible for 1.7B. On the 30-minute fixture it
returned 5,313 hypothesis tokens versus 4,415 for 0.6B and preserved many
question/answer exchanges that are structurally useful for an interview. But
the participant-only score penalized the added audible interviewer speech:
1.7B had 1,453 insertions versus 604 for 0.6B, producing 74.09% versus 54.46%
MER. It also had fewer aligned deletions (112 versus 161) and slightly fewer
substitutions (1,378 versus 1,398), but edit-path components cannot isolate
speaker-specific quality once the streams contain different speech. Therefore
these monolithic labels do not prove that 1.7B is worse ASR. The subsequent
same-turn attributed score closes this gate and justifies its extra cost for
the measured mono-interview route.

The three-minute comparison shows the same pattern: 1.7B scored 81.30% MER
versus 49.87% for 0.6B, with 538 versus 410 hypothesis tokens and 159 versus 32
insertions. Deletions were 6 versus 7 and substitutions 148 versus 153. Do not
interpret 1.7B's lower 9.21-second runner wall versus 24.55 seconds for 0.6B as
a model speed win: the earlier 0.6B observation paid 19.24 seconds of first-job
audio preparation, while 1.7B ran after caches were warm. The warmed 30-minute
comparison above is the less-confounded throughput comparison.

Do not cite the Qwen first/middle/last window values stored by the generic
scorer. Qwen's “segments” are approximately three-minute processing containers,
not speech intervals. A 180-second scorer window can overlap two containers and
therefore receive roughly six minutes of hypothesis text, producing bogus
values above 100%. Only the full concatenated 30-minute score is in scope here.
The 60-minute duplicate was not rescored because it adds no independent
reference content.

Whisper's standard compression-ratio temperature fallback removed the
catastrophic repetition produced by a discarded greedy-only diagnostic. The
production-configured result remained unusable at 95.58% error with severe
Mandarin-normalization and mixed-script corruption. Its timestamps were native
and covered through 179.98 seconds, but accurate timing cannot compensate for
bad text. It was not advanced to 30 minutes.

Whisper turbo 8-bit remained cached but unmeasured. The bounded funnel pruned
it after the production-configured 4-bit model failed the quality gate; no
claim about 8-bit quality follows from that.

## Recommended fast-interview interface

Use Qwen3-ASR 1.7B as the preferred quality/speed worker behind segmentation,
with 0.6B retained as the smallest and fastest option. Neither model is the
component responsible for diarization or time alignment:

1. Preserve separate recorder channels when possible; otherwise run VAD and
   diarization first.
2. Cut speaker-attributed speech turns while retaining absolute turn bounds and
   anonymous speaker labels.
3. Keep one MLX Qwen worker resident and transcribe turns with explicit
   `Cantonese` conditioning. Use 1.7B by default when transcript agreement
   matters; select 0.6B when minimum latency or memory is the priority.
4. Reattach the upstream speaker and turn bounds. Run the forced aligner per
   turn only when word-level editing timestamps are needed.
5. Keep any behavioral interpretation uncertainty-aware and human-reviewed;
   none of these ASR measurements validate hiring conclusions.

The controlled same-turn participant score is complete. Remaining gates are a
measured 1.7B sequential end-to-end run, independently labeled interviewer
turns, a physical 16 GB host, dense-turn validation, and non-oracle role
mapping. The separately measured FluidAudio wrapper plus 1.7B ASR components
sum to about 69.16 seconds, but that is planning arithmetic across different
runs, not a measured end-to-end wall.

## Reproduction

`run_mlx_asr.py` requires an absolute existing snapshot path and sets the Hub,
Transformers, and Datasets offline flags before loading mlx-audio. Example:

```bash
<mlx-python> model_tests/benchmark/run_mlx_asr.py \
  --model-path <cached-qwen3-asr-0.6b-8bit-snapshot> \
  --audio model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_participant_24k.wav \
  --output model_tests/benchmark_runs/mlx_qwen3_asr_0.6b_8bit_spice30m.json \
  --language Cantonese \
  --qwen-chunk-seconds 180 \
  --qwen-batch-size 1 \
  --max-tokens 16384

# For the duration/resource stress gate, substitute the frozen duplicated
# 60-minute participant-microphone fixture and a distinct output path.

python3 model_tests/benchmark/score_asr.py \
  --reference model_tests/benchmark/manifests/spice_vf19a_cantonese_interview30m.json \
  --run model_tests/benchmark_runs/mlx_qwen3_asr_0.6b_8bit_spice30m.json \
  --output model_tests/benchmark_runs/mlx_qwen3_asr_0.6b_8bit_spice30m_score.json
```

Raw transcripts and memory timelines stay in ignored `benchmark_runs/`; the
tracked compact result binds them by SHA-256.
