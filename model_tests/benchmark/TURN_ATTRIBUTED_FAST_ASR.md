# Turn-attributed fast interview ASR

This continuation measures the concrete local interview route proposed by the
separate ASR and diarization studies:

1. FluidAudio emits anonymous, overlap-preserving speaker intervals on the
   frozen 30-minute SpiCE canonical mono mix.
2. A deterministic reconciler converts those intervals into non-overlapping
   speaker turns and an explicit abstention ledger.
3. One persistent Qwen3-ASR MLX worker transcribes each accepted turn with
   `Cantonese` conditioning and reattaches its absolute FluidAudio bounds and
   anonymous label. The settled turn plan is tested with both 0.6B and 1.7B
   8-bit checkpoints.

The 0.6B ASR job alone took **29.90 seconds fresh** and **28.46 seconds after model
load** for the actual 30-minute interview. It emitted all 195 accepted turns.
Its maximum observed MLX active-plus-cache allocator footprint was **3.20 GiB**
after clearing the reusable cache between each batch-one turn; sampled process
RSS peaked at **1.66 GiB**. These metrics overlap and must not be added. They are
encouraging for a 16 GB target, but they are not a physical-16-GB result and do
not include memory held by Core ML services during a preceding FluidAudio job.

A controlled model-only swap on the exact same 195 turns changes the quality
decision. Qwen 1.7B took **53.77 seconds fresh**, peaked at **3.02 GiB sampled
RSS / 4.86 GiB MLX active-plus-cache**, and reduced the oracle-participant error
from **52.64% to 33.56%**. It is 1.80x the fresh ASR wall of 0.6B but remains
33.5x real time and far inside the five-minute target. On this evidence, 1.7B
earns its extra cost as the default mono-interview transcript worker; 0.6B is
the lower-latency/lower-memory fallback. Physical 16 GB remains unvalidated.

The full end-to-end sequential wall measurement is recorded separately below
after the ASR-only integration gate. Exact values, hashes, and artifact
provenance are in `results/2026-08-13-turn-attributed-fast-asr.json`.

## Frozen turn policy

The policy was declared before the three-minute smoke and was not selected
against transcript or diarization quality scores:

- convert FluidAudio bounds to integer samples on the canonical 16 kHz clock;
- retain source bounds within the requested prefix and abstain on invalid or
  empty bounds;
- remove raw diarizer fragments shorter than 250 ms into the abstention ledger;
- sweep interval endpoints into exact active-speaker sets;
- never transcribe a multi-speaker overlap twice or assign it arbitrarily;
- merge same-label regions only across at most 300 ms of true silence—never
  across another active speaker, overlap, or filtered-fragment region;
- abstain on final turn windows shorter than 500 ms;
- crop every accepted turn exactly once, bucket deterministically by duration
  for inference, then restore chronological order;
- keep labels anonymous. The SpiCE participant mapping is an oracle used only
  in a separate score artifact.

The runner checks that accepted windows do not overlap and that accepted active
speech, bridge silence, short-turn abstentions, overlap abstentions,
filtered-fragment-only spans, and unclaimed gaps form an exact sample-level
partition of the input.

## Measurements

All rows are one fresh process on an M4 Max with 64 GiB. The FluidAudio output
was already staged for the ASR-only rows, so their wall excludes diarization.

| Configuration | Fresh ASR wall | Service job after model load | Peak RSS | MLX active + cache | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| 3-minute smoke, batch 4 | 3.38 s | 1.92 s | 1.50 GiB | 5.29 GiB | 27/27 turns |
| 30-minute batch 4, cache retained | 17.19 s | 15.74 s | 1.64 GiB | **21.87 GiB** | 195/195 turns |
| 30-minute batch 1, cache retained | 29.09 s | 27.28 s | 1.82 GiB | **18.22 GiB** | 195/195 turns |
| **30-minute batch 1, cache cleared per turn** | **29.90 s** | **28.46 s** | **1.66 GiB** | **3.20 GiB** | **195/195 turns** |
| **30-minute 1.7B, batch 1, cache cleared per turn** | **53.77 s** | **52.27 s** | **3.02 GiB** | **4.86 GiB** | **195/195 turns** |

The faster batch-four route is not the memory-bounded default. Its high value
was reusable MLX allocator cache, not sampled process RSS or live tensors.
Batch one still accumulated 18.22 GiB when the installed private helper looped
over all turns without clearing that cache. The tracked runner now calls the
same inspected helper once per outer batch, synchronously samples immediately
before and after `mx.clear_cache()`, and preserves the global token budget.

The resource-aware output exactly matches the cache-retaining batch-one output:
the complete text and every one of the 195 segment dictionaries are equal, with
normalized segment SHA-256
`bde962f3db664b3e72b9797c3dc496b7ee2bf81a4fbc0d129597e2d2071afaa0`.
Prompt and generation counts also match at 25,157 and 6,752; 9,632 of the
16,384-token global generation budget remained.

The resource-aware run has three distinct memory observations:

- 100 ms background sampling saw a 1.93 GiB active peak;
- MLX's explicit active-memory high-water counter saw 2.29 GiB;
- synchronous samples after each batch completed, before cache clearing, saw a
  maximum 0.94 GiB active plus 2.26 GiB cache; every post-clear observation saw
  zero cache. The maximum active-plus-cache value across all samples was
  3.20 GiB.

These counters expose allocator behavior, not physical RAM residency.

### Controlled 0.6B versus 1.7B turn A/B

The 1.7B run replaced only the cached model snapshot. Audio, FluidAudio
artifact, turn thresholds, language, ordering, batch size, cache-clearing
policy, and token budget were identical. Both artifacts record turn-plan hash
`1b69d915…569b8`; sorted serialization of all 195 accepted turn records is
also identical under `jq -S` (`380c05c1…55bf`). Both processed all turns with
no host-swap
growth.

| Same-turn metric | Qwen 0.6B 8-bit | Qwen 1.7B 8-bit | 1.7B tradeoff |
| --- | ---: | ---: | ---: |
| Fresh ASR wall | 29.90 s | 53.77 s | 1.80x |
| Inference wall | 28.34 s | 52.15 s | 1.84x |
| Peak sampled RSS | 1.66 GiB | 3.02 GiB | 1.82x |
| MLX active + cache | 3.20 GiB | 4.86 GiB | 1.52x |
| S1 oracle-participant MER | 52.64% | **33.56%** | **19.08 points / 36.25% relative reduction** |
| S1 edit path (D / I / S) | 276 / 50 / 1,765 | **241 / 27 / 1,065** | 758 fewer edits total |

The error reduction appears in every predeclared 180-second diagnostic window:
45.19% to 35.32% at the beginning, 55.11% to 31.59% in the middle, and 61.45%
to 34.81% at the end. Because the same oracle-selected S1 intervals feed both
models, this comparison separates ASR quality from interviewer bleed handling
far better than the earlier monolithic participant-microphone comparison.

This supersedes the earlier monolithic disposition that kept 1.7B only as an
unproven challenger. In that test, 1.7B's 74.09% participant-only MER versus
0.6B's 54.46% was dominated by the larger model transcribing more audible
interviewer speech. Once both models receive identical isolated turns, 1.7B
wins decisively at 33.56% versus 52.64%; it is therefore the preferred
quality/speed interview worker rather than merely a challenger.

The 1.7B end-to-end FluidAudio-plus-ASR wall was not measured. The measured
53.77-second ASR stage leaves wide headroom under five minutes. For planning
only, adding it to the separately measured 15.39-second FluidAudio wrapper
stage gives **69.16 seconds (RTF 0.03842)**. This is a non-measured component
sum across different runs and cache states, not a measured 1.7B end-to-end
pipeline observation.

### Actual sequential end to end

A fresh orchestration then ran the cached/staged FluidAudio quality job and,
only after it exited, launched a fresh resource-aware Qwen process against the
new diarization artifact. External wall was **45.67 seconds** for 30 minutes of
audio (RTF 0.02537, 39.41x real time):

- FluidAudio wrapper subprocess: 15.39 seconds external; its fresh CLI wall was
  15.21 seconds and sampled CLI RSS peaked at 0.55 GiB;
- Qwen wrapper subprocess: 30.28 seconds external; its internal fresh-runner
  wall was 29.54 seconds and service job after model load was 28.07 seconds;
- the Qwen stage saw 1.66 GiB peak sampled process RSS, 3.20 GiB maximum
  active-plus-cache, and zero host swap growth.

The subprocesses were strictly sequential, so this is a measured pipeline wall,
not an arithmetic projection. Their memory counters are still not additive:
FluidAudio's CLI RSS omits Core ML service memory, while Qwen RSS and MLX
allocator counters overlap.

The repeat was content-stable. The newly emitted 589 FluidAudio intervals were
exactly equal to the settled normalized output; the turn-plan hash remained
`1b69d915…569b8`; and the Qwen text plus all 195 segment dictionaries remained
exact, including the `bde962f3…aa0` hash and token counts. Whole artifact and
raw FluidAudio hashes changed because they include run-specific paths, timing,
and resource observations. No content nondeterminism was observed in this
repeat.

## Coverage and transcript score

On the 30-minute file, the reconciler accepted 1,645.62 seconds of anonymous
single-speaker activity (91.42% of the file) in 195 turns. It retained 2.24
seconds of short bridge silence as turn context. The abstention ledger contains
3.70 seconds of overlap, 6.55 seconds of final short turns, and 14.77 seconds
where only filtered raw fragments were active. Another 127.11 seconds was
unclaimed gap. The categories sum exactly to 1,800 seconds.

The corpus has a participant transcript but no interviewer transcript. All
scores below use the same orthography-sensitive Han-character/Latin-token
metric; none measure semantic equivalence or behavioral-analysis validity.

| Hypothesis scope | Mixed-token error | Interpretation |
| --- | ---: | --- |
| Full S1 + S2 transcript | 103.07% | Not a valid quality comparison: interviewer text is penalized against a participant-only reference. |
| **S1 only** | **52.64%** | Participant label selected by the independent FluidAudio interval-overlap oracle; 2,091 edits / 3,972 reference tokens. |
| S2 only | 92.47% | Transparent counterfactual label score, not a second identity claim. |

The S1 result is 1.81 percentage points better than the monolithic Qwen 0.6B
participant-microphone result (54.46%), but it remains worse than FireRed on the
participant microphone (49.02%) and VibeVoice's internally attributed
participant output (45.29%). These are not clean model-only comparisons:
FluidAudio + Qwen receives the canonical two-microphone downmix and an
oracle-selected anonymous label, while the other rows receive the participant
microphone, and VibeVoice has a different speaker-attribution mechanism.

The controlled A/B establishes that 1.7B is the better Qwen worker for these
speaker-isolated Cantonese turns while still beating the service target by a
wide margin. It does not establish a global ASR ranking: VibeVoice and FireRed
were scored on the participant microphone with different attribution and
timing contracts. For mono interview analysis, FluidAudio plus 1.7B is now the
recommended transcript-quality route, with 0.6B retained when minimum latency
or memory matters. VibeVoice remains the product-editing route, and FireRed
remains the dialect-focused balanced route. FluidAudio's weak dense-turn
CantoMap change score still rules out treating these bounds as validated
interruption or rapid-backchannel labels.

## API and reproduction

`mlx-audio` 0.4.5's public Qwen `generate()` batches only chunks produced from
one input waveform. The runner therefore calls the inspected private
`_generate_chunks_batched` method to batch already bounded turns. It validates
the method signature before inference and records the installed source hash
(`c0826905…c45d250`). This coupling is pinned evidence, not a stable public API;
a dependency upgrade must fail closed until revalidated.

```bash
<mlx-python> model_tests/benchmark/run_turn_attributed_mlx_asr.py \
  --model-path <cached-qwen3-asr-0.6b-8bit-snapshot> \
  --audio model_tests/benchmark_data/spice/VF19A_Cantonese_interview30m_canonical_mix_16k.wav \
  --diarization-run model_tests/benchmark_runs/fluidaudio_quality_spice30m.json \
  --output model_tests/benchmark_runs/qwen_fluidaudio_turns_spice30m.json \
  --language Cantonese --batch-size 1 --max-tokens 16384

python3 model_tests/benchmark/score_asr.py \
  --reference model_tests/benchmark/manifests/spice_vf19a_cantonese_interview30m.json \
  --run model_tests/benchmark_runs/qwen_fluidaudio_turns_spice30m.json \
  --hypothesis-speaker S1 \
  --output model_tests/benchmark_runs/qwen_fluidaudio_turns_spice30m_s1_score.json
```

Before production, repeat the whole pipeline on a physical 16 GB Mac, replace
the SpiCE partial reference with independently checked two-role labels, and add
forced alignment only where word-level editing times are required. Downstream
analysis should retain anonymous labels and uncertainty; these measurements do
not validate personality, competence, deception, or hiring-suitability claims.
