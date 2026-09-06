# CantoMap stereo channel-structure diagnostic

## Outcome

The original CantoMap recording contains useful speaker-correlated stereo
structure, and the canonical benchmark preparation removes it by converting
44.1 kHz stereo to 16 kHz mono. That makes channel preservation a credible
follow-up for dense conversation. It does **not** establish that downmix caused
FluidAudio's collapse: no channel-aware FluidAudio counterfactual was run, and
a deterministic raw-energy baseline was itself a poor diarizer.

On reference-exclusive speech frames, opaque corpus speaker `F` was
right-dominant by a median 3.55 dB and `G` was left-dominant by 5.22 dB. The
association is imperfect: the opposite channel was louder in 29.55% of `F`
frames and 20.21% of `G` frames. These are oracle diagnostics from one labeled
slice, not capture metadata or inferred identities. The checked local corpus
README and EAF identify annotation tiers `F`/`G` but do not map those IDs to
left/right channels.

The practical conclusion is:

- preserve stereo or explicit microphone-role metadata at ingestion;
- do not treat instantaneous channel dominance as diarization;
- if a channel-aware path is pursued, combine per-channel speech activity with
  hysteresis/minimum-duration constraints and explicit bleed suppression or
  abstention;
- choose thresholds on other labeled conversations, then evaluate this slice
  only as held-out evidence.

## Provenance

The frozen CantoMap window is 30.500--180.400 seconds of
`160818_009F37_38_D.wav`.

| Artifact | Format | SHA-256 |
| --- | --- | --- |
| Corpus source | 289.732 s, 44.1 kHz, stereo PCM16 | `84d6edb64f9115992c89a0ca8e72c0f53259117e00139e0d463201f802b96d98` |
| Existing canonical fixture | 149.900 s, 16 kHz, mono PCM16 | `39d206733b9d8dea84b2b115253150256ad49adffade6eab7298596d41367fb8` |
| Prepared left-only follow-up | 149.900 s, 16 kHz, mono PCM16 | `0f8b15328a0f4cca5c0aa4f0ebf0542e3e74d494acb8ba261f8149f9e0b0cf8c` |
| Prepared right-only follow-up | 149.900 s, 16 kHz, mono PCM16 | `3b9cdadd70dab6b1b4826217f3c049e2e812dd55d0037725c3255fac49fa827f` |

The existing preparation invokes ffmpeg with `-ac 1`; its result is an
irreversible downmix for diarization purposes. The stereo source remains
canonical. The left/right fixtures are ignored local files and have not been
passed to FluidAudio or another diarization model.

Across the analyzed stereo window, left RMS was -23.81 dBFS, right RMS was
-26.92 dBFS, and uncentered channel correlation was 0.129. Overall level and
correlation alone do not assign speakers.

## Predeclared diagnostic

Before inspecting reference-conditioned channel dominance, the following
ladder was declared:

- non-overlapping 10 ms PCM frames;
- active when the louder channel is at least -45 dBFS;
- dominance threshold of 0, 3, or 6 dB;
- emit the louder anonymous channel when its advantage is greater than the
  threshold; emit both anonymous channel labels when an active frame is inside
  the threshold; emit neither below the activity gate.

Reference labels are used in two bounded ways: to summarize channel dominance
on frames where exactly one annotation tier is active, and by the existing
scorer to find the optimal anonymous channel-to-tier mapping. No threshold is
selected as a production default from these same labels.

Oracle-only dominance summary:

| Reference tier | Exclusive denominator | Median left-minus-right | P10--P90 | More-often-louder channel |
| --- | ---: | ---: | ---: | --- |
| `F` | 23.01 s / 2,301 frames | **-3.55 dB** | -10.51 to +5.82 dB | Right, 70.45% |
| `G` | 72.14 s / 7,214 frames | **+5.22 dB** | -3.95 to +11.87 dB | Left, 79.79% |

## Scored ladder

All agreement scores use the existing 10 ms CantoMap scorer and its optimal
anonymous mapping (`left → G`, `right → F` in every row). The zero-collar
exclusive denominator is 95.15 reference-speaker seconds across 145.76 seconds
of evaluated audio. The 250 ms collar retains only 63.66 reference-speaker
seconds / 86.64 evaluated seconds, so the lower collar number cannot stand
alone.

| Dominance threshold | Output channel-seconds L / R | Output intervals | Exclusive error, 0 collar | Exclusive error, 250 ms | Predicted / matched changes | Change F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 dB | 91.10 / 56.86 | 3,485 | **74.83%** | 51.49% | 3,422 / 75 | 4.29% |
| 3 dB | 108.41 / 77.65 | 3,389 | **104.48%** | 75.53% | 2,159 / 75 | 6.71% |
| 6 dB | 123.86 / 102.91 | 2,943 | **140.03%** | 107.21% | 1,412 / 75 | 10.09% |

The reference contains 75 annotation-order changes. Matching all 75 at these
rows is not success: precision is only 2.19%, 3.47%, and 5.31%, respectively,
because 10 ms energy volatility creates thousands of false changes. Higher
thresholds emit both channels more often, which reduces confusion but sharply
increases false-alarm speaker time. This is why the error rate can exceed 100%.

For context, FluidAudio quality on the mono fixture emitted 34 changes, matched
3, and reported 47.52% exclusive zero-collar error. The raw channel baseline
does not outperform it. What the baseline adds is evidence that the discarded
stereo cue is associated with the two reference tiers—not evidence that the cue
alone can recover stable turns.

## Reproduction

The PCM-only diagnostic uses no learned model:

```bash
python3 model_tests/benchmark/run_channel_energy_diarization.py \
  --audio model_tests/benchmark_data/cantomap/ConversationData/Subjects-37_38/160818_009F37_38_D.wav \
  --reference model_tests/benchmark_data/prepared/cantomap_yue_hk_37_38_d_030500_180400/reference.json \
  --output-dir model_tests/benchmark_runs/cantomap_channel_energy \
  --frame-ms 10 --activity-dbfs -45 --dominance-db 0 3 6

for threshold in 0 3 6; do
  python3 model_tests/benchmark/score_diarization.py \
    --reference model_tests/benchmark_data/prepared/cantomap_yue_hk_37_38_d_030500_180400/reference.json \
    --run model_tests/benchmark_runs/cantomap_channel_energy/channel_energy_dominance_${threshold}db.json \
    --output model_tests/benchmark_runs/cantomap_channel_energy/channel_energy_dominance_${threshold}db_score.json \
    --frame-ms 10 --collar-ms 250 --speaker-change-tolerance-s 1
done
```

The exact compact result and ignored-artifact hashes are frozen in
`results/2026-08-13-cantomap-channel-structure.json`.

## Next controlled study

Do not tune another ladder on this slice. A meaningful next experiment should
freeze one channel-aware algorithm in advance—per-channel VAD, attack/release
hysteresis, minimum active/silent durations, and a bleed/ambiguity policy—then
score it across multiple CantoMap pairs or another fully labeled stereo
interview set. Role or identity should come only from capture metadata,
enrollment, or human confirmation; left/right energy is not identity.

Only after that algorithm is fixed should mono versus channel-aware processing
be compared as a causal ablation. Until then, the defensible product action is
simply to retain original channels and avoid unnecessary downmixing.
