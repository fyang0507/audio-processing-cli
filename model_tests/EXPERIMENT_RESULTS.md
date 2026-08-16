# ASR experiment results

This is the concise evidence companion to
[DECISION_REPORT.md](DECISION_REPORT.md), distilled from
[FINDINGS.md](FINDINGS.md). Unless stated otherwise, runs used fresh processes
on an Apple M4 Max with 64 GiB unified memory. RSS and allocator counters have
different scopes: do not add them or treat them as physical-16-GiB proof.

## Decision evidence

| Decision | Experiment run | Measured result | What it supports |
| --- | --- | --- | --- |
| **Treat Qwen as a transcript worker, not a full pipeline.** | Both MLX 8-bit checkpoints received the same 139.284-second Mandarin-English, Sichuanese-bearing, two-speaker clip with `language=None`, one 180-second container, batch 1. | Both emitted mixed Han/Latin text. 0.6B returned `Chinese`; 1.7B returned `English`. Each emitted one full-file container, `speaker=null`, and no speech/word times. 1.7B retained observed `耍啥子`; 0.6B rendered `刷啥子`. | Automatic language output exists but is neither localized nor a safe code-switch router. Mixed-script retention is verified; code-switch accuracy and dialect breadth are not. Use external timing/speaker capabilities. |
| **Use Qwen 1.7B by default for Cantonese interviews; 0.6B for footprint/latency.** | FluidAudio produced one fixed 195-turn plan from the 30-minute Cantonese SpiCE canonical mix. Both Qwen sizes received identical turns, ordering, `Cantonese` hint, batch-one/cache-clear policy, and token budget. | 1.7B: **33.56% (1,333/3,972)** oracle-participant mixed-token error, **53.77 s** ASR wall, 3.02 GiB RSS / 4.86 GiB MLX proxy. 0.6B: **52.64% (2,091/3,972)**, **29.90 s**, 1.66 GiB / 3.20 GiB. | The 19.08-point reduction supports 1.7B for this Cantonese route. It does not establish the same gap for Mandarin, English, or other dialects. The participant mapping is an oracle, not role identification. |
| **One interview use case can support both capture layouts.** | A strictly sequential fresh FluidAudio → reconciliation → Qwen 0.6B run processed the same 30-minute mix; the channel-first route was separately inspected for dominance and bleed. | Mono pipeline: **45.67 s** wall, 195/195 turns, 91.42% single-speaker activity accepted. Capture channels preserve role provenance but still need bleed gates and deduplication. | Mixed audio and dedicated channels should converge on the same Qwen worker contract after producing trusted external bounds. The complete two-channel merge remains unmeasured. |
| **Expose FireRedASR2S as its full offering; load LID only on demand.** | Paired full-pipeline FireRed CPU float32 batch-4 runs on the same 139.284-second clip, LID on vs off; a separate LID-off run covered a 30-minute SpiCE participant channel. | LID-on vs off: **162.09 vs 84.24 s** inference and **12.27 vs 9.15 GiB** RSS; ASR text and all 246 word texts/times were identical. The 30-minute LID-off run took **665.26 s**, used 9.12 GiB RSS, and emitted native word intervals; an exact-repeat hour retained text with ≤1 ms drift. | FireRedVAD/LID/ASR/Punc is a capability-rich pipeline for region LID and native word time/confidence, not a speaker-aware or low-latency default. Repeat audio validates duration stability, not accuracy. |
| **Prefer VibeVoice over FireRed except for dialect or resource constraints.** | Compare both on a 27.8-second Mandarin-English/Sichuanese probe, a 139.3-second editing probe, frozen CantoMap Cantonese, and separate 30-minute SpiCE configurations. | VibeVoice normalized Sichuanese `看哈→看一下`; FireRed retained `看哈`. VibeVoice retained `Fortnite`; FireRed produced `fornight`/`for for night`. On identical CantoMap audio, VibeVoice scored **44.88% (171/381)** versus FireRed **47.77% (182/381)**. SpiCE results were VibeVoice **45.29% (1,799/3,972)** on an oracle-selected mixed-channel stream and FireRed **49.02% (1,947/3,972)** on the participant microphone, so they are not a clean head-to-head. FireRed was faster/lighter; VibeVoice OOMed under a strict 16 GiB MPS cap. | VibeVoice is the stronger structure/general-transcript substitute on these fixtures, while FireRed is the explicit dialect-form/LID/native-time or constrained-resource route. The small same-audio Cantonese gap and two short Sichuanese examples cannot rank the models across Chinese varieties. |
| **Abstain on dense conversational turns.** | A 149.9-second CantoMap conversation contained 75 annotation-order speaker changes. | VibeVoice matched 39/75; the best sherpa row 9/75; FluidAudio quality 3/75. | None is validated for rapid backchannels, interruptions, or dense overlap. Prefer channels or review. |
| **Do not offboard FluidAudio by default yet.** | Matched direct `pyannote/speaker-diarization-community-1` revision `3533c8cf…54ee` MPS regular-output runs with the same known-two-speaker CantoMap and canonical SpiCE inputs as FluidAudio quality. | CantoMap: Community-1 improved 250-ms-collar agreement error / change F1 to **19.20% / 42.00%**, versus FluidAudio **22.31% / 5.50%**, but took **10.80 s / 1.30 GiB RSS** versus **1.26 s / 313 MiB**. SpiCE: Community-1 took **43.18 s / 1.51 GiB** and reached **94.67%** participant-interval F1, versus FluidAudio **14.74 s / 560 MiB / 95.42%**. | Direct pyannote is a feasible, modular fallback, not a measured simplification on this Mac. Keep FluidAudio behind a diarizer adapter; do not broaden FluidAudio into a general platform or migrate the standalone Silero VAD. |
| **Use Silero only when a standalone speech map is required.** | Silero VAD 6.2.1 ONNX on the prepared 149.9-second CantoMap slice with the product-CLI thresholds. | **0.369 s** inference, 115.36 MiB RSS; against the union of ELAN speaker intervals: 76.55% precision, 95.67% recall, 85.05% F1. | This configuration is a fast speech-activity component, not diarization. The ELAN union is not independently adjudicated VAD ground truth. |

## Evidence boundaries

- No frozen, held-out Mandarin-English switch-span, filler, repair, or broad
  Chinese-dialect suite; the Qwen capability probe verifies interfaces only.
- No human word-boundary labels for FireRed or ForcedAligner MAE/P95.
- No physical 16 GiB machine validation. FluidAudio RSS omits Core ML service
  memory, and MLX/PyTorch allocator counters overlap process memory.
- No non-oracle candidate/interviewer role mapping or completed two-channel
  merged evaluation.
- Exact-repeat hour tests measure duration/resource stability, not a second
  transcript sample. Dense-turn and overlap behavior remain abstention cases.

## Primary records

- [Qwen capability probe](benchmark/results/2026-08-16-qwen-capabilities.json)
- [Same-turn Qwen comparison](benchmark/TURN_ATTRIBUTED_FAST_ASR.md)
- [Long-form ASR results](benchmark/SPICE.md)
- [Diarization results](benchmark/DIARIZATION.md)
- [Full controlled findings](FINDINGS.md)
