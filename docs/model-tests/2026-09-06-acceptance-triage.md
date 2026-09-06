# Acceptance triage, 6 September 2026

This review follows the [full-source acceptance record](../../model_tests/benchmark/results/2026-09-06-pr49-full-e2e.json) on candidate `00873235aed5c1ac17e53baaa2a1c687eaa5e26b`. The initial audit inspected source at `e42dcad5a72357d7e4d1752ffb3382ae1441eec2` without rerunning audio. A subsequently authorized diagnostic reproduction at `fe64fd0296a33318e3145c72f59261cbc5f9c784` captured the boundary failure described below, with temporary instrumentation removed afterward. No production timing repair or new acceptance pass is claimed. The [developer acceptance gate](../agent-acceptance.md) defines future scope and verdicts explicitly.

## What Qwen's partial result means

| Component or delivery | Recorded outcome | What it establishes |
| --- | --- | --- |
| Plain Qwen ASR | Run exited 0; 15 text segments | Plain full-source execution completed. Recognition accuracy is not established by that exit. |
| Diarized Qwen ASR | Run exited 0; 33 text segments | Selected ASR units completed. `complete: true` is independent of alignment success. |
| Diarization and overlap | Two anonymous labels, 31 turns, one overlap interval; both capabilities report `produced` | These components produced results. Five non-alignment abstentions preserve overlap/short-turn/fragment limits; full attribution or speaker accuracy is not proven. |
| Word alignment | 32 segments have word streams; one lacks timing; `word_timestamps: abstained` | Complete word-timing delivery was not achieved. Text and its assigned speaker remain available. |
| Untimed native exports | TXT and Markdown delivered before the timed refusal | Useful speaker-labeled transcription survived. |
| Timed Markdown | Exit 2, `timing_required_for_timestamps`, `segment_id: seg_27` | The exporter correctly refused an unbounded text segment. Required timing delivery failed; this is not a total ASR/diarization failure. |
| SRT/VTT | Not attempted after the stopping condition | Blocked dependent steps, not observed subtitle-command failures. |

The direct evidence is the local canonical `tests/artifacts/e2e-pr49-2026-09-06/04-qwen-diarization/diarized.canonical.json`, SHA256 `96dcd460b0892b2febe6bc1cde46dd99493aca31088cb8993d035b07f572fbb3`, and the case's native `commands/017-diarized-run/` and `commands/020-diarized-timed-md/` streams. The compact tracked record preserves their counts, outcomes, hashes and locations; raw private transcripts remain untracked.

`seg_27` contains `Like.` with speaker `S1`, but has no `words`, `start` or `end`. The unique alignment abstention is 96.570437–97.996625 seconds, matching `turn_25`. The canonical schema does not explicitly join the segment to that abstention: the association follows from the unique missing segment, unique alignment abstention and host construction. These bounds describe the processing/abstention scope; they are not a measured extent of the word.

## Detection needs no other ASR model

The [Qwen orchestrator](../../src/audio_cli/transcribe/orchestrator/qwen.py) sends completed ASR units to alignment, validates the returned words, and reconciles them with sentence text. It then tests for ordinary lexical text without a `words` field and records `alignment_unavailable` at the affected unit's bounds. [Publication](../../src/audio_cli/transcribe/execution/publication.py) records the capability as abstained and counts the segment without words. This is internal result validation against the same run, not a comparison with FireRed or VibeVoice.

Separately, [readable export timing](../../src/audio_cli/export/timing.py) requires supplied segment bounds or the extent of a nonempty real word stream. Neither exists for `seg_27`; the [CLI](../../src/audio_cli/cli.py) translates the resulting error to the recorded refusal. The exporter does not need to listen to the audio to detect missing metadata. It also cannot judge whether the text, speaker or supplied word boundaries are accurate merely because the metadata exists.

At the initial audit, the reason alignment became unusable was unresolved. The source supports three distinct paths:

- The [aligner stage](../../src/audio_cli/transcribe/stages/aligner.py) can catch a per-unit inference/conversion exception and return `words: null` with error type/message while the stage itself exits 0.
- The [alignment adapter](../../src/audio_cli/transcribe/adapters/aligner.py) can reject invalid tokens, bounds, containment or ordering and withhold the unit's words.
- The [sentence adapter](../../src/audio_cli/transcribe/adapters/qwen.py) can reject words that do not reproduce the recognized text after punctuation/whitespace/case normalization, including leftover words or an empty stream for lexical text, and retain the sentence without timing.

The original retained diagnostics do not identify which path occurred. The aligner stdout log is empty and stderr only records loading and progress through 31 units. [Transport](../../src/audio_cli/transcribe/transport/service.py) reads the structured aligner response from the orchestrator's temporary working directory, which is removed at exit; the durable log option retains stdout/stderr separately. The original bundle contains no structured aligner response or recorded host rejection reason. Therefore “the model returned no timestamps” would overstate that evidence. The separately authorized reproduction below captured a returned timestamp that the host rejected.

## Diagnostic reproduction: a returned endpoint outside the unit

The [boundary investigation record](../../model_tests/benchmark/results/2026-09-06-qwen-alignment-boundary.json) records one new full-source CLI run with the same stack, wants and fixture hash. Temporary host-only instrumentation copied the aligner request/response before cleanup, without changing inference, input, model configuration or normalization. It was removed immediately after capture and the transport's original hash verified. The original acceptance remains unchanged; this is development diagnosis, not a fallback acceptance attempt.

The new canonical segments, turns, abstentions and overlap intervals exactly match the original run. The captured request for `turn_25` contains `Like.`, speaker `S1`, and bounds 96.570437–97.996625 seconds. The aligner response contains the word `Like` at 96.570–98.010 seconds. Of all returned words, this is the only one exceeding a unit boundary by more than the adapter's 0.501 ms serialization-rounding allowance. Its start overrun is 0.437 ms, which the adapter can reconcile. Its end overrun is **13.375 ms**, which fails the end-within-unit check; the adapter withholds the whole word stream and the orchestrator records the abstention. No ASR text or speaker was missing.

The pinned ForcedAligner model configuration uses `timestamp_segment_time: 80` milliseconds. The installed `mlx-audio` 0.4.5 runner multiplies predicted timestamp classes by that value before converting to seconds; its source/configuration hashes are in the investigation record. The returned local endpoint is consistent with 1.440 seconds on this 80 ms grid, while this input unit lasts approximately 1.426188 seconds. The observed failure is an out-of-unit forced-alignment estimate rejected by strict host containment. The coarse output grid explains the precision mismatch; it does not prove the true acoustic word boundary or establish that clipping every overrun is safe.

Issue #51 now has a concrete prevention target: preserve the response and unit/segment rejection reason, and define evidence-backed handling of a quantized endpoint near the supplied audio boundary without widening the generic guard or fabricating timing. Controlled edge/large-overrun cases and a subsequent original-source CLI run must validate any proposed repair. No production timing policy was changed in this investigation. The shared forced-aligner path also serves Qwen 0.6B and VibeVoice when word timing is requested; this exact failure was reproduced only on Qwen 1.7B, and FireRed supplies native word timing instead.

## Feedback dispositions

| Feedback | Triage | Follow-up |
| --- | --- | --- |
| Capability outcomes are difficult to see in a receipt | Genuine gap: the receipt already counts six abstentions but omits which requested capability abstained. | [#50: requested capability outcomes in receipts](https://github.com/fyang0507/audio-processing-cli/issues/50). Preserve existing counters and canonical bytes. |
| Alignment abstention has no diagnostic cause/segment join; export fix requests timing already requested | Genuine diagnosis/remediation gap. The separate reproduction identified a returned word endpoint 13.375 ms outside its input unit. The timed refusal itself correctly protects the contract. | [#51: traceable unavailable alignment and accurate remediation](https://github.com/fyang0507/audio-processing-cli/issues/51). Preserve rejection evidence and validate a boundary-integration remedy without guessing timing. |
| Adding diarization changes text and can withhold scope | Genuine operator-guidance gap. The [orchestrator](../../src/audio_cli/transcribe/orchestrator/qwen.py) changes ASR units when diarization is selected; existing skill cautions do not make that consequence explicit. | [#52: explain diarization's effect on text and scope](https://github.com/fyang0507/audio-processing-cli/issues/52). No model/workflow redesign is implied. |
| Reports mix original/intermediate/delivered evidence and repeat large structures | Genuine navigation gap after the new summary/comparison features: demo summary has 22 indexed limit entries; comparisons are 137,482 and 122,505 bytes. | [#53: compact navigation by measurement stage and scope](https://github.com/fyang0507/audio-processing-cli/issues/53). Retain original pointers and independent region/speech references. |
| Inspect/enhance compact receipts | An optional entry point for #53, not a separate missing processing capability. | Evaluate within #53; do not create a duplicate issue. |
| Add abstention counts to receipts | Already implemented in [receipt construction](../../src/audio_cli/transcribe/execution/receipt.py), and present in this run's raw stdout. | No new counter issue. #50 is narrowly about capability outcomes. |
| Add source/stack/timing provenance, compact plans, run progress and a transcription log destination | Implemented by PR #49 within #46/#48. | No duplicates. The missing structured alignment response is the distinct #51 gap. |
| Explain known VibeVoice configuration warnings | Source-backed explanation already exists in [CLI feedback](../cli-feedback.md) and its pinned-source links. | No new warning-classification issue. Warnings are not presumed harmless; generic warning summarization is optional. |
| Add whole-source coverage or source hash fields everywhere | Not demonstrated necessary by this run. Source identity was recorded by the harness; existing range/completion/absence contracts are deliberate. | No schema expansion based solely on this suggestion. Revisit only with a concrete consuming workflow. |
| Report zero-duration word counts | Sixteen were observed; zero duration is not by itself a proven boundary error without reference annotations. | Preserve as a timing-review observation; do not create a correctness issue or clamp/invent durations. #51 can retain relevant rejection evidence. |
| Projected costs differ from observed runtime | A single run with queue waiting separated is not a controlled performance benchmark. | No performance defect established. |
| Noise effectiveness, speech preservation and exact A/V sync are unmeasured | These are explicit evidence limits. RNNoise applied; absent quality evidence is not skipped processing. | No duplicate denoising/sync issue; #41/#42 already addressed their agreed scope. |
| Demo LRA/calibration limits and fresh-detection differences | Recorded limits remained; media bytes match the previously accepted render. Fixed and fresh scopes are different. | Preserve user acceptance and the numerical limits separately. Do not infer a new DSP regression. |
| Missing non-audio bootstrap command records | Harness limitation, not a CLI defect. | The new acceptance guide starts recording before operator activity and requires disclosure of missing records. |

These four issues extend the already implemented scope of #46/#48 without reopening their delivered features. They are future work, not fixes claimed by PR #49 or by this documentation change. Filler/editorial editing remains outside scope.

## Which stacks the issues affect

| Issue | Qwen 0.6B / 1.7B | FireRed | VibeVoice |
| --- | --- | --- | --- |
| #50: capability outcomes in receipts | Both use the shared receipt builder | Same shared builder | Same shared builder |
| #51: alignment diagnostics and remediation | Both use ForcedAligner and the shared bounds adapter; overrun observed on 1.7B | Native word timing, so this ForcedAligner overrun path does not apply; diagnostic infrastructure and generic export remedy are shared | Same ForcedAligner/bounds adapter when word timing is requested; native segment bounds can still support timed Markdown independently |
| #52: effect of diarization on text/scope | Both replace fixed ASR units with diarizer units | FluidAudio add-on is supported, but ASR remains VAD-based and speakers are joined afterward | Native speaker generation; the diarization request does not reslice ASR input; optional overlap handling may withhold attribution while retaining text |
| #53: enhancement report navigation | Independent of ASR stack | Independent of ASR stack | Independent of ASR stack |

This is source-level applicability, not evidence that the same recording failed on every exposed stack. The shared [receipt builder](../../src/audio_cli/transcribe/execution/receipt.py), [Qwen workflow](../../src/audio_cli/transcribe/orchestrator/qwen.py), [FireRed workflow](../../src/audio_cli/transcribe/orchestrator/firered.py), [VibeVoice workflow](../../src/audio_cli/transcribe/orchestrator/vibevoice.py) and enhancement [report package](../../src/audio_cli/pipeline/reports/) establish the ownership. In particular, do not generalize Qwen's ASR reslicing/text-withholding behavior to FireRed or VibeVoice.
