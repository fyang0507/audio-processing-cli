# Diagnosing and enhancing audio

Use this lane for recording diagnosis, speech cleanup, leveling, and a proxy for machine listening. For a scoped time or frequency correction, also read [targeted-fixes.md](targeted-fixes.md).

## Work from the original and enhance last

Diagnosis alone needs inspection, not a render. When transcription is requested or needed, save the original-source result before enhancement or reuse its saved result; follow [transcribe.md](transcribe.md). Pure enhancement does not require transcription. The enhancement loop is:

1. **Inspect the original.** Read measured facts and the profile's conclusions. Explain them and stop if the user only asked for diagnosis.
2. **Resolve without rendering.** Read the dry run's operations, predictions, component evaluations, and any adjustment scopes before creating media.
3. **Render from the original.** Enhancement is the final media-changing step. A revision starts from that same source, not a previous render.
4. **Verify the delivered file.** Read the emitted report, by default `OUTPUT.report.json`, rather than quoting dry-run predictions as achieved measurements.

## Choose by destination

`transcription` prepares a speech proxy for machine listening; `product-demo` treats speech and non-overlapping program audio for human listeners. The profile sets eligible stages, targets, and safe bounds. A disabled stage reporting `no_op` / `disabled_by_profile` is expected. Keep eligible stages unless the user asks to skip them; do not skip a guard merely to force a desired number.

Automatic balancing can miss quiet intended music or machine audio below its detection threshold. A region outside detected speech is not necessarily noise: it may contain wanted program audio, and the activity detector does not know music from a fan. Use user feedback, known event ranges, or other evidence for a scoped correction instead of reclassifying the whole recording.

Gain and EQ affect overlapping speech and music together. Do not promise separate control without separate sources. Broadband suppression is likewise bounded cleanup, not source separation; its own reported operation and evaluation determine whether it ran. A high-pass or hum filter alone is not evidence that broadband noise was suppressed.

Choose the broadband method before processing and inspect the selected invocation's live help. The stationary method needs usable reference evidence; the optional RNNoise method needs an explicitly provisioned model. Use [model-packages.md](model-packages.md) for that lifecycle. An abstention does not authorize switching methods when the task forbids fallback. Model-guided processing can still alter wanted sounds within speech. Its reported mask bound is not guaranteed delivered noise reduction or preserved intelligibility, and `applied` does not certify either.

## Read parent and child outcomes

Use `audio report summary` to navigate a saved enhancement report without processing the audio again; read its help for the input and output surface. Start with its compact navigation view to locate delivered program metrics, separate phase/scope groups, and nested unmeasured checks. Follow its report pointers for full measurements and operation details. Repeated evidence values can share a finding index while their occurrences retain different scopes: an indexed-entry count is not a count of delivered failures. A missing measurement phase remains unknown. The summary preserves recorded outcomes and does not certify overall success.

Stages can be `applied`, `no_op`, `skipped`, `abstained`, or `failed`. `applied` means some work ran, not that every goal was met. Read `stages[].component_evaluations[]` for child statuses and reasons: `environment-denoise` can apply a filter while `broadband-denoise` abstains. Quote the actual child outcome and resolved parameters rather than inferring them from the parent.

For source balancing, also read `stages[].final_region_evaluations[]` when present. `bounded_outside_target` means the safe gain bound was reached without attaining that region's numeric target; `abstained_overlap` preserves a region that could not be adjusted independently. The top-level `unresolved[]` collects explicit component abstentions and unmet measured regional or loudness-range targets. Read its scope, status, reason, and `measured_at` when supplied: `predicted_pre_encode` is a prediction and `encoded_output` is the delivered-file check. An empty list does not establish perceptual quality or rule out undetected problems.

Read each navigation outcome count's collection pointer. A stage's `inside_target_regions` count describes its own recorded stage decision, while `final_region_evaluations` is a separate collection. Neither replaces a fresh inspection's rule evaluations or speech reference. Keep original rule misses, intermediate limits, predictions and delivered measurements separate when explaining remaining targets.

Report these remaining limits even after a successful render. Do not enlarge gains or add a second pass simply to make every row green. An approved listening result remains useful evidence.

When a component supplies `noise_reference_scopes`, read each scope's eligibility or rejection reason with the component's final decision. Locally eligible references can still fail pooled checks; rejected scopes explain abstention without authorizing a fallback tool, model, or stronger correction.

## Measure the file that was delivered

| Report surface | Interpretation |
| --- | --- |
| `source`, `engine`, `profile` | original identity, tool/profile version, targets, and safe bounds |
| `observations`, `regions`, `rule_evaluations` | original measurements, detected scopes, and profile conclusions |
| `stages`, `adjustments` | resolved operations and their scoped outcomes |
| `measurements.before`, `.predicted`, `.after` | source measurements, dry-run predictions, and post-encode checks respectively |
| `resolved_operations_sha256` | identity of the resolved processing plan |

Quote `measurements.after.program_actual.integrated_loudness_lufs`, `.true_peak_dbtp`, and `.loudness_range_lu` for the delivered file. The corresponding `before.program_actual` measures the original; `predicted.program_actual` still describes a simulation despite its field name.

Older reports lack `program_actual`. In their `measurements.after.program` block, `input_i` is the delivered file's integrated loudness in LUFS, `input_tp` its true peak in dBTP, and `input_lra` its loudness range in LU. The names come from measuring that file as input to the analyzer. Raw `output_i`, `output_tp`, and related values describe normalization diagnostics, not the file that was delivered; the legacy fields retain that meaning. Never turn an absent or null measurement into zero.

Check `rendered`, `timeline_verification`, an available `timeline_preserved`, `measurements.after.duration_delta_ms`, and `final_peak_validation` along with the actual after measurements. A dry run has only predictions: `predicted_pass` is not post-encode validation. The CLI enforces its active publication checks; report the actual target/limit and any skipped stage rather than imposing every profile preference as an unconditional acceptance gate. Successful publication does not certify every child target or prove that someone prefers the sound.

## Duration is not alignment evidence

Read `duration_basis` before comparing durations. The outer `source`/`output` durations are probed audio-stream or container metadata; their `decoded_audio` objects describe actual decoded samples. `timeline_verification.scope: decoded_audio_duration_only` and its tolerance qualify `timeline_preserved`: a pass checks decoded length only. Dry runs report `not_run` and omit the boolean. Older reports without this scope checked probed duration only and could mark a dry run true.

For video, compare available `source.timing` and `output.timing` audio/video starts and `primary_audio_start_minus_video_start_seconds`. These describe metadata origins. Read the explicit `content_alignment` and `av_sync` abstentions even when durations or starts match. A changed start, codec delay, or padding does not by itself prove clipping or a content shift; never claim exact sync, add an inferred offset to transcript bounds, or shift timestamps to repair an assumed defect. Report the available evidence and unresolved alignment instead.

## Fixed regions and fresh detection answer different questions

`region_basis` identifies the source timeline, original detection, fixed source regions, and report-local IDs. Before/after regional measurements reuse those original scopes and IDs to show how the same intervals changed. A fresh `inspect` of the enhanced output runs detection again and may split, merge, or reclassify intervals; its region IDs are local to that new analysis. Compare source-timeline intervals, not equal-looking IDs across reports.

Use `audio report compare` for saved reports with compatible identity and timeline evidence. Its compact navigation view starts with actual overlap, ambiguity and unmatched counts; follow the side-specific pointers to their intervals. Pair counts differ from region counts, and ambiguous overlaps have no preferred match. It keeps each report's intervals, measurements, and speech reference separate. A refusal for missing identity or timeline evidence is not permission to guess a link. A matching digest and decoded duration still do not establish content alignment or A/V sync.

ASR wording and VAD or program classification can change after enhancement. That alone is expected, not evidence of corruption or a reason for speculative re-editing. Keep the canonical original transcript, report substantive listening findings and measured limits, and do not require classification invariance. Enhancement preserves time; it does not remove fillers or other words.

## When it refuses

Use [failures.md](failures.md) for the error channel and safe remedy handling. Common domain refusals protect the original input, reject a marked prior render, or require a distinct writable output/report destination. Choose a fresh output when replacement was not requested.

An invalid adjustment belongs in [targeted-fixes.md](targeted-fixes.md). A very short clip can lack an integrated loudness measurement despite a real non-silent peak; preserve that distinction and read the stated remedy instead of describing the signal as silent.
