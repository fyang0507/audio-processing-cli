# Interpret saved enhancement reports

Use `audio report summary` for one saved inspection/enhancement report and `audio report compare` for compatible reports. Start with compact navigation and follow its pointers only where the question needs more evidence. These commands require no reprocessing; use live help for output choices.

## Answer the user's question

Report what changed, what remains unresolved, and which conclusions still need listening evidence. Keep original measurements, dry-run predictions, intermediate treatment checks, and delivered-file measurements separate. A stage applying or an empty unresolved list does not establish that every component succeeded or that the sound is preferred; inspect child limits relevant to the request.

Use recorded post-encode measurements for the delivered file. In older reports without `program_actual`, the analyzer's `input_i`, `input_tp`, and `input_lra` measure the file it was given; `output_*` describes normalization diagnostics. Do not substitute those diagnostics for delivered measurements.

## Compare like scopes

Before/after regional checks in one enhancement report reuse fixed original intervals. A fresh inspection detects regions again and can split, merge, or reclassify them. Compare source-time intervals, not report-local IDs or raw success counts. Keep each report's speech reference and measured scopes separate; an ambiguous overlap has no preferred match.

Fixed-region improvement and newly detected out-of-target regions can both be true. Neither collection is an overall success rate. A scope-comparison result concerns time coverage, not equality of levels, targets, or sound quality.

## Respect evidence limits

A decoded-duration pass establishes length within the reported tolerance, not content alignment or A/V sync. Matching stream starts do not establish sync either; do not infer an offset or shift transcript times from metadata differences.

Changed ASR wording or activity detection after enhancement alone is not evidence of corruption. Preserve the original-source transcript and distinguish fresh detection from treatment verification. Keep absent measurements unknown; a short clip with no integrated loudness measurement can still have a real non-silent peak.
