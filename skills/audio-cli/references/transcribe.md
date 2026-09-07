# Transcribe original media and export saved results

## Save once, reuse for delivery

1. Reuse unchanged original-source JSON when its scope and produced capabilities meet the request.
2. Otherwise discover stacks, inspect the selected stack's capabilities, and plan the original input with only the capabilities needed. Use live help/catalogs for choices and syntax; a suggested minimal plan may need capabilities added for the user's deliverables.
3. Provision the plan's missing packages through [model packages](model-packages.md), then run recognition and save canonical JSON. When enhancement is also requested, transcribe first and enhance last.
4. Read the transcript, requested-capability outcomes, coverage, and abstentions before exporting. Processing completion does not guarantee complete timing, speech coverage, or attribution.
5. Export readable/subtitle deliverables from that JSON with `audio transcribe export`; retain it for subsequent exports without rerunning models or editing recognized text.

Choose based on required output and the catalog's stated limits. Package size and a declared capability do not establish accuracy on new media. For subtitles, request real `word_timestamps`. For speaker labels, request diarization; labels remain anonymous without independent identity evidence.

## Decisions that need care

- On Qwen, adding diarization changes recognition input spans and can change the text. Treat independently requested variants as separate results; do not require byte-identical wording or splice them together.
- Canonical output and `verbatim` do not guarantee filler, repetition, or dialect recall. Do not clean the canonical text or describe omitted words as intentional cuts.
- Published timing uses the original decoded-audio timeline. Do not pre-clip input for a retry, manually offset bounds, or add container/stream starts. Duration equality does not certify subtitle-to-video sync.
- Read abstentions even on successful completion. Text can survive unavailable alignment; speaker labels on emitted segments do not establish attribution of all original speech. An empty overlap ledger does not establish no overlap if the plan did not detect it.
- Never invent word timing from segment/turn extents or source duration. A capability that abstained remains unmet even when other output is useful; explain its affected scope.

## Recover without changing the task silently

For partial results, retain the JSON and follow a concrete range-resume remedy against the original. If the run made no progress, repeating the same request is not a remedy. Use [failures](failures.md) for command failures and bounded repairs.

When the task permits alignment-boundary recovery, choose a clipping limit from the recorded overrun evidence and live help. Inspect recorded corrections and outcomes afterward. Clipping an estimate is not measuring its true acoustic endpoint, and a larger limit cannot fix missing output or text mismatches.

A changed stack or alignment policy is a separate attempt with new result, export, and diagnostic destinations. Preserve the earlier evidence and required scope/capabilities. Check the new result before export; do not splice text, timing, or anonymous speaker labels across attempts. Stop when the authorized recovery is exhausted.

## Export prerequisites

SRT/VTT need real word timing; inspect export omissions as well as the produced cues. Timed text/Markdown need supplied segment or word timing for every segment. Follow the CLI's refusal when those prerequisites are unmet; do not manufacture bounds or silently downgrade a required timed deliverable.

Merge only compatible disjoint results through the exporter. Independent VibeVoice runs with native diarization cannot be merged because equal-looking anonymous labels do not establish shared identity. Export them separately or plan the needed scope together.

Use distinct output destinations and explain recognition, timing, and coverage limits in the reply. Keep the canonical JSON unchanged.
