---
name: audio-cli
description: Measure, enhance, and transcribe local audio or video with the audio CLI; export transcripts and subtitles, apply scoped corrections, and manage required model packages. Use for recording diagnosis, speech cleanup, transcription, and subtitle requests.
---

# Audio work with the `audio` CLI

Use the `audio` CLI for audio inspection, processing, transcription, export, and model management. Use its live help for syntax and its responses for measurements, limitations, and remedies. This skill supplies the workflow and decisions around those commands.

Use the supplied executable consistently, or `audio` from the current directory if none is specified. Use absolute media/output paths across directories. No repository checkout or implementation imports are needed.

## Read for the task

| Request | Reference |
| --- | --- |
| Diagnose or enhance a recording; choose a treatment | [Enhancement](references/enhance-audio.md) |
| Correct a particular interval or frequency | [Targeted fixes](references/targeted-fixes.md) |
| Interpret or compare saved inspection/enhancement reports | [Saved reports](references/saved-reports.md) |
| Transcribe original media or export saved JSON | [Transcription and export](references/transcribe.md) |
| Provision, verify, or reclaim packages | [Model packages](references/model-packages.md) |
| Resolve command setup or a failure | [Failures](references/failures.md) |

Read only what the task needs. Diagnosis needs no render; enhancement needs no transcription; exporting saved results needs no model run.

## Working rules

- Preserve the original media and its timeline. Resolve every enhancement revision from the original. When transcription is also needed, save or reuse its original-source JSON first, then enhance last.
- Keep canonical JSON unchanged; put interpretation in the reply. Never invent missing text, measurements, timing, or speakers. Canonical means unedited original-source output, not guaranteed verbatim recognition.
- Separate measured results from listening preferences. Preserve an approved result instead of chasing every numeric target. Enhancement makes no editorial or filler-removal decisions by design.
- Provision models and runtimes only through `audio packages pull`. The sole automatic exception is the small, hash-verified Silero speech-activity model on first use, including inspection/enhancement.
- Follow actionable CLI remedies within the user's scope. Do not silently change the requested method, stack, capabilities, or deliverables to get a successful exit, or substitute another audio tool.
