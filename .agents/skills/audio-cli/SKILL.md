---
name: audio-cli
description: Measure, enhance, and transcribe local audio or video with the audio CLI; export transcripts and subtitles, apply scoped audio corrections, and manage required model packages. Use for recording diagnosis, speech cleanup, transcription, and subtitle requests.
---

# Audio work with the `audio` CLI

Use only the `audio` CLI for audio processing, inspection, transcription, export, and model management. Use saved JSON for evidence. This skill supplies decisions and interpretation; the CLI's `--help` supplies commands, flags, and defaults.

## Use the command from any directory

Use `audio` from the current working directory. If the task supplies an absolute path to an `audio` executable, use that executable consistently, including for remedies printed as `audio ...`. Use absolute media and output paths when working across directories.

Read its help to discover the available surface. `audio doctor` reports the command's path, version, and host readiness. Neither step requires a repository checkout or runtime imports.

If the command is missing, cannot start, or lacks a required operation, follow [references/readiness.md](references/readiness.md) and report the blocker. Do not switch to an alternative audio tool or backend when a CLI result is unsatisfactory.

## Route by the request

| The request | Read before running it |
| --- | --- |
| Diagnose, enhance, or prepare an audio/video recording | [references/enhance-audio.md](references/enhance-audio.md) |
| Correct a particular region or measured frequency | [references/targeted-fixes.md](references/targeted-fixes.md) |
| Transcribe original media or export a saved result | [references/transcribe.md](references/transcribe.md) |
| Prepare packages, check readiness, or reclaim managed disk | [references/model-packages.md](references/model-packages.md) |
| A command failed | [references/failures.md](references/failures.md) |

Read only the relevant references. Diagnosis or enhancement alone does not require transcription; export of an existing result does not require running models.

## Preserve the evidence

- **The original media is canonical.** When transcription is requested or needed, save its original-source result before enhancement or reuse the saved result; see [references/transcribe.md](references/transcribe.md). Enhancement is the final media-changing step. Resolve each revision from the original, preserving its timeline, rather than chaining renders. Enhanced-media ASR or VAD checks may differ and never replace the canonical transcript.
- **Absence is meaningful.** Never fill an omitted measurement, timestamp, or speaker with a null, zero, or inferred value. Preserve the machine JSON unchanged and put interpretation in the reply.
- **A measurement is not a preference.** Separate report checks from listening evidence. Preserve an approved result rather than chasing every numeric preference.
- **`abstained` is unresolved.** Seek evidence for a scoped correction instead of forcing larger values. Gain and EQ cannot independently control speech and music that overlap in one track.
- **Provision through `audio packages pull`.** The sole automatic exception is the small, hash-verified Silero speech-activity model on first use, including inspection or enhancement. Other models and runtimes require explicit provisioning; never hand-download weights, hand-create a runtime, or edit its pins.
- **Enhancement does not delete fillers.** Canonical means unedited original-source output, not guaranteed verbatim recognition. Filler editing is future scope; video would require synchronized audio/video cuts, not audio-only muting or deletion.
