---
name: audio-cli
description: Measure, enhance, and transcribe local audio or video with the audio CLI; export transcripts and subtitles, apply scoped audio corrections, and manage required model packages. Use for recording diagnosis, speech cleanup, transcription, and subtitle requests.
---

# Audio work with the `audio` CLI

Use the CLI for audio operations and saved JSON for evidence. This skill supplies the decisions
and interpretation; the selected invocation's `--help` supplies commands, flags, and defaults.

## Select the invocation first

- **Using or testing a checkout:** select `uv run audio` from that checkout's root even when
  `command -v audio` succeeds. From elsewhere use `uv run --project` with the actual absolute
  checkout path, followed by `audio`. A PATH installation may expose older commands.
- **Using an installed tool:** resolve it with `command -v audio` and read that executable's help.
  If it is missing or cannot start, read [references/install.md](references/install.md).

Keep the selected launcher and working directory for every command, including remedies printed as
`audio ...`. To verify checkout identity, run this from its root and confirm the module path belongs
to that checkout:

```bash
uv run python -c "import audio_cli; print(audio_cli.__file__)"
uv run audio --help
```

A mismatched PATH tool is not a reason to reinstall it. Below, `audio` means the selected invocation.

## Route by the request

| The request | Read before running it |
| --- | --- |
| Diagnose, enhance, or prepare an audio/video recording | [references/enhance-audio.md](references/enhance-audio.md) |
| Correct a particular region or measured frequency | [references/targeted-fixes.md](references/targeted-fixes.md) |
| Transcribe original media or export a saved result | [references/transcribe.md](references/transcribe.md) |
| Prepare packages, check readiness, or reclaim managed disk | [references/model-packages.md](references/model-packages.md) |
| A command failed | [references/failures.md](references/failures.md) |

Read only the relevant references. Diagnosis or enhancement alone does not require transcription;
export of an existing result does not require running models.

## Preserve the evidence

- **The original media is canonical.** When transcription is requested or needed, save its
  original-source result before enhancement or reuse the saved result; see
  [references/transcribe.md](references/transcribe.md). Enhancement is the final media-changing
  step. Resolve each revision from the original, preserving its timeline, rather than chaining
  renders. Enhanced-media ASR or VAD checks may differ and never replace the canonical transcript.
- **Absence is meaningful.** Never fill an omitted measurement, timestamp, or speaker with a null,
  zero, or inferred value. Preserve the machine JSON unchanged and put interpretation in the reply.
- **A measurement is not a preference.** Separate report checks from listening evidence. Preserve
  an approved result rather than chasing every numeric preference.
- **`abstained` is unresolved.** Seek evidence for a scoped correction instead of forcing larger
  values. Gain and EQ cannot independently control speech and music that overlap in one track.
- **Provision through `audio packages pull`.** The sole automatic exception is the small,
  hash-verified Silero speech-activity model on first use, including inspection or enhancement.
  Other models and runtimes require explicit provisioning; never hand-download weights,
  hand-create a runtime, or edit its pins.
- **Enhancement does not delete fillers.** Canonical means unedited original-source output, not
  guaranteed verbatim recognition. Filler editing is future scope; video would require synchronized
  audio/video cuts, not audio-only muting or deletion.
