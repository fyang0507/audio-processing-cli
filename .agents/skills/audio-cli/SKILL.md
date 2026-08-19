---
name: audio-cli
description: Do audio work on local media with the `audio` CLI — diagnose what is wrong with a recording, clean up and level speech in a video or podcast, render an ASR-ready proxy before transcribing, apply an evidence-backed fix to one region or one frequency, and provision the local models transcription needs. Use this whenever someone wants audio measured, cleaned up, made clearer or more consistent, or prepared for transcription, even if they never mention this CLI, a profile, or a model — every result comes back as JSON you can act on, and the failure modes here are the kind that produce confident wrong answers.
---

# Audio work with the `audio` CLI

Someone hands you media and a complaint: *the audio in this demo is rough*, *make this usable for
transcription*, *why does this sound bad?* This CLI is how you answer — it measures, decides,
renders, and re-measures deterministically, and prints JSON at every step so you can act on facts
rather than impressions. Your part is judgment: pick the right lane, read the output honestly, and
say what you actually know.

`audio --help` and `audio <command> --help` are the command surface. Every subcommand, flag, and
default is there, so read it rather than guessing, and expect nothing here to repeat it. This skill
carries what help text cannot: which lane a request belongs in, what the output means, and where
the honest limits are.

## Start

```bash
command -v audio
```

If that finds nothing, read [references/install.md](references/install.md) — except inside a
checkout of this repository, where `uv run audio ...` works from the root without installing.

## Route by what the person wants

| The request | Your lane | Read before running it |
| --- | --- | --- |
| "what's wrong with this audio?" | measure, then explain in prose | [references/enhance-audio.md](references/enhance-audio.md) |
| "fix / clean up / level this recording or video" | inspect, resolve, render, verify | [references/enhance-audio.md](references/enhance-audio.md) |
| "get this ready for transcription" | the same loop, transcription profile | [references/enhance-audio.md](references/enhance-audio.md) |
| "this part is too quiet", "there's a hum at 60 Hz" | a scoped gain, authorized by evidence | [references/targeted-fixes.md](references/targeted-fixes.md) |
| "transcribe this", "what will this download?" | provision models explicitly | [references/model-packages.md](references/model-packages.md) |
| "free up disk", or a command reports something missing | check state, then pull, repair, or reclaim | [references/model-packages.md](references/model-packages.md) |
| the command is missing or will not run | install and verify | [references/install.md](references/install.md) |

Read the lane's file before running its commands. Each is short, and each exists because that lane
has a way of producing a plausible wrong answer that `--help` cannot warn you about.

## Three invariants

- **The original media is canonical.** Nothing modifies a source file, and a render is never the
  input to another render — a report describes the *original's* timeline, so a second pass would be
  measuring a measurement.
- **Absence is meaningful.** A field the tool did not supply stays absent rather than becoming a
  null, a zero, or a default that reads like a measurement. Never fill one in when relaying results.
- **stdout JSON belongs to the caller.** Pass it through unchanged and put interpretation in your
  reply, not in the payload.

## The honest limits

These are the four claims that get made wrongly. Knowing them is most of the skill:

- **A measurement is not a preference.** The numbers that chose the processing cannot also prove it
  sounds better. When someone wants to know whether it sounds good, ask them to listen, or say
  plainly that you verified conformance and not taste.
- **`abstained` means unresolved, not permission to force.** A stage that abstained found no
  correction it could make safely. The next move is evidence, not a bigger number.
- **Overlapping sources in one mixed track cannot be separated.** Gain and EQ move speech and music
  together when they overlap in time. Preserve the source or abstain; never imply independent
  control you do not have.
- **Models arrive only when explicitly pulled, and there is no transcription command yet.**
  Provisioning ships ahead of it, so never describe `audio transcribe` as runnable, and never
  hand-download weights or hand-build an environment to work around a missing package.

When a command fails, it prints one JSON error object on stderr. If that object carries a `fix`,
run that string verbatim — it is the tool naming its own remedy, and improvising instead is how one
problem becomes two.
