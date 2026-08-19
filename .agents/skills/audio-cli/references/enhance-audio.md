# Diagnosing and fixing audio

The lane for *"what's wrong with this?"*, *"clean this up"*, and *"make it transcription-ready"*.

## The loop

Measure first, resolve without rendering, render once from the original, then verify from the
report — in that order, because each step is only trustworthy if the one before it held.

1. **Measure.** `inspect` reports facts and, given a profile, what that profile concludes from them.
   It writes nothing to the media, so run it freely. This alone answers "what's wrong with this
   audio?" — read the observations back in prose and stop there if no fix was requested.
2. **Resolve.** A dry run evaluates every eligible stage, resolves the exact operations, and predicts
   the resulting measurements without producing media. Read it before rendering: this is where you
   discover that a stage abstained, or that a correction you expected is not going to happen.
3. **Render.** From the canonical original, never from a previous render.
4. **Verify.** Read the report that lands beside the output as `OUTPUT.report.json`, and quote your
   numbers from it rather than from the dry run's predictions.

## Choosing the profile

The profile is a declaration of what the render is *for*, and it fixes targets, bounds, and which
stages are eligible.

- **`transcription`** — a speech proxy for machine listening. Intelligibility over polish.
- **`product-demo`** — a single speaker plus non-overlapping machine audio or music, delivered to
  human ears.

Pick from the destination of the audio, not from how bad it sounds. A consequence worth expecting:
`source-balance` is a `product-demo` stage, so a `transcription` run reports it `no_op` with reason
`disabled_by_profile`. That is the profile working, not a stage failing.

## What the five stages do

They always run in this order, and the report names them, so you need the vocabulary to explain a
result:

| Stage | Corrects |
| --- | --- |
| `channel-balance` | correlated channel-level mismatch, nothing else |
| `environment-denoise` | DC offset, rumble, hum, and broadband noise, scoped to speech |
| `voice-enhance` | bounded presence, leveling, and compression inside detected speech |
| `source-balance` | detected non-overlapping program audio, against treated speech |
| `program-loudness` | EBU R128 loudness and true-peak limiting, without undoing the above |

Keep every eligible stage in play unless the person explicitly names ones to skip. The bounds, not
your selection, are what keep the result conservative.

Each stage reports one of `applied`, `no_op`, `skipped`, `abstained`, or `failed`, and an individual
region may report `abstained_overlap`. Only `failed` is a failure. `abstained` is the one to slow
down on: the stage had evidence that something was wrong and no correction it could make safely, so
the answer is more evidence (see [targeted-fixes.md](targeted-fixes.md)), never a larger number.

## What automatic enhancement will not catch

Say these out loud when they apply, because each one is a place where a satisfied-looking report and
an unhappy listener coexist:

- **Quiet intended music or machine audio may sit below the balancing threshold.** It then gets the
  global loudness treatment and no regional correction. This is the most common gap, and the
  targeted-fix lane exists for it.
- **"Not speech" never means "noise."** Speech detection answers one question. A region it did not
  claim may be a laptop speaker, a demo sound, or a room the person wanted kept.
- **Non-speech program audio is selected by how loud it is, not by what it is.** The detector
  promotes activity that stands above the noise floor; it does not know music from a fan.
- **One miss is not a reason to loosen the profile.** Route that evidence into a scoped fix so the
  correction stays with the media that needed it.
- **Overlapping speech and music cannot be pulled apart** with gain or EQ. Separate stems, or
  abstain.

## Reading the report

| Surface | Holds |
| --- | --- |
| `source`, `engine`, `profile` | hashes, versions, thresholds, bounds, processing order |
| `observations` | the measured facts: channels, environment, speech, program audio, loudness |
| `regions` | detected speech and non-speech-program scopes, with the IDs measurements reuse |
| `rule_evaluations` | what the profile concluded from those observations |
| `stages` | per stage: status, reason, exact operations, resolved transitions |
| `measurements.before` / `.predicted` / `.after` | verification, program-wide and per region |
| `adjustments` | any scoped fixes you authorized, with where they landed |
| `resolved_operations_sha256` | the processing plan, for reproducibility |

Before reporting a render as good, confirm: `rendered` is `true`; `timeline_preserved` is `true` with
`duration_delta_ms` inside 50 ms; `final_peak_validation.status` is `pass`; integrated loudness is
within 0.6 LU of the profile target; every eligible stage has a status and a reason; before/after
regional measurements reuse the same region IDs.

Two things to know about that list. In a dry run, peak validation reads `predicted_pass` — a
prediction, and the only thing available before an encode exists. And the checks are enforced, not
merely reported: the render encodes to a temporary file, verifies it, and moves it into place only
if it passes, so a failed verification leaves no partial output, no report, and no clobbered
previous render. A zero exit with a written report means the list above held; read the values
anyway, because which number you quote is your responsibility.

## When it refuses

Every failure here exits **2** with one JSON object on stderr. Unlike provisioning errors, most
carry only `type` and `message` — no `code`, no `fix` — so read the message rather than waiting for
a machine-readable remedy. The refusals worth recognizing:

- **input is a marked render.** Renders are stamped, and enhancing one is refused, because the
  report chain describes the original. Go back to the source file. There is a flag to override it,
  for the case the person asks for by name.
- **output would overwrite the input**, or an output path is missing for a real render. Both protect
  the canonical original.
- **the output or the report already exists.** Nothing is replaced unless the person asked for it,
  and the message names the flag that does. `inspect` checks its report destination before it
  decodes anything, so this refusal costs a retry rather than the analysis.
- **the audio is too short to have a loudness measurement.** Integrated loudness is gated in 400 ms
  blocks, so a clip shorter than one has none however loud it is. The refusal quotes the true peak
  that proves the signal is not silent and names the stage to skip. Relay it that way: reporting a
  short clip as silent is the wrong answer this message exists to prevent.
- **an invalid adjustment file**, which is its own lane — see [targeted-fixes.md](targeted-fixes.md).
