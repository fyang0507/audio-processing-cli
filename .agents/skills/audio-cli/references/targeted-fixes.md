# Fixing one region or one frequency

The lane for *"the music at the end is too quiet"* and *"there's a hum through the whole thing"* —
a bounded gain you authorize because you have evidence the automatic profile cannot see. The file
format is not in `--help`, so it is here in full.

## Earn the authorization first

Automatic enhancement is deliberately conservative, so a miss is expected rather than a bug. What
turns a miss into a fix is evidence from outside the tool's own heuristics: the person told you what
they heard, the video shows what the audio should be doing, source metadata or a known event range
pins the time, or a production requirement states the target.

Never generalize from one file. Quiet external-speaker music is intentional in one demo and unwanted
bleed in the next, and only the person can say which. And never build a frequency correction from
intent alone — a hum you were told about but did not measure is a hum you cannot scope.

Work from the canonical original, and inspect it first so the scope you claim traces to measured
facts. Then resolve without rendering and read the `adjustments` block back: placement, fade, and
predicted regional measurements are all visible before anything is written.

## The file

Exactly one top-level `adjustments` array. Each item holds exactly `type`, `gain_db`, and `scope`;
each scope holds exactly `time` and `frequency`. Unknown keys are rejected rather than ignored, so
do not add comments, labels, or ids of your own — the report assigns each adjustment a stable id.

```json
{
  "adjustments": [
    {
      "type": "gain",
      "gain_db": 5.0,
      "scope": {
        "time": {"start": 42.1, "end": 55.8},
        "frequency": "all"
      }
    }
  ]
}
```

- `type` is `gain`. It is the only type.
- `gain_db` is a finite number in `[-24, +12]`.
- `time` is `"all"`, or an object satisfying `0 <= start < end <= duration`.
- `frequency` is `"all"`, or an object holding exactly `low_hz`, `high_hz`, and `shape`.
- Frequency bounds satisfy `20 <= low_hz < high_hz < 24000`, whatever the source sample rate.
- `shape` is `band` or `notch`.

## Placement is derived from scope, not declared

Where the gain lands in the processing order follows from what you scoped, and the report prints the
result as `placement`:

| Scope | `placement` | Meaning |
| --- | --- | --- |
| frequency-scoped | `before-voice-enhance` | tonal correction, before speech treatment sees it |
| full-band, time-scoped | `after-source-balance` | regional level change, after automatic balancing |
| full-band, full-time | `before-program-loudness` | program-wide level change, before normalization |

Time-scoped gains take the boundary fade *inside* the scope you declared. When the audible event has
to receive full gain from its first sample to its last, widen the scope into surrounding silence you
have verified is silent, then confirm the resolved fade in the report.

## Rejection is structured, and nothing is repaired for you

Scopes are treated as untrusted input and validated against the media's real duration before any
model loads or anything renders. An invalid file exits **2**, prints one JSON error, and creates
neither media nor a report. Values are never clamped, `start` and `end` are never swapped, units are
never reinterpreted, and an adjustment is never silently dropped — a wrong scope is worse than no
adjustment, so the tool refuses instead of guessing.

| `code` | Cause |
| --- | --- |
| `scope_out_of_range` | a time or frequency scope outside the media or the 20–24000 Hz bounds |
| `gain_out_of_range` | `gain_db` outside `[-24, +12]` |
| `non_finite_number` | NaN or an infinity, echoed back as `"NaN"` / `"Infinity"` |
| `invalid_number` | a string, boolean, or null where a number belongs |
| `invalid_adjustment` | unreadable file, malformed JSON, wrong key set, or a `type` other than `gain` |

The first four name the exact `field`, echo a JSON-safe `provided`, and state the `allowed` range, so
the correction is usually mechanical:

```json
{
  "error": {
    "type": "AdjustmentError",
    "code": "scope_out_of_range",
    "message": "adjustments[0].scope.time must satisfy 0 <= start < end <= 5.000",
    "field": "adjustments[0].scope.time",
    "provided": {"start": 4.0, "end": 6.0},
    "allowed": {
      "minimum_start_seconds": 0.0,
      "maximum_end_seconds": 5.0,
      "constraint": "start < end"
    }
  }
}
```

`invalid_adjustment` carries a message only — it is the shape-of-the-file error, so there is no one
field to blame. Preserve an abstention when no valid scope can be established.

## Two worked cases

**Quiet intended program.** Inspection did not promote quiet music to non-speech program audio, but
the person or the video confirms 42.1–55.8 s is intended playback. Authorize a bounded full-band
correction over a slightly wider window, using the extra headroom only where inspection shows
silence:

```json
{
  "adjustments": [
    {
      "type": "gain",
      "gain_db": 5.0,
      "scope": {"time": {"start": 42.0, "end": 55.9}, "frequency": "all"}
    }
  ]
}
```

Then compare the region against the speech and program targets in the report. Do not assume the
requested gain survives limiting unchanged — loudness and peak control run afterwards.

**A measured narrow hum.** Correct it without touching unrelated frequencies:

```json
{
  "adjustments": [
    {
      "type": "gain",
      "gain_db": -8.0,
      "scope": {"time": "all", "frequency": {"low_hz": 55, "high_hz": 65, "shape": "notch"}}
    }
  ]
}
```

## Afterwards

Confirm the report records every adjustment with its id, exact scope, resolved placement, and gain.
Recheck loudness, true peak, the regional relationships, and timeline preservation, then have someone
listen to the adjusted region *and both of its boundaries* — pumping, boosted room noise, tonal
coloration, and limiter interaction all show up at the edges rather than in the middle. Never use a
gain to claim independent control of overlapping sources.
