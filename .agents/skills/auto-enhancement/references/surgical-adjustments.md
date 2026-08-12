# Inspect and Apply Surgical Adjustments

## Contents

- Authority
- Workflow
- Adjustment schema
- Defensive validation response
- Placement semantics
- Quiet intended program example
- Frequency correction example
- Verification and failure boundaries

## Authority

Use a surgical adjustment only when evidence exists beyond the automatic profile's own heuristics. Valid evidence includes explicit user listening feedback, video context, source metadata, a known event range, or an independently supplied production requirement.

Do not convert a profile miss into a universal automatic rule. A quiet external-speaker music segment may be intentional in one product demo and unwanted background in another.

## Workflow

1. Inspect the canonical original and persist the facts:

   ```bash
   audio inspect INPUT --profile PROFILE --report INSPECTION.json
   ```

2. Combine `observations` and `regions` with the external evidence. Resolve exact start and end times and decide whether the correction is full-band or frequency-scoped.
3. Create an adjustment file with only the supported schema below.
4. Resolve the complete pipeline without rendering:

   ```bash
   audio enhance INPUT --profile PROFILE \
     --adjustments ADJUSTMENTS.json --dry-run
   ```

5. Review the resolved `adjustments`, their placement, stage statuses, predicted regional measurements, and peak validation implications.
6. Render from the same original:

   ```bash
   audio enhance INPUT --profile PROFILE \
     --adjustments ADJUSTMENTS.json -o OUTPUT
   ```

7. Inspect `OUTPUT.report.json`, decode the complete output, and listen to the adjusted region plus both boundaries.

## Adjustment schema

Use exactly one top-level `adjustments` array. Each item must contain exactly `type`, `gain_db`, and `scope`.

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

Apply these constraints:

- Set `type` to `gain`.
- Keep `gain_db` between −24 and +12 dB.
- Set `time` to `all` or an object satisfying `0 <= start < end <= duration`.
- Set `frequency` to `all` or an object containing exactly `low_hz`, `high_hz`, and `shape`.
- Keep frequency bounds within `20 <= low_hz < high_hz < Nyquist`.
- Set `shape` to `band` or `notch`.
- Reject unknown keys, invalid scopes, unsafe gain, and malformed JSON rather than guessing.

## Defensive validation response

Treat the declared time and frequency scopes as untrusted input. The CLI validates them against the probed media duration and the 48 kHz processing Nyquist boundary before loading the VAD model or rendering.

For an invalid scope, require all of the following behavior:

- Exit with status 2.
- Emit one JSON error object on stderr.
- Set `type` to `AdjustmentError` and `code` to `scope_out_of_range` or `non_finite_number`.
- Identify the exact `field`, echo a JSON-safe `provided` value, and state the `allowed` range.
- Create neither output media nor the default report.
- Do not clamp values, swap start/end, reinterpret units, or silently drop the adjustment.

Example response for a six-second end time on five-second media:

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

Correct the scope from inspection or external evidence and submit a new dry run. Preserve an abstention when a valid scope cannot be established.

## Placement semantics

The CLI resolves placement from scope:

- Frequency-scoped gain runs before `voice-enhance` as a minimum-phase peaking biquad.
- Time-scoped full-band gain runs after `source-balance`.
- Full-time, full-band gain runs before `program-loudness`.

Time-scoped adjustments use the profile boundary fade inside the declared scope. Expand the declared start and end into verified surrounding silence when the audible event itself must receive full gain from its first through last sample. Confirm the resolved fade in the report.

## Quiet intended program example

If inspection does not promote quiet music to `non_speech_program`, but user or video context confirms that 42.1–55.8 seconds is intended playback, authorize a bounded full-band correction:

```json
{
  "adjustments": [
    {
      "type": "gain",
      "gain_db": 5.0,
      "scope": {
        "time": {"start": 42.0, "end": 55.9},
        "frequency": "all"
      }
    }
  ]
}
```

Use the extra boundary headroom only when inspection confirms it is silent or otherwise safe. Compare the adjusted region against speech and program targets after the full pipeline; do not assume the requested gain survives limiting unchanged.

## Frequency correction example

Represent a measured narrow hum correction without changing unrelated frequencies:

```json
{
  "adjustments": [
    {
      "type": "gain",
      "gain_db": -8.0,
      "scope": {
        "time": "all",
        "frequency": {"low_hz": 55, "high_hz": 65, "shape": "notch"}
      }
    }
  ]
}
```

Do not create a frequency correction from intent alone; require measured spectral evidence.

## Verification and failure boundaries

- Start from the canonical original, not a prior enhanced render.
- Confirm the report records every adjustment with a stable ID, exact scope, resolved placement, filter or transition, and gain.
- Recheck integrated loudness, encoded true peak, regional relationships, and timeline preservation.
- Listen for boundary pumping, boosted room noise, tonal coloration, and limiter interaction.
- Do not use gain to claim independent control of overlapping sources.
- Preserve an abstention when the requested correction lacks reliable scope, evidence, or headroom.
