# Correct a region or frequency

Use scoped gain when user feedback, a known event range, or a production requirement identifies a correction automatic treatment cannot resolve. Inspect the original to establish its bounds. A reported hum still needs a measured frequency; do not invent a band from the description alone.

Reuse the user's existing authorization for the correction. Do not generalize wanted music or unwanted noise from one recording to another, or increase gains simply to satisfy every numeric target.

## Adjustment input

The adjustment-file format is not supplied by CLI help. Pass a JSON file with exactly this structure; the numbers are examples, not measurements of the current recording:

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

- `type` is `gain`; `gain_db` is finite and within `[-24, +12]`.
- `time` is `"all"` or exactly `{"start": seconds, "end": seconds}`, with `0 <= start < end <= duration`.
- `frequency` is `"all"` or exactly `{"low_hz": 55, "high_hz": 65, "shape": "notch"}`. Bounds satisfy `20 <= low_hz < high_hz < 24000`; `shape` is `band` or `notch`.
- Keep the shown key sets; do not add labels, comments, or IDs. Multiple corrections are separate entries in the array.

For a measured narrow hum, use a negative gain and the measured frequency scope; use `time: "all"` only if the correction is warranted throughout.

## Resolve and check

Follow the [enhancement workflow](enhance-audio.md). Dry-run from the original and inspect the resolved scope, placement, fade, and predicted effect before rendering. Placement follows the scope; it is not an input choice.

Boundary fades occur inside a time scope. If an event needs full gain throughout, extend the scope only into verified surrounding silence. Loudness and peak control run afterward, so the delivered change may differ from the requested gain.

Correct invalid input from the CLI's stated field and allowed bounds. If the evidence cannot establish a valid scope, leave the correction unresolved. After rendering, check the measured effect and obtain listening feedback on the region and both boundaries. Gain/EQ affects all overlapping sources within the selected scope.
