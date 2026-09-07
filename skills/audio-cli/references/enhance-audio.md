# Diagnose and enhance a recording

## Choose the treatment

Choose the profile by destination: `product-demo` for speech and non-overlapping program audio intended for people; `transcription` for a speech proxy intended for machine listening. A proxy never replaces the original-source transcript.

Automatic treatment covers channel balance, environmental cleanup, voice EQ and dynamics, and program loudness/peak control. `product-demo` also balances detected non-speech program audio against speech. Discover eligible stages with the selected profile's `--list-stages`; retain them unless the request calls for a skip.

Environmental cleanup includes high-pass/hum filtering and a selectable broadband denoiser. Choose the broadband method before processing:

| Method | When to choose it | Prerequisite or limit |
| --- | --- | --- |
| `stationary` | Reference-based cleanup of a steady background; the default when no method is specified | Needs usable noise-only reference intervals; can abstain when that evidence is unavailable or unsuitable |
| `rnnoise` | Model-guided speech cleanup when a stationary reference is unsuitable, or the user requests it | Explicitly provision `rnnoise-voice` through [model packages](model-packages.md); can alter wanted sound within speech |

Neither method separates overlapping sources or guarantees intelligibility. A filter applying does not establish that broadband denoising ran. An abstention calls for a decision about the unmet request, not an automatic method switch.

For a known quiet interval or measured narrow hum, use [targeted fixes](targeted-fixes.md). Automatic detection cannot decide whether quiet music or machine audio is wanted; use the user's intent and scoped evidence. Gain/EQ cannot independently control sources that overlap in one track.

## Inspect → resolve → render → verify

1. Inspect the original with the chosen profile. For diagnosis alone, explain the findings and stop.
2. Dry-run the selected treatment and any adjustments. Check what will apply, abstain, or remain outside target before rendering.
3. Render to a distinct destination from the original. Start every revision from that same source.
4. Read the delivered-file report. Distinguish actual post-encode measurements from predictions; include unresolved treatment limits in the reply. Do not present publication success as proof of preferred sound.

Use [saved reports](saved-reports.md) when interpreting detailed evidence or comparing results. Seek listening feedback for perceptual judgments, especially for wanted background audio. A numeric miss alone does not justify stronger processing or another pass.
