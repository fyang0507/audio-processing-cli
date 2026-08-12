---
name: auto-enhancement
description: Inspect, automatically enhance, and surgically tune local audio or video with the repository's profile-driven `audio` CLI. Use for transcription proxies, single-speaker product demos, loudness or source balancing, deterministic dry runs, enhancement reports, user-reported audio problems, or evidence-backed time and frequency gain adjustments.
---

# Auto Enhancement

Use the CLI as the deterministic measurement, decision, rendering, and verification surface. Keep the original media canonical.

## Core workflow and commands

1. Verify the standalone entry point:

   ```bash
   command -v audio
   audio --help
   ```

   If either command fails, read [references/setup.md](references/setup.md), complete setup, and verify the installed entry point before continuing. Use `uv run audio ...` only while developing an uninstalled checkout.

2. Choose `transcription` for an ASR-oriented speech proxy or `product-demo` for a single-speaker delivery render with non-overlapping machine audio or music.

3. Inspect the original before changing it. Persist the inspection when another step or agent needs the facts:

   ```bash
   audio inspect INPUT --profile PROFILE
   audio inspect INPUT --profile PROFILE --report INSPECTION.json
   ```

4. Query the installed profile version and stage order when compatibility matters, then resolve every eligible stage without rendering:

   ```bash
   audio enhance --profile PROFILE --list-stages
   audio enhance INPUT --profile PROFILE --dry-run
   ```

5. Review every stage status. Treat `abstained` as unresolved rather than permission to force a change.

6. Render from the canonical original. Skip only stages the user explicitly rejects:

   ```bash
   audio enhance INPUT --profile PROFILE -o OUTPUT
   audio enhance INPUT --profile PROFILE \
     --skip=channel-balance,program-loudness -o OUTPUT
   ```

   Use `--force` only to replace a known output and report. Use `--vad-model PATH` only to supply the pinned Silero ONNX model from an explicit local path.

7. Verify the durable `OUTPUT.report.json`, timeline, encoded true peak, and measured before/after program and regional values.

8. Listen before claiming perceptual preference.

## Authority boundaries

- Keep all standard stages eligible unless the user explicitly requests `--skip=stage-a,stage-b`.
- Let auto-enhance remain bounded. A missed quiet but intentional sound is not evidence that the automatic detector should universally promote quiet background audio.
- Use `inspect` plus user feedback, transcript or video context, source metadata, or other independent evidence to authorize a surgical adjustment.
- Do not infer that non-speech audio is unwanted environment. It may be intended program audio that falls below the automatic source-balance threshold.
- Do not claim independent control over overlapping sources in one mixed track.
- Do not enhance an enhanced render unless the user explicitly requests it.
- Preserve stdout JSON for agent consumption and communicate interpretation separately.

## Detailed references

- Read [references/setup.md](references/setup.md) only when the `audio` command is unavailable or broken.
- Read [references/auto-enhance-workflow.md](references/auto-enhance-workflow.md) when deciding whether automatic enhancement is effective for a particular edit, or when explaining its stage logic, detection methodology, report semantics, and coverage limits. It is an internal-process reference, not a command guide.
- Read [references/surgical-adjustments.md](references/surgical-adjustments.md) when inspection or external intent supports a targeted correction, especially for quiet intended program audio, a known time range, or a frequency-specific defect.
