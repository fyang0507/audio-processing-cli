# audio-processing-cli

`audio` is a local-first utility layer for agents that need to inspect and improve audio without turning an editing agent into a DAW operator. Its first implemented surface is a narrow, profile-driven enhancement loop:

```text
original + profile
  -> measure
  -> resolve bounded operations
  -> render from the original timeline
  -> remeasure and verify
```

The CLI reports what it measured, which versioned rule matched, the exact DSP parameters it resolved, and whether the result conforms to the selected profile. It does not label audio universally “good” or “bad,” and it abstains where a mixed track cannot be changed safely.

This implements [Issue #4 — Profile-driven automatic audio enhancement](https://github.com/fyang0507/audio-processing-cli/issues/4) within the product boundary established by [Issue #1](https://github.com/fyang0507/audio-processing-cli/issues/1).

## Install

Requirements:

- FFmpeg and FFprobe on `PATH`
- [`uv`](https://docs.astral.sh/uv/) as the installer and tool manager

```bash
brew install ffmpeg
uv tool install .
```

This performs a non-editable installation into an isolated user-level environment and exposes `audio` through `~/.local/bin`. The repository does not need to be the current directory when using the installed command, and `uv` is not part of normal invocation:

```bash
audio --help
audio enhance --profile product-demo --list-stages
```

If `~/.local/bin` is not already on `PATH`, run `uv tool update-shell` once and open a new shell. Rebuild and reinstall the current checkout after an unversioned local change with `uv tool install --force --reinstall .`; remove it with `uv tool uninstall audio-processing-cli`.

After the repository is published, the same isolated installation can be performed without cloning it first:

```bash
uv tool install "git+https://github.com/fyang0507/audio-processing-cli.git"
```

Use `uv run` only while developing an uninstalled checkout:

```bash
uv sync --extra dev
uv run audio enhance --list-stages --profile product-demo
```

The first inspection downloads the pinned 2.2 MB Silero VAD 6.2.1 ONNX model from its official repository and verifies its SHA-256 digest. Set `AUDIO_PROCESSING_VAD_MODEL` or pass `--vad-model` to use a pre-populated local model. No PyTorch runtime is required.

## Use

Inspect facts without selecting a quality policy:

```bash
audio inspect demo.mp4
```

Evaluate the same observations against a versioned profile:

```bash
audio inspect demo.mp4 --profile product-demo
```

Resolve every eligible standard stage without rendering:

```bash
audio enhance demo.mp4 --profile product-demo --dry-run
```

Render and verify a delivery-oriented video while copying its video stream:

```bash
audio enhance demo.mp4 \
  --profile product-demo \
  -o demo-enhanced.mp4
```

The durable report is written beside the output as `demo-enhanced.mp4.report.json`. JSON is also emitted on stdout for agents.

For a speech-oriented transcription proxy:

```bash
audio enhance meeting.m4a \
  --profile transcription \
  -o meeting-enhanced.wav
```

Skip exactly the named stages when user intent requires it:

```bash
audio enhance demo.mp4 \
  --profile product-demo \
  --skip=channel-balance,program-loudness \
  -o demo-with-skips.mp4
```

If `program-loudness` is skipped, true-peak validation still runs and the command fails instead of silently limiting a clipping signal.

## Profiles and stages

Every stage ends as `applied`, `no_op`, `skipped`, `abstained`, or `failed`. The fixed processing order is:

1. `channel-balance` — corrects a level mismatch only when the channels are correlated enough to treat it as unintended.
2. `environment-denoise` — evaluates speech-scoped high-pass/de-hum cleanup and explicitly abstains from unreliable broadband denoising.
3. `voice-enhance` — treats VAD regions as seeds, expands them to silence-anchored acoustic voice boundaries, then applies bounded presence correction, leveling, and compression at full strength throughout the resolved treatment region. Equal-power transitions finish before the guarded voice onset and begin after the guarded voice offset.
4. `source-balance` — balances non-overlapping machine-audio regions against treated speech for `product-demo`; it is disabled for `transcription`.
5. `program-loudness` — uses EBU R128 measurement to resolve fixed gain, then iterates an oversampled true-peak limiter without undoing earlier region balance.

`transcription@4` targets −23 LUFS / −3 dBTP. `product-demo@4` targets −16 LUFS / −1.5 dBTP and keeps detected machine audio between 4 and 2 dB below the treated speech reference. Version 3 grows reliable VAD seeds to neighboring acoustic activity, reserves silent guard time, and places speech-treatment fades outside that guard. Version 4 changes no threshold: it makes `vad_min_silence_ms` the only thing that decides where a speech region breaks, where a second hard-coded merge had previously required 540 ms of silence to split a region the profile said should split at 300. Renders therefore differ from version 3 wherever a pause falls between those figures — the local 27.8 s fixture moves from 6 speech regions to 10. Every threshold and bound is emitted in the report’s `profile` object.

## Constrained adjustments

Explicit user or agent evidence may be represented as a bounded gain adjustment:

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
    },
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

```bash
audio enhance demo.mp4 \
  --profile product-demo \
  --adjustments adjustments.json \
  -o demo-rerendered.mp4
```

Gain is limited to −24…+12 dB. Regional operations receive equal-power boundary fades; frequency operations resolve to minimum-phase biquads. Unknown fields and out-of-range scopes fail validation.

Invalid time or frequency scopes fail before rendering with exit status 2 and a structured JSON error on stderr containing `code`, `field`, `provided`, and `allowed`. The CLI does not clamp or reorder invalid boundaries and creates no output or report for the rejected request.

## Verification contract

A successful render verifies:

- the output duration remains within 50 ms of the original audio timeline;
- integrated loudness reaches the selected profile within 0.6 LU;
- encoded true peak stays at or below the declared target (the render reserves 1 dB of codec headroom before lossy encoding);
- every eligible stage has an explicit terminal status;
- the before/after speech and machine-region measurements reuse the same stable region IDs;
- the source hash, profile version, resolved operations, output hash, and runtime versions are recorded.

The report demonstrates deterministic profile conformance. Perceptual preference still requires human listening evidence; overlapping sources still require separate tracks or a future separation capability.

## Test

```bash
uv run --extra dev pytest
```

One agent skill travels with the CLI in the source distribution.
[`audio-cli`](.agents/skills/audio-cli/SKILL.md) is the onboarding surface for an agent asked to fix
or measure someone's audio: it routes by request — diagnose and enhance, apply a targeted fix,
provision transcription models, install the command — and holds only what `--help` cannot say, which
is the judgment, the report semantics, and the limits worth admitting to a user.
