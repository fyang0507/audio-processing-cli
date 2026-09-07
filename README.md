# audio-processing-cli

`audio` is a local-first CLI for audio inspection, profile-driven enhancement, transcription, deterministic transcript export, and explicit model provisioning. Enhancement follows a bounded loop:

```text
original + profile
  -> measure
  -> resolve bounded operations
  -> render from the original timeline
  -> remeasure and verify
```

The CLI reports what it measured, which versioned rule matched, the exact DSP parameters it resolved, and whether the result conforms to the selected profile. It does not label audio universally “good” or “bad,” and it abstains where a mixed track cannot be changed safely.

Transcription uses an explicit stack and produces a reusable JSON result for offline export. `audio transcribe capabilities` describes a stack, `audio transcribe plan` resolves an exact request, and `audio transcribe run` recognizes original media using one of four stacks: `qwen-1.7b`, `qwen-0.6b`, `firered`, and `vibevoice`. `audio transcribe export` renders normalized results without model work. `audio doctor` reports what the machine supplies, and `audio packages` installs, verifies, and reclaims the pinned packages and runtimes. Provider ASR, alignment, and diarization downloads require an explicit `audio packages pull`. The small shared Silero VAD bootstrap used by inspection, enhancement, or transcription is the sole implicit model fetch and is described under Install.

## Install

This section covers operator setup. Audio-processing agents use the installed `audio` command and report setup blockers through its [readiness guidance](skills/audio-cli/references/readiness.md).

Requirements:

- Python 3.11+ (managed by `uv` when needed)
- FFmpeg and FFprobe on `PATH`; optional RNNoise enhancement also requires FFmpeg’s `arnndn` filter
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

Install directly from this repository without cloning it first:

```bash
uv tool install "git+https://github.com/fyang0507/audio-processing-cli.git"
```

Use `uv run` only while developing an uninstalled checkout:

```bash
uv sync --extra dev
uv run audio enhance --list-stages --profile product-demo
```

When no valid local copy is available, the first VAD use downloads the pinned 2.2 MB Silero VAD 6.2.1 ONNX model from its official repository and verifies its SHA-256 digest. It is the only model this CLI fetches implicitly; `audio packages pull silero-vad` provisions it explicitly instead. The [VAD implementation](src/audio_cli/vad.py) enforces that SHA-256 pin. Set `AUDIO_PROCESSING_MODEL_CACHE` to select the shared provisioning root, or `AUDIO_PROCESSING_VAD_MODEL` to use a pre-populated copy of that same hash-pinned model outside the cache; an arbitrary ONNX file is refused rather than run under false 6.2.1 provenance. See the [migration guide](docs/vad-model-migration.md) for replacing the removed `--vad-model` option with a command-local environment assignment. No PyTorch runtime is required.

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

Use `audio report summary demo-enhanced.mp4.report.json` to locate recorded component outcomes, unresolved scopes, and measurement blocks without processing the media again. It prints a concise JSON projection with pointers into the original report and preserves abstentions; `--navigation` groups the recorded phases and scopes. Compare that saved render report with a fresh inspection of the delivered media:

```bash
audio inspect demo-enhanced.mp4 --report demo-delivered.inspect.json
audio report compare demo-enhanced.mp4.report.json demo-delivered.inspect.json --navigation
```

Comparison uses recorded source identity and interval evidence. Fixed before/after regions and fresh detection scopes remain separate; differing speech references or non-speech intervals do not establish changed audio quality. See [report navigation](docs/cli-feedback.md) and [saved-report comparison](docs/report-comparison.md).

For an enhanced speech-listening copy (keep canonical transcription on the original):

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

If `program-loudness` is skipped, integrated loudness is not a pass/fail target, but encoded true peak must remain at or below −0.1 dBTP. The command fails instead of silently limiting a signal above that ceiling.

Integrated loudness is gated in 400 ms blocks by EBU R128, so an input shorter than one block has none to normalize however loud it is. The refusal says so, quotes the true peak that proves the signal is not silent, and names the flag that skips the stage; a short clip is never reported as silence.

## Profiles and stages

Every stage ends as `applied`, `no_op`, `skipped`, `abstained`, or `failed`. The fixed processing order is:

1. `channel-balance` — corrects a level mismatch only when the channels are correlated enough to treat it as unintended.
2. `environment-denoise` — evaluates speech-scoped high-pass/de-hum cleanup and bounded broadband denoising. The default `stationary` method requires a guarded noise reference; explicit `rnnoise` selection uses a provisioned model as a spectral guide. Component outcomes report application, no-op, or abstention separately from the parent stage.
3. `voice-enhance` — treats VAD regions as seeds, expands them to silence-anchored acoustic voice boundaries, then applies bounded presence correction, leveling, and compression at full strength throughout the resolved treatment region. Equal-power transitions finish before the guarded voice onset and begin after the guarded voice offset.
4. `source-balance` — balances non-overlapping machine-audio regions against treated speech for `product-demo`; it is disabled for `transcription`.
5. `program-loudness` — uses EBU R128 measurement to resolve fixed gain, then iterates an oversampled true-peak limiter without undoing earlier region balance.

See [Enhancement capabilities and algorithms](docs/enhancement.md) for each stage's mechanism, adjustments, report semantics, and recorded evidence, including when noise cleanup abstains.

`transcription@5` targets −23 LUFS / −3 dBTP. `product-demo@5` targets −16 LUFS / −1.5 dBTP and aims to keep detected machine audio between 4 and 2 dB below the treated speech reference within its correction bounds. The [profile definitions](src/audio_cli/profiles.py) own the thresholds, and every threshold and bound is emitted in the report’s `profile` object. Speech treatment uses silence-anchored boundaries; broadband processing retains the source timeline and is delivered only inside those treatment regions.

The default `--denoiser stationary` needs no additional denoising model. To select the optional RNNoise guide explicitly:

```bash
audio packages pull rnnoise-voice
audio enhance demo.mp4 --profile product-demo --denoiser rnnoise -o demo-rnnoise.mp4
```

The [model-guided workflow](src/audio_cli/pipeline/denoise.py) uses continuous FFmpeg `arnndn` output to guide bounded spectral suppression; it reports that clean-reference noise reduction and speech preservation were not measured. It never silently falls back to the stationary method. See [RNNoise provisioning](docs/packages/rnnoise.md) and [denoising evidence](docs/rnnoise-denoising.md) for requirements and scope limits.

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

The [output publication checks](src/audio_cli/pipeline/publication.py) verify that:

- the encoded output decoded-audio duration remains within 50 ms (inclusive) of the original decoded-audio duration;
- integrated loudness reaches the selected profile within 0.6 LU when `program-loudness` is enabled and not skipped;
- encoded true peak stays at or below the profile target, or −0.1 dBTP when `program-loudness` is skipped (normalization reserves 1 dB of codec headroom before lossy encoding);
- every eligible stage has an explicit terminal status;
- the before/after speech and machine-region measurements reuse the same stable region IDs;
- the source hash, profile version, resolved operations, output hash, and CLI/FFmpeg versions are recorded.

`timeline_verification` declares the duration-only scope and tolerance, and explicitly abstains on content alignment and A/V sync. Dry runs omit `timeline_preserved`. Read [duration and alignment evidence](docs/timing-evidence.md) for probed versus decoded duration bases, original/output stream timing, codec padding limits, and the content-bearing CLI regression fixtures.

Render success verifies the mandatory output gates; it does not imply every component achieved its preferred target. Read `unresolved` for abstained denoise components, bounded regional corrections, and loudness-range limits left unresolved to preserve balance. Use `measurements.after.program_actual` for the encoded render's measured LUFS, LRA, and true peak. The legacy `program` object retains raw FFmpeg diagnostics; its `output_*` fields describe a hypothetical additional normalization, not the saved render.

`region_basis` identifies the source intervals used for before/after comparison. A fresh `inspect` on an enhanced file runs detection again, so region ids and classifications may differ. Keep any requested canonical transcription on the original media, and enhance last. Human listening judges preference; overlapping sources require separate tracks or a future separation capability. Enhancement preserves fillers. Editorial and filler-removal decisions are outside the CLI scope; issues #1 and #39 are closed as `NOT_PLANNED`.

## Model packages

`audio packages pull` is the only operation that fetches provider models or prepares their runtimes. A missing provider package is a refusal carrying a `fix`, never a background download. The shared, hash-pinned Silero VAD bootstrap described under Install remains the sole implicit model fetch. [ENVIRONMENTS.md](docs/ENVIRONMENTS.md) describes the host `core` environment and four managed runtime environments, their pins, and provisioning layout. RNNoise is an enhancement-only package in `core`; it belongs to no transcription stack.

```bash
audio doctor                            # tools, toolchains, memory, disk, and provisioning state
audio packages list                     # what is provisioned, and what it occupies
audio packages path                     # resolved managed root and package locations
audio packages pull --stack qwen-1.7b   # provision every package that stack can use
audio packages verify                   # re-check what is provisioned; exit 3 if a check fails
audio packages purge --dry-run          # what a teardown would reclaim, reclaiming nothing
```

Six behaviours to know before dispatching on the payloads:

- `pull` accepts package ids **or** `--stack`, never both, and it refuses `--want` at exit 2 rather than accepting a capability filter it does not implement. Use `transcribe plan` to find the exact package set, then pull those ids, or pull the complete stack.
- A package the registry already calls `ready` is reported under `skipped` and not re-materialized; it contributes nothing to `pulled_known_bytes`. `pull --repair PACKAGE` forces the work anyway, re-downloading a Hub snapshot and re-cloning a checkout rather than trusting what is on disk.
- `--stack` tolerates a package its toolchain blocks: the rest of the stack provisions, the exit stays 0, and the blocked package appears in `warnings` with `blocking: true`. Naming that package on the command line is an instruction rather than a guess, so there an absent toolchain is exit 3.
- `verify` publishes one verdict per environment — `ok`, `drifted`, `blocked`, or `absent`. Only `drifted` is repairable here, with `verify --repair`. `blocked` means a provisioning or repair tool is off `PATH` and no ready built runtime can substitute for it; a directly launchable FluidAudio product therefore remains `ok` without Swift. The blocked verdict exits 0 and says so rather than naming a fix this CLI cannot perform.
- Verification reports the identity it actually checks: Silero publishes `digest: "ok"` for its SHA-256 pin; RNNoise publishes `revision`, `git_blob_sha1`, and `bytes` for its Git blob identity. Hub packages publish pinned revision information after live cache binding and allowlist/size checks. The [artifact verifier](src/audio_cli/packages/artifacts.py) and [package verifier](src/audio_cli/packages/package_verification.py) own these distinct verdicts.
- `remove` resolves every name before deleting anything. Local targets and environment ownership come from the installed manifest, not mutable registry paths. A Hub revision is eligible only when this root recorded downloading it, the current manifest still pins it for that package, and it was not pre-existing; `purge --dry-run` reports both the eligible and retained sets before a caller promises a total.

## Transcription

The stack is an explicit quality choice. Inspect its capabilities, resolve the packages and stages for one request, provision what the plan names, then run:

```bash
audio transcribe stacks
audio transcribe capabilities --input meeting.m4a --stack qwen-1.7b
audio transcribe plan --input meeting.m4a --stack qwen-1.7b \
  --want diarization,word_timestamps --language Cantonese
audio packages pull --stack qwen-1.7b
audio transcribe run --input meeting.m4a --stack qwen-1.7b \
  --want diarization,word_timestamps --language Cantonese \
  -o meeting.timed.json
```

Runs emit reusable normalized JSON to stdout and also save it when `--output` is supplied; a successful stdout-only run creates no automatic file. The explicit `--format json` spelling remains accepted for scripts. Migrate old `run --format txt/md` commands by saving JSON first, then using `transcribe export` with a distinct readable destination. Removed formats fail with an argument error before media probing or model execution.

Existing transcript and partial-result paths are preserved unless `--force` is explicit; no flag can make the output overwrite an input transcript, its derived partial path, or canonical source media.

Publication protects canonical media and saved inputs across path aliases and directory changes. See [output refusals](docs/transcribe-contract/40-refusals.md) for replacement and destination rules, and [environments](docs/ENVIRONMENTS.md) for managed-runtime integrity and offline execution.

Omitting `--want` requests only the stack’s declared floors. `--language` is an optional closed Qwen hint and does not reach the forced aligner. Missing packages fail at exit 3 with an explicit `audio packages pull` fix before decode or model load. The [orchestrator](src/audio_cli/transcribe/orchestrator/__init__.py) dispatches all four stacks; [capability declarations](src/audio_cli/transcribe/stacks.json) and live `capabilities` output describe their requested features. Keep the original media as input, including when continuing a partial result with `--range`.

During transcription, host stage and elapsed-time progress goes to stderr, while stdout remains JSON. Raw backend stdout/stderr is retained at announced paths, including on failure. Use `--log-dir PATH` for durable storage; the default is temporary. Preserve those logs with the canonical result; warnings alone do not establish recognition quality. [Alignment diagnostics](docs/alignment-diagnostics.md) connects rejected or explicitly corrected bounds to their raw stage evidence.

Use `transcribe plan --compact` to omit the generated sample while keeping all decisions and provisioning guidance. With `transcribe run --output PATH --receipt`, stdout becomes a concise JSON receipt and the saved canonical JSON stays unchanged. An incomplete run still exits 4 with its refusal on stderr and a receipt naming the actual partial file and coverage.

Before exporting, inspect coverage, requested-capability outcomes and abstentions in the saved JSON: processing completion does not guarantee that requested timing or speakers were delivered. Retain the canonical result for subsequent exports.

Export saved results without running models (`audio export` remains a compatibility alias):

```bash
audio transcribe export --input meeting.timed.json --format txt -o meeting.txt
audio transcribe export --input meeting.timed.json --format srt -o meeting.srt
audio transcribe export --input meeting.timed.json --format md --timestamps -o meeting.md
audio transcribe export --input meeting.timed.json --format md --provenance -o meeting.with-source.md
```

Repeat `--input` in source-timeline order to merge compatible continuations. `--provenance` adds saved source/stack/timing-basis headers and each input's coverage to Markdown or text. `--timestamps` applies only to Markdown and text and requires real segment or word times. Subtitle export uses real word streams; unsupported timing is refused rather than invented. See the [export contract](docs/transcribe-contract/30-export.md), [cue construction](src/audio_cli/export/cues.py), and [timing validation](src/audio_cli/export/timing.py) for event and abstention handling.

## Test

Use [agent acceptance](docs/agent-acceptance.md) as the final verdict for user-facing feature work, after the implementation checks below. It defines fresh operator contexts, the regression media matrix, evidence requirements and explicit PASS/FAIL/BLOCKED criteria.

```bash
uv run --extra dev pytest
```

Install the pinned Ruff hooks once per checkout, then run the same deterministic lint and format gate over the repository before submitting a change:

```bash
uv run --extra dev pre-commit install
uv run --extra dev pre-commit run --all-files
```

One agent skill is developed and shipped with the CLI under `skills/audio-cli/` in the source distribution. [`audio-cli`](skills/audio-cli/SKILL.md) is the onboarding surface for an agent asked to fix or measure someone's audio: it routes to enhancement, targeted corrections, original-source transcription and export, saved-report interpretation and comparison, provisioning, readiness, and command failures. Its references explain judgment, result semantics, and evidence limits; live `--help` owns command syntax, flags, and defaults. This layout keeps CLI user guidance outside development-agent auto-discovery; building the package does not install the skill into another workspace.

## Repository layout

See the [documentation index](docs/README.md) for contracts, implementation notes, and research reports, and the [test layout](docs/testing.md) for source ownership, fixtures, and local media.
