---
name: model-packages
description: Provision, verify, and remove the model packages and runtime environments the repository's `audio` CLI needs for transcription work. Use before running any stack, when a command reports packages_not_provisioned or package_integrity_failed, when asked what a request will download or which environments it spans, when reclaiming disk, or when diagnosing a machine with `audio doctor`.
---

# Model Packages

Weights and runtimes are provisioned **explicitly**. `audio packages pull` is the only thing that
downloads weights, creates an environment, applies a patch, or builds a Swift product. A
transcription request never triggers any of them — it fails with exit 3 and prints the `pull`
line that would fix it. Never work around that by fetching weights yourself.

Use the installed entry point. `uv run audio ...` is for an uninstalled development checkout
only; see [references/setup.md](../auto-enhancement/references/setup.md) if `audio` is missing.

## Start by looking

```bash
audio doctor                  # tool, ffmpeg/swift/uv/git, memory, disk, root, every state
audio packages list           # what is provisioned, its size, its declared license
audio packages path           # resolved root and per-package locations
```

`doctor` is the first command on an unfamiliar machine, and the only one that tells you whether
a missing tool will block a request. All three are read-only and safe to run at any time.

## Translate the request into package ids

A user names a stack and what they want from it. Provision by id when you know them, or let a
stack select everything it can use:

```bash
audio packages pull vibevoice-asr-7b qwen3-forcedaligner   # exactly these
audio packages pull --stack firered                        # everything firered can use
```

| Package id | Supplies | Environment | Download |
| --- | --- | --- | --- |
| `qwen3-asr-1.7b-8bit` | ASR for the `qwen-1.7b` stack | `mlx` | 2.30 GiB |
| `qwen3-asr-0.6b-8bit` | ASR for the `qwen-0.6b` stack | `mlx` | 0.94 GiB |
| `qwen3-forcedaligner` | word timestamps on any stack without native word timing | `mlx` | 1.19 GiB |
| `vibevoice-asr-7b` | ASR **and** native speaker structure for `vibevoice` | `torch-vibevoice` | 16.16 GiB |
| `firered-asr2s` | the whole `firered` stack: VAD, language id, ASR, punctuation | `torch-firered` | 8.93 GiB |
| `fluidaudio` | speaker diarization for stacks without it natively | `swift` | unsized build |
| `speaker-diarization-coreml` | the diarizer's model; comes with `fluidaudio` | `swift` | 0.12 GiB |
| `silero-vad` | speech-activity regions | `core` | 2.3 MB, auto-fetches |

Two things to read off that table rather than guess:

- **`firered-asr2s` is one package supplying four roles.** A whole-pipeline FireRed request —
  VAD, language id, ASR, punctuation — is that one id. Do not look for separate punctuation or
  language-id packages; there are none, and punctuation always runs because punctuated text is
  a floor rather than an option.
- **`qwen3-forcedaligner` is the add-on for word timing**, and it is needed by `vibevoice` and
  both Qwen stacks. It lives in `mlx` even when the ASR does not, so pulling it alongside
  `vibevoice-asr-7b` creates two environments. That is expected, not a mistake.

`silero-vad` is the only package that fetches itself on first use, because it is small and
hash-pinned. Everything else fails closed.

`--stack` accepts `qwen-1.7b`, `qwen-0.6b`, `vibevoice`, `firered`. It over-provisions
deliberately: it pulls every package that stack *can* use, including diarization and the
aligner. Prefer explicit ids when the user's request is narrow and the difference is gigabytes.
`--want` is reserved and currently requires `--stack`; it does not yet narrow the set.

## Provision

```bash
audio packages pull vibevoice-asr-7b qwen3-forcedaligner
```

Progress goes to stderr; stdout is a JSON receipt naming each package, the revision pulled,
the environments created, and any warnings. Then confirm:

```bash
audio packages verify
```

`verify` re-checks artifact digests, that each environment still matches its locked dependency
set, that patches are still applied, and the pinned private API the Qwen timing figures depend
on. It exits **3** if anything failed and names a `fix` for each. `--repair` re-syncs a drifted
environment from its lock:

```bash
audio packages verify --repair
```

Expect `license_unreviewed` in `pull` warnings. It is non-blocking and it means the license a
model card declares has not been reviewed by a human. Report it; do not treat it as an error,
and do not describe a declared license as a cleared one.

## Reclaim

```bash
audio packages remove vibevoice-asr-7b   # one package
audio packages purge --dry-run           # what a full teardown would free
audio packages purge                     # everything this tool provisioned
```

An environment is deleted exactly when its last package goes; the report names what was kept
and why. Both commands delete only the Hub revisions this tool recorded as materialized here,
leave other revisions of the same repository alone, and never touch user media or transcript
output. `purge` works from the registry, so it finds everything even in a session that never
ran `pull`.

## Exit codes

| Code | Meaning | What to do |
| --- | --- | --- |
| 0 | done | continue |
| 2 | the request was wrong — unknown package or stack, a package that was never provisioned | read the `allowed` list in the error and correct the name |
| 3 | provisioning is incomplete or broken — `packages_not_provisioned`, `package_integrity_failed`, a missing tool, a failed check | run the `fix` the payload names, verbatim |

Every error is JSON on stderr with a `code`, a `detail`, and usually a `fix`. Prefer running the
`fix` over composing your own command.

## Authority boundaries

- Never download weights, create a virtual environment, or install a package by hand to work
  around exit 3. The whole point of the failure is that provisioning is explicit and pinned.
- Never edit a lock file under `src/audio_cli/environments/locks/` to make an install succeed.
  A lock reproduces the configuration the recorded measurements came from.
- `swift` is required only by `fluidaudio` and `speaker-diarization-coreml`. If it is absent,
  `doctor` says so and those two fail with `requires_tool: ["swift"]`; every other package still
  provisions. Report the blocked capability rather than substituting a different diarizer.
- A crashed or interrupted `pull` leaves the package not-ready, which still reads as absent.
  Re-run the same `pull` to finish it; do not clean up by hand.
- Report the numbers the commands print. Do not add up per-package sizes yourself when a
  payload already gives a total, and say when something is unsized rather than omitting it.
- Purge before uninstalling the tool, or the provisioning root outlives the only thing that
  knows how to describe it.

## Detailed references

- Read [references/environments.md](references/environments.md) when you need to explain *why*
  a request spans the environments it does, or when a user asks whether two stacks could share
  one runtime. It is background, not a command guide.
