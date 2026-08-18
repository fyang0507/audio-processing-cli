---
name: model-packages
description: Provision, verify, and remove the model packages and runtime environments the repository's `audio` CLI needs for transcription work. Use before running any stack, when a command reports packages_not_provisioned or package_integrity_failed, when asked what a request will download or which environments it spans, when reclaiming disk, or when diagnosing a machine with `audio doctor`.
---

# Model Packages

Weights and runtimes are provisioned **explicitly**. `audio packages pull` is the only thing that
downloads weights, creates an environment, applies a patch, or builds a Swift product. A
transcription request never triggers any of them — it fails with exit 3 and prints the `pull`
line that would fix it. Never work around that by fetching weights yourself.

Invoke it as `audio ...` when installed. In a development checkout it is not installed — check
with `command -v audio`, and fall back to `uv run audio ...` from the repository root. If
neither works, install with `uv tool install .` from the checkout, or
`uv tool install "git+https://github.com/fyang0507/audio-processing-cli.git"` without one.

**Everything provisioned lives under one root**, printed by `audio packages path` and
`audio doctor`. It defaults to a per-platform cache directory and is overridden by the
`AUDIO_PROCESSING_MODEL_CACHE` environment variable — set that to put the environments and the
registry on another disk. Weights are the exception and the thing most likely to confuse you:
they live in the **shared Hugging Face cache**, not under the root, so a root holding a couple
of gigabytes after a 17 GiB pull is normal. Trust the per-package `location` from
`audio packages path`, not the root's size.

## Start by looking

```bash
audio doctor                  # tool, ffmpeg/swift/uv/git, memory, disk, root, every state
audio packages list           # what is provisioned, its size, its declared license
audio packages path           # resolved root and per-package locations
```

`doctor` is the first command on an unfamiliar machine, and the only one that tells you whether
a missing tool will block a request. All three are read-only and safe to run at any time.

Two fields in that output are worth knowing in advance. `doctor` marks an environment
`provisional: true` when a future change might move its packages elsewhere — it is a note about
the roadmap, changes nothing about any command, and is not a warning. And `path`'s `models`
entry covers hash-pinned single-file artifacts only (currently just `silero-vad`); it is often
absent, which is not evidence that a pull failed.

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

The two VAD entries are not a choice you make at provisioning time. FireRed brings its own, so
a FireRed request needs `firered-asr2s` and nothing else; `silero-vad` is the add-on for stacks
that have no native VAD, and it arrives by itself when one is used. Pulling it alongside
`firered-asr2s` is harmless but pointless.

`--stack` accepts `qwen-1.7b`, `qwen-0.6b`, `vibevoice`, `firered`. It over-provisions
deliberately: it pulls every package that stack *can* use, including diarization and the
aligner. Prefer explicit ids when the user's request is narrow and the difference is gigabytes.
`--want` is reserved and currently requires `--stack`; it does not yet narrow the set.

## Provision

```bash
audio packages pull vibevoice-asr-7b qwen3-forcedaligner
```

Progress goes to stderr; stdout is a JSON receipt naming each package, the environments
created, the bytes this pull added as `pulled_known_bytes`, and any warnings. Each package
reports either a `revision` or — for `firered-asr2s`, which spans four repositories — a
`revisions` list. A package already present in the shared cache reports
`hub_revisions_pre_existing` instead of being downloaded again. Then confirm:

```bash
audio packages verify
```

`verify` re-checks artifact digests, that each environment still matches its locked dependency
set, that patches are still applied, and the pinned private API the Qwen timing figures depend
on. It exits **3** if anything failed and names a `fix` for each.

`--repair` exists on both commands, and they fix different things:

```bash
audio packages verify --repair            # re-sync a drifted environment from its lock
audio packages pull --repair PACKAGE      # re-materialize a package the registry calls ready
```

Use `pull --repair` for `package_integrity_failed` — a digest that changed, or weights that
disappeared from the shared cache. Use `verify --repair` when the failure names an environment.

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
and why. Neither command touches user media or transcript output, and `purge` works from the
registry, so it finds everything even in a session that never ran `pull`.

**Read the teardown report before believing what it freed.** Weights are in the shared Hugging
Face cache, so teardown reaches outside the root, and it draws one line: a revision **this root
downloaded** is deleted, while a revision that was **already cached** when this root pulled it
is kept and listed under `hub_revisions_retained`. That is why a `purge` can legitimately report
freeing far less than the packages' sizes — someone else's copy was already there. Run
`purge --dry-run` first; it prints `would_remove.hub_revisions` against
`would_keep.hub_revisions` so you can see the split before anything goes.

One case that line does not cover: if two roots each pulled a revision, the one that downloaded
it owns it, and purging that root removes weights the other still expects. The other root's
`audio packages verify` will then report `package_integrity_failed`, and
`audio packages pull --repair PACKAGE` restores it. If you are provisioning a throwaway root
beside a populated cache, prefer `remove` for the packages you added over a blanket `purge`.

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
  Re-run the same `pull` to finish it; do not clean up by hand. "By hand" means deleting
  environment directories or cache blobs yourself — `remove` and `purge` are the supported
  verbs, and they are the only things that keep the registry honest about what is gone.
- Report the numbers the commands print, and use the right one: `pulled_known_bytes` covers
  only the packages in that `pull`, `total_known_bytes` from `list` is cumulative, and
  `reclaimed_bytes` from a teardown is measured after the fact and includes environment
  directories, so it will not match either. Say when something is unsized rather than omitting
  it — `fluidaudio` always is, because it is a build product.
- Purge before uninstalling the tool, or the provisioning root outlives the only thing that
  knows how to describe it.

## Detailed references

- Read [references/environments.md](references/environments.md) when you need to explain *why*
  a request spans the environments it does, or when a user asks whether two stacks could share
  one runtime. It is background, not a command guide.
- Installing or repairing the `audio` command itself is covered by the sibling
  auto-enhancement skill's [setup reference](../auto-enhancement/references/setup.md). Read it
  only if `audio` is missing and `uv run audio` is not an option.
