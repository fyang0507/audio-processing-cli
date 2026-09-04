
## Packages and environments

Everything provisioned lives under one root: `AUDIO_PROCESSING_MODEL_CACHE`
when set, otherwise the per-platform cache directory the Silero backend already
uses. Nothing is provisioned beside a source checkout the way the `model_tests/`
experiments were.

```text
<root>/
  registry.json       # provisioned packages and environments, revisions, digests, byte sizes
  models/             # pinned single-file artifacts, e.g. silero-vad-6.2.1.onnx
  envs/
    mlx/              # uv environment from the repository-tracked lock
    torch-firered/    # uv environment, plus the FireRed source checkout
    torch-vibevoice/  # uv environment, plus the VibeVoice checkout and its applied patch
    swift/            # FluidAudio built product and its Core ML package
```

**Four** provisioned environments, and the grouping is not a choice. Packages are
grouped into as few environments as their dependencies permit, and what they permit
was resolved rather than argued: every non-empty grouping was put to `uv`, ten of
fifteen conflict, and all ten conflict on `transformers`
(`model_tests/benchmark/results/2026-08-17-environment-partition.json`). Four
packages carry three mutually exclusive ranges — `>=5.5.0,<5.13.0` via
`mlx-audio`, `==5.1.0` for FireRed, `==4.57.6` via `qwen-asr`, and `<5.0.0` for
VibeVoice — so the minimal partition is three Python environments plus `swift`, and
the probe confirms that partition is **unique**. Re-run it when an upstream pin
moves; it re-derives the layout instead of inviting anyone to re-argue it.

| Environment | Packages | Constraint |
| --- | --- | --- |
| core | `silero-vad` | The `audio` tool's own environment; numpy, scipy, onnxruntime. Nothing to provision. |
| `mlx` | `qwen3-asr-1.7b-8bit`, `qwen3-asr-0.6b-8bit`, `qwen3-forcedaligner` | `mlx` plus `mlx-audio==0.4.5`, whose `transformers>=5.5.0` nothing else accepts; deliberately torch-free, and now torch-free in fact. |
| `torch-firered` | `firered-asr2s` | `transformers==5.1.0`, pinned exactly, compatible with nothing else here. It also cannot leave PyTorch: `mlx-audio` implements FireRed's AED but no punctuator, and punctuated text is a floor. |
| `torch-vibevoice` | `vibevoice-asr-7b` | `transformers<5.0.0`. **Provisional** — `mlx-audio` runs VibeVoice-ASR with no added dependencies at 12.4 GB against 20.8 GB, but produces a different transcript, so the move waits on re-measurement rather than on a lock. |
| `swift` | `fluidaudio`, `speaker-diarization-coreml` | Swift toolchain build; no Python, so no lock. |

The forced aligner sits in `mlx`, not beside VibeVoice where the recorded pipeline
ran it. `mlx-audio`'s implementation reproduced the recorded torch alignment token
for token — identical text on all 246 aligned tokens, no punctuation-invariant
violations, bound agreement median 0 s with a P95 of 80 ms
(`model_tests/benchmark/results/2026-08-17-mlx-collapse-probes.json`). The
consequence is the point: `word_timestamps` on a Qwen stack no longer pulls in a
PyTorch environment, so the fast long-form path spans two environments rather than
three. The mechanism is in [ENVIRONMENTS.md](ENVIRONMENTS.md).

`fluidaudio` provisions the built Swift product plus one Core ML model package,
`FluidInference/speaker-diarization-coreml`. FluidAudio also ships a Core ML VAD
that its own `VadManager` resolves; the comparison probe exercised it, but this tool
has never provisioned it as a named package, so it is not one. If `--vad fluidaudio`
ships, it becomes a package and gets its own registry entry.

Licenses are a registry field, reported by `audio packages list` and not by a plan —
a plan resolves a pipeline, not a redistribution question. Every package now carries
the license its model card states at the pinned revision: Qwen ASR and the aligner
Apache-2.0, FireRed's four checkpoints Apache-2.0, the combined VibeVoice package
`mixed: mit + apache-2.0` because it carries a pinned Qwen tokenizer subset, and Silero MIT
read from the tagged `LICENSE` rather than a card. Two were read rather than reported — the
FluidAudio SDK is Apache-2.0 and `speaker-diarization-coreml` is CC-BY-4.0
(`model_tests/benchmark/DIARIZATION.md`).

So the field is two fields, because a card is not a review: `license_declared` is what
the card says, and `license_reviewed` stays false until someone reads the terms. A
declared license is evidence that one exists, not a redistribution clearance, and
collapsing the two would turn a scraped string into a cleared one. `unreviewed` as a
single value is retired: it could not distinguish "nobody looked" from "the card says
Apache-2.0 and nobody has checked what that obliges".

Rules:

- Environment dependency sets are locked in this repository. `pull` materializes
  a lock; it never resolves "latest". This is what keeps the `mlx-audio` private
  batched API at the one version the source-hash guard expects.
- Hub weights stay in the Hugging Face cache. The registry records which
  revisions this tool materialized rather than duplicating a snapshot, while the live cache
  index binds each repository and pinned revision to its recorded snapshot path.
- Applying the VibeVoice patch and building the FluidAudio product happen in
  `pull` and nowhere else. Neither is ever triggered by a transcription request.
- `remove` is reference-counted from installed-manifest package identities: an environment
  survives while another provisioned package still needs it. Local deletion targets also come
  from that manifest, never from mutable registry paths. A Hub revision is deletion-eligible only
  when the receipt records that this root downloaded it, the current manifest still pins it for
  that package, and it was not recorded as pre-existing; everything else is retained because the
  Hugging Face cache may be shared with other tools.
- `purge` reads `registry.json`, not shell history, so a session that never ran
  `pull` can still enumerate its recorded state and safely reclaim manifest-owned artifacts.
  It reports reclaimable package bytes before removing anything.
- `verify` re-checks artifact digests, Hub cache identity, the `mlx-audio` private-API source
  hash, exact source-checkout HEAD and tracked names, manifest-owned post-patch hashes plus their
  receipt copy, ordinary and ignored untracked files, and that the one contained FluidAudio
  product from its exact managed checkout runs.

Lifecycle: `audio packages list | pull | verify | remove | purge | path`. `pull`
accepts package ids **or** `--stack`, never both, and it does not accept `--want`:
narrowing a stack to the capabilities a plan actually uses belongs to the planner
(#12), so until that exists a stack provisions every package it can use and the flag
is refused rather than accepted and ignored. `path` prints the resolved root and
per-package locations so a session with no provisioning history can still locate
everything.

### States, and which command publishes which

Two enumerations, and they are not the same enumeration. Conflating them is what let an
unusable environment read as a working one.

**Registry state**, recorded in `registry.json` and republished verbatim by `doctor`,
`packages list`, and `packages path`. A package is `pulling` then `ready`; an environment is
`creating` then `ready`; either is **absent** by having no entry at all. Intent is written
before any bytes move, so a crashed `pull` is nameable, and anything not `ready` counts as
absent to a run and as reclaimable to `purge`.

**Verify verdict**, published only by `packages verify`, one per provisioned environment:

| Verdict | Means |
| --- | --- |
| `ok` | every check that applies to this environment passed. |
| `drifted` | its installed set differs from its lock. Repairable: `verify --repair` re-syncs it. |
| `blocked` | a provisioning or repair tool is not on `PATH` and no ready built runtime can substitute for it. Not repairable by this tool — installing a toolchain is not something `audio` does. |
| `absent` | its environment registry entry is missing or not `ready`. |

A `ready` package cannot promote that verdict. When one depends on an `absent` environment,
`verify` emits the typed failure `environment_not_ready` with `packages` and `fix`, and performs
no freeze, checkout, interpreter, or built-product probe beneath that environment root.

`blocked` exists because the two enumerations answer different questions and only the second
one is a claim about usability. A stack pull provisions the packages that need no toolchain and
reports the blocked one, so `swift` can legitimately be `ready` in the registry while holding
only model weights and no executable. Once FluidAudio has been built, runtime and `verify` launch
that product directly: a live product makes the environment usable even if Swift later leaves
`PATH`; an absent or nonlaunching product does not. `doctor` says whether the current filesystem
has that built-runtime substitute when publishing `blocked_by_missing_tool`, and a bare `ok` from
`verify` requires the direct launch to succeed. The word is the one this tool already uses for
the condition, in `doctor`'s `blocked_by_missing_tool` and in a pull warning's `blocking: true`;
it is not a new concept, only a missing spelling.

A registry-state field is **not** restated as `blocked`. It reports what the registry holds,
which is a different and still-useful fact, and three commands publish it — rewriting one of
them would leave the other two disagreeing while nothing published the registry's own field.
Which tool is missing is static per environment (`requires_tool` is manifest data), so `verify`
keeps a flat enum a caller can switch on rather than an object restating it.
