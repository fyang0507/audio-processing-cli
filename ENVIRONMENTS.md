# Provisioned environments

How the tool creates and manages the runtimes its backends need, and why the layout is what it
is. Issue [#11](https://github.com/fyang0507/audio-processing-cli/issues/11).

[VOCABULARY.md](VOCABULARY.md) defines `package` and `environment` and states the rules this
document implements. Where the two disagree, VOCABULARY is the naming contract and this is the
mechanism. The capability half of the stack table belongs to the planner
([#12](https://github.com/fyang0507/audio-processing-cli/issues/12)); the package and
environment half is `src/audio_cli/environments/manifest.json`, described below.

## The layout

| Environment | Interpreter | Packages | Why it is separate |
| --- | --- | --- | --- |
| `core` | the tool's own | `silero-vad` | Not provisioned. numpy, scipy, onnxruntime. The only place anything auto-fetches. |
| `mlx` | 3.13.9 | `qwen3-asr-1.7b-8bit`, `qwen3-asr-0.6b-8bit`, `qwen3-forcedaligner` | `mlx-audio==0.4.5` requires `transformers>=5.5.0,<5.13.0`. Torch-free by intent, and now torch-free in fact. |
| `torch-firered` | 3.12.12 | `firered-asr2s` | FireRed pins `transformers==5.1.0` exactly. It cannot join anything, and it cannot leave PyTorch — see the punctuator finding below. |
| `torch-vibevoice` | 3.12.12 | `vibevoice-asr-7b` | VibeVoice requires `transformers>=4.51.3,<5.0.0`. **Provisional**: see the VibeVoice finding. |
| `swift` | none | `fluidaudio`, `speaker-diarization-coreml` | A Swift build product plus one Core ML package. No interpreter, so no lock. |

Four provisioned environments, where VOCABULARY declared three and marked the reason unverified.
The count is the same as the split it anticipated, but the membership is not: the forced
aligner moved to `mlx`, which is what matters, because it takes PyTorch out of the fast
long-form path entirely.

| Request | Environments spanned | Known download |
| --- | --- | --- |
| `qwen-1.7b`, floors only | `mlx` | 2.30 GiB |
| `qwen-1.7b` + `diarization` + `word_timestamps` | `mlx`, `swift` | 3.61 GiB + one unsized build |
| `vibevoice` + `word_timestamps` | `mlx`, `torch-vibevoice` | 17.35 GiB |
| `firered` + `lid` | `torch-firered` | 8.93 GiB |

## How the layout was derived

Not by judgement. VOCABULARY says packages are grouped "into as few environments as their
dependencies permit", which is a claim a resolver can check, so
`model_tests/benchmark/run_env_partition_probe.py` checks it: it resolves **every** non-empty
subset of the dependency-contributing packages and reports the smallest grouping in which
every group resolves. The result is
`model_tests/benchmark/results/2026-08-17-environment-partition.json`.

Fifteen groupings tested, ten conflict, and **all ten conflict on `transformers`** — not one
is about torch, numpy, or a platform wheel. Four packages carry three mutually exclusive
ranges:

| Package | Requires `transformers` | Via |
| --- | --- | --- |
| Qwen ASR checkpoints | `>=5.5.0,<5.13.0` | `mlx-audio==0.4.5` |
| `firered-asr2s` | `==5.1.0` | its own `pyproject.toml` |
| forced aligner (torch path) | `==4.57.6` | `qwen-asr==0.0.6` |
| `vibevoice-asr-7b` | `>=4.51.3,<5.0.0` | its own `pyproject.toml` |

Only the last two intersect, so the minimal partition is three Python environments, and the
probe confirms it is **unique** — there is no equally small alternative grouping, so the layout
carries no undocumented choice. Re-run the probe when any upstream pin moves; it re-derives the
partition rather than asking anyone to re-argue it.

Two traps that probe exists to avoid, both live:

- **FireRed ships two contradictory dependency files.** `requirements.txt` pins
  `torch==2.1.0+cu118` and `transformers==4.51.3` against a CUDA index — impossible on Apple
  Silicon and disagreeing with the `pyproject.toml` the working venv was built from. The
  as-built environment is the artifact; an upstream requirement file is not.
- **FireRed's LID stage has an undeclared dependency.** `fireredlid/data/feat.py:7` imports
  `kaldi_native_fbank`, which `pyproject.toml` never lists. Provision strictly from upstream
  metadata and `--lid on` fails at import on an otherwise correct environment.

## What `mlx-audio` changed, and what it did not

`mlx-audio==0.4.5` — already pinned, already required — ships MLX implementations of all three
PyTorch-side components: `qwen3_forced_aligner` (774 lines), `vibevoice_asr` (955), and
`fireredasr2` (665). That makes "how few environments are possible" a different question from
"how few do the measured implementations permit", so all three were probed. Results:
`model_tests/benchmark/results/2026-08-17-mlx-collapse-probes.json`.

**The forced aligner is equivalent, and moved.** On the recorded case — same wav, same segment
list, same language rule, same offset arithmetic — the MLX 8-bit aligner produced **identical
token text on all 246 aligned tokens**, across all 17 word-bearing segments, leaving both
non-speech segments wordless, with no punctuation-invariant violations. Bounds agree at a
median of 0 s, P95 80 ms, max 1.6 s, with the tail concentrated in long English filler-heavy
segments. Neither path is scored against labels, so the tail is a difference and not an error —
and it is the honest cost of the move. What the move buys: `word_timestamps` on a Qwen stack no
longer pulls in a PyTorch environment, so the fast long-form path spans two environments
instead of three.

**VibeVoice is not equivalent, and did not move.** On the recorded CantoMap fixture, verified
by sha256 before loading, MLX produced 47 segments where torch produced 49 — with full
coverage on both sides, so different segmentation rather than a truncated decode — shifted some
orthography to traditional forms (`大樹` for `大树`), disagreed on at least one lexical item
(`男人車站` for `蓝印车站`), and labelled one non-speech interval `[Human Sounds]` where torch
said `[Silence]`. 8-bit and bf16 MLX produce **identical text**, so this is the implementation
and not quantization.

The memory case is nevertheless real, and it is the one the spec documents worry about:

| Configuration | Peak | Generate (149.9 s audio) |
| --- | --- | --- |
| torch bf16 + patch | 20.84 GB MPS | 79.7 s |
| MLX bf16 | 20.74 GB | 275.1 s |
| MLX 8-bit | 12.43 GB | 56.3 s |

MLX and PyTorch counters have different scopes and must not be differenced across columns, but
within its own counter MLX 8-bit is the only attractive MLX configuration, and 12.4 GB is the
first VibeVoice figure that does not immediately rule out a 16 GiB machine. Adopting it
replaces every recorded VibeVoice figure, so it is a re-measurement decision, not a lock
change. `torch-vibevoice` exists until that decision; when it is made, the environment and its
lock are deleted rather than edited.

**FireRed cannot move at all.** `mlx-audio`'s `fireredasr2` is the AED only — Conformer
encoder, transformer decoder, beam search — and `mlx-audio` ships no punctuation restoration
model anywhere; every `punc` hit in the package is TTS text normalization. `mlx-community`
publishes `FireRedASR2-AED-mlx` and nothing for FireRedPunc, FireRedLID, or FireRedVAD.
Punctuated text is a **floor**, so a FireRed stack without FireRedPunc is not a conforming
backend, and moving only the AED would split one stack across two environments — worse than
not moving it. Third-party ONNX and MLX conversions of the auxiliary models exist and none has
been run here; an ONNX FireRedPunc would fit in `core`, which already carries onnxruntime.
That is an option, not a plan.

Recorded for the same reason: `mlx-audio` also ships VAD models (`silero_vad`, `fsmn`,
`smart_turn`, `sortformer`) and LID models (`ecapa_tdnn`, `wav2vec2`). `sortformer` is a
diarizer, so even `swift` has an MLX candidate. FluidAudio was chosen on measured evidence in
`model_tests/benchmark/DIARIZATION.md`; displacing it means re-running that comparison, and
none of it is done.

## Creating an environment

`uv venv --python <pinned>` then `uv pip sync <lock>`. Nothing else — no resolution at
provisioning time, no "latest", no build step for a Python environment.

Validated end to end while writing this: building `mlx` from `locks/mlx.txt` produced
mlx 0.32.0, mlx-audio 0.4.5, mlx-lm 0.31.3, transformers 5.12.1, huggingface-hub 1.27.0,
sentencepiece 0.2.2, scipy 1.18.0, tokenizers 0.22.2, numpy 2.5.2, Python 3.13.9, and no
torch — matching the as-built environment on every package checked. The
`mlx-audio` private-API source hash over that fresh install is
`c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250`, **equal** to the pinned
guard, so an environment built from the lock is byte-identical on the decode path the recorded
figures came from. Both probes in this document then ran in it.

Why `uv pip sync` rather than `uv sync` over a project, or extras on this repo's own
`pyproject.toml`:

- The four environments need four interpreters and three mutually exclusive dependency sets.
  Extras cannot express that; they share one resolution and one interpreter.
- A project-per-environment needs a `pyproject.toml` written into the cache root, so the tool
  would author build metadata at runtime to install other people's packages.
- `uv pip sync` is convergent, not additive: it makes the environment *equal* to the lock,
  removing anything extra. `verify --repair` is then the same operation as `pull`, which is one
  code path rather than two.
- `--generate-hashes` costs about two seconds per lock and makes every wheel digest-checked by
  the installer. Provisioning integrity is the resolver's, not ours.

### Locks

```text
src/audio_cli/environments/
  manifest.json                  # environments, packages, sources, sizes, licenses
  requirements/<env>.in          # inputs, with a citation per pin
  locks/<env>.txt                # uv pip compile --generate-hashes output
  patches/<name>.patch           # applied by pull, verified by verify
```

They live inside the package, not at the repository root, because `uv tool install` users have
no checkout to read from — the locks have to ship in the wheel.

Each `.in` file cites, per pin, the file or installed distribution it came from. Where upstream
left a dependency unpinned, the lock pins it to the **as-built** version rather than to
whatever is newest: left alone, the resolver picked numpy 2.5.2 and librosa 1.0.0 for
`torch-vibevoice` where 2.4.6 and 0.11.0 are what ran, and a lock that does not reproduce the
measured environment is not evidence of anything.

Regenerate with:

```bash
uv pip compile src/audio_cli/environments/requirements/<env>.in \
  -o src/audio_cli/environments/locks/<env>.txt \
  --python-version <major.minor> --generate-hashes
```

Adding a package: write its `.in` fragment, run the partition probe, and let it say which
environment the package joins or whether it needs a new one. That is the whole procedure, and
it is why the probe reads dependency sets rather than a hand-maintained grouping.

## The registry

`<root>/registry.json`, where `<root>` is `AUDIO_PROCESSING_MODEL_CACHE` when set and
otherwise the per-platform cache directory `src/audio_cli/vad.py` already resolves.

```jsonc
{
  "schema_version": 1,
  "tool_version": "0.2.0",
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "environments": {
    "mlx": {
      "state": "ready",              // creating | ready
      "path": "envs/mlx",
      "python": "3.13.9",
      "lock_sha256": "199236e5...",  // the lock this was synced from
      "created_utc": "2026-08-17T21:04:11Z"
    }
  },
  "packages": {
    "qwen3-asr-1.7b-8bit": {
      "state": "ready",              // pulling | ready
      "environment": "mlx",
      "kind": "weights",
      "source": {"type": "huggingface", "repo": "...", "revision": "a8379a2e..."},
      "materialized": {"hub_revisions": ["a8379a2e..."], "bytes": 2467859030},
      "license_declared": "apache-2.0",
      "license_reviewed": false,
      "pulled_utc": "2026-08-17T21:06:02Z"
    }
  }
}
```

Written by atomic rename through a `.tmp` sibling, the same way `vad.py` writes a downloaded
model. That alone is not enough, so:

**A crashed `pull` cannot read as provisioned.** An entry is written with `state` set to
`pulling` or `creating` *before* any bytes move, and flipped to `ready` only after its digest
or its lock check passes. Anything not `ready` counts as absent to `run`'s exit-3 check, and
counts as reclaimable to `purge`. Recording the intent first is what makes a half-finished
download discoverable at all — VOCABULARY requires `purge` to find everything from the
registry alone, and a registry that only gains entries on success leaves orphaned bytes that
nothing can name.

**Reference counts are derived, never stored.** `remove <package>` deletes that package's
artifacts, then removes its environment only if no other non-absent package targets it. A
stored count is a second copy of a fact the table already holds, and it would eventually
disagree with it.

**`purge` reads the registry, reports reclaimable bytes, and touches no media or output.**

## `verify`

Four checks, all cheap, none loading weights:

1. **Artifact digests** for every `weights` package, against what `pull` recorded.
2. **Environment equality with its lock** — `uv pip freeze` against the locked set. A drifted
   environment is a repairable state, not a fatal one; `--repair` re-syncs.
3. **The `mlx-audio` private-API guard** — the source hash over
   `mlx_audio/stt/models/qwen3_asr/qwen3_asr.py`, plus the signature of
   `Qwen3ASRModel._generate_chunks_batched`. Both are readable by importing the class, so
   verification needs no checkpoint: confirmed, the signature check passes on a fresh
   environment without loading the 2.3 GiB model.
4. **Patches applied, and the Swift product runs.** A patch is verified by the post-patch
   digest of each file it touches, recorded at `pull` time.

Each check has a failure that must be reachable, not merely described: reverting the patch,
deleting a wheel, or bumping `mlx-audio` each has to make exactly one of these fail.

## Running a stage in another environment

The recorded evidence used **strictly sequential fresh subprocesses** — that is what
`run_interview_pipeline.py` measured and what the per-stage memory figures assume. So:

- One subprocess per stage, spawned as `<root>/envs/<env>/bin/python <stage script> <request>
  <result>`, where the stage script is a file inside the installed wheel, passed by absolute
  path. `audio_cli` is *not* installed into provisioned environments: it would drag
  onnxruntime and a conflicting numpy into each one.
- Request in, result out, both as JSON files; progress on stderr; exit code as the signal.
- Residency is enforced by process exit rather than by discipline. "No two model stages
  resident at once" is a consequence of the transport, not a rule someone must remember.

This also makes the adapter-normalization floor structural. A model-specific object cannot
cross a process boundary, so the stage script must serialize normalized output — the floor is
satisfied by construction instead of by review. A persistent worker would give that up and
would be outside what was measured; if one is ever needed for load time, it needs its own
evidence.

`swift` has no interpreter, so its stage invokes the built product directly. That asymmetry is
contained in one place: the transport chooses an executable per environment, and every
environment's stage speaks the same JSON.

## What is still open

- **`audio packages` and `audio doctor` are not implemented.** The data and the mechanism they
  need exist and are validated; the commands are not written.
- **The VibeVoice re-measurement decision**, above. It is the only thing standing between four
  provisioned environments and three.
- **`fluidaudio` remains unsized.** It is a build product; `pull` records its size once, and
  it is the only entry left in `unsized_packages` after this pass. Every other package now has
  a real byte count read from the Hub at its pinned revision — the spec documents' figures were
  illustrative, and several were wrong: `microsoft/VibeVoice-ASR` is 16.16 GiB rather than the
  quoted 17.0, and `speaker-diarization-coreml` is 129 MB rather than 84.
- **Licenses are declared, not reviewed.** Every package now carries the license its card
  states at the pinned revision — Qwen and FireRed apache-2.0, VibeVoice mit,
  `speaker-diarization-coreml` cc-by-4.0, FluidAudio apache-2.0, Silero mit read from the
  tagged LICENSE. `license_reviewed` stays false except where the terms were actually read.
  A declared license is evidence that one exists, not a redistribution clearance.
- **The FireRed auxiliary conversions** (ONNX punctuator, MLX LID, CoreML VAD) are unevaluated.
  If the punctuator works in `core`, `torch-firered` becomes questionable too.
