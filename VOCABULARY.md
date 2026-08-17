# Transcription vocabulary

Agreed 2026-08-16. This is the naming contract for the `transcribe` surface and
the schema beneath it. It consolidates terms already used in
[Issue #1](https://github.com/fyang0507/audio-processing-cli/issues/1),
[HANDOFF.md](HANDOFF.md), and
[model_tests/DECISION_REPORT.md](model_tests/DECISION_REPORT.md), and resolves
the collisions between them. `transcribe` is the most demanding command, so it
defines the vocabulary; other commands follow it. The command sequences that
exercise it are in [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md).

Keep HANDOFF's three kinds of statement separate everywhere below: a declared
**capability** is an interface, a **measured** result is tied to a recorded
fixture and configuration, and an **unresolved** item still needs evidence. A
declared interface is not evidence of output quality.

## Terms

**stack** — the core ASR family together with the auxiliary processors it ships
with. `firered` is a four-model stack (VAD, LID, AED, punctuation); `qwen-1.7b`
and `qwen-0.6b` are single-model stacks; `vibevoice` is one model that also
carries native speaker structure. Choosing the stack is the caller's first and
required decision, because it fixes transcript quality, language and dialect
behavior, and which capabilities arrive natively. None of that is derivable from
a requirement list.

**role** — a functional slot in a pipeline. The enumeration is fixed and small:
`decode`, `vad`, `diarizer`, `reconciler`, `asr`, `aligner`, `punctuator`,
`lid`. This is what Issue #1 §4.3 and §5 call a "component".

**backend** — an adapter that fills exactly one role and normalizes its output
to the schema. Issue #1 §8's `ASRBackend` is the role's interface; a backend is
an implementation of it. Normalize at the adapter boundary; never leak a
model-specific object past it.

**add-on** — a backend the planner adds because a requirement cannot be met
natively by the chosen stack. Add-ons are derived mechanically, never chosen by
preference: `word_bounds` on a stack without native word timing forces the
aligner, `speaker_attribution` on a stack without native speaker structure forces
a diarizer and the reconciler.

**package** — the provisionable unit: pinned weights, the environment they target,
and any toolchain build. One package may supply several backends. `firered`
supplies `vad`, `lid`, `asr`, and `punctuator`; `fluidaudio` supplies `diarizer`
and a candidate `vad`. Packages are what `audio packages pull` operates on and
what a provisioning error names.

**environment** — a provisioned runtime that several packages share. Its
dependency set is locked in this repository, not resolved at provisioning time.
Packages are grouped into as few environments as their dependencies permit.

**capability** — a typed output property, requested by the caller and declared
by a backend. Names are flat `snake_case`, matching the existing report keys.

**satisfaction** — how a requested capability was met: `native`, `derived`,
`abstained`, or `failed`. It is defined only for a capability that was actually
requested, so it has no `unavailable` value: a capability the chosen stack cannot
satisfy fails the request with exit 2 before anything loads, and a capability
nobody asked for is simply absent from the output. `abstained` is a principled
refusal on a successful run; `failed` is a backend error, which exits 1 and
writes no result.

**availability** — the catalog counterpart to satisfaction, used when nothing has
been requested yet: `native`, `requires_add_on`, or `impossible`. Step-one
planning reports this for every capability in the namespace, which is how an
agent discovers what a stack can do without triggering an error.

Issue #1 §2.2's `computed` / `not_computed` / `available_but_not_computed` is a
third, separate axis describing Observation Store state; all three survive.

**evidence** — the two independent things a declaration can be backed by, kept
apart because conflating them is how a declared interface gets read as a quality
result: `evidence.interface` is `verified` or `unverified` — did a recorded run
confirm the output exists — and `evidence.quality` is `measured` or `unmeasured` —
does a number tied to a fixture exist. A capability can be interface-verified and
quality-unmeasured, which describes most of this stack's timing claims.

`evidence` is **required** wherever satisfaction or availability is `native` or
`derived`. Omitting it is not a shorthand for "fine"; an absent marker would let
a capability with a recorded weak result read cleaner than one that merely lacks
labels. Where a quality number exists and is poor, say so in a `measured_limit`
alongside `quality: "measured"` — measured-and-weak is not unmeasured. The
diarizer matching 3 of 75 annotated speaker changes on a dense conversation is
the case this rule exists for.

**plan** — the resolved instance: the chosen stack, the requirements, the roles
they imply, the backends filling those roles with revisions and configuration,
the policy block, and the packages required. Computed before any model loads, so
it also drives the provisioning check; emitted afterwards as Issue #1 §11.2
provenance. Plan and provenance are the same object at two points in time.

Planning is answered in two steps. Given only a stack, it returns that stack's
capability catalog — what is native, what an add-on would supply, what is
impossible — with no input file, since that is a static property. Given a stack
and requirements, it returns the resolved plan plus a `sample_output` block built
by serializing a placeholder result through the same serializer `run` uses. The
sample is generated, never hand-written, so output shape stays a pure function of
the resolved capability set and no combination needs enumerating. It guarantees
key sets, types, and which fields are absent; it does not predict cardinality,
whether abstentions occur, or runtime degradation.

**policy** — the genuine choices applied regardless of which backend runs:
abstain on ambiguous overlap, the minimum turn duration and what happens below
it, and failing closed on protected intervals. Emitted in the plan alongside the
`floors` array, and not caller-selectable in v1.

Policy holds only what could legitimately have been decided otherwise. Two things
therefore do **not** belong in it: a floor, because a floor is not a choice and
encoding one as a boolean makes an invariant read like a setting; and anything
already determined by the request, because that would create a second source of
truth. Clean versus verbatim rendering is the second case — it is the `verbatim`
capability, and its absence means clean.

A backend whose inference is not deterministic runs at a fixed internal seed so
repeated runs agree and a downstream `word_id` stays stable. VibeVoice needs
this because its acoustic tokenizer samples a Gaussian latent. The seed is
recorded in provenance and is not a caller-facing knob; every backend declares
`deterministic` alongside it.

**abstention** — a recorded refusal to assert, carrying an interval and a
reason. Abstentions must survive to the output.

## Floors

Floors are not capabilities. A caller never opts into them and never opts out,
and a backend that cannot meet one is not a conforming backend:

- **Punctuated, sentence-segmented text.** Nothing downstream wants raw text —
  not a reader, not an LLM, not a cue splitter that needs sentence boundaries.
  FireRed ships punctuation as a separate stage, so its adapter always runs
  FireRedPunc.
- **Punctuation stays attached to its word.** `"Fred,"`, never `"Fred"` followed
  by a separate `","` token. Subtitle cue splitting reads sentence-final and
  clause punctuation off the word token, so a parallel punctuation stream makes
  correct cue breaking impossible no matter how good the splitter is. FireRed is
  the specific risk, because FireRedPunc emits punctuation with its own bounds;
  its adapter reattaches rather than passing that stream through. Asserted at the
  adapter boundary for every stack.

  Where this meets `no_synthesized_bounds`: the word keeps its own `start` and
  `end` unchanged, and the mark's bounds are dropped. Extending the word's `end`
  over the punctuation would synthesize a bound, and cue splitting needs only the
  mark's presence in the token, not its timing. Confidence, where a backend emits
  it, stays the word's; a mark carries none.
- **The original source timeline is canonical.** Every bound refers to it. The
  transcription path never modifies the source.
- **No synthesized bounds.** A timing field absent from the backend stays absent.
- **Abstentions survive to the output.**
- **Normalization at the adapter boundary.** No model-specific object travels
  past it.

## Capabilities

Requestable, because a caller genuinely chooses them:

| Capability | Meaning | Why it is its own name |
| --- | --- | --- |
| `verbatim` | Disfluency- and dialect-form-preserving text. Implies a transcript. | Backends differ: VibeVoice normalized `看哈→看一下` where FireRed retained it. Absent this, text is clean-rendered. |
| `speaker_attribution` | Anonymous speaker label on transcript text. | The outcome both multi-speaker paths deliver. Whether it was native or reconciled from a diarizer is provenance, not a separate request. |
| `turn_bounds` | Diarization-grade speaker turn intervals in time, independent of text. | Cutting on a speaker change needs time, not text. ASR segment bounds are not diarization-grade. |
| `overlap_intervals` | Cross-speaker overlap. | Feeds the abstention ledger; also the basis for refusing to attribute overlapping speech. |
| `speech_bounds` | Speech-activity regions. | Speech activity only; not turns and not events. |
| `segment_bounds` | ASR segment extents. | Cheap coarse timing where a stack emits it natively. Cannot produce subtitle-grade cues. |
| `word_bounds` | Word or character intervals. | The only timing that supports subtitle cues or word-level editing. Never inferred from `container_bounds`. |
| `word_confidence` | Per-word confidence. | An audit signal, distinct from the bounds themselves. |
| `region_language` | Region language label and confidence. | Region-level; cannot locate a switch. Costs roughly double inference on FireRed. |
| `token_language` | Per-token language. | **No backend provides this.** Named so the catalog can report it `impossible` and a request for it can fail loudly with `capability_unsupported`, rather than the assumption being drawn silently from code-switching support. |

Provenance-only. These appear in every plan so they can be audited and so they
can be refused as a timing or routing source. They are never requestable:

| Capability | Why it exists | Why it is not requestable |
| --- | --- | --- |
| `container_bounds` | Records the processing container a stack used. | Not time evidence. Never promotable to any other `*_bounds`. |
| `container_language` | Records a stack's single language label. | No output artifact needs it, and it disagreed across Qwen sizes on one clip, so it is not a routing oracle. |

Declared in the namespace but not implemented in v1:

| Capability | Blocked on |
| --- | --- |
| `capture_role` | The dedicated-channel merge and non-oracle role mapping are unmeasured. Never inferred; only capture metadata may establish it. |
| `filler_candidates`, `repetition_candidates`, `false_start_candidates` | Need a `DisfluencyAnnotator` role. `verbatim` preserves fillers in text without annotating them. Belongs to `analyze`/`inspect`. |

Requesting any of these fails with exit 2 `capability_unsupported`, `allowed: []`,
`reason: "not_implemented_v1"`. Step one's catalog reports them
`availability: impossible` with the same reason, so their status is discoverable
without a failed command.

Speaker identity and semantic role are not capabilities at all. They stay
external to ASR unless capture metadata establishes them, in which case they
appear as `capture_role`.

## Resolution

The caller chooses a stack, then states requirements. The planner derives the
add-ons. There is no default stack, no preference scalar, and no tie-break
ordering, because a stack choice is a quality judgement the planner cannot make.

1. `--stack` is required. Omitting it fails with the stack list rather than
   guessing.
2. Requirements the stack satisfies natively cost nothing.
3. Requirements the stack cannot satisfy natively force specific add-ons.
4. Requirements the chosen stack cannot satisfy fail the request with exit 2
   before any model loads: `capability_unsatisfiable_on_stack` with a non-empty
   `allowed` when another stack could serve it, `capability_unsupported` with
   `allowed: []` and a reason when none can. Step one reports the same facts as
   `availability: impossible` without erroring, so discovery never costs a
   failed command.
5. An add-on with more than one implementation may be pinned (`--diarizer`).
   With one implementation it is forced, and the pin exists for later ones.

This table is the planner's whole logic and the source the anti-fabrication test
is parametrized over, so it must distinguish the two failure codes rather than
render them alike.

| Requirement | `qwen-1.7b`, `qwen-0.6b` | `vibevoice` | `firered` |
| --- | --- | --- | --- |
| `verbatim` | native; interface verified, quality unmeasured | native; interface verified, quality unmeasured — normalization observed | native; interface verified, quality unmeasured — dialect form retained |
| `speaker_attribution` | + `fluidaudio` + reconciler | native | + `fluidaudio` + reconciler |
| `turn_bounds` | + `fluidaudio` | + `fluidaudio` | + `fluidaudio` |
| `overlap_intervals` | + `fluidaudio` | + `fluidaudio` | + `fluidaudio` |
| `speech_bounds` | + `silero-vad` | + `silero-vad` | native |
| `segment_bounds` | exit 2: unsatisfiable_on_stack | native | native |
| `word_bounds` | + `qwen3-forcedaligner` | + `qwen3-forcedaligner` | native |
| `word_confidence` | exit 2: unsatisfiable_on_stack | exit 2: unsatisfiable_on_stack | native |
| `region_language` | exit 2: unsatisfiable_on_stack | exit 2: unsatisfiable_on_stack | + FireRedLID stage |
| `token_language` | exit 2: unsupported | exit 2: unsupported | exit 2: unsupported |
| `capture_role`, `*_candidates` | exit 2: unsupported | exit 2: unsupported | exit 2: unsupported |

`unsatisfiable_on_stack` carries a non-empty `allowed`, so the fix is to switch
stacks. `unsupported` carries `allowed: []` and a reason —
`no_backend_declares` for `token_language`, `not_implemented_v1` for the rest.

## Packages and environments

Everything provisioned lives under one root: `AUDIO_PROCESSING_MODEL_CACHE`
when set, otherwise the per-platform cache directory the Silero backend already
uses. Nothing is provisioned beside a source checkout the way the `model_tests/`
experiments were.

```text
<root>/
  registry.json     # provisioned packages and environments, revisions, digests, byte sizes
  models/           # pinned single-file artifacts, e.g. silero-vad-6.2.1.onnx
  envs/
    mlx/            # uv environment from the repository-tracked lock
    torch/          # uv environment, plus source checkouts and applied patches
    swift/          # FluidAudio built product and its Core ML package
```

Three provisioned environments is the floor: MLX and PyTorch cannot share one,
and FluidAudio has no Python at all.

| Environment | Packages | Constraint |
| --- | --- | --- |
| core | `silero-vad` | The `audio` tool's own environment; numpy, scipy, onnxruntime. Nothing to provision. |
| `mlx` | `qwen3-asr-1.7b-8bit`, `qwen3-asr-0.6b-8bit` | `mlx` plus `mlx-audio==0.4.5`; deliberately torch-free. |
| `torch` | `vibevoice-asr-7b`, `qwen3-forcedaligner`, `firered-asr2s` | Shared. The joint resolution of VibeVoice's pinned transformers, `qwen-asr`, and FireRed's requirements is **unverified**; if it fails, split and record why. |
| `swift` | `fluidaudio` | Swift toolchain build; no Python. |

Rules:

- Environment dependency sets are locked in this repository. `pull` materializes
  a lock; it never resolves "latest". This is what keeps the `mlx-audio` private
  batched API at the one version the source-hash guard expects.
- Hub weights stay in the Hugging Face cache. The registry records which
  revisions this tool materialized rather than duplicating a snapshot.
- Applying the VibeVoice patch and building the FluidAudio product happen in
  `pull` and nowhere else. Neither is ever triggered by a transcription request.
- `remove` is reference-counted: an environment survives while another
  provisioned package still needs it. It deletes only the Hub revisions the
  registry records as materialized here, and says so, because the Hugging Face
  cache may be shared with other tools.
- `purge` reads `registry.json`, not shell history, so a session that never ran
  `pull` can still find and free everything. It reports reclaimable bytes before
  removing anything.
- `verify` re-checks artifact digests, the `mlx-audio` private-API source hash,
  whether tracked patches are applied, and that the Swift product runs.

Lifecycle: `audio packages list | pull | verify | remove | purge | path`. `pull`
accepts package ids, or the same `--stack` and `--want` arguments `transcribe`
takes so the provisioning set comes from the plan rather than from memorized ids.
`path` prints the resolved root and per-package locations so a session with no
provisioning history can still locate everything.

## Retired and reserved words

- **route**, **recipe**, **preset** — not CLI concepts. `route` remains prose in
  DECISION_REPORT for describing recommendations to people. `stack` was
  previously retired alongside them and is now a defined term above; use it only
  in that sense, not for a runtime-footprint claim.
- **feature** — not used on the request axis; it stays available for acoustic
  features. Issue #1 §2.2's `analyze --features` should become a capability list.
- **component** — Issue #1 §4.3 and §5 use it for what this document calls a
  role. Prefer `role`, `backend`, or `add-on` in new work.
- **profile** — reserved for `enhance`'s conformance target, which declares
  loudness and true-peak bounds to conform to. It is not a stack or a backend
  selection and does not migrate to `transcribe`.
- **view** — reserved for Observation Store projections (`inspect --view`).
- **model** — too coarse for the schema and wrong for the command surface: the
  provisioned set includes a Swift build product, two Core ML bundles, and two
  dependency environments containing no weights. Prefer `package`, `stack`, or
  `backend`. The command group is `audio packages`.
- **punctuation** — not a capability. See Floors.

## Versioning

The output schema and the plan carry versions so consumers can detect
incompatibility. Stacks are named, not versioned: a different model size is a
different stack id. `enhance --profile transcription@3` keeps its own versioning.
