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

**satisfaction** — how a requested capability is met: `native` or `derived`.
Decided at plan time, before anything loads. It is defined only for a capability
that was actually requested, so it has no `unavailable` value: a capability the
chosen stack cannot satisfy fails the request with exit 2, and a capability nobody
asked for is simply absent from the output.

**outcome** — what became of that capability on a run: `produced` or `abstained`.
Keep it separate from satisfaction, which is a plan-time resolution and cannot
describe a run. `abstained` is a principled refusal on a *successful* run, and the
abstention ledger names the capability and the interval. A backend error is not an
outcome: it exits 1 with a `backend_failed` payload and writes no result, so no
artifact can carry it and no enum member is needed for it.

**availability** — the catalog counterpart to satisfaction, used when nothing has
been requested yet: `native`, `requires_add_on`, or `impossible`. Step-one
planning reports this for every capability in the namespace, which is how an
agent discovers what a stack can do without triggering an error.

Issue #1 §2.2's `computed` / `not_computed` / `available_but_not_computed` is a
third, separate axis describing Observation Store state; all three survive.

**evidence** — the two independent things a declaration can be backed by, kept
apart because conflating them is how a declared interface gets read as a quality
result. `evidence.interface` is `verified` or `unverified` — did a recorded run
confirm the output exists. `evidence.quality` is `measured`, `unmeasured`, or
`refuted` — does a number tied to a fixture exist, or did a recorded run instead
show the property failing. A capability can be interface-verified and
quality-unmeasured, which describes most of this stack's timing claims.

`evidence` is **required** wherever satisfaction or availability is `native` or
`derived`. Omitting it is not a shorthand for "fine"; an absent marker would let
a capability with a recorded weak result read cleaner than one that merely lacks
labels. Each non-neutral quality value carries a companion field, and the
companion is required:

- `quality: "measured"` with a poor number carries `measured_limit`.
  Measured-and-weak is not unmeasured. The diarizer matching 3 of 75 annotated
  speaker changes on a dense conversation is the case this rule exists for.
- `quality: "refuted"` carries `observed_limit`: no number exists, but a recorded
  run exhibited the property failing. `vibevoice` normalizing `看哈→看一下` and
  `qwen-0.6b` rendering `刷啥子` where `qwen-1.7b` retained `耍啥子` are both
  refutations of `verbatim`, not absences of a measurement. Filed as `unmeasured`
  they would read as untested, which is the inversion this object exists to
  prevent — one level below where it was first caught.

**plan** — the resolved instance: the chosen stack, the requirements, the roles
they imply, the backends filling those roles with revisions and configuration,
the policy block, and the packages required. Computed before any model loads, so
it also drives the provisioning check; emitted afterwards as Issue #1 §11.2
provenance. Plan and provenance are the same object at two points in time, and
the second point in time contributes exactly one thing: provenance carries an
`outcome` per capability, a plan does not.

Planning is answered in two steps. Given only a stack, it returns that stack's
capability catalog — what is native, what an add-on would supply, what is
impossible — with no input file, since that is a static property. Because step one is
the only place a caller can decide step two, the catalog carries the context that
decision needs and not just the `availability` enum: the stack's `determinism` and
measured resource envelope, whether it accepts a language input, an `add_on_cost` per
capability that requires one (packages, environment, external tools, measured stage
time and memory), a `shares_stage_with` list where several capabilities come from one
add-on run, and a `timing_precision` block where a capability is native but its
accuracy is unmeasured. An `availability` value alone cannot distinguish a free add-on
from one that pulls a second runtime, nor tell a caller whether native is good enough.

Given a stack
and requirements, it returns the resolved plan plus a `sample_output` block built
by serializing a placeholder result through the same serializer `run` uses. The
sample is generated, never hand-written, so output shape stays a pure function of
the resolved capability set and no combination needs enumerating. It guarantees
key sets, types, and which fields are absent; it does not predict cardinality,
whether abstentions occur, or runtime degradation.

**policy** — the genuine choices applied regardless of which backend runs: abstain
on ambiguous overlap, the recorded turn thresholds and what happens below each of
them, and whether anything in this plan can detect overlap at all. Emitted in the
plan alongside the `floors` array, and not caller-selectable in v1. Each threshold
is named separately; collapsing them into one `min_turn_ms` silently picks one of
three recorded values.

Policy holds only what could legitimately have been decided otherwise. Two things
therefore do **not** belong in it: a floor, because a floor is not a choice and
encoding one as a boolean makes an invariant read like a setting; and anything
already determined by the request, because that would create a second source of
truth. Rendering is the second case — it is the `verbatim` capability, so `policy`
carries no rendering field. In v1 the capability's absence does not mean clean,
because nothing cleans; it means the plan was not asked to answer for text
fidelity.

A backend whose inference is not deterministic runs at a fixed internal seed so
repeated runs agree and a downstream `word_id` stays stable. VibeVoice needs
this because its acoustic tokenizer samples a Gaussian latent. The seed is
recorded in provenance and is not a caller-facing knob; every backend declares
`deterministic` alongside it, plus a `determinism_basis` that cites either a
repeat-hash measurement or the decode configuration in the runner that produced
the recorded figures. "By construction" without a citation is not a basis.

`deterministic` is qualified by `determinism_tolerance_ms`, because one boolean cannot
carry a measured near-miss. `0.0` claims byte-identical normalized output on repeat.
FireRed declares `1.0`: its exact-repeat run reproduced text exactly and timestamps
only to within a millisecond. Publish what a tolerance means for the caller rather than
the number alone — 1 ms is a fraction of a video frame and irrelevant to subtitle cues,
while being fatal to a `word_id` keyed on a start time.

**language input** — a hint passed *to* an ASR, distinct from every language
capability, which is an output. Keep the two apart in naming: `--language` is an input
that constrains a decode; `region_language`, `container_language`, and `token_language`
are outputs that report one. A stack declares whether it accepts an input in its
`language_input` block, and passing the flag where it is not accepted is an error
rather than a silently ignored argument. This is the only caller-settable model input
in v1.

**execution** — the plan's statement of stage order and model residency. Stages run
strictly sequentially and no two model stages are resident at once, so a request costs
the sum of the stage walls and the maximum of the stage peaks. This is declared rather
than left implicit because every recorded figure was produced by strictly sequential
fresh subprocesses whose record states that the stages did not overlap; per-stage memory
peaks published without it invite summing. It is also what keeps the largest stack's
memory claim honest, since VibeVoice and the aligner have never been measured resident
together.

**abstention** — a recorded refusal to assert, carrying an interval and a
reason. Abstentions must survive to the output.

## Floors

Floors are not capabilities. A caller never opts into them and never opts out,
and a backend that cannot meet one is not a conforming backend:

- **Punctuated, sentence-segmented text.** Nothing downstream wants raw text —
  not a reader, not an LLM, not a cue splitter that needs sentence boundaries.
  FireRed ships punctuation as a separate stage, so its adapter always runs
  FireRedPunc.
- **Punctuation is sentence-level.** Marks live in sentence and segment `text`.
  They never form a parallel stream with their own bounds, they are never attached
  to a word token by this tool, and punctuation is not a request: floor one already
  puts it in the text. Both backends that produce word bounds already agree.
  FireRed builds its word stream from pre-punctuation AED timestamps
  (`fireredasr2system.py:181-184`), so word tokens carry only the intra-word marks
  the ASR itself produced — 20 of 12,370 recorded word tokens, every one an English
  contraction such as `it's`. Qwen3-ForcedAligner emits one token per
  non-punctuation character, so `好，现在开始。` aligns as five tokens.

  What cue splitting needs is not a mark on a token but a sound mapping from a
  mark's position in the sentence text to a word index, and that rests on one
  invariant: **stripping punctuation and whitespace from a sentence's `text` yields
  exactly the concatenation of its word `text` values, compared
  case-insensitively.** Verified on all five recorded FireRed artifacts (12,370
  words) and all 17 aligned segments of the forced-aligner artifact. Assert it at
  the adapter boundary per stack; it is the testable statement the old
  attach-to-word rule was reaching for, and unlike that rule it is not vacuous on
  the stack it names.

  Case-insensitively is not a hedge. FireRedPunc lowercases its input and then
  re-capitalizes sentence starts and standalone `i`
  (`fireredpunc/punc.py:349-382`), so sentence text and word text differ in case by
  construction — 234 characters across those artifacts. Neither stream is derivable
  from the other: the sentence text holds the marks and the casing, the word stream
  holds the bounds. Both are carried; the sentence text is canonical for reading and
  for subtitles. A segment may legitimately have no word stream at all — the
  forced-aligner artifact has two such segments, both VibeVoice non-speech event
  tags — so the invariant binds only where words exist.
- **The original source timeline is canonical.** Every bound refers to it. The
  transcription path never modifies the source.
- **No synthesized bounds.** A timing field absent from the backend stays absent.
- **Abstentions survive to the output.**
- **Normalization at the adapter boundary.** No model-specific object travels past
  it, no backend scaffolding survives it, and a backend's default-filled field is not a
  value. Three verified instances, one per stack. Qwen's private batched API returns the
  model's own scaffold inside its text — `language English<asr_text>` — which the public
  path strips and an adapter on the batched path must strip too. FireRed emits
  `lang: null, lang_confidence: 0` on every sentence whether or not LID ran
  (`fireredasr2system.py:149-150`), so with LID off the adapter drops both rather
  than publish a zero that reads as a measured confidence. VibeVoice emits
  `Speaker: "N/A"` on non-speech segments; that is the absence of a label and must
  not become a speaker id.

## Capabilities

Requestable, because a caller genuinely chooses them:

| Capability | Meaning | Why it is its own name |
| --- | --- | --- |
| `verbatim` | The stack **can produce** verbatim text: it emits what it heard, disfluencies included, rather than a cleaned rendering. How faithfully it does so is the quality axis, not this one. | Interface verified on all four stacks — 24 to 28 filler hits on one probe, none of them cleaning and none of them complete. No backend exposes a verbatim switch and nothing in v1 cleans, so requesting this asserts an interface rather than selecting a mode, and the plan answers for fidelity separately: `quality: "refuted"` on the two stacks a recorded run caught normalizing a dialect form. |
| `speaker_attribution` | Anonymous speaker label on transcript text. | The outcome both multi-speaker paths deliver. Whether it was native or reconciled from a diarizer is provenance, not a separate request. |
| `turn_bounds` | Diarization-grade speaker turn intervals in time, independent of text. | Cutting on a speaker change needs time, not text. ASR segment bounds are not diarization-grade. |
| `overlap_intervals` | Cross-speaker overlap. | Feeds the abstention ledger; also the basis for refusing to attribute overlapping speech. |
| `speech_bounds` | Speech-activity regions. | Speech activity only; not turns and not events. |
| `segment_bounds` | ASR segment extents. | Cheap coarse timing where a stack emits it natively. Cannot produce subtitle-grade cues. |
| `word_bounds` | Word or character intervals. | The only timing that supports subtitle cues or word-level editing. Never inferred from `container_bounds`. May be legitimately absent on a segment that has no speech to align. |
| `region_language` | Region language label and confidence. | Region-level; cannot locate a switch. Costs roughly double inference on FireRed. FireRed carries the label on each *sentence*, but it is produced once per VAD region and copied onto the sentences inside it (`fireredasr2system.py:129-155`), so per-sentence variation would be fabricated. |
| `token_language` | Per-token language. | **No backend provides this.** Named so the catalog can report it `impossible` and a request for it can fail loudly with `capability_unsupported`, rather than the assumption being drawn silently from code-switching support. |

Provenance-only. These appear in every plan so they can be audited and so they
can be refused as a timing or routing source. They are never requestable:

| Capability | Why it exists | Why it is not requestable |
| --- | --- | --- |
| `container_bounds` | Records the processing container a stack used. | Not time evidence. Never promotable to any other `*_bounds`. |
| `container_language` | Records a stack's single language label. | No output artifact needs it. On one Mandarin-majority clip `qwen-1.7b` reported English and `qwen-0.6b` reported Chinese, so it is not a routing oracle. It is read off the model's own output scaffold, not a separate detector. |

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
5. A role with more than one implementation may be pinned. `--vad` is the case
   that exists today: `silero-vad` is the default and FluidAudio ships a second
   Core ML implementation. `--diarizer` has one implementation and is therefore
   forced; the pin is defined so that adding a second one is not a surface change.

This table is the planner's whole logic and the source the anti-fabrication test
is parametrized over, so it must distinguish all three refusal codes rather than
render them alike.

| Requirement | `qwen-1.7b`, `qwen-0.6b` | `vibevoice` | `firered` |
| --- | --- | --- | --- |
| `verbatim` | native | native | native |
| `speaker_attribution` | + `fluidaudio` + reconciler | native | + `fluidaudio` + reconciler |
| `turn_bounds` | + `fluidaudio` | + `fluidaudio` | + `fluidaudio` |
| `overlap_intervals` | + `fluidaudio` | + `fluidaudio` | + `fluidaudio` |
| `speech_bounds` | + `silero-vad` | + `silero-vad` | native |
| `segment_bounds` | exit 2: unsatisfiable_on_stack | native | native |
| `word_bounds` | + `qwen3-forcedaligner` | + `qwen3-forcedaligner` | native |
| `region_language` | exit 2: unsatisfiable_on_stack | exit 2: unsatisfiable_on_stack | native (FireRedLID stage) |
| `token_language` | exit 2: unsupported | exit 2: unsupported | exit 2: unsupported |
| `container_bounds`, `container_language` | exit 2: not_requestable | exit 2: not_requestable | exit 2: not_requestable |
| `capture_role`, `*_candidates` | exit 2: unsupported | exit 2: unsupported | exit 2: unsupported |

Four cell forms, and a test parametrized over this table must accept exactly these:
`native`; `native (<stage>)`; `+ <package>` or `+ <package> + <role>`; and
`exit 2: <code>`. `native (<stage>)` is not an add-on. FireRedLID ships inside the
`firered` package and fills the `lid` role the stack already declares, so requesting
`region_language` adds nothing to the plan the stack did not already contain — it
changes cost, not composition, which is why the add-on `+` notation would misreport
it.

The three refusal codes: `unsatisfiable_on_stack` carries a non-empty `allowed`, so
the fix is to switch stacks; `unsupported` carries `allowed: []` and a reason —
`no_backend_declares` for `token_language`, `not_implemented_v1` for the rest;
`not_requestable` carries `allowed: []` and an `alternatives` list, because the
provenance-only pair is present in the output and merely cannot be asked for.

Cells carry resolution only. Evidence lives in the per-stack catalog, because it
differs where resolution does not: `verbatim` resolves `native` on all four stacks
and has a recorded refutation on two of them. That is also why the two Qwen sizes
share one column and get separate catalogs — their resolution is identical for
every capability here and their observed text fidelity is not. Across two probes and
two dialect lexemes the split is consistent: `firered` and `qwen-1.7b` retained `看哈`
and `耍啥子`, while `vibevoice` normalized `看哈` to `看一下` and both it and
`qwen-0.6b` rendered `刷啥子`. Filler retention, by contrast, separates nothing — 24
to 28 hits across all four — so the differentiating half of text fidelity is dialect
form, and the catalog says which half a figure describes.

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

`fluidaudio` provisions the built Swift product plus one Core ML model package,
`FluidInference/speaker-diarization-coreml`. FluidAudio also ships a Core ML VAD
that its own `VadManager` resolves; the comparison probe exercised it, but this tool
has never provisioned it as a named package, so it is not one. If `--vad fluidaudio`
ships, it becomes a package and gets its own registry entry.

Licenses are a registry field, reported by `audio packages list` and not by a plan —
a plan resolves a pipeline, not a redistribution question. Two are recorded today:
the FluidAudio SDK is Apache-2.0 and `speaker-diarization-coreml` is CC-BY-4.0
(`model_tests/benchmark/DIARIZATION.md`). Every other package reports
`license: "unreviewed"`, which is a stated gap rather than an omission that could be
read as cleared.

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
  provisioned set includes a Swift build product, one Core ML model package, and
  two dependency environments containing no weights. Prefer `package`, `stack`, or
  `backend`. The command group is `audio packages`.
- **punctuation** — not a capability, and not a flag. See Floors.
- **word_confidence** — retired as a capability. FireRed was the only claimed
  source and it emits no per-word confidence: every word is exactly
  `{start_ms, end_ms, text}` across 12,370 recorded tokens
  (`fireredasr2system.py:181-184`). What it does emit is `asr_confidence` per
  *sentence*, which is a different granularity and is not requestable in v1. Do
  not reintroduce the name for the sentence value.

## Versioning

The output schema and the plan carry versions so consumers can detect
incompatibility. Stacks are named, not versioned: a different model size is a
different stack id. `enhance --profile transcription@3` keeps its own versioning.
