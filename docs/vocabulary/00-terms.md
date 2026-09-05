# Transcription vocabulary

Agreed 2026-08-16. This is the naming contract for the `transcribe` surface and
the schema beneath it. It consolidates terms already used in
[Issue #1](https://github.com/fyang0507/audio-processing-cli/issues/1),
[HANDOFF.md](../../HANDOFF.md), and
[model_tests/DECISION_REPORT.md](../../model_tests/DECISION_REPORT.md), and resolves
the collisions between them. `transcribe` is the most demanding command, so it
defines the vocabulary; other commands follow it. The command sequences that
exercise it are in [TRANSCRIBE_CONTRACT.md](../../TRANSCRIBE_CONTRACT.md).

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

**role** — a functional slot filled by a *provisioned* backend. The enumeration is fixed and
small: `decode`, `vad`, `diarizer`, `asr`, `aligner`, `punctuator`, `lid`. This is what
Issue #1 §4.3 and §5 call a "component".

Deterministic glue is not a role. Reconciling diarizer turns onto ASR text is real work with
a real guarantee — an exact partition of the timeline, so no span is transcribed twice and no
gap is invented — but it has no package, no revision, no environment, and no measurable cost,
so listing it beside four models implied a fifth model that does not exist. It is stated in
the `diarization` capability's note instead. `decode` stays a role because ffmpeg is an
external binary with a configuration that changes the audio every later stage sees.

**backend** — an adapter that fills exactly one role and normalizes its output
to the schema. Issue #1 §8's `ASRBackend` is the role's interface; a backend is
an implementation of it. Normalize at the adapter boundary; never leak a
model-specific object past it.

**add-on** — a backend the planner adds because a requirement cannot be met
natively by the chosen stack. Add-ons are derived mechanically, never chosen by
preference: `word_timestamps` on a stack without native word timing forces the
aligner, `diarization` on a stack without native speaker structure forces
a diarizer.

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
provenance. Provenance embeds the executed plan verbatim and adds only what running it
revealed: the `stack` that ran, an `outcomes` map saying what became of each requested
capability, and an `observed` block of stage walls, peaks, and cardinalities. It does not
restate the plan's `satisfaction`, `backend`, or `evidence` around each outcome — those are
in the embedded plan, and repeating them is how two copies of one fact start to disagree.

Two questions, two commands, and no order between them. `audio transcribe capabilities
--stack S --input F` answers what a stack can do with a file: the capability catalog, how the
audio will be partitioned, what the run will cost, and whether a failure leaves anything
usable. `audio transcribe plan --stack S --input F [--want ...]` resolves one request into the
backends that will run, the packages to provision, and the shape of the output. Both require a
stack and an input; neither reads media beyond a metadata probe; neither provisions anything.

An earlier draft made both of these `plan`, distinguished by whether `--want` was present, and
called them step one and step two. That was wrong twice: one command answered two different
questions depending on an absent argument, and the numbering implied a sequence nothing
enforced. An agent that knows what it needs goes straight to `plan`. On `plan`, an omitted
`--want` means the floors-only request — a punctuated transcript and nothing optional — which
is the same meaning the flag's absence has on `run`, not a request for the menu.

Because `capabilities` is where a request gets chosen, it carries the context that choice needs
and not just the `availability` enum: `cost` at stack and add-on scope, `timing_precision`
where a capability is native but its accuracy is unmeasured, `failure_recovery`, and the
languages the stack actually handles. An `availability` value alone cannot distinguish a free
add-on from one that pulls a second runtime, nor tell a caller whether native is good enough.

**Structure only where something dispatches on it.** The report carries three enums and two
numbers — `availability`, `processing.unit`, `failure_recovery.partial_results`, the unit
count, and `projected_seconds` — and says everything else in one sentence per capability.
Citations are not payload: `model_tests/` paths belong in this repository's documents and will
not exist in a shipped tool's output, so a sentence names the fixture in words ("on a
30-minute sample") and the research record holds the artifact. `cost` states a run that
actually happened before it states a rate, in units a reader can hold — seconds and gibibytes
against a named sample — because nobody should multiply an RTF by a duration to learn whether
a job takes a minute or an hour.

The sentences still have to do the work the retired objects were built to enforce: say what
was measured and on what fixture, distinguish a run that **refuted** a property from one that
merely never measured it, and state the consequence rather than the forensics. A plan is the
opposite case and keeps its structure, because it is dispatched on rather than read: `roles`
with revisions and configuration is audited provenance, and `satisfaction`, `outcome`, and
`evidence` are asserted by tests.

A plan also carries a `sample_output` block built by serializing a placeholder result through
the same serializer `run` uses. The sample is generated, never hand-written, so output shape
stays a pure function of the resolved capability set and no combination needs enumerating. It
guarantees key sets, types, and which fields are absent; it does not predict cardinality,
whether abstentions occur, or runtime degradation.

**refusal** — a request the tool declines, always with a `fix`. `fix` is a runnable command
wherever a configuration exists that would work, and a sentence naming the nearest available
output where none does; a plausible command that fails again teaches a caller nothing. The two
capability refusals additionally carry `available_on_stack` — every name the chosen stack
accepts, split into native, requires-an-add-on, and impossible — because a caller who got
`--want` wrong needs the menu for the stack it picked, not a second command to go find it.

**policy** — the decisions the tool makes that no caller can change: abstain on ambiguous
overlap or an unavailable requested alignment, the three recorded turn thresholds and what
happens below each, and whether anything in a plan can detect overlap at all. Recorded here and in the research record, and **not
printed**, on the same reasoning as the floors: a caller cannot select any of it, so putting it
in every plan invited a reader to mistake it for a set of settings, and the thresholds are
fixed by the tool version rather than by the request.

The three thresholds are `raw_fragment_min_ms: 250`, `accepted_turn_min_ms: 500`, and
`same_label_merge_max_ms: 300`, exactly as declared by
`model_tests/benchmark/turn_attributed_mlx_asr/runtime.py:76-90`. A raw diarizer fragment below
250 ms is not transcribed; same-label fragments separated by at most 300 ms are merged; a
resulting turn below 500 ms is not transcribed. The two declined cases become `raw_fragment`
and `short_turn` abstentions respectively. Ambiguous multi-speaker activity becomes `overlap`.

The one part a caller must not miss is a trap, so it is surfaced where it is actionable rather
than buried in a block: when nothing in a plan detects overlap, an empty abstention ledger
means *undetected*, not *absent*. That is a plan warning and a line in the
`overlapped_speech` catalog note.

Each threshold is still named separately wherever it is written down. Collapsing
`raw_fragment_min_ms`, `accepted_turn_min_ms`, and `same_label_merge_max_ms` into one
`min_turn_ms` silently picks one of three recorded values, which is a defect this project made
once already.

Two things were never policy and are worth keeping out of it: a floor, because a floor is not a
choice; and anything already fixed by the request, which would create a second source of truth.
Rendering is the second case — it is the `verbatim` capability. Note that in v1 its absence does
not mean clean, because nothing cleans; it means the plan was not asked to answer for text
fidelity.

A backend whose inference is not deterministic runs at a fixed internal seed so a run is
reproducible at all. VibeVoice needs this because its acoustic tokenizer samples a
Gaussian latent. The seed is recorded in provenance and is not a caller-facing knob; each
backend declares `deterministic`, a `determinism_tolerance_ms`, and a
`determinism_basis` that cites either a repeat-hash measurement or the decode
configuration in the runner that produced the recorded figures. "By construction" without
a citation is not a basis. The tolerance exists because one boolean cannot carry a
measured near-miss: `0.0` claims byte-identical normalized output on repeat, while FireRed
declares `1.0` because its exact-repeat run reproduced text exactly and timestamps only to
within a millisecond.

All of that is **provenance, not a decision input**, and it lives in the executed plan's
`roles` rather than in the step-one catalog. Its only consumer is an auditor asking whether
two artifacts should be byte-comparable. It is specifically *not* the answer to "do I need
an aligner for accurate timing" — that is boundary error against labels, which the catalog
states in prose and which is unmeasured on every path here. Reproducibility is not accuracy;
conflating the two is easy and was done here once.

**language input** — a hint passed *to* an ASR, distinct from every language
capability, which is an output. Keep the two apart in naming: `--language` is an input
that constrains a decode; `lid` and `token_lid`
are outputs that report one. A stack declares whether it accepts an input in its
`languages` capability, and passing the flag where it is not accepted is an error
rather than a silently ignored argument. This is the only caller-settable model input
in v1.

Qwen accepts exactly these 30 names, case-insensitively and normalized to the spelling shown:
`Chinese`, `English`, `Cantonese`, `Arabic`, `German`, `French`, `Spanish`, `Portuguese`,
`Indonesian`, `Italian`, `Korean`, `Russian`, `Thai`, `Vietnamese`, `Japanese`, `Turkish`,
`Hindi`, `Malay`, `Dutch`, `Swedish`, `Danish`, `Finnish`, `Polish`, `Czech`, `Filipino`,
`Persian`, `Greek`, `Romanian`, `Hungarian`, `Macedonian`. The source is `support_languages`
in both pinned checkpoint configs:
[`Qwen3-ASR-1.7B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit/blob/a8379a2e2f9e313c9292cdf1af4055ab56d50d55/config.json)
and [`Qwen3-ASR-0.6B-8bit`](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit/blob/89e96d92ba34aca20b3e29fb10cc284097d1219f/config.json).

FireRed accepts no language input. Its `lid` output uses the 115 labels after the five special
tokens in the pinned
[`FireRedLID` dictionary](https://huggingface.co/FireRedTeam/FireRedLID/blob/1bb4d285c8456429385d9c0810300df4297bc11b/dict.txt):

```text
en es fr zh other xinan ja ko ru mandarin min wu xiang yue north de pt ab af am ar as az ba be bg bn br ca cs cy da el eo et eu fa gl gn ha iw hi ht hu hy ia id is it ka kk lo lt lv mk ml mn mr mt no ne nl nn oc pa pl ps ro sd sk sl sq sr sv sw ta te tg th tk tr tt uk ur uz vi yi yo kn so ceb jw mi hr bs tl ln my fi sn lb gu ms km bo fo gv haw la mg sa sco si su war
```

These vocabularies are not reconciled: Qwen calls its input `Cantonese`, while FireRed emits
`yue`. Translating between them would be a separate declared policy, not spelling cleanup.

**unit** — the piece of audio a stack actually works on, and the granularity at which a
run can be partial. Qwen works in diarized turns when speaker structure is requested and
in fixed chunks otherwise; FireRed works in VAD regions; VibeVoice is handed whole media
in a single call and therefore has exactly one unit. The unit is a stack property, but
*how many* units a given file yields is a property of the pair, which is why an input is
required to plan. Where the count depends on content the plan has not decoded, it is
reported as the fixed structural field `unit_count: null` rather than guessed.

**coverage** — which parts of the source a result actually covers. Every result carries the
boolean `complete`; `coverage` is present if and only if `complete` is `false`. It holds a
`covered_through_seconds` watermark — the end of the longest
contiguous transcribed prefix, the one number a caller can act on without reasoning about
gaps — plus explicit `covered_intervals` and `missing_intervals`, because completion is not
necessarily contiguous in time: the recorded Qwen runner processes turns in
duration-bucketed order and restores chronological order afterwards.

**resume** — finishing a partial run without redoing it, via `--range <start>[:<end>]` on
the original input. Ranges rather than pre-clipped audio, because clipping shifts the
timeline and would force a consumer to re-offset every bound by hand before merging, which
is arithmetic on timestamps performed outside the tool and precisely what the
canonical-timeline floor exists to prevent.

**failure_recovery** — a stack's declared answer to what a failure leaves behind:
`partial_results` of `per_unit` or `prefix_only`, and whether it is `resumable`. It belongs in
the `capabilities` report because it is a stack-choice input, not a run-time discovery. Qwen
and FireRed preserve completed independent units. VibeVoice is `prefix_only`: a generation-cap
truncation can salvage complete parsed segments before the cut and resume the remainder, while
a failure before any complete segment — including OOM at model load — leaves no result.

**execution** — the plan's statement of stage order and residency at environment-process
granularity. Environment processes run strictly sequentially, so a request costs the sum of
their walls and the maximum of their peaks. Qwen and VibeVoice use one fresh process per model
stage. FireRed is the deliberate exception: one `torch-firered` process loads its VAD, LID,
ASR, and punctuator models co-resident, and only that process-level peak is meaningful. The
plan must never turn one package or one environment into a claim that those models were loaded
one at a time. VibeVoice and the later MLX aligner remain separate environment processes and
have never been measured co-resident.

**abstention** — a recorded refusal to assert, carrying an interval and a `reason` from exactly
four allowed values: `overlap` for ambiguous multi-speaker activity, `short_turn` for an
accepted turn below 500 ms, `raw_fragment` for a span whose only activity was a sub-250 ms
diarizer fragment, and `alignment_unavailable` for an ordinary VibeVoice speech segment whose
requested aligner result is absent, invalid, or does not reproduce the segment text after
punctuation and whitespace are removed. The last uses the segment's native bounds, preserves
its text, omits `words`, and makes the run-level `word_timestamps` outcome `abstained`; bracketed
non-speech event tags are not sent to the aligner and are not abstentions. Abstentions must
survive to the output. Budget-unprocessed intervals are coverage, not abstentions, because the
tool did not reach them rather than declining to assert.
