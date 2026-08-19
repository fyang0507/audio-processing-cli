# `transcribe` command contract

**Status: specification.** None of the commands below are implemented yet. This
is the agent-facing sequence the v1 transcription work must produce, written end
to end for all four v1 stack ids: from a machine with nothing installed, through
provisioning and execution and export, to teardown. Terms are defined in
[VOCABULARY.md](VOCABULARY.md); the backend evidence is in
[model_tests/DECISION_REPORT.md](model_tests/DECISION_REPORT.md).

Two decisions shape every sequence below:

- **`plan` and `run` are separate subcommands, not a flag.** They have different
  side effects, different output shapes, and different exit-code contracts:
  `plan` never reports a provisioning failure, it *is* the provisioning report.
  Only commands that select backends need the split, so `export` has none.
- **`--stack` and `--input` are both required, on `plan` as well as `run`.** The stack
  fixes transcript quality, language and dialect behavior, and which capabilities arrive
  natively. The input fixes the processing detail — how the audio is partitioned into
  units, how many there are, what the run will cost, and whether a failure is
  recoverable — and none of that is a static property of the stack. Omitting either
  fails with what was missing.

`--want` is optional. Omitting it requests nothing beyond the floors, which is a
legitimate and useful request: a punctuated, sentence-segmented transcript on the
canonical timeline with an abstention ledger and no optional capability at all. It is not
a shorthand for "everything" — that would provision a diarizer and an aligner nobody
asked for — and it is not an error.

There is no batch surface. One invocation takes one input; whether the backend then
segments that input and batches the pieces is its own business, but it is *disclosed* in
the plan rather than hidden, because the partition determines both the cost and the
failure mode. A caller with a directory of interviews loops. A caller with a three-hour
file learns from the plan how it will be cut up and what happens if a piece fails.

Both deviate from Issue #1 §2.1, which writes bare `audio transcribe meeting.m4a`
as the simplest command. The deviation is deliberate.

| Exit | Meaning |
| --- | --- |
| 0 | Success. |
| 1 | Runtime or backend failure with nothing salvageable. No result is written. Distinct from a principled abstention, which is a successful run. |
| 2 | Request or validation error: missing stack or input, unknown capability, a capability the chosen stack cannot satisfy, an option the stack does not accept, a pin that conflicts with a requirement, absent word timing on export. |
| 3 | A required package is not provisioned, or a provisioned one failed its integrity check. Only `run` can return these. |
| 4 | Incomplete: some units were transcribed and some were not. A partial result **is** written, with a coverage ledger and a resume command. Only `run` can return this, and only on a stack whose work is partitioned. |

Every error payload carries `code` and `fix`. **`fix` is a runnable command wherever a
configuration exists that would work**, so a caller can copy one line and be right on the next
attempt; where nothing would work it is a sentence saying so and naming the nearest available
output. Emitting a plausible command that fails again is worse than admitting there is none.
The remaining fields are fixed per code, and this table is the contract — a payload with a
field not listed for its code, or missing one that is, is a defect:

| Code | Exit | Fields beyond `code` and `fix` |
| --- | --- | --- |
| `stack_required` | 2 | `field`, `allowed`, `stacks` (id → one-line characterization) |
| `input_required` | 2 | `field`, `note` |
| `capability_unknown` | 2 | `field`, `provided`, `available_on_stack`, `did_you_mean` when a near name exists |
| `option_unsupported_on_stack` | 2 | `field`, `provided`, `allowed` (empty), `stacks_accepting` |
| `capability_unsatisfiable_on_stack` | 2 | `capability`, `allowed` (non-empty), `available_on_stack` |
| `capability_unsupported` | 2 | `capability`, `allowed` (empty), `reason` |
| `pin_conflicts_with_native_capability` | 2 | `field`, `provided`, `allowed`, `capability` |
| `timing_required_for_format` | 2 | `field`, `provided`, `requires_capability`, `found`, `note` |
| `packages_not_provisioned` | 3 | `missing`, `total_known_download_bytes`, `unsized_packages` |
| `package_integrity_failed` | 3 | `failed` (package, check, expected, actual) |
| `backend_failed` | 1 | `role`, `backend`, `detail` |
| `run_incomplete` | 4 | `role`, `backend`, `detail`, `coverage`, `output` |

`allowed` means different things by code and is never a free-text field: the stack ids for
`stack_required`, the stacks that could serve the request for
`capability_unsatisfiable_on_stack`, and empty where switching stacks cannot help.
`capability_unknown` covers a name that is not in the namespace at all, which is distinct from
a name that is real but unsatisfiable here.

Both capability errors also carry **`available_on_stack`**, the full set of names this stack
will accept, split into `native`, `requires_add_on`, and `impossible`. A caller who got the
`--want` wrong needs to know what it can ask for *here*, not only which other stack would have
worked, and the split says which of those choices are free. It makes the error self-sufficient:
correcting a request should not require a second command to find the menu.

Three unrelated things were previously all called `requires`. They are now
`packages` (the plan's provisioning list), `requires_tool` (an external toolchain a
package needs, such as `swift`), and `requires_capability` (the capability an
export format needs).

Machine-readable output goes to stdout; human progress goes to stderr, so an
agent can pipe stdout safely. Note the distinction from a plan's `warnings`
array, which is a stdout field of the plan document, not a stderr message.

`run` defaults to `--format json` on stdout, matching the existing CLI's
machine-readable convention. `--format md|txt` are for human consumption.

`--language` is the one caller-settable model input, and it is a hint passed to the
ASR rather than a capability. Only the Qwen stacks accept it; `vibevoice` advertises
code switching without language selection, and FireRed's ASR takes no language
argument at all — its language is an output of the optional LID stage, not an input.
Each stack's `languages` catalog entry says whether it takes one, and passing the flag to a
stack that does not is `option_unsupported_on_stack` rather than a silently ignored
argument.

Every recorded interview figure was produced with `--language Cantonese`, so exposing
the flag is what makes those numbers reachable from the CLI at all; without it the
measured configuration would be one no caller could ask for. Omitting the flag is a
real second configuration, not a default: the model still emits a language label with
no hint, and the two Qwen sizes disagreed with each other on the same recording when
run that way, which is an argument for stating the language you know and against
trusting the label you get back.

## Two questions, two commands

Nothing sequences these. An earlier draft called them "step one" and "step two" and made both
forms of `plan`, distinguished by whether `--want` was present — which meant `plan` answered
two different questions depending on an absent argument, and implied an order the tool never
enforced. An agent that already knows what it needs should go straight to a plan; an agent that
does not needs somewhere to look. Those are different questions, so they are different
commands, and neither is a gate on the other.

- `audio transcribe capabilities --stack S --input F` — what can this stack do with this file,
  what would each addition cost, and what happens if the run dies.
- `audio transcribe plan --stack S --input F [--want ...]` — resolve this exact request: the
  backends that will run, the packages to provision, and the shape of the output.

Both require a stack and an input, neither reads media beyond a metadata probe, and neither
provisions anything. With `--want` omitted, `plan` resolves the floors-only request — a
punctuated transcript and nothing optional — which is a real request and the same meaning the
flag's absence has on `run`. It is not a request for the menu.

### `capabilities` — what can this stack do with this file?

The capability catalog is a static property of the stack, but the processing detail is not: the
backend reads the input's metadata, decides how it will partition the audio, and discloses that
along with what the run will cost and whether a failure leaves anything usable.

```bash
audio transcribe capabilities --stack firered --input field.wav
```

This is not an availability lookup. It is the only place an agent can find what it needs to
choose a request, so it carries the context that choice needs: what the stack
already produces and how precisely, what each add-on would cost in packages, time,
and memory, and what a recorded run showed about quality. An agent that only learns
`requires_add_on` cannot tell a free add-on from one that pulls a second runtime, and
an agent that only learns `native` cannot tell whether native is *good enough* for
what it is building. Both are request decisions, so both belong here.

The catalog uses its own axis, `availability`, because nothing has been requested
yet and `satisfaction` is defined only for a requested capability:

```json
{
  "stack": "firered",
  "family": "FireRedASR2S",
  "environment": "torch-firered",
  "roles": "vad, asr and punctuator always; lid as well when lid is requested",
  "input": {"path": "field.wav", "duration_seconds": 27.8, "container": "wav",
            "sample_rate_hz": 48000, "channels": 1},
  "processing": {
    "unit": "vad_region",
    "unit_count": null,
    "note": "FireRedVAD segments the audio and every later stage runs per region, four at a time. The count depends on speech activity, so it is not known until the VAD runs; a recorded 30-minute channel produced 55 regions."
  },
  "failure_recovery": {
    "partial_results": "per_unit",
    "note": "Regions are independent, so a failure leaves the finished ones usable and --range addresses the rest."
  },
  "cost": {
    "proved": "11 min 5 s and 9.1 GiB peak to transcribe a 30-minute sample on an M4 Max, CPU float32 with LID off and batch size 4. Peak is dominated by model weights rather than duration, so it does not shrink with a shorter input.",
    "projected_seconds": 10.3
  },
  "capabilities": {
    "languages": {"availability": "native",
                  "note": "Advertises Mandarin, English, code-switching and 20+ Chinese dialects, with the VAD and LID stages claiming 100+ languages; only Mandarin, English and Cantonese have actually been run here. The one accuracy figure is Cantonese at 49.02% mixed-token error on a 30-minute participant channel, and dialect breadth is untested beyond two lexemes. Takes no --language hint: on this stack language is an LID output, not an input."
                  },
    "verbatim": {"availability": "native",
                 "note": "Emits disfluencies rather than cleaning them, and retained the dialect form on both probed clips. No stack has a measured filler recall, and two lexemes cannot rank varieties."
                 },
    "word_timestamps": {"availability": "native",
                        "note": "Monotonic across the 30- and 60-minute runs, but never scored against hand-labelled boundaries. Neither is the forced aligner, so switching stacks for timing accuracy would trade one unmeasured number for another."},
    "vad": {"availability": "native",
            "note": "FireRedVAD, with no measured accuracy. The Silero add-on has a measured 0.8505 frame-level F1 on one fixture, so --vad silero-vad buys evidence rather than quality; pick it if you need a number."},
    "segment_timestamps": {"availability": "native",
                           "note": "Sentence extents from the punctuation stage. Unmeasured against labels."},
    "lid": {"availability": "native",
            "note": "One label per VAD region, copied onto every sentence inside it, so per-sentence variation would be fabricated. Adds 78 s on a 139-second sample — 162 s with the stage against 84 s without — and about 16 s on this input. Its weights are fetched only when this capability is requested and their size is unrecorded."},
    "diarization": {"availability": "requires_add_on",
                    "note": "Adds FluidAudio, which needs a Swift toolchain and a second environment: 15 s and 0.55 GiB peak on a 30-minute sample, and that same run also serves overlapped_speech. Produces speaker labels on the text and the turn intervals together, mapped onto the transcript by an exact partition of the timeline so no span is transcribed twice and no gap is invented. Measured 95.42% participant-interval F1 on a 30-minute interview but matched only 3 of 75 annotated speaker changes on a dense two-speaker conversation — strong on long turns, unsuitable where turns are short or overlapping. RSS excludes memory held by system Core ML services."
                    },
    "overlapped_speech": {"availability": "requires_add_on",
                          "note": "Comes out of the same FluidAudio run as diarization, so asking for both costs one stage. Unmeasured. Without it nothing in the plan detects overlap, so an empty abstention ledger means undetected rather than absent."},
    "token_lid": {"availability": "impossible", "reason": "no_backend_declares",
                  "note": "Named only so a request fails loudly. Code-switching support does not imply per-token labels, and no backend here produces them."}
  },
  "next": "audio transcribe plan --input field.wav --stack firered --want <capabilities>"
}
```

`processing` and `failure_recovery` are why `--input` is required rather than optional.
Neither is derivable from the stack: the unit a stack works in is a stack property, but
how many units *this* file yields, what it will cost, and whether a failure is survivable
are properties of the pair. `unit_count_known_at_plan_time: false` is the honest answer
wherever the partition depends on content the plan has not decoded — VAD regions here,
diarized turns on a Qwen plan that requests `diarization` — and it is stated
rather than guessed, because a fabricated count is worse than an absent one.

`failure_recovery` varies by stack and is the field to read before committing to a long
file. It is `per_unit` here and on Qwen; it is **`none` on `vibevoice`**, which is handed
whole media in a single `generate` call, so there is no partition to salvage and a failure
at minute forty of a forty-one-minute run yields nothing. See §5.1.

Everything in that document is there because a caller acts on it, and almost none of it is
structured. Three enums carry the decisions a program branches on — `availability`,
`processing.unit`, and `failure_recovery.partial_results` — plus two numbers, the unit count
and the projected seconds. Everything else is a sentence.

That is deliberate, and it is a correction. Earlier drafts gave each capability an
`evidence` object, a `cost` object with seven keys, a `timing_precision` object holding one
null and one string, an `alternative` object wrapping one sentence, and `measured_limit`
beside `observed_limit` beside `note` beside `interface_basis`. None of that nesting had a
consumer: no caller branches on `interface: "verified"`, and a reader who wants to know
whether a stack suits them reads the sentence either way. Structure that no one dispatches
on is ceremony, and it makes the payload longer to read and easier to get subtly wrong —
which it did, repeatedly, in this document's own history.

What the sentences must still do is what the retired objects were built to enforce. Say
what was measured and on what, say when a run *refuted* rather than merely failed to
measure something, and state the consequence rather than the forensics: "matched only 3 of
75 annotated speaker changes on a dense two-speaker conversation — strong on long turns,
unsuitable where turns are short or overlapping" is actionable, while the fixture, the collar,
and the diarizer preset are not — they live in the research record under `model_tests/`, which
is where an auditor looks and which will not exist in a shipped tool's output. Two facts here
are worth
noticing because they are the sort a structured field would have hidden: FireRed's native
word timing has never been scored against hand-labelled boundaries and neither has the
aligner, so switching stacks for timing accuracy trades one unmeasured number for another;
and `vad` is the inverse case, where the add-on has a measured figure and the native stage
has none.

A plan keeps its structure, because that document *is* dispatched on: `roles`
with backends, revisions, and configuration is audited provenance, and `satisfaction`,
`outcome`, and `evidence` are asserted by tests. A catalog is read to make a choice; a plan
is read by a machine that has to reproduce a run.

There is no provenance-only section, and there used to be. It carried the extents of
Qwen's processing units and the single language label read off its output scaffold, on the
argument that publishing them let them be audited and refused as a timing source. Neither
earned the space: a caller can act on neither, the extents are not speech timing and are
not published as any kind, and the label is not a detector. Removing them also removed a
refusal code, since a capability that does not exist cannot be a capability that exists
but cannot be asked for. Discovering that `token_lid` is `impossible` is what `capabilities` is for; *requesting* it is
an error, not a field (see Refusals).

`roles` is one sentence rather than two arrays. FireRedLID is inside the stack but runs only
when `lid` is requested, so listing it unconditionally would promise weights this request
will not fetch while calling it an add-on would promise a package the stack already
contains. Saying "lid as well when lid is requested" is both shorter and truer than a pair
of keys.

Two things FireRed emits that this catalog deliberately does not offer.
`asr_confidence` is real but *sentence*-level, and there is no requestable
capability at that granularity in v1 — see the retired `word_confidence` entry in
[VOCABULARY.md](VOCABULARY.md), which was the wrong name for it. And `lang` /
`lang_confidence` appear on every sentence in the raw output even when LID never
ran, defaulted to `null` and `0`; the adapter drops both rather than publishing a
zero confidence that reads as measured.

### `plan` — what will this request produce?

Adds the resolved roles, the packages to provision, and a `sample_output` block.

### How `sample_output` is produced, and what it guarantees

The sample is built by populating the real result object with one placeholder
entity per requested capability — plus the floor artifacts every conforming run
carries, such as the abstention ledger — and serializing it through **the same
serializer `run` uses**. It is never a hand-written example, and there is no second code
path. Consequently the output shape is a pure function of the resolved capability
set, so all combinations are generated on demand rather than enumerated.

Guaranteed identical to a real run: key sets, nesting, types, which fields are
absent because a capability was not requested, which are `native` versus
`derived`, and the provenance structure.

Not predictable, and therefore not claimed: cardinality of segments, words, and
turns; whether the abstention ledger is populated, since that depends on the audio
actually containing overlap; whether a given segment has a word stream at all,
since a segment with no speech to align has none — the forced-aligner artifact has
two, both VibeVoice non-speech event tags such as `[Environmental Sounds]`; and
output quality wherever a capability's `evidence.quality` is `unmeasured`. The
sample is the contract for a successful run; a backend failure exits 1 and writes
nothing.

One field a run adds that a plan does not have: each capability in the embedded
provenance gains an `outcome` of `produced` or `abstained`. That is the only
difference between the two documents, which is why the key-set test must compare
against a real run's provenance rather than the elided placeholder printed below.

Placeholder timing and text values are `null`, never `0.0` or a plausible
string. `0.0` is a legal timestamp and would violate the `no_synthesized_bounds`
floor the moment a consumer read it as measured. Real metadata that the plan
genuinely has — duration, path — is populated rather than stubbed.

Enum-valued fields are the one exception: they show one legal member rather than
`null`, so a consumer can see the field is categorical. So `"reason": "overlap"` is
shape, while `"text": null` is content withheld. Free-text and numeric fields are
always `null`. One member is not the member set, so the sample is not where a
consumer learns it: the plan warns when nothing in it can detect overlap, which is the
case where the ledger cannot fill at all.

The binding test is that the sample's key set equals a real run's key set, and that
no key exists for a capability that was not requested. That is the anti-fabrication
guarantee, and it is parametrized over the derivation table: each stack against each
capability it satisfies natively, each capability requiring an add-on, and each of
both refusal codes, which are distinct and must not be collapsed into one
"unsatisfiable" case.

## 0. Once per machine

```bash
brew install ffmpeg
uv tool install .
audio --help
audio doctor
```

`audio doctor` reports tool version and path, ffmpeg/ffprobe/swift/uv presence,
platform, total and available memory, available disk, the resolved root, and
per-environment and per-package state. Absent `swift` is reported, not fatal: it
blocks only the packages that need it.

## 1. Interview — `qwen-1.7b`

Fast long-form transcript with anonymous speaker attribution.

### 1.1 Plan before committing to anything

```bash
audio transcribe plan --input meeting.m4a \
  --stack qwen-1.7b \
  --want diarization
```

Exits 0 whether or not anything is provisioned:

```json
{
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "asr":        {"backend": "qwen3-asr-1.7b-8bit", "environment": "mlx",
                   "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
                   "config": {"batch_size": 1, "clear_mlx_cache_after_every_batch": true,
                              "language": null,
                              "api_path": "_generate_chunks_batched"},
                   "adapter_strips": ["language <label><asr_text> scaffold"],
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 0.0,
                   "determinism_basis": "sampler=make_sampler(temp=0.0), i.e. argmax decode (run_turn_attributed_mlx_asr.py:661); back-to-back calls in one process produced byte-identical text, cross-process repetition untested"
                   },
    "diarizer":   {"backend": "fluidaudio", "version": "0.15.5",
                   "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
                   "environment": "swift",
                   "config": {"preset": "quality", "step_ratio": 0.1,
                              "min_segment_duration": 0.0, "output": "regular",
                              "threshold": 0.6, "num_speakers": 2},
                   "config_note": "every cited diarization measurement used a known two-speaker prior; num_speakers must be supplied or the measured_limit figures do not apply",
                   "selected_by": "add_on_required_by:diarization"}
  },
  "execution": {
    "stage_order": ["decode", "diarizer", "asr"],
    "residency": "one_model_stage_at_a_time",
    "note": "stages run strictly sequentially and no two model stages are resident together; wall time adds across stages, peak memory does not, and the per-stage peaks below must not be summed"
  },
  "capabilities": {
    "diarization": {"satisfaction": "derived", "backend": "fluidaudio",
                    "evidence": {"interface": "verified", "quality": "measured"},
                    "note": "Anonymous labels reconciled sample-exactly onto the ASR text, plus the diarizer's turn intervals. This preset matched 3 of 75 annotated speaker changes on a dense conversation and is not validated for rapid backchannels, interruptions, or dense overlap."
                    }
  },
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "kind": "weights",
     "bytes": 2463307541, "provisioned": false},
    {"package": "fluidaudio", "environment": "swift", "kind": "toolchain",
     "requires_tool": ["swift"], "bytes": null, "provisioned": false},
    {"package": "speaker-diarization-coreml", "environment": "swift", "kind": "weights",
     "bytes": null, "provisioned": false}
  ],
  "total_known_download_bytes": 2463307541,
  "unsized_packages": ["fluidaudio", "speaker-diarization-coreml"],
  "warnings": [],
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "source": {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "speaker": null}
    ],
    "turns": [
      {"turn_id": "turn_0", "speaker": null, "start": null, "end": null}
    ],
    "abstentions": [
      {"abstention_id": "ab_0", "reason": "overlap", "start": null, "end": null}
    ],
    "provenance": "<stack, outcomes, observed, and the executed plan; elided in print>"
  }
}
```

`turns` is present because `diarization` was requested, and the two arrive together: the
speaker labels land on `segments[].speaker` and the intervals the diarizer measured land in
`turns`. They are separate arrays because they are separate measurements at separate
granularities — a turn can span several sentences — and neither is derived from the other.

Two things the sample deliberately does *not* contain. Segments carry no `words` array,
because `word_timestamps` was not requested. And they carry no `start` or `end`: those are
`segment_timestamps`, which is exit-2 on this stack, since Qwen emits no segment extents and
the chunk boundaries it works in are not speech timing. A segment exists as a floor artifact;
its time extents do not come free with it. Keys are absent rather than null-valued at the
container level, so absence can never read as a measured value.

`provenance` is shown as a string here purely to keep the example readable. In a
real result it is the full executed plan, and the key-set test must compare against
that, not against this placeholder.

The `abstentions` ledger *is* present, and not as an exception to that rule: it is
a floor artifact rather than a requested capability, so the one-placeholder-per-request
rule does not reach it. It is also
genuinely producible from this exact request, since FluidAudio emits
overlap-permitting output. Whether it fills depends on the audio.

`duration_seconds` is populated because `plan` reads container metadata rather than
stubbing what it already knows. The value `1794.2` is illustrative — this document
has no real `meeting.m4a` — and is deliberately not the 30-minute reference fixture behind
the `capabilities` report's `cost.proved`, which describes a recorded run rather than this input.

There is no `measured` block here, and there was one. It restated the `capabilities` report's timing and
memory figures inside every plan, which duplicated the one place those figures belong now
that a plan cannot be reached without a stack and an input. The ASR stage dominates the cost
on every stack, and the `capabilities` report's `cost.proved` already says what that stage did on a named
sample.

`api_path` is in the config for a reason that is not a detail. Qwen has two decode entry
points and they do not produce the same text: on identical input, weights, and greedy
settings, the public `generate()` path and the private `_generate_chunks_batched` path agreed
on **every word** and differed by **two Chinese commas**, at 242 versus 240 generated tokens.
Lexical evidence therefore transfers between the paths and punctuation does not — and
punctuation is what cue splitting breaks on, so a plan has to say which path ran.

`adapter_strips` records a live instance of the adapter-normalization floor. The private
batched API returns the model's own scaffold inside its text — the observed prefix is
literally `language English<asr_text>` — where the public path strips it. An adapter on the
batched path that forgets to do the same produces a transcript beginning with the scaffold.
The scaffold also carries the model's own language guess, which is where the `languages`
entry's warning against trusting it comes from: on one Mandarin-majority clip 1.7B read
English and 0.6B read Chinese. It is stripped with the rest and never published.

`execution` states what every recorded figure in this document already assumed and no
earlier draft declared. The orchestrator that produced the end-to-end interview
measurement ran *strictly sequential fresh subprocesses*, and its record says in as many
words that the stages did not overlap. Publishing per-stage memory peaks without
declaring that is how a reader ends up summing them. Two consequences worth stating
outright: a request spanning several provisioned environments costs the sum of the
stage walls and the maximum of the stage peaks, not the sum of both; and strict
sequencing is load-bearing for the memory story rather than an implementation detail,
because `vibevoice` at 20.28 GiB and the aligner have never been measured resident at
the same time and nothing here should imply they can be.

### 1.2 What `run` does with packages absent

```bash
audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization
```

Exit 3. Nothing computed, nothing downloaded, stderr:

```json
{
  "code": "packages_not_provisioned",
  "missing": [
    {"package": "qwen3-asr-1.7b-8bit", "kind": "weights", "bytes": 2463307541},
    {"package": "fluidaudio", "kind": "toolchain", "requires_tool": ["swift"], "bytes": null},
    {"package": "speaker-diarization-coreml", "kind": "weights", "bytes": null}
  ],
  "total_known_download_bytes": 2463307541,
  "unsized_packages": ["fluidaudio", "speaker-diarization-coreml"],
  "fix": "audio packages pull --stack qwen-1.7b"
}
```

### 1.3 Provision, verify, execute

```bash
audio packages pull --stack qwen-1.7b
audio packages verify
audio packages list
audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization \
  --format md
audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization \
  --format json -o meeting.transcript.json
```

The JSON result carries the executed plan as `provenance`, anonymous speaker
labels, and the overlap abstention ledger. Anonymous labels are never mapped to a
person or a role.

### 1.4 Adding timing, and what it costs

Qwen has no native word timing, so `word_timestamps` forces the aligner. It does **not** force
another environment: the aligner runs in `mlx`, the same environment the ASR is already in, so
the cost is 1.19 GiB of weights and one more stage rather than a second runtime. The request
spans `mlx` and `swift`.

That was not true when this document was written, and it is worth saying why it changed. The
aligner was measured through `qwen-asr`, a PyTorch package, and the same alignment is available
from `mlx-audio` — the library the Qwen stacks already pin. Rerunning the recorded case through
it produced identical token text on all 246 aligned tokens
(`model_tests/benchmark/results/2026-08-17-mlx-collapse-probes.json`), so the move costs no
transcript change; bounds agree to a median of 0 s with a P95 of 80 ms, and boundary accuracy
is unmeasured against labels on both paths, which is the same caveat `evidence.quality` already
carries. [ENVIRONMENTS.md](ENVIRONMENTS.md) holds the layout.

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b \
  --want diarization,word_timestamps,overlapped_speech,vad
audio packages pull --stack qwen-1.7b
audio transcribe run --input meeting.m4a --stack qwen-1.7b \
  --want diarization,word_timestamps,overlapped_speech,vad \
  --format json -o meeting.timed.json
```

`word_timestamps` arrives `derived` with `evidence.quality: "unmeasured"`, because
boundary MAE/P95 has no labels.

The roles and packages this adds to §1.1 — two of the four roles not yet shown, plus
the one package that auto-fetches:

```json
{
  "roles": {
    "vad":     {"backend": "silero-vad", "environment": "core",
                "version": "silero-vad-6.2.1",
                "config": {"threshold": 0.5, "exit_threshold": 0.35,
                           "min_speech_ms": 100, "min_silence_ms": 300,
                           "speech_pad_ms": 120},
                "selected_by": "add_on_required_by:vad"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "mlx",
                "config": {"scope": "all_segments"},
                "selected_by": "add_on_required_by:word_timestamps"}
  },
  "capabilities": {
    "vad": {"satisfaction": "derived", "backend": "silero-vad",
            "evidence": {"interface": "verified", "quality": "measured"},
            "note": "0.8505 frame-level F1 at 0.7655 precision and 0.9567 recall, against the union of 83 annotated utterance intervals on one 150-second fixture. That is an activity gate for this exact five-value configuration and says nothing about language coverage, turns, overlap, or chained VAD-plus-ASR behaviour."
            }
  },
  "packages": [
    {"package": "silero-vad", "environment": "core", "kind": "weights",
     "bytes": null, "provisioned": true, "auto_fetch": true,
     "note": "hash-pinned single file; fetched on first use, so it never returns exit 3"},
    {"package": "qwen3-forcedaligner", "environment": "mlx", "kind": "weights",
     "bytes": null, "provisioned": false}
  ]
}
```

`vad` is the one derived capability with a real number behind it, so the
whole recorded configuration is declared rather than just the threshold: precision is
0.77 against a union-of-speakers reference, which means the gate over-includes, and
that is a property of these five values together. Declaring one of them and citing the
F1 of all five is how a measured figure ends up attributed to a configuration that
never produced it.

`silero-vad` is the one package that can be `provisioned: true` on a machine that
never ran `pull`, because it is the small hash-pinned artifact the existing Silero
backend already fetches on demand. Everything else fails closed.

`--vad` selects among the implementations offered as add-ons: `silero-vad` today, plus
FluidAudio's Core ML VAD once that ships as a package. FireRed's own VAD is inside its
stack and is not offered to other stacks. That makes `vad` the only role with a real
choice, which is why the pin is defined there while `--diarizer` remains a forced
single implementation with a pin reserved for later ones.

### 1.5 Smaller stack, named directly

`qwen-0.6b` is a separate package from `qwen-1.7b`, so it needs its own pull —
without it this exits 3:

```bash
audio packages pull --stack qwen-0.6b
audio transcribe run --input meeting.m4a --stack qwen-0.6b --want diarization \
  --language Cantonese --format md
```

Measured on the 30-minute Cantonese SpiCE fixture, Apple M4 Max / 64 GiB, batch 1
with the MLX cache cleared after every batch: `qwen-0.6b` ran 29.90 s at 1.66 GiB RSS
versus `qwen-1.7b` at 53.77 s and 3.02 GiB, and scored 52.64% mixed-token error
versus 33.56%. Those are ASR-stage walls on identical diarized turns, and the accuracy
comparison is Cantonese-only.

`--language Cantonese` is on that command deliberately: it is the configuration those
figures were measured under, and the flag exists so a caller who knows the language
can state it instead of leaving the model to guess. The plan echoes it as
`roles.asr.config.language`, so the executed provenance records which of the two
configurations ran. Nothing else in the plan changes — a hint is an input to the ASR,
not a capability, so it adds no role, no package, and no output field.

`qwen-0.6b` is not simply a smaller `qwen-1.7b`. On the 139.284-second probe it
rendered `刷啥子` where 1.7B retained `耍啥子`, so its `verbatim` catalog entry carries
`quality: "refuted"` with that `observed_limit` while 1.7B's carries `unmeasured`.
Identical resolution, different recorded fidelity — which is why the two share a
column in the derivation table and not a catalog. Their filler retention is identical
at 26 hits, so the interface half of `verbatim` is the same on both; only the dialect
half separates them.

## 2. Product-demo editing — `vibevoice`

Verbatim-oriented text with native anonymous speaker structure and word
intervals for an editing agent.

```bash
audio transcribe plan --input demo.mp4 --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps
```

`diarization` and `segment_timestamps` are `native`, so this stack needs no
diarizer at all. Only `word_timestamps` adds the aligner. **Abridged to the
fields that differ from §1.1** — the envelope, `packages`, and
`sample_output` all take the same shape.

```json
{
  "roles": {
    "decode":  {"backend": "ffmpeg",
                "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "asr":     {"backend": "vibevoice-asr-7b", "environment": "torch-vibevoice",
                "revision": "d0c9efdb8d614685062c04425d91e01b6f37d944",
                "source_commit": "94da20d98b2fa7688e9cbfaf7692ddb4954f7600",
                "patch": "vibevoice-logits-to-keep",
                "config": {"device": "mps", "dtype": "bfloat16", "attention": "sdpa",
                           "seed": 1234},
                "deterministic": true,
                "determinism_tolerance_ms": 0.0,
                "determinism_basis": "three seeded repeats shared one normalized-output hash; text decode is do_sample=False (run_vibevoice.py:267)",
                "determinism_note": "acoustic tokenizer samples a Gaussian latent; fixed seed required",
                "selected_by": "stack"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "mlx",
                "config": {"scope": "all_segments"},
                "selected_by": "add_on_required_by:word_timestamps"}
  },
  "capabilities": {
    "verbatim":           {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "refuted"},
                           "note": "Emits disfluencies rather than cleaning them — 28 filler hits on the probe, the highest of the four stacks — but a recorded run refuted dialect preservation twice: 看哈 became 看一下 and 耍啥子 became 刷啥子, both retained by firered."
                           },
    "diarization":        {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "measured"},
                           "note": "Native speaker labels and segment bounds; turns group adjacent same-speaker segments, so no bound is synthesized. Matched 39 of 75 annotated speaker changes on a dense conversation and is not validated for rapid backchannels, interruptions, or dense overlap."
                           },
    "segment_timestamps": {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "word_timestamps":    {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                           "evidence": {"interface": "verified", "quality": "unmeasured"},
                           "note": "Boundary error against labels is unmeasured, and absent on any segment with no speech to align."}
  },
  "unsized_packages": ["vibevoice-asr-7b", "qwen3-forcedaligner"],
  "warnings": [
    {"code": "measured_peak_exceeds_target", "blocking": false,
     "detail": "measured 20.28 GiB live MPS allocation on spice-30min-participant; a strict 16 GiB MPS cap OOMs at model load, measured on a 27.8 s probe, while an 18 GiB cap passed that probe"}
  ]
}
```

```bash
audio packages pull --stack vibevoice
audio packages verify
audio transcribe run --input demo.mp4 --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps \
  --format json -o demo.transcript.json
```

`verbatim` is the reason this stack's catalog is worth reading before choosing it, and
it is worth being precise about what the capability claims. It claims the stack **can
produce** verbatim text — that it emits what it heard rather than a cleaned rendering —
and that is now measured rather than assumed: 28 filler hits here, 26 on both Qwen
sizes, 24 on FireRed, no stack cleaning and no stack complete. Accuracy is a separate
story, carried by `quality`. So `verbatim` resolves `native` here exactly as it does on
`firered`, and the entire difference is that a recorded run *refuted* the quality half
twice, on two clips and two lexemes. `quality: "refuted"` with an `observed_limit` is
not the same statement as `unmeasured`; filing the normalization as unmeasured would
have made the stack that failed the probe read like the stack that was never tested.

Nothing selects this. No backend exposes a verbatim switch — four verbatim-requesting
system prompts left Qwen's output byte-identical to its unprompted baseline — and
nothing in v1 cleans, so the request asserts an interface and the plan answers for
fidelity. It is the one requestable capability that never changes plan composition, by
design rather than by oversight.

Two adapter obligations this stack creates, both from its recorded output.
VibeVoice emits `Speaker: "N/A"` on non-speech segments; that is the absence of a
label, so the adapter emits no speaker rather than a speaker whose id is `"N/A"`.
And it emits bracketed non-speech event tags such as `[Environmental Sounds]` as
segment `text`. Those segments are real segments with real bounds and no words, so
they survive into the transcript and `export` decides whether to render them —
which is a subtitle convention question, parked in issue #10, not a transcription
one.

The memory warning is advisory by explicit product decision, and it is emitted
from the plan rather than as a mid-run OOM. Its reference run took roughly
fourteen minutes of generation for thirty minutes of audio — an RTF near 0.47,
which is the figure to scale by; the plan cannot know `demo.mp4`'s duration cost
in advance. Cut and rerender from the original media; this command only reads it.

Neither `vibevoice-asr-7b` nor `qwen3-forcedaligner` has a byte size from the
reproducible harness, so both appear in `unsized_packages`. What exists is a
pre-harness table of disk notes — `~17 GB` and `~1.8 GB` respectively — plus a
16.157 GiB BF16 weight floor. That table's own document marks it as history rather
than decision evidence, and none of its figures is a provisioning measurement, but
it is the same table this contract cites for FireRed, so it should not be described
as nonexistent here and authoritative there.

## 3. Dialect and audit — `firered`

The only stack with native word timing, native speech bounds, and a region language
label.

```bash
audio transcribe plan --input field.wav --stack firered \
  --want verbatim,word_timestamps,vad,segment_timestamps
```

Every requirement is `native`, so there are no add-ons at all — and this is the only
plan in this document that resolves `punctuator`, and the only one whose `vad` comes
from inside the stack rather than as an add-on. Abridged to `roles` and `execution`:

```json
{
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "vad":        {"backend": "firered-vad", "environment": "torch-firered",
                   "selected_by": "stack"},
    "asr":        {"backend": "firered-asr2-aed", "environment": "torch-firered",
                   "config": {"device": "cpu", "dtype": "float32", "batch_size": 4,
                              "return_timestamp": true},
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 1.0,
                   "determinism_basis": "exact-repeat 60-minute fixture: both halves' text sequences equal the standalone 30-minute run, maximum rebased timestamp drift 1.0 ms within a declared 2.0 ms tolerance; normalized segments are therefore not byte-equal"},
    "punctuator": {"backend": "firered-punc", "environment": "torch-firered",
                   "config": {"batch_size": 4},
                   "selected_by": "floor:punctuated_sentence_segmented_text",
                   "recases_text": true}
  }
}
```

`punctuator` is the one role in any plan selected by a floor rather than by the
stack or a requirement. It is not optional and not requestable: floor one requires
punctuated, sentence-segmented text, and on this stack that means FireRedPunc always
runs.

`determinism_tolerance_ms` is why `deterministic: true` means something here.
`0.0` claims byte-identical normalized output on repeat, which is what VibeVoice's
three seeded repeats measured. FireRed declares `1.0` because its recorded
exact-repeat run reproduced text exactly and timestamps only to within a
millisecond, so its normalized segments are *not* byte-equal. A single boolean would
have had to either overclaim that or discard a real result; a downstream `word_id`
scheme has to know which.

```bash
audio packages pull --stack firered
audio packages verify
audio transcribe run --input field.wav --stack firered \
  --want verbatim,word_timestamps,vad,segment_timestamps \
  --format json -o field.transcript.json
```

`firered-asr2s` is one package pinning four repositories and `pull` materializes all of
them, LID weights included, whatever the plan asked for: narrowing a pull to the roles a
plan actually uses is what `--want` is reserved for, and `pull` refuses that flag today
rather than appearing to honour it. Neither the whole-package figure nor a narrowed one is
recorded in a tracked artifact — the only tracked source is a pre-harness "~9.2 GB" note that
its own document marks as history rather than decision evidence — so both appear
as `approximate, unrecorded` until per-artifact sizes are recorded the way the
MLX runs record `weight_bytes`.

### 3.1 What the punctuation floor actually requires here

This is the stack `punctuation_is_sentence_level` is aimed at, and it is worth being
exact about why, because the earlier draft of this floor named a risk FireRed does
not have and prescribed a rule that could never fire.

FireRedPunc does not emit marks with their own bounds. It returns punctuated
*sentence* strings with *sentence* bounds (`fireredpunc/punc.py:109-119`), while
`words` is built separately from the pre-punctuation AED timestamps
(`fireredasr2system.py:181-184`). There is no parallel per-mark stream, so there is
nothing to strip bounds from: measured on the recorded artifacts, 0 of 379 and 0 of
246 word tokens carry a sentence mark, and the only punctuation that appears inside
any word token across all 12,370 recorded words is the apostrophe in 20 English
contractions, which the ASR itself produced.

What the adapter must actually guarantee is the invariant cue splitting depends on:
stripping punctuation and whitespace from a sentence's `text` yields exactly the
concatenation of its word `text` values, compared case-insensitively. Case-insensitively,
because `RuleBaedTxtFix.fix` lowercases the ASR text and then re-capitalizes sentence
starts and standalone `i` (`fireredpunc/punc.py:349-382`) — 234 characters differ by
case across the recorded artifacts, which is why the role above declares
`recases_text: true`. Sentence text carries the marks and the casing; the word stream
carries the bounds; neither is derivable from the other, and the sentence text is
canonical for reading and for subtitles.

Adding the region language label pulls LID and roughly doubles inference: 162.09
versus 84.24 seconds on the 139.284-second probe, CPU float32 at batch size 4,
with identical ASR text and all 246 word texts and times in both runs. The label is
produced once per VAD region and copied onto every sentence in that region, so a
consumer reading per-sentence `lang` as per-sentence detection would be reading
variation that was never measured.

```bash
audio transcribe plan --input field.wav --stack firered \
  --want verbatim,word_timestamps,lid
audio packages pull --stack firered
audio transcribe run --input field.wav --stack firered \
  --want verbatim,word_timestamps,lid --format json -o field.lid.json
```

That plan adds the eighth and last role, and it is the one case where a requirement
turns on a stage the stack already contains rather than adding a package:

```json
{
  "roles": {
    "lid": {"backend": "firered-lid", "environment": "torch-firered",
            "config": {"batch_size": 4},
            "selected_by": "requirement:lid",
            "granularity": "vad_region",
            "cost_note": "162.09 s with LID versus 84.24 s without, on the 139.284 s probe"}
  },
  "capabilities": {
    "lid": {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "one label per VAD region, copied onto the sentences inside it; no per-sentence detection was measured"}
  }
}
```

Note `selected_by: "requirement:lid"` rather than
`add_on_required_by:lid`. Nothing was added — the `lid` role is declared
by the stack and named as conditional in the `capabilities` report's `roles` sentence. The five
`selected_by` forms are `stack`, `requirement:<capability>`,
`add_on_required_by:<capability>`, `floor:<floor>`, and `pin:<flag>`. `decode` is the
one role that carries no `selected_by` at all, because it is unconditional; the field
exists to explain why a role that could have been absent is present.

FireRed has no speaker output, so speaker attribution here is an add-on like it
is on Qwen. Note the `pull` — entering at this section without it exits 3, since
nothing earlier in §3 provisioned a diarizer:

```bash
audio packages pull --stack firered
audio transcribe run --input interview.wav --stack firered \
  --want verbatim,word_timestamps,diarization,overlapped_speech \
  --format json -o interview.firered.json
```

## 4. Export

Deterministic post-processing. No stack, no packages, no `plan`/`run` split.

```bash
audio export --input meeting.timed.json --format srt -o meeting.srt
audio export --input meeting.timed.json --format vtt -o meeting.vtt
audio export --input meeting.transcript.json --format md
audio export --input meeting.transcript.json --format txt
audio export --input meeting.transcript.json --format jsonl
```

Subtitle formats require word timing and refuse without it.
`meeting.transcript.json` from §1.3 has none, so:

```bash
audio export --input meeting.transcript.json --format srt
```

Exit 2, stderr:

```json
{
  "code": "timing_required_for_format",
  "field": "--format",
  "provided": "srt",
  "requires_capability": "word_timestamps",
  "found": [],
  "note": "container bounds are processing extents, not cue timing",
  "fix": "audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization,word_timestamps --format json -o meeting.timed.json"
}
```

`md` and `txt` are for people. `jsonl` is one segment object per line, ordered by
start time — the same segment objects the JSON result carries, without the envelope
or the provenance — so a consumer can stream or `grep` a long transcript without
parsing the whole document. It has no timing requirement, and because it drops the
provenance it is an export for reading, not an artifact to audit against.

VTT carries speaker labels as voice tags when `diarization` is present,
which is a commitment to VTT as a real format rather than SRT with dots. Cue
segmentation is deterministic and belongs here; its parameters, break-priority
order, and millisecond-quantization invariants are specified in issue #10, and v1
may ship them hard-coded.

Two things issue #10 must handle that only became visible from the recorded
artifacts. Breaking at punctuation means locating a mark in the sentence text and
mapping it to a word index, which is sound only under the punctuation floor's
invariant and only case-insensitively, because FireRed's punctuator recases. And a
segment may carry text with no word stream — VibeVoice's non-speech event tags — so
the splitter needs a rule for those rather than assuming every segment yields cues.

Timing quality is not yet validated: boundary MAE/P95 is unmeasured for both
FireRed's native times and the aligner, so these files are producible but not yet
claimed broadcast-acceptable.

## 5. Refusals

Each of these is shown as a caller sees it, with the corrected command, in
[TRANSCRIBE_HAPPY_PATH.md](TRANSCRIBE_HAPPY_PATH.md) §4.

```bash
audio transcribe run --input meeting.m4a --want diarization
```

Exit 2, `code: "stack_required"`, `field: "--stack"`, `allowed` listing the four
stack ids, and `stacks` mapping each to a one-line characterization plus a pointer to
the decision report.

```bash
audio transcribe plan --stack qwen-1.7b --want diarization
```

Exit 2, `code: "input_required"`, `field: "--input"`, and a `note` saying why a stack
alone cannot be planned: the partition, the unit count, the projected cost, and whether a
failure is recoverable are all properties of the input. Reported by `plan`, not only by
`run`, because `plan` is the command whose whole job is to answer those questions.

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want segment_timestamps
```

Exit 2, `code: "capability_unsatisfiable_on_stack"`, `capability: "segment_timestamps"`,
`allowed: ["vibevoice", "firered"]`, plus `available_on_stack` listing everything Qwen would
accept instead. Reported by `plan` as well as `run`, before
anything loads, and the fix is actionable: switch stacks. Qwen's only time-like
output is the processing container, and promoting that to a segment extent is the
fabrication this code exists to prevent.

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want word_timing
```

Exit 2, `code: "capability_unknown"`, `field: "--want"`, `provided: "word_timing"`,
`did_you_mean: "word_timestamps"`, and `available_on_stack` listing every name this stack
accepts. A misspelling and an unsatisfiable
requirement are different failures: this one has no `capability` field, because the
name is not one.

```bash
audio transcribe plan --input demo.mp4 --stack vibevoice --want verbatim --language Cantonese
```

Exit 2, `code: "option_unsupported_on_stack"`, `field: "--language"`,
`provided: "Cantonese"`, `allowed: []`, `stacks_accepting: ["qwen-1.7b", "qwen-0.6b"]`.
VibeVoice takes no language argument, so the alternative to refusing is accepting a
flag that does nothing — which would let a caller believe it had constrained a decode
it had not touched.

```bash
audio transcribe plan --input meeting.m4a --stack firered --want token_lid
```

Exit 2, `code: "capability_unsupported"`, `capability: "token_lid"`,
`allowed: []`, `reason: "no_backend_declares"`. An impossible request is an error
rather than a silently `unavailable` field, so an agent gets an explicit answer
instead of assuming code-switching support implies per-token labels. `capabilities` is where
this is *discovered* without erroring.

```bash
audio transcribe plan --input meeting.m4a --stack vibevoice \
  --want diarization --diarizer fluidaudio
```

Exit 2, `code: "pin_conflicts_with_native_capability"`, `field: "--diarizer"`,
`provided: "fluidaudio"`, `allowed: []`, `capability: "diarization"`, because
`vibevoice` satisfies it natively and no diarizer role exists in this plan. Pins
select among implementations of a role the plan actually contains.

A backend crash is the one failure that is not a refusal:

```json
{
  "code": "backend_failed",
  "role": "asr",
  "backend": "vibevoice-asr-7b",
  "detail": "MPS backend out of memory during generate",
  "fix": "retry with --stack qwen-1.7b --want word_timestamps, or free memory; the plan's measured_peak_exceeds_target warning applies"
}
```

Exit 1, and no result is written. This must stay distinguishable from an abstention,
which is a *successful* run that declines to assert something: exit 0, a result, and
a ledger entry. Collapsing the two would make a crash and a principled refusal look
identical to a caller.

A corrupt package is a third thing again — provisioned, but not usable:

```json
{
  "code": "package_integrity_failed",
  "failed": [
    {"package": "qwen3-forcedaligner", "check": "weight_digest",
     "expected": "9f2c1d…", "actual": "4be0a7…"}
  ],
  "fix": "audio packages pull --repair qwen3-forcedaligner"
}
```

Exit 3, and nothing loads. `run` does not hash multi-gigabyte weights on every
invocation; it checks presence and the registry, so this surfaces either from an explicit
`audio packages verify` or from the cheap check catching a size or revision mismatch. A
corruption subtle enough to pass the cheap check fails at model load instead, which is
`backend_failed` at exit 1 with `fix` pointing at `audio packages verify` — the same
condition, found later, reported as what it looked like from where it was found.

### 5.1 Incomplete runs

Long-form transcription has partial-completion mechanisms that are real and recorded, not
hypothetical. Two are visible in the harness today:

- **The Qwen path carries a global generation budget** and stops when it runs out, keeping
  the turns it finished. The recorded runner tracks `input_turns`, `processed_turns`, and
  `unprocessed_turns`, and every recorded run has `unprocessed_turns: []` — but the
  60-minute stress run consumed 10,169 of 16,384 tokens, so at that token rate the budget
  exhausts somewhere near **1.6 hours** of comparable material. A three-hour interview
  runs into this before it runs into anything else.
- **VibeVoice has a single generation cap** and a `hit_max_new_tokens` detector, applied to
  one `generate` call over the whole file. Recorded runs set it between 1,024 and 16,384
  depending on fixture. Hitting it truncates the transcript.

Add out-of-memory to those — the product-demo route measured 20.28 GiB live MPS on thirty
minutes and OOMs at model load under a strict 16 GiB cap — and interruption, and a failure
in any one stage of a five-stage chain.

**What a partial run must leave behind.** On a stack whose work is partitioned, an
incomplete run writes its result and exits 4 rather than throwing the work away:

```json
{
  "code": "run_incomplete",
  "role": "asr",
  "backend": "qwen3-asr-1.7b-8bit",
  "detail": "global generation budget exhausted after 148 of 195 turns",
  "coverage": {
    "covered_through_seconds": 1402.88,
    "covered_fraction": 0.782,
    "covered_intervals": [[0.0, 1402.88]],
    "missing_intervals": [[1402.88, 1794.2]],
    "units_total": 195,
    "units_completed": 148
  },
  "output": "meeting.partial.json",
  "fix": "audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization,word_timestamps --language Cantonese --range 1402.88: -o meeting.rest.json"
}
```

Four properties that matter more than the shape.

`covered_through_seconds` is the end of the longest **contiguous prefix** that is fully
transcribed, which is the number an agent can act on without reasoning about gaps. It is
not the same as "the last unit that finished": the recorded runner processes turns in
duration-bucketed order and restores chronological order afterwards, so completion is not
contiguous in time. `covered_intervals` and `missing_intervals` therefore carry the exact
truth, and a resume that only honours the watermark is correct but may redo work the
ledger shows was already done.

`--range <start>[:<end>]` is the resume mechanism, and it exists so the agent does **not**
clip the audio. Clipping shifts the timeline, which means every bound in the second result
would need re-offsetting by hand before the two could be merged — arithmetic on
timestamps, performed by a consumer, which is exactly what the canonical-timeline floor
exists to prevent. With `--range`, bounds in the second result are already on the original
timeline and merging is concatenation.

The partial result is a **conforming result document** with `complete: false` and the same
`coverage` block, so every floor still holds inside it: no synthesized bounds, abstentions
survive, punctuation invariant intact for the units that ran. It is not a debug dump.

Ids are document-scoped, so merging two results means re-numbering. `export` accepts
several transcripts in timeline order and re-ids as it goes, which covers the subtitle
case without anyone hand-editing JSON:

```bash
audio export --input meeting.partial.json --input meeting.rest.json \
  --format srt -o meeting.srt
```

**On `vibevoice` none of this applies.** `failure_recovery.partial_results` is `none`
there, because the stack is handed whole media in one call and has no partition to
salvage. That is the sharpest reason `capabilities` reports the field: on a long file the
most likely failure — memory — lands on the one stack that cannot resume.

## 6. Teardown

Everything provisioned is discoverable from the registry, so a session that never
ran `pull` can still find and remove it.

```bash
audio packages path
audio packages list
audio packages remove vibevoice-asr-7b     # torch env survives; aligner and firered need it
audio packages purge --dry-run             # reports reclaimable bytes
audio packages purge
uv tool uninstall audio-processing-cli
```

`remove` and `purge` delete the environments, Swift build products, and checkouts
this tool created, plus the Hub revisions the registry records as materialized
here, stating that the Hugging Face cache may be shared with other tools. Neither
touches user media or output artifacts.

Purge before uninstalling, or the resolved root outlives the only tool that knows
how to describe it.

## Capability coverage

Where each capability in the namespace is exercised above:

| Capability | Exercised | Shown as |
| --- | --- | --- |
| `verbatim` | requested in §2 and §3; evidence divergence in §1.5 | native on all four; `quality: "refuted"` on `vibevoice` and `qwen-0.6b` |
| `diarization` | §1, §2, §3 | derived on Qwen and FireRed, native on VibeVoice; yields `segments[].speaker` and `turns[]` together |
| `overlapped_speech` | §1.4, §3 | derived |
| `vad` | §1.4, §3 | derived on Qwen via `silero-vad`, native on FireRed |
| `segment_timestamps` | §2, §3, §5 | native on VibeVoice and FireRed; exit 2 `unsatisfiable_on_stack` on Qwen |
| `word_timestamps` | §1.4, §2, §3 | derived on Qwen and VibeVoice, native on FireRed |
| `lid` | §3 | native on FireRed only, with its inference cost and region granularity |
| `token_lid` | `capabilities`, §5 | `impossible` in the catalog; exit 2 `unsupported` when requested |

All seven roles appear in a resolved plan: `decode` and `asr` in §1.1, §2 and §3;
`diarizer` in §1.1; `vad` in §1.4 (`silero-vad`) and §3
(`firered-vad`); `aligner` in §1.4 and §2; `punctuator` in §3; `lid` in §3's LID
variant. Four of the five `selected_by` forms appear in a resolved plan above —
`stack`, `requirement`, `add_on_required_by`, and `floor`; the fifth, `pin`, appears
only in §5 where a pin is rejected. Every error code in the table at the top of this
document is shown with its payload or its field list.
