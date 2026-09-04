# `transcribe` command contract

**Status: v1 implemented.** `capabilities`, `plan`, and `run` work for all four stack ids, and
`export` writes every contracted format. This is the agent-facing sequence the complete v1 transcription
surface must produce: from a machine with nothing installed, through provisioning and execution
and export, to teardown. Terms are defined in
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
| 2 | Request or validation error: missing stack or input, unknown capability, a capability the chosen stack cannot satisfy, an option the stack does not accept, a pin that conflicts with a requirement, an unsafe or unwritable output destination, invalid or incompatible export inputs, inert export `--force`, or absent word timing on export. |
| 3 | A required package is not provisioned, or a provisioned one failed its integrity check. Only `run` can return these. |
| 4 | Incomplete: zero or more units were transcribed and at least one remains. A partial result **is** written, with a coverage ledger and a resume command. Only `run` can return this, and only on a stack whose work is partitioned. |

Every error payload carries `code` and `fix`. **`fix` is a runnable command wherever the current
request can be repaired without inventing a caller-owned path**, so a caller can copy one line and
be right on the next attempt; otherwise it is a sentence naming the decision or nearest available
output. Emitting a plausible command that fails again is worse than admitting there is none.
The remaining fields are fixed per code, and this table is the contract — a payload with a
field not listed for its code, or missing one that is, is a defect:

| Code | Exit | Fields beyond `code` and `fix` |
| --- | --- | --- |
| `stack_required` | 2 | `field`, `allowed`, `stacks` (id → one-line characterization) |
| `input_required` | 2 | `field`, `note` |
| `capability_unknown` | 2 | `field`, `provided`, `available_on_stack`, `did_you_mean` when a near name exists |
| `option_unsupported_on_stack` | 2 | `field`, `provided`, `allowed` (empty), `stacks_accepting` |
| `option_value_unsupported` | 2 | `field`, `provided`, `allowed`, `did_you_mean` when a near value exists |
| `capability_unsatisfiable_on_stack` | 2 | `capability`, `allowed` (non-empty), `available_on_stack` |
| `capability_unsupported` | 2 | `capability`, `allowed` (empty), `reason` |
| `pin_conflicts_with_native_capability` | 2 | `field`, `provided`, `allowed`, `capability` |
| `range_invalid` | 2 | `field`, `provided`, `reason` |
| `output_exists` | 2 | `field`, `provided`, `existing` |
| `output_is_canonical_input` | 2 | `field`, `provided`, `resolved_target` |
| `output_path_invalid` | 2 | `field`, `provided`, `target`, `reason` |
| `output_required_for_force` | 2 | `field`, `provided`, `requires` |
| `export_input_invalid` | 2 | `field`, `provided`, `reason` |
| `export_inputs_incompatible` | 2 | `field`, `provided`, `reason` |
| `timing_required_for_format` | 2 | `field`, `provided`, `requires_capability`, `found`, `note` |
| `packages_not_provisioned` | 3 | `missing`, `total_known_download_bytes`, `unsized_packages` |
| `package_integrity_failed` | 3 | `failed` (package, check, expected, actual) |
| `package_build_unusable` | 3 | `package`, `product`, `built`, `fix` |
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
An existing explicit output or its derived partial-result path is refused before decode unless
`--force` is present. When `-o` is omitted, incomplete runs choose an unused sibling partial path
instead of replacing an earlier attempt. Even with `--force`, `--output` may never resolve to an
input transcript, its derived partial path, or canonical source media.

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

The accepted `--language` values are case-insensitive but are echoed with this exact spelling:
`Chinese`, `English`, `Cantonese`, `Arabic`, `German`, `French`, `Spanish`, `Portuguese`,
`Indonesian`, `Italian`, `Korean`, `Russian`, `Thai`, `Vietnamese`, `Japanese`, `Turkish`,
`Hindi`, `Malay`, `Dutch`, `Swedish`, `Danish`, `Finnish`, `Polish`, `Czech`, `Filipino`,
`Persian`, `Greek`, `Romanian`, `Hungarian`, and `Macedonian`. Those 30 names are the
`support_languages` arrays in the pinned
[`Qwen3-ASR-1.7B-8bit` config](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit/blob/a8379a2e2f9e313c9292cdf1af4055ab56d50d55/config.json)
and [`Qwen3-ASR-0.6B-8bit` config](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit/blob/89e96d92ba34aca20b3e29fb10cc284097d1219f/config.json),
not a vocabulary invented by this CLI. FireRed accepts no language input; its `lid` output is
the 115 labels after the five special tokens in the pinned
[`FireRedLID` dictionary](https://huggingface.co/FireRedTeam/FireRedLID/blob/1bb4d285c8456429385d9c0810300df4297bc11b/dict.txt).

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
            "note": "One label per VAD region, copied onto every sentence inside it, so per-sentence variation would be fabricated. On the same 139.284-second probe, LID on measured 162.09 s and 12.26 GiB peak RSS versus 84.24 s and 9.16 GiB with LID off. Its weights are fetched only when this capability is requested."},
    "diarization": {"availability": "requires_add_on",
                    "note": "Adds FluidAudio, which needs a Swift toolchain and a second environment: 15 s and 0.55 GiB peak on a 30-minute sample. Produces speaker labels on the text and the turn intervals together, mapped onto the transcript by an exact partition of the timeline so no span is transcribed twice and no gap is invented. The published 95.42% participant-interval F1 and 3-of-75 speaker-change figures came from a run with a two-speaker prior and overlap detection enabled; the shipped default supplies no speaker-count prior and enables overlap detection only when overlapped_speech is requested, so its quality is unmeasured. RSS excludes memory held by system Core ML services."
                    },
    "overlapped_speech": {"availability": "requires_add_on",
                          "note": "Enables FluidAudio's overlapping-segments mode in the same stage as diarization. The recorded 95.42% participant-interval F1 and downstream 33.56% MER used this mode plus a two-speaker prior; the shipped no-prior configuration is unmeasured. Without this request nothing in the plan detects overlap, so an empty abstention ledger means undetected rather than absent."},
    "token_lid": {"availability": "impossible", "reason": "no_backend_declares",
                  "note": "Named only so a request fails loudly. Code-switching support does not imply per-token labels, and no backend here produces them."}
  },
  "next": "audio transcribe plan --input field.wav --stack firered --want <capabilities>"
}
```

`processing` and `failure_recovery` are why `--input` is required rather than optional.
Neither is derivable from the stack: the unit a stack works in is a stack property, but
how many units *this* file yields, what it will cost, and whether a failure is survivable
are properties of the pair. The fixed structural field is `unit_count: null` wherever the
partition depends on content the plan has not decoded — VAD regions here, diarized turns on a
Qwen plan that requests `diarization` — because a fabricated count is worse than an explicit
unknown.

`failure_recovery` varies by stack and is the field to read before committing to a long
file. It is `per_unit` here and on Qwen; it is **`prefix_only` on `vibevoice`**, which is handed
whole media in a single `generate` call. A generation-cap truncation can salvage every complete
segment before the cut and resume from that prefix watermark, while a model-load failure still
leaves nothing. See §5.1.

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
turns; whether the abstention ledger is populated, since that depends on detected
overlap and on whether every requested per-segment alignment conforms; whether a given
segment has a word stream at all, since a segment with no speech to align has none —
the forced-aligner artifact has two, both VibeVoice non-speech event tags such as
`[Environmental Sounds]` — and an ordinary VibeVoice speech segment may instead lack
words only with a same-bounds `alignment_unavailable` ledger entry; and output quality
wherever a capability's `evidence.quality` is `unmeasured`. The
sample is the contract for a successful run; a backend failure exits 1 and writes
nothing.

One field a run adds that a plan does not have: each capability in the embedded
provenance gains an `outcome` of `produced` or `abstained`. A requested VibeVoice word
stream that is absent or nonconforming on even one ordinary speech segment makes the
run-level `word_timestamps` outcome `abstained`, while conforming word streams on other
segments remain. That is the only difference between the two documents, which is why
the key-set test must compare against a real run's provenance rather than the elided
placeholder printed below.

Placeholder timing and text values are `null`, never `0.0` or a plausible
string. `0.0` is a legal timestamp and would violate the `no_synthesized_bounds`
floor the moment a consumer read it as measured. Real metadata that the plan
genuinely has — duration, path — is populated rather than stubbed.

Enum-valued fields are the one exception: they show one legal member rather than
`null`, so a consumer can see the field is categorical. So `"reason": "raw_fragment"` is
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
                              "api_path": "_generate_chunks_batched", "max_tokens": 16384},
                   "adapter_strips": ["language <label><asr_text> scaffold"],
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 0.0,
                   "determinism_basis": "sampler=make_sampler(temp=0.0), i.e. argmax decode (run_turn_attributed_mlx_asr.py:661); back-to-back calls in one process produced byte-identical text, cross-process repetition untested"
                   },
    "diarizer":   {"backend": "fluidaudio", "version": "0.15.5",
                   "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
                   "environment": "swift",
                   "config": {"step_ratio": 0.1,
                              "min_segment_duration": 0.0,
                              "threshold": 0.6, "batch_size": 32},
                   "config_note": "the shipped default supplies no speaker-count prior and enables overlapping_segments only when overlapped_speech is requested; the cited quality figures used both --num-speakers 2 and --overlapping-segments, so they do not measure this request",
                   "selected_by": "add_on_required_by:diarization"}
  },
  "execution": {
    "stage_order": ["decode", "diarizer", "asr"],
    "residency": "one_model_stage_at_a_time",
    "note": "stages run strictly sequentially and no two model stages are resident together; wall time adds across stages, peak memory does not, and the per-stage peaks below must not be summed"
  },
  "capabilities": {
    "diarization": {"satisfaction": "derived", "backend": "fluidaudio",
                    "evidence": {"interface": "verified", "quality": "unmeasured"},
                    "note": "Anonymous labels are reconciled sample-exactly onto the ASR text, plus the diarizer's turn intervals. The shipped no-prior, overlap-off configuration is unmeasured; the 3-of-75 speaker-change result used a two-speaker prior with overlap detection enabled and does not apply to this request."
                    }
  },
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "kind": "weights",
     "bytes": 2467859030, "provisioned": false},
    {"package": "fluidaudio", "environment": "swift", "kind": "toolchain",
     "requires_tool": ["swift"], "bytes": null, "provisioned": false},
    {"package": "speaker-diarization-coreml", "environment": "swift", "kind": "weights",
     "bytes": 21599417, "provisioned": false}
  ],
  "total_known_download_bytes": 2489458447,
  "unsized_packages": [],
  "warnings": [],
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "complete": true,
    "source": {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "speaker": null}
    ],
    "turns": [
      {"turn_id": "turn_0", "speaker": null, "start": null, "end": null}
    ],
    "abstentions": [
      {"abstention_id": "ab_0", "reason": "raw_fragment", "start": null, "end": null}
    ],
    "provenance": "<stack, outcomes, observed, and the executed plan; elided in print>"
  }
}
```

`turns` is present because `diarization` was requested, and the two arrive together: the
speaker labels land on `segments[].speaker` and the intervals the diarizer measured land in
`turns`. They are separate arrays because they are separate measurements at separate
granularities — a turn can span several sentences — and neither is derived from the other.
When `overlapped_speech` is requested, a segment intersecting any detected overlap that intersects
the selected document scope keeps its text and any bounds but omits `speaker`, even if that overlap
starts before the range. This attribution mask is intersection-based; publication in
`overlapped_speech[]` and the `overlap` abstention ledger remains start-owned. A known
multi-speaker interval is never collapsed into one anonymous identity merely because diarization
was also requested.

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
On `run`, the same source-audio duration is recomputed from the canonical decode's PCM frame
count so range and coverage arithmetic use the timeline actually processed. `source.path` is the
resolved absolute path of the original media, so a later `export` invocation can protect that
canonical input even from a different working directory. `source` publishes no temporary format,
sample-rate, or channel fields; the working WAV remains an unpublished transport artifact.

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

Each RSS cell is the total resident footprint of the process executing that stage, including
its runtime and orchestration overhead; it is not allocation attributed only to the backend.
The in-process VAD cell is the core process high-water mark observed through that stage, while
fresh model and Swift stage cells measure those child processes. `peak_mps_live_bytes_by_stage`
is different: it is the MLX allocator's live-device-memory high-water mark inside each MLX
stage, excludes ordinary RSS and non-MLX processes, and is also aggregated with `max`, never sum.

### 1.2 What `run` does with packages absent

```bash
audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization
```

Exit 3. Nothing computed, nothing downloaded, stderr:

```json
{
  "code": "packages_not_provisioned",
  "missing": [
    {"package": "qwen3-asr-1.7b-8bit", "kind": "weights", "bytes": 2467859030},
    {"package": "fluidaudio", "kind": "toolchain", "requires_tool": ["swift"], "bytes": null},
    {"package": "speaker-diarization-coreml", "kind": "weights", "bytes": 21599417}
  ],
  "total_known_download_bytes": 2489458447,
  "unsized_packages": [],
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
                "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
                "config": {"scope": "all_segments",
                           "language_rule": "Chinese when text matches [一-鿿], otherwise English; the ASR --language hint is never forwarded"},
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
                "tokenizer": {"materialized_role": "tokenizer",
                              "repository": "Qwen/Qwen2.5-7B",
                              "revision": "d149729398750b98c0af14eb82c78cfe92750796"},
                "source_commit": "94da20d98b2fa7688e9cbfaf7692ddb4954f7600",
                "patch": "vibevoice-logits-to-keep",
                "config": {"device": "mps", "dtype": "bfloat16", "attention": "sdpa",
                           "seed": 1234, "max_new_tokens": 16384},
                "deterministic": true,
                "determinism_tolerance_ms": 0.0,
                "determinism_basis": "three seeded repeats shared one normalized-output hash; text decode is do_sample=False (run_vibevoice.py:267)",
                "determinism_note": "acoustic tokenizer samples a Gaussian latent; fixed seed required",
                "selected_by": "stack"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "mlx",
                "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
                "config": {"scope": "all_segments",
                           "language_rule": "Chinese when text matches [一-鿿], otherwise English; the ASR --language hint is never forwarded"},
                "selected_by": "add_on_required_by:word_timestamps"}
  },
  "capabilities": {
    "verbatim":           {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "refuted"},
                           "note": "Emits disfluencies rather than cleaning them — 28 filler hits on the probe, the highest of the four stacks — but a recorded run refuted dialect preservation twice: 看哈 became 看一下 and 耍啥子 became 刷啥子, both retained by firered."
                           },
    "diarization":        {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "measured"},
                           "note": "Each bounded speech segment forms its own native turn, so no gap or non-speech event is filled."
                           },
    "segment_timestamps": {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "word_timestamps":    {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                           "evidence": {"interface": "verified", "quality": "unmeasured"},
                           "note": "Boundary error against labels is unmeasured; non-speech events remain wordless, while unavailable ordinary-speech alignment records alignment_unavailable."}
  },
  "unsized_packages": [],
  "warnings": [
    {"code": "measured_peak_exceeds_target", "blocking": false,
     "detail": "measured 20.28 GiB live MPS allocation on spice-30min-participant; a strict 16 GiB MPS cap OOMs at model load, measured on a 27.8 s probe, while an 18 GiB cap passed that probe"}
  ]
}
```

The aligner rule is executable provenance, not a language-quality claim: the recorded probe
selects `Chinese` when its text regex sees a CJK ideograph and `English` otherwise
(`model_tests/benchmark/run_mlx_forced_aligner_probe.py:55,94`). It does not forward Qwen's
`Cantonese` hint. The pinned aligner implementation branches only for Japanese and Korean;
Chinese, Cantonese, English, and every other value use `tokenize_space_lang`, whose own CJK
splitter handles ideographs (`qwen3_forced_aligner.py:129-145,236-247`). There is therefore no
Chinese-only path and no Cantonese-specific tokenization failure. A future adapter declares
this rule because it is the configuration the recorded probe actually ran, not because the
pinned source proves it is better than forwarding the ASR hint.

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

Three adapter obligations this stack creates, all from its recorded output.
VibeVoice emits `Speaker: "N/A"` on non-speech segments; that is the absence of a
label, so the adapter emits no speaker rather than a speaker whose id is `"N/A"`.
And it emits bracketed non-speech event tags such as `[Environmental Sounds]` as
segment `text`. Those segments are real segments with real bounds and no words; they
are not sent to the aligner, survive into the transcript, and are not abstentions.
Finally, if an ordinary speech segment's requested alignment stream is absent or
nonconforming, the adapter preserves the text and native bounds, omits `words`, records
one `alignment_unavailable` abstention at those exact bounds, and marks the run-level
`word_timestamps` outcome `abstained`. Valid word streams on other segments remain.

The memory warning is advisory by explicit product decision, and it is emitted
from the plan rather than as a mid-run OOM. Its reference run took roughly
fourteen minutes of generation for thirty minutes of audio — an RTF near 0.47,
which is the figure to scale by; the plan cannot know `demo.mp4`'s duration cost
in advance. Cut and rerender from the original media; this command only reads it.

The live catalog's `failure_recovery.note` also scopes the cap risk: on
`spice-30min-participant`, 11,345 generated tokens over 1,800 seconds is 6.30
tokens/second, so at that observed rate the declared 16,384-token cap projects to
about 43 minutes of comparable audio. That is a rate extrapolation, not an observed
truncation.

Both `vibevoice-asr-7b` and `qwen3-forcedaligner` are sized from their pinned Hub snapshots, so
neither appears in `unsized_packages`. The VibeVoice package total includes its explicit offline
Qwen tokenizer subset as well as the ASR checkpoint; the receipt records both revisions.

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
                   "revision": "7990aaccc6b7aec1e527743bd30201f2c4a03b8c",
                   "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
                   "selected_by": "stack"},
    "asr":        {"backend": "firered-asr2-aed", "environment": "torch-firered",
                   "revision": "2304afed56eacfee6256dee5937ed22ffa0b64ec",
                   "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
                   "config": {"device": "cpu", "dtype": "float32", "batch_size": 4,
                              "return_timestamp": true, "beam_size": 3, "nbest": 1,
                              "decode_max_len": 0, "softmax_smoothing": 1.25,
                              "aed_length_penalty": 0.6, "eos_penalty": 1.0},
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 2.0,
                   "determinism_basis": "exact-repeat 60-minute fixture repeated the text and speaker-null sequences; maximum rebased timestamp drift was 1.0000000000002037 ms, within the frozen 2.0 ms tolerance, so normalized segments were not byte-equal"},
    "punctuator": {"backend": "firered-punc", "environment": "torch-firered",
                   "revision": "e448fd967f44182a1c323cc30f5d89f2400c28da",
                   "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
                   "config": {"batch_size": 4},
                   "selected_by": "floor:punctuated_sentence_segmented_text",
                   "recases_text": true}
  },
  "execution": {
    "stage_order": ["decode", "vad", "asr", "punctuator"],
    "residency": "one_environment_process_at_a_time",
    "environments_spanned": ["torch-firered"],
    "note": "FireRed runs one process with VAD, ASR and punctuator co-resident; requesting lid loads that model into the same process. The measured LID-off peak is 9830449152 bytes (9.16 GiB), not a maximum of imaginary per-role processes."
  }
}
```

`punctuator` is the one role in any plan selected by a floor rather than by the
stack or a requirement. It is not optional and not requestable: floor one requires
punctuated, sentence-segmented text, and on this stack that means FireRedPunc always
runs.

`determinism_tolerance_ms` is why `deterministic: true` means something here.
`0.0` claims byte-identical normalized output on repeat, which is what VibeVoice's
three seeded repeats measured. FireRed declares the artifact's frozen `2.0` ms policy:
its recorded exact-repeat run repeated the text and speaker-null sequences, while the
maximum rebased timestamp drift was `1.0000000000002037` ms, so its normalized segments
are *not* byte-equal. A single boolean would
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
            "revision": "1bb4d285c8456429385d9c0810300df4297bc11b",
            "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
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
  "note": "subtitle cue bounds come from word timestamps and are never synthesized",
  "fix": "audio transcribe run --input /Users/you/recordings/meeting.m4a --stack qwen-1.7b --want diarization,word_timestamps --language Cantonese -o meeting.timed.json"
}
```

A contradictory legacy or hand-edited result can record `word_timestamps: "produced"` while
ordinary sentence text carries no real word stream. That document violates the result contract;
export rejects the input before considering the requested format:

```bash
audio export --input ordinary.sentences.json --format srt
```

Exit 2, stderr:

```json
{
  "code": "export_input_invalid",
  "field": "--input",
  "provided": "ordinary.sentences.json",
  "reason": "ordinary speech without words requires an alignment_unavailable abstention and an abstained word_timestamps outcome",
  "fix": "regenerate or repair the input transcript before exporting it"
}
```

Changing formats cannot repair contradictory evidence. A valid requested-alignment failure uses
`word_timestamps: "abstained"` and a same-bounds `alignment_unavailable` abstention; export may
then preserve ordinary text in text formats and omit it from subtitle cues without mistaking the
absence for successful timing.

A result written by an older CLI may instead lack a safe durable source identity. If word timing
is missing and `source.path` is relative, missing, not a file, or cannot be resolved, export must
not reinterpret it from the current working directory and prescribe a rerun against the wrong
media:

```bash
audio export --input legacy.transcript.json --format srt
```

Exit 2, stderr:

```json
{
  "code": "timing_required_for_format",
  "field": "--format",
  "provided": "srt",
  "requires_capability": "word_timestamps",
  "found": [],
  "note": "subtitle cue bounds come from word timestamps and are never synthesized",
  "fix": "regenerate this transcript with the current audio CLI before rerunning or exporting it; its relative, missing, non-file, or unresolvable source.path cannot safely identify the original media"
}
```

There is one narrow nonempty wordless exception. If `word_timestamps` is recorded as produced
and **every** segment is a positive-duration, bracketed non-speech event such as `[Music]`, with
source-relative `start`/`end` bounds but no `speaker` and no `words`, SRT/VTT export succeeds with
an empty subtitle rather than turning the event's container bounds into a cue. Any ordinary
sentence among those wordless segments restores the refusal above. Once at least one real timed
word stream exists, export emits cues for timed segments and omits every wordless segment — both
bounded events and bounded ordinary speech carrying an exact same-bounds
`alignment_unavailable` abstention — rather than inventing bounds. Schema v1 has no association
between an unbounded Qwen segment and the processing-unit interval in its abstention ledger, so a
mixed result containing that shape refuses SRT/VTT instead of using an unrelated row to justify
dropping text. The ledger and capability outcome retain the incomplete-timing evidence.

`md` and `txt` are for people. `jsonl` is one segment object per line, ordered by
start time — the same segment objects the JSON result carries, without the envelope
or the provenance — so a consumer can stream or `grep` a long transcript without
parsing the whole document. It has no timing requirement, and because it drops the
provenance it is an export for reading, not an artifact to audit against.

VTT carries speaker labels as voice tags when `diarization` is present,
which is a commitment to VTT as a real format rather than SRT with dots. V1 ships a
fixed deterministic cue policy: at most 7 seconds and 2 lines, 16 CJK characters per
line, sentence-end then clause-punctuation then word-gap break priority, no speaker
change inside a cue, and 1 ms quantization. Issue #10 tracks future tuning rather than
an unimplemented prerequisite.

Three artifact-derived rules are already enforced. Breaking at punctuation maps marks
in sentence text to word indexes under the punctuation floor's case-insensitive
invariant, because FireRed's punctuator recases. And a segment may carry text with no
word stream, so the splitter omits it rather than assuming every segment yields cues.
Finally, result validation permits the recorded FireRed integer-millisecond seam where a
word edge escapes its sentence by at most 1 ms, while refusing anything larger; the raw
runner artifacts and exact affected rows are recorded in [HANDOFF.md](HANDOFF.md).

When `--output` is supplied, export serializes UTF-8 text and publishes it atomically.
An existing regular file is refused unless `--force` is explicit; a directory is never
replaceable. Neither mode may target any input transcript or the canonical source media,
including aliases, and `--force` does not override that protection. The writer carries
descriptor-captured device/inode identities through rendering. A forced replacement uses the
platform's atomic entry-exchange primitive, inspects the displaced inode, and atomically rolls
back when it is protected or non-regular; an absent destination is claimed with an exclusive
hard link. A protected file renamed onto the output mid-command is therefore refused without an
unaddressable or partial-output window. Without `--output`, the same serialized content is
written to stdout. The boundary is accidental and public-destination retargeting, not a hostile
same-credential process changing a random private sibling after its final identity check; POSIX
has no portable conditional-unlink operation, and that process already has direct authority to
remove the canonical file.

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
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --language EN
```

Exit 2, `code: "option_value_unsupported"`, `field: "--language"`, `provided: "EN"`,
`allowed` containing the exact 30-name Qwen vocabulary, and `did_you_mean: "English"`.
This is distinct from `option_unsupported_on_stack`: Qwen accepts the option, but not that
value. `english` is accepted case-insensitively and normalized to `English` in the plan.

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

A malformed or nonintersecting resume range is a request error, not an FFmpeg failure:

```json
{
  "code": "range_invalid",
  "field": "--range",
  "provided": "400:",
  "reason": "range does not intersect the source duration",
  "fix": "audio transcribe run --input demo.mp4 --stack qwen-0.6b"
}
```

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
    {"package": "vibevoice-asr-7b", "check": "hub_snapshot_integrity",
     "expected": "snapshot directories, manifest-filtered files, and recorded byte size",
     "actual": ["Qwen/Qwen2.5-7B is missing allow_pattern tokenizer.json"]}
  ],
  "fix": "audio packages pull --repair vibevoice-asr-7b"
}
```

Exit 3, and nothing loads. For Hub materializations, both explicit `verify` and run preflight
require the cache index to bind each repository and pinned revision to the recorded snapshot path,
all manifest `allow_patterns`, and the tree-byte total recorded at pull. These are cheap live
checks, not a fabricated weight
digest: a same-size content mutation can still pass and then fail at model load as
`backend_failed` exit 1. The hash-pinned URL artifact has a different boundary: pull accepts it
only at the exact manifest-derived models path as a contained, non-symlink regular file, including
after download; Silero's runtime auto-fetch applies the same rule. Source-backed native packages
additionally inspect the live Git HEAD,
derive the exact tracked file set from the installed patch, require every manifest-owned
post-patch SHA256 and the same exact values in the receipt, and reject ordinary
and Git-ignored untracked artifacts. Ignored bytecode and extensions remain importable, so they
cannot hide behind `.gitignore`. A legacy receipt may record `checkout_commit` as the manifest's
short `commit` alias or its full `resolved_commit`; new pulls record the full value, and the live
HEAD must always equal the full resolved commit. Run preflight also launches every selected managed
interpreter and requires exactly one contained, non-symlink FluidAudio product from its exact
managed checkout before decode. Before provisioning, freeze, interpreter launch, or decode, every
managed environment root is independently derived from the manifest and must be the exact
non-symlink directory resolving under the provisioning root; a normal venv `bin/python` symlink is
allowed inside that trusted root. `pull` also refuses a redirected `envs` parent before creation or
installation. `verify` reports a redirected ready root as `drifted`, and run refuses it before
decode rather than following the registry or filesystem redirect.
A runtime abort's `fix` is deliberately a sentence
directing the caller to the reported condition; a package verification command can legitimately
print `ok` after an OOM and therefore cannot be advertised as its repair.

A built package has a fourth failure of its own: the build succeeded and the executable it
produced does not launch.

```json
{
  "code": "package_build_unusable",
  "package": "fluidaudio",
  "product": "fluidaudiocli",
  "built": true,
  "fix": "audio packages pull --repair fluidaudio"
}
```

Exit 3, and the registry entry stays `pulling`, so `list` and `run` both report the package
absent rather than ready. `built: true` is kept because it is the useful half of the diagnosis —
a compile failure and a product that will not start need different responses, and this is the
second. The condition was real: the product name was hard-coded as `fluidaudio` while
`Package.swift` at the pinned commit declares `fluidaudiocli`, so `swift run` refused a perfectly
good build. It is pinned in the manifest beside the commit now, and `pull` refuses instead of
recording `product_runs: false` and exiting 0.

### 5.1 Incomplete runs

Long-form transcription has partial-completion mechanisms that are real and recorded, not
hypothetical. Two are visible in the harness today:

- **The Qwen path carries a global generation budget** and stops when it runs out. The stage
  records every turn it finished; the partial document retains the longest chronological prefix
  so its resume is disjoint. The recorded runner tracks `input_turns`, `processed_turns`, and
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
    "scope_intervals": [[0.0, 1794.2]],
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

`scope_intervals` records the complete source-timeline span of the processing units this invocation
selected. A fixed-unit ordinary Qwen run covers the whole source; FireRed can begin at its first
native VAD region and end at its last, leaving leading or trailing silence outside the processing
ledger. For a ranged run it exactly matches the recorded `selected_unit_scope`. Covered and missing
intervals partition that scope exactly; material outside it is neither claimed nor counted in
`covered_fraction`.

`covered_through_seconds` is the end of the longest **contiguous prefix** that is fully
transcribed, which is the number an agent can act on without reasoning about gaps. It is
not the same as "the last unit that finished": the runner processes turns in duration-bucketed
order and restores chronological order afterwards. On an incomplete Qwen run, completed units
after the first gap are deliberately omitted from the partial document and counted as missing;
the emitted resume therefore produces a disjoint suffix. `covered_intervals` and
`missing_intervals` still carry the exact artifact truth, and the schema can represent
non-contiguous coverage for later partitioned backends without pretending it is a prefix.

`--range <start>[:<end>]` is the resume mechanism, and it exists so the caller or downstream
consumer does **not** pre-clip the audio. A user-supplied clip would shift the timeline and force
every bound in the second result to be re-offset by hand before merge — consumer timestamp
arithmetic that the canonical-timeline floor exists to prevent. The CLI may still make an
unpublished internal clip when a whole-media backend requires one: ranged VibeVoice generation
runs against that transport clip, then the adapter adds the validated range start exactly once
to every relative segment/event bound before anything becomes public. Its aligner receives those
restored source-timeline bounds, the result's `source.path` still names the original canonical
media, and no temporary clip path escapes. With `--range`, the consumer therefore receives bounds
already on the original timeline and merging remains concatenation.

A ranged run also records the selection in the embedded executed plan:

```json
{
  "range": {
    "requested": [1402.88, 1794.2],
    "selected_unit_scope": [1402.88, 1794.2]
  }
}
```

`requested` is the validated interval (an open end resolves to the canonical WAV duration),
while `selected_unit_scope` can expand to whole processing-unit boundaries. On an incomplete
Qwen run, only the longest chronological prefix is published; later duration-bucketed successes
are rerun so the `--range` continuation is disjoint rather than asking export to guess duplicates.
Auxiliary VAD, overlap, and abstention spans use the same expanded scope and are owned by their
start. An open range reaches the canonical duration even when the last diarized turn ends earlier,
so tail evidence belongs to the continuation instead of disappearing between partial documents.
For an incomplete hand-written range that begins in a diarization gap, coverage remains bounded to
the selected processing units while auxiliary ownership begins at the requested bound; the gap's
evidence is preserved without claiming it was transcribed.

The partial result is a **conforming result document** with `complete: false` and the same
`coverage` block, so every floor still holds inside it: no synthesized bounds, abstentions
survive, punctuation invariant intact for the units that ran. It is not a debug dump.
If `units_completed` is zero, `fix` is deliberately a sentence rather than a `--range` command:
the deterministic run has no later unit to skip to, so replaying the same request cannot be
presented as a remedy.

Ids are document-scoped, so merging two results means re-numbering. `export` accepts
several transcripts in timeline order and re-ids as it goes, which covers the subtitle
case without anyone hand-editing JSON:

```bash
audio export --input meeting.partial.json --input meeting.rest.json \
  --format srt -o meeting.srt
```

One stack has a deliberate exception. VibeVoice's anonymous native speaker labels are local to
each independent generation: `Speaker 0` in a resumed range is not evidence for the same person as
`Speaker 0` in the partial run. Multi-input VibeVoice results that requested `diarization` are
therefore refused as `export_inputs_incompatible`; export them separately or rerun the desired
ranges together as one generation. Single-input export and multi-input VibeVoice results without
native diarization remain supported.

**On `vibevoice`, recovery is `prefix_only`.** The upstream parser yields no structured result
when generation stops inside an unterminated segment, so the adapter must parse the raw output
and retain every complete segment before the cut. That prefix is a conforming result with
`complete: false`, a coverage watermark at its end, and a `--range` fix for the remainder.
This does not turn every failure into a partial result: an OOM at model load has decoded no
prefix, writes no result, and remains exit 1.

## 6. Teardown

Everything provisioned is discoverable from the registry, so a session that never
ran `pull` can still find and remove it.

```bash
audio packages path
audio packages list
audio packages remove vibevoice-asr-7b     # removes its unique torch environment and checkout
audio packages purge --dry-run             # reports reclaimable bytes
audio packages purge
uv tool uninstall audio-processing-cli
```

`packages path` uses `location` for a single materialized source and a repository-to-path
`locations` map for a multi-repository Hub package; it never emits a null singular alias beside
that map. A native package also carries `checkout`, because the executable source is part of the
live provenance that `verify` checks. Fields for materializations a package does not have stay
absent.

`remove` and `purge` treat the registry as an ownership receipt, not path authority. Local
package and environment targets come from the installed manifest and stay inside the managed
root. A Hub revision is deletion-eligible only when the receipt says this root downloaded it,
the current manifest still pins it for that package, and it was not already cached before pull;
all other claimed revisions are retained. An unknown or retired registry package loses its
entry without following any recorded path. A failed local deletion leaves its registry owner
and contributes no reclaimed bytes. Neither command touches user media or output artifacts.

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
