# `transcribe` command contract

**Status: v1 implemented.** `capabilities`, `plan`, and `run` work for all four stack ids, and `export` writes every contracted format. This is the agent-facing sequence the complete v1 transcription surface must produce: from a machine with nothing installed, through provisioning and execution and export, to teardown. Terms are defined in [VOCABULARY.md](../VOCABULARY.md); the backend evidence is in [docs/model-tests/DECISION_REPORT.md](../model-tests/DECISION_REPORT.md).

Two decisions shape every sequence below:

- **`plan` and `run` are separate subcommands, not a flag.** They have different side effects, different output shapes, and different exit-code contracts: `plan` never reports a provisioning failure, it *is* the provisioning report. Only commands that select backends need the split, so `export` has none.
- **`--stack` and `--input` are both required, on `plan` as well as `run`.** The stack fixes transcript quality, language and dialect behavior, and which capabilities arrive natively. The input fixes the processing detail — how the audio is partitioned into units, how many there are, what the run will cost, and whether a failure is recoverable — and none of that is a static property of the stack. Omitting either fails with what was missing.

`--want` is optional. Omitting it requests nothing beyond the floors, which is a legitimate and useful request: a punctuated, sentence-segmented transcript on the canonical timeline with an abstention ledger and no optional capability at all. It is not a shorthand for "everything" — that would provision a diarizer and an aligner nobody asked for — and it is not an error.

There is no batch surface. One invocation takes one input; whether the backend then segments that input and batches the pieces is its own business, but it is *disclosed* in the plan rather than hidden, because the partition determines both the cost and the failure mode. A caller with a directory of interviews loops. A caller with a three-hour file learns from the plan how it will be cut up and what happens if a piece fails.

Both deviate from Issue #1 §2.1, which writes bare `audio transcribe meeting.m4a` as the simplest command. The deviation is deliberate.

| Exit | Meaning |
| --- | --- |
| 0 | Success. |
| 1 | Runtime or backend failure with nothing salvageable. No result is written. Distinct from a principled abstention, which is a successful run. |
| 2 | Request or validation error: missing stack or input, unknown capability, a capability the chosen stack cannot satisfy, an option the stack does not accept, a pin that conflicts with a requirement, an unsafe or unwritable output destination, invalid or incompatible export inputs, inert export `--force`, or absent word timing on export. |
| 3 | A required package is not provisioned, or a provisioned one failed its integrity check. Only `run` can return these. |
| 4 | Incomplete: zero or more units were transcribed and at least one remains. A partial result **is** written, with a coverage ledger and a resume command. Only `run` can return this, and only on a stack whose work is partitioned. |

Every error payload carries `code` and `fix`. **`fix` is a runnable command wherever the current request can be repaired without inventing a caller-owned path**, so a caller can copy one line and be right on the next attempt; otherwise it is a sentence naming the decision or nearest available output. Emitting a plausible command that fails again is worse than admitting there is none. The remaining fields are fixed per code, and this table is the contract — a payload with a field not listed for its code, or missing one that is, is a defect:

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
| `timing_required_for_timestamps` | 2 | `field`, `provided`, `input`, `segment_id`, `requires_any_capability`, `note` |
| `timestamps_unsupported_for_format` | 2 | `field`, `provided`, `format`, `allowed_formats` |
| `packages_not_provisioned` | 3 | `missing`, `total_known_download_bytes`, `unsized_packages` |
| `package_integrity_failed` | 3 | `failed` (package, check, expected, actual) |
| `package_build_unusable` | 3 | `package`, `product`, `built`, `fix` |
| `backend_failed` | 1 | `role`, `backend`, `detail` |
| `run_incomplete` | 4 | `role`, `backend`, `detail`, `coverage`, `output` |

`allowed` means different things by code and is never a free-text field: the stack ids for `stack_required`, the stacks that could serve the request for `capability_unsatisfiable_on_stack`, and empty where switching stacks cannot help. `capability_unknown` covers a name that is not in the namespace at all, which is distinct from a name that is real but unsatisfiable here.

Both capability errors also carry **`available_on_stack`**, the full set of names this stack will accept, split into `native`, `requires_add_on`, and `impossible`. A caller who got the `--want` wrong needs to know what it can ask for *here*, not only which other stack would have worked, and the split says which of those choices are free. It makes the error self-sufficient: correcting a request should not require a second command to find the menu.

Three unrelated things were previously all called `requires`. They are now `packages` (the plan's provisioning list), `requires_tool` (an external toolchain a package needs, such as `swift`), and `requires_capability` (the capability an export format needs).

Machine-readable output goes to stdout; human progress goes to stderr, so an agent can pipe stdout safely. Note the distinction from a plan's `warnings` array, which is a stdout field of the plan document, not a stderr message.

`run` defaults to `--format json` on stdout, matching the existing CLI's machine-readable convention. `--format md|txt` are for human consumption. An existing explicit output or its derived partial-result path is refused before decode unless `--force` is present. When `-o` is omitted, incomplete runs choose an unused sibling partial path instead of replacing an earlier attempt. Even with `--force`, `--output` may never resolve to an input transcript, its derived partial path, or canonical source media.

`--language` is the one caller-settable model input, and it is a hint passed to the ASR rather than a capability. Only the Qwen stacks accept it; `vibevoice` advertises code switching without language selection, and FireRed's ASR takes no language argument at all — its language is an output of the optional LID stage, not an input. Each stack's `languages` catalog entry says whether it takes one, and passing the flag to a stack that does not is `option_unsupported_on_stack` rather than a silently ignored argument.

Every recorded interview figure was produced with `--language Cantonese`, so exposing the flag is what makes those numbers reachable from the CLI at all; without it the measured configuration would be one no caller could ask for. Omitting the flag is a real second configuration, not a default: the model still emits a language label with no hint, and the two Qwen sizes disagreed with each other on the same recording when run that way, which is an argument for stating the language you know and against trusting the label you get back.

The accepted `--language` values are case-insensitive but are echoed with this exact spelling: `Chinese`, `English`, `Cantonese`, `Arabic`, `German`, `French`, `Spanish`, `Portuguese`, `Indonesian`, `Italian`, `Korean`, `Russian`, `Thai`, `Vietnamese`, `Japanese`, `Turkish`, `Hindi`, `Malay`, `Dutch`, `Swedish`, `Danish`, `Finnish`, `Polish`, `Czech`, `Filipino`, `Persian`, `Greek`, `Romanian`, `Hungarian`, and `Macedonian`. Those 30 names are the `support_languages` arrays in the pinned [`Qwen3-ASR-1.7B-8bit` config](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit/blob/a8379a2e2f9e313c9292cdf1af4055ab56d50d55/config.json) and [`Qwen3-ASR-0.6B-8bit` config](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit/blob/89e96d92ba34aca20b3e29fb10cc284097d1219f/config.json), not a vocabulary invented by this CLI. FireRed accepts no language input; its `lid` output is the 115 labels after the five special tokens in the pinned [`FireRedLID` dictionary](https://huggingface.co/FireRedTeam/FireRedLID/blob/1bb4d285c8456429385d9c0810300df4297bc11b/dict.txt).

## Discovery and input-specific requests

`audio transcribe stacks` prints the declared stack ids, their existing characterization, and the `native`, `requires_add_on`, and `impossible` capability groups as JSON. It requires no input, metadata probe, registry read, model, or provisioning. These are declared interface facts, not installed readiness or new accuracy measurements. Every transcription `--stack` help points to this command; request commands retain the same semantic missing-stack refusal. The complete discovery output and progress/log behavior are documented in [CLI feedback](../cli-feedback.md).

### Two input-specific questions, two commands

Nothing sequences these. An earlier draft called them "step one" and "step two" and made both forms of `plan`, distinguished by whether `--want` was present — which meant `plan` answered two different questions depending on an absent argument, and implied an order the tool never enforced. An agent that already knows what it needs should go straight to a plan; an agent that does not needs somewhere to look. Those are different questions, so they are different commands, and neither is a gate on the other.

- `audio transcribe capabilities --stack S --input F` — what can this stack do with this file, what would each addition cost, and what happens if the run dies.
- `audio transcribe plan --stack S --input F [--want ...]` — resolve this exact request: the backends that will run, the packages to provision, and the shape of the output.

Both require a stack and an input, neither reads media beyond a metadata probe, and neither provisions anything. With `--want` omitted, `plan` resolves the floors-only request — a punctuated transcript and nothing optional — which is a real request and the same meaning the flag's absence has on `run`. It is not a request for the menu.

### `capabilities` — what can this stack do with this file?

The capability catalog is a static property of the stack, but the processing detail is not: the backend reads the input's metadata, decides how it will partition the audio, and discloses that along with what the run will cost and whether a failure leaves anything usable.

```bash
audio transcribe capabilities --stack firered --input field.wav
```

This is not an availability lookup. It is the only place an agent can find what it needs to choose a request, so it carries the context that choice needs: what the stack already produces and how precisely, what each add-on would cost in packages, time, and memory, and what a recorded run showed about quality. An agent that only learns `requires_add_on` cannot tell a free add-on from one that pulls a second runtime, and an agent that only learns `native` cannot tell whether native is *good enough* for what it is building. Both are request decisions, so both belong here.

The catalog uses its own axis, `availability`, because nothing has been requested yet and `satisfaction` is defined only for a requested capability:

```json
{
  "stack": "firered",
  "family": "FireRedASR2S",
  "environment": "torch-firered",
  "roles": "vad, asr and punctuator always; lid as well when lid is requested",
  "input": {"path": "field.wav", "duration_seconds": 27.8, "container": "wav",
            "sample_rate_hz": 48000, "channels": 1, "duration_basis": "probed_audio_stream",
            "timing": {"basis": "probed_timestamps", "audio": [{"stream_index": 0, "duration_seconds": 27.8}]}},
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
  "next": "audio transcribe plan --input field.wav --stack firered"
}
```

`processing` and `failure_recovery` are why `--input` is required rather than optional. Neither is derivable from the stack: the unit a stack works in is a stack property, but how many units *this* file yields, what it will cost, and whether a failure is survivable are properties of the pair. The fixed structural field is `unit_count: null` wherever the partition depends on content the plan has not decoded — VAD regions here, diarized turns on a Qwen plan that requests `diarization` — because a fabricated count is worse than an explicit unknown.

`failure_recovery` varies by stack and is the field to read before committing to a long file. It is `per_unit` here and on Qwen; it is **`prefix_only` on `vibevoice`**, which is handed whole media in a single `generate` call. A generation-cap truncation can salvage every complete segment before the cut and resume from that prefix watermark, while a model-load failure still leaves nothing. See §5.1.

Everything in that document is there because a caller acts on it, and almost none of it is structured. Three enums carry the decisions a program branches on — `availability`, `processing.unit`, and `failure_recovery.partial_results` — plus two numbers, the unit count and the projected seconds. Everything else is a sentence.

That is deliberate, and it is a correction. Earlier drafts gave each capability an `evidence` object, a `cost` object with seven keys, a `timing_precision` object holding one null and one string, an `alternative` object wrapping one sentence, and `measured_limit` beside `observed_limit` beside `note` beside `interface_basis`. None of that nesting had a consumer: no caller branches on `interface: "verified"`, and a reader who wants to know whether a stack suits them reads the sentence either way. Structure that no one dispatches on is ceremony, and it makes the payload longer to read and easier to get subtly wrong — which it did, repeatedly, in this document's own history.

What the sentences must still do is what the retired objects were built to enforce. Say what was measured and on what, say when a run *refuted* rather than merely failed to measure something, and state the consequence rather than the forensics: "matched only 3 of 75 annotated speaker changes on a dense two-speaker conversation — strong on long turns, unsuitable where turns are short or overlapping" is actionable, while the fixture, the collar, and the diarizer preset are not — they live in the research record under `model_tests/`, which is where an auditor looks and which will not exist in a shipped tool's output. Two facts here are worth noticing because they are the sort a structured field would have hidden: FireRed's native word timing has never been scored against hand-labelled boundaries and neither has the aligner, so switching stacks for timing accuracy trades one unmeasured number for another; and `vad` is the inverse case, where the add-on has a measured figure and the native stage has none.

A plan keeps its structure, because that document *is* dispatched on: `roles` with backends, revisions, and configuration is audited provenance, and `satisfaction`, `outcome`, and `evidence` are asserted by tests. A catalog is read to make a choice; a plan is read by a machine that has to reproduce a run.

There is no provenance-only section, and there used to be. It carried the extents of Qwen's processing units and the single language label read off its output scaffold, on the argument that publishing them let them be audited and refused as a timing source. Neither earned the space: a caller can act on neither, the extents are not speech timing and are not published as any kind, and the label is not a detector. Removing them also removed a refusal code, since a capability that does not exist cannot be a capability that exists but cannot be asked for. Discovering that `token_lid` is `impossible` is what `capabilities` is for; *requesting* it is an error, not a field (see Refusals).

`roles` is one sentence rather than two arrays. FireRedLID is inside the stack but runs only when `lid` is requested, so listing it unconditionally would promise weights this request will not fetch while calling it an add-on would promise a package the stack already contains. Saying "lid as well when lid is requested" is both shorter and truer than a pair of keys.

Two things FireRed emits that this catalog deliberately does not offer. `asr_confidence` is real but *sentence*-level, and there is no requestable capability at that granularity in v1 — see the retired `word_confidence` entry in [VOCABULARY.md](../VOCABULARY.md), which was the wrong name for it. And `lang` / `lang_confidence` appear on every sentence in the raw output even when LID never ran, defaulted to `null` and `0`; the adapter drops both rather than publishing a zero confidence that reads as measured.

### `plan` — what will this request produce?

Adds the resolved roles, the packages to provision, and a `sample_output` block. When the registry snapshot marks packages missing, top-level `next` is a runnable `audio packages pull` command naming exactly those package ids in plan order. It is absent when all planned packages are provisioned. This is provisioning guidance, not an integrity check, and is not copied into the durable result's embedded plan. `capabilities.next` is a runnable floors-only plan; add `--want` deliberately for capabilities beyond the floors.

`audio transcribe plan --compact` omits only `sample_output` and does not generate it. The default is unchanged. `roles`, `execution`, `capabilities`, `packages`, `total_known_download_bytes`, `unsized_packages`, `warnings`, and conditional `next` retain exactly their default values. This is a presentation choice; request validation, stack selection, package selection, estimates, and warnings are unchanged.

### How `sample_output` is produced, and what it guarantees

The sample is built by populating the real result object with one placeholder entity per requested capability — plus the floor artifacts every conforming run carries, such as the abstention ledger — and serializing it through **the same serializer `run` uses**. It is never a hand-written example, and there is no second code path. Consequently the output shape is a pure function of the resolved capability set, so all combinations are generated on demand rather than enumerated.

Guaranteed identical to a real run: key sets, nesting, types, which fields are absent because a capability was not requested, which are `native` versus `derived`, and the provenance structure.

Not predictable, and therefore not claimed: cardinality of segments, words, and turns; whether the abstention ledger is populated, since that depends on detected overlap and on whether every requested per-segment alignment conforms; whether a given segment has a word stream at all, since a segment with no speech to align has none — the forced-aligner artifact has two, both VibeVoice non-speech event tags such as `[Environmental Sounds]` — and an ordinary VibeVoice speech segment may instead lack words only with a same-bounds `alignment_unavailable` ledger entry; and output quality wherever a capability's `evidence.quality` is `unmeasured`. The sample is the contract for a successful run; a backend failure exits 1 and writes nothing.

One field a run adds that a plan does not have: each capability in the embedded provenance gains an `outcome` of `produced` or `abstained`. A requested VibeVoice word stream that is absent or nonconforming on even one ordinary speech segment makes the run-level `word_timestamps` outcome `abstained`, while conforming word streams on other segments remain. That is the only difference between the two documents, which is why the key-set test must compare against a real run's provenance rather than the elided placeholder printed below.

Placeholder timing and text values are `null`, never `0.0` or a plausible string. `0.0` is a legal timestamp and would violate the `no_synthesized_bounds` floor the moment a consumer read it as measured. Real metadata that the plan genuinely has — duration, path — is populated rather than stubbed.

Enum-valued fields are the one exception: they show one legal member rather than `null`, so a consumer can see the field is categorical. So `"reason": "raw_fragment"` is shape, while `"text": null` is content withheld. Free-text and numeric fields are always `null`. One member is not the member set, so the sample is not where a consumer learns it: the plan warns when nothing in it can detect overlap, which is the case where the ledger cannot fill at all.

The binding test is that the sample's key set equals a real run's key set, and that no key exists for a capability that was not requested. That is the anti-fabrication guarantee, and it is parametrized over the derivation table: each stack against each capability it satisfies natively, each capability requiring an add-on, and each of both refusal codes, which are distinct and must not be collapsed into one "unsatisfiable" case.

### Duration bases

Capability and plan estimates use the media-owned probed duration with `duration_basis: probed_audio_stream` or `probed_container`; capability `input.timing` also exposes available stream origins. Runs instead label `source.duration_basis: canonical_decoded_pcm` and use the canonical mono 16 kHz PCM16 frame count. Zero is the first decoded sample, not a container timestamp. [Duration and alignment evidence](../timing-evidence.md) defines these fields, omission rules, and why duration or start differences cannot establish clipping, missing words, or a justified timestamp shift. Legacy v1 sources without a basis remain readable without a fabricated basis.

Request validation remedies correct the offending fields in prose and tell the caller to repeat its original command with every other argument preserved. They never change `run` to `plan`, drop range/output/format/language options, or silently choose a full-media run. This keeps fixed refusal payloads independent of execution-only options. Concrete provisioning and output retry commands remain runnable when their builders have the full context.

### Published-run receipts

`audio transcribe run --receipt` requires `--output PATH` and JSON format (`--format json`, including its default). It applies to all four stacks. The saved normalized JSON remains byte-for-byte identical to the same run without `--receipt`; only stdout becomes a JSON receipt. Default run stdout remains the full result. For example, a complete floors-only Qwen run with three segments and no abstentions returns:

```json
{
  "output": "meeting.json",
  "source": {
    "path": "/Users/you/recordings/meeting.m4a",
    "duration_seconds": 361.0,
    "duration_basis": "canonical_decoded_pcm",
    "timebase": "seconds"
  },
  "stack": "qwen-0.6b",
  "complete": true,
  "counts": {"segments": 3, "abstentions": 0}
}
```

`output` is the actual published path. `source`, `stack`, and `complete` copy saved facts. Counts always include `segments` and `abstentions`; `words` counts supplied word objects only when a word stream is present, and `turns`, `vad_regions`, `lid_regions`, and `overlapped_speech` appear only when their saved collections exist. Missing collections and source metadata stay absent. Counts do not measure recognition accuracy or missing spoken words; `complete` describes processing completion, and the saved abstention ledger and capability outcomes remain the evidence for capability limitations.

An incomplete run still exits 4 and prints the unchanged `run_incomplete` refusal, coverage, and resume fix to stderr. Receipt stdout names the actual partial JSON file, sets `complete: false`, and includes its saved `coverage` and counts. The requested complete output is not reported as published. Backend and publication failures emit no receipt.

Invalid receipt combinations refuse at exit 2 before request resolution, media probing, or model work. `--receipt` without `--output` in default JSON format produces:

```json
{
  "code": "receipt_options_invalid",
  "field": "--receipt",
  "provided": true,
  "output_supplied": false,
  "format": "json",
  "requires": ["--output", "--format json"],
  "fix": "use --receipt with --output PATH and --format json, or remove --receipt; repeat the original command, preserving every other argument"
}
```
