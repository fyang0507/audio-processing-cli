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
- **`--stack` is required.** The stack fixes transcript quality, language and
  dialect behavior, and which capabilities arrive natively; none of that is
  derivable from a requirement list. Omitting it lists the stacks rather than
  guessing.

Both deviate from Issue #1 §2.1, which writes bare `audio transcribe meeting.m4a`
as the simplest command. The deviation is deliberate.

| Exit | Meaning |
| --- | --- |
| 0 | Success. |
| 1 | Runtime or backend failure. Partial results are not written. Distinct from a principled abstention, which is a successful run. |
| 2 | Request or validation error: missing stack, unknown capability, a capability the chosen stack cannot satisfy, a pin that conflicts with a requirement, absent word timing on export. |
| 3 | A required package is not provisioned. Only `run` can return this. |

Every error payload carries `code` and `fix`. The remaining fields are fixed per
code, and this table is the contract — a payload with a field not listed for its
code, or missing one that is, is a defect:

| Code | Exit | Fields beyond `code` and `fix` |
| --- | --- | --- |
| `stack_required` | 2 | `field`, `allowed`, `stacks` (id → one-line characterization) |
| `capability_unknown` | 2 | `field`, `provided`, `allowed` |
| `capability_unsatisfiable_on_stack` | 2 | `capability`, `allowed` (non-empty) |
| `capability_unsupported` | 2 | `capability`, `allowed` (empty), `reason` |
| `capability_not_requestable` | 2 | `capability`, `allowed` (empty), `alternatives` |
| `pin_conflicts_with_native_capability` | 2 | `field`, `provided`, `allowed`, `capability` |
| `timing_required_for_format` | 2 | `field`, `provided`, `requires_capability`, `found`, `note` |
| `packages_not_provisioned` | 3 | `missing`, `total_known_download_bytes`, `unsized_packages` |
| `backend_failed` | 1 | `role`, `backend`, `detail` |

`allowed` means different things by code and is never a free-text field: the stack
ids for `stack_required`, the capability namespace for `capability_unknown`, the
stacks that could serve the request for `capability_unsatisfiable_on_stack`, and
empty for the three cases where switching stacks cannot help. `capability_unknown`
covers a name that is not in the namespace at all, which is distinct from a name
that is real but unsatisfiable here.

Three unrelated things were previously all called `requires`. They are now
`packages` (the plan's provisioning list), `requires_tool` (an external toolchain a
package needs, such as `swift`), and `requires_capability` (the capability an
export format needs).

Machine-readable output goes to stdout; human progress goes to stderr, so an
agent can pipe stdout safely. Note the distinction from a plan's `warnings`
array, which is a stdout field of the plan document, not a stderr message.

`run` defaults to `--format json` on stdout, matching the existing CLI's
machine-readable convention. `--format md|txt` are for human consumption.

## Planning in two steps

`plan` answers progressively, matching the order the interface enforces. Neither
form reads media beyond metadata, and neither provisions anything.

### Step one — what can this stack do?

No input file required; this is a static property of the stack, answerable before
any media exists.

```bash
audio transcribe plan --stack firered
```

The catalog uses its own axis, `availability`, because nothing has been requested
yet and `satisfaction` is defined only for a requested capability:

```json
{
  "catalog_version": 1,
  "stack": "firered",
  "family": "FireRedASR2S",
  "roles_included": ["vad", "asr", "punctuator"],
  "roles_conditional": {"lid": "region_language"},
  "environment": "torch",
  "floors": ["punctuated_sentence_segmented_text", "punctuation_is_sentence_level",
             "canonical_timeline", "no_synthesized_bounds", "abstentions_survive",
             "adapter_normalization"],
  "capabilities": {
    "verbatim":            {"availability": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "retained 看哈 on the 27.8 s Sichuanese probe; two examples cannot rank varieties"},
    "word_bounds":         {"availability": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "speech_bounds":       {"availability": "native", "stage": "FireRedVAD",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "segment_bounds":      {"availability": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "region_language":     {"availability": "native", "stage": "FireRedLID",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "cost": "~2x inference; additional weights, size unrecorded",
                            "granularity_note": "one label per VAD region, copied onto every sentence in that region; per-sentence variation would be fabricated"},
    "speaker_attribution": {"availability": "requires_add_on", "add_on": ["fluidaudio", "reconciler"],
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "measured_limit": "3 of 75 annotated speaker changes matched on the 149.9 s CantoMap dense conversation",
                            "record": "model_tests/benchmark/DIARIZATION.md"},
    "turn_bounds":         {"availability": "requires_add_on", "add_on": ["fluidaudio"],
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "measured_limit": "95.42% participant-interval F1 on the 30-minute SpiCE mix, but 5.50% speaker-change F1 at a one-second tolerance on the CantoMap dense conversation; strong on long turns, weak on rapid change detection",
                            "record": "model_tests/benchmark/DIARIZATION.md"},
    "overlap_intervals":   {"availability": "requires_add_on", "add_on": ["fluidaudio"],
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "token_language":      {"availability": "impossible", "reason": "no_backend_declares"},
    "capture_role":        {"availability": "impossible", "reason": "not_implemented_v1"},
    "filler_candidates":       {"availability": "impossible", "reason": "not_implemented_v1"},
    "repetition_candidates":   {"availability": "impossible", "reason": "not_implemented_v1"},
    "false_start_candidates":  {"availability": "impossible", "reason": "not_implemented_v1"}
  },
  "provenance_only": {},
  "next": "audio transcribe plan INPUT --stack firered --want <capabilities>"
}
```

`provenance_only` is present in every plan and every catalog, and empty on every
stack except Qwen: `container_bounds` and `container_language` record a processing
container, Qwen is the only stack that has one (180-second chunks, or the diarized
turns when timing is supplied externally), and VibeVoice and FireRed pass whole
media in a single call. Empty is therefore the common case, not the exception.
Discovering that `token_language` is `impossible` is what step one is for;
*requesting* it is an error, not a field (see Refusals).

`roles_conditional` is separate from `roles_included` because FireRedLID is inside
the stack but runs only when `region_language` is requested. Listing `lid`
unconditionally would promise weights this request will not fetch; calling it an
add-on would promise a package the stack does not already contain. It is neither.

Two things FireRed emits that this catalog deliberately does not offer.
`asr_confidence` is real but *sentence*-level, and there is no requestable
capability at that granularity in v1 — see the retired `word_confidence` entry in
[VOCABULARY.md](VOCABULARY.md), which was the wrong name for it. And `lang` /
`lang_confidence` appear on every sentence in the raw output even when LID never
ran, defaulted to `null` and `0`; the adapter drops both rather than publishing a
zero confidence that reads as measured.

### Step two — what will this request produce?

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
consumer learns it: `policy.abstention_reasons` publishes the set this plan can
actually produce, and an empty array there means the ledger cannot fill.

The binding test is that the sample's key set equals a real run's key set, and that
no key exists for a capability that was not requested. That is the anti-fabrication
guarantee, and it is parametrized over the derivation table: each stack against each
capability it satisfies natively, each capability requiring an add-on, and each of
the three refusal codes, which are distinct and must not be collapsed into one
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
audio transcribe plan meeting.m4a \
  --stack qwen-1.7b \
  --want speaker_attribution
```

Exits 0 whether or not anything is provisioned:

```json
{
  "plan_version": 1,
  "request": {"input": "meeting.m4a", "stack": "qwen-1.7b",
              "want": ["speaker_attribution"]},
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "asr":        {"backend": "qwen3-asr-1.7b-8bit", "environment": "mlx",
                   "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
                   "config": {"batch_size": 1, "clear_mlx_cache_after_every_batch": true,
                              "language": null},
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 0.0,
                   "determinism_basis": "sampler=make_sampler(temp=0.0), i.e. argmax decode (run_turn_attributed_mlx_asr.py:661); a decode-configuration citation, not a repeat-hash measurement"},
    "diarizer":   {"backend": "fluidaudio", "version": "0.15.5",
                   "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
                   "environment": "swift",
                   "config": {"preset": "quality", "step_ratio": 0.1,
                              "min_segment_duration": 0.0, "output": "regular",
                              "threshold": 0.6, "num_speakers": 2},
                   "config_note": "every cited diarization measurement used a known two-speaker prior; num_speakers must be supplied or the measured_limit figures do not apply",
                   "selected_by": "add_on_required_by:speaker_attribution"},
    "reconciler": {"backend": "sample-exact-turn-partition",
                   "config": {"partition": "sample_exact"},
                   "selected_by": "add_on_required_by:speaker_attribution"}
  },
  "floors": ["punctuated_sentence_segmented_text", "punctuation_is_sentence_level",
             "canonical_timeline", "no_synthesized_bounds", "abstentions_survive",
             "adapter_normalization"],
  "policy": {
    "policy_version": 1,
    "overlap": "abstain",
    "overlap_detection": "fluidaudio",
    "raw_fragment_min_ms": 250,
    "accepted_turn_min_ms": 500,
    "same_label_merge_max_ms": 300,
    "below_threshold": "abstain",
    "abstention_reasons": ["overlap", "raw_fragment", "short_turn"]
  },
  "capabilities": {
    "speaker_attribution": {"satisfaction": "derived", "backend": "fluidaudio",
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "note": "reconciled sample-exactly onto ASR text; anonymous labels only",
                            "measured_limit": "on the 149.9 s CantoMap conversation with 75 annotated speaker changes this diarizer preset matched 3; not validated for rapid backchannels, interruptions, or dense overlap",
                            "record": "model_tests/benchmark/DIARIZATION.md"}
  },
  "provenance_only": {
    "container_bounds":   {"backend": "qwen3-asr-1.7b-8bit",
                           "note": "processing container extents; not time evidence"},
    "container_language": {"backend": "qwen3-asr-1.7b-8bit",
                           "note": "single label; disagreed across Qwen sizes on one clip"}
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
  "measured": {
    "reference_run": "spice-30min-canonical-mix",
    "fixture_duration_seconds": 1800.0,
    "hardware": "apple-m4-max-64gib",
    "config": "batch 1; MLX cache cleared after every batch, 195 clear observations over 195 accepted turns; language hint \"Cantonese\"",
    "asr_stage_wall_seconds": 53.77,
    "peak_rss_bytes": 3241689088,
    "end_to_end_wall_seconds": null,
    "end_to_end_note": "never measured for 1.7B; the record's 69.16 s arithmetic sum adds a 15.39 s diarization stage taken from a separate 0.6B end-to-end run and carries measured_end_to_end: false",
    "record": "model_tests/benchmark/results/2026-08-13-turn-attributed-fast-asr.json"
  },
  "warnings": [],
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "source": {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "speaker": null}
    ],
    "abstentions": [
      {"abstention_id": "ab_0", "reason": "overlap", "start": null, "end": null}
    ],
    "provenance": "<the full executed plan; elided in this printed example only>"
  }
}
```

Two things the sample deliberately does *not* contain. There is no `turns` array,
because `turn_bounds` was not requested — §1.4 adds it explicitly. And segments
carry no `words` array, because `word_bounds` was not requested. They also carry no
`start` or `end`: those are the `segment_bounds` capability, which is exit-2 on
this stack, and Qwen's only time-like output is the container extents this document
forbids promoting. A segment exists as a floor artifact; its time extents do not
come free with it. Keys are absent rather than null-valued at the container level,
so absence can never read as a measured value.

`provenance` is shown as a string here purely to keep the example readable. In a
real result it is the full executed plan, and the key-set test must compare against
that, not against this placeholder.

The `abstentions` ledger *is* present, and not as an exception to that rule: it is
a floor artifact, governed by `policy.overlap` above rather than by any requested
capability, so the one-placeholder-per-request rule does not reach it. It is also
genuinely producible from this exact request, since FluidAudio emits
overlap-permitting output. Whether it fills depends on the audio.

`duration_seconds` is populated because `plan` reads container metadata rather than
stubbing what it already knows. The value `1794.2` is illustrative — this document
has no real `meeting.m4a` — and is deliberately *not* the `measured` block's
`fixture_duration_seconds: 1800.0`, which describes the reference fixture, not this
input.

One gap the `measured` block now states rather than hides: every figure in it comes
from a run that passed the language hint `"Cantonese"`, while the plan passes
`language: null`. The MER figures in §1.5 therefore do not describe the no-hint
path. Whether a language hint should be caller-settable is an open surface question,
not a flag this document defines; the recorded capability probe passed no hint and
both Qwen sizes still emitted a language label, disagreeing with each other on the
same recording.

### 1.2 What `run` does with packages absent

```bash
audio transcribe run meeting.m4a --stack qwen-1.7b --want speaker_attribution
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
  "fix": "audio packages pull --stack qwen-1.7b --want speaker_attribution"
}
```

### 1.3 Provision, verify, execute

```bash
audio packages pull --stack qwen-1.7b --want speaker_attribution
audio packages verify
audio packages list
audio transcribe run meeting.m4a --stack qwen-1.7b --want speaker_attribution \
  --format md
audio transcribe run meeting.m4a --stack qwen-1.7b --want speaker_attribution \
  --format json -o meeting.transcript.json
```

The JSON result carries the executed plan as `provenance`, anonymous speaker
labels, and the overlap abstention ledger. Anonymous labels are never mapped to a
person or a role.

### 1.4 Adding timing, and what it costs

Qwen has no native word timing, so `word_bounds` forces the aligner and its
`torch` environment. One request then spans all three provisioned environments.

```bash
audio transcribe plan meeting.m4a --stack qwen-1.7b \
  --want speaker_attribution,word_bounds,turn_bounds,overlap_intervals,speech_bounds
audio packages pull --stack qwen-1.7b \
  --want speaker_attribution,word_bounds,turn_bounds,overlap_intervals,speech_bounds
audio transcribe run meeting.m4a --stack qwen-1.7b \
  --want speaker_attribution,word_bounds,turn_bounds,overlap_intervals,speech_bounds \
  --format json -o meeting.timed.json
```

`word_bounds` arrives `derived` with `evidence.quality: "unmeasured"`, because
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
                "selected_by": "add_on_required_by:speech_bounds"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "torch",
                "config": {"scope": "all_segments"},
                "selected_by": "add_on_required_by:word_bounds"}
  },
  "capabilities": {
    "speech_bounds": {"satisfaction": "derived", "backend": "silero-vad",
                      "evidence": {"interface": "verified", "quality": "measured"},
                      "measured_limit": "0.8505 frame-level F1 (0.7655 precision, 0.9567 recall, 10 ms frames) on the 149.9 s CantoMap downmix against the union of 83 ELAN utterance intervals; an activity gate for this exact configuration only, and no evidence of language or dialect coverage, turns, overlap, or chained VAD-plus-ASR behaviour",
                      "record": "model_tests/benchmark/results/2026-08-15-silero-vad.json"}
  },
  "packages": [
    {"package": "silero-vad", "environment": "core", "kind": "weights",
     "bytes": null, "provisioned": true, "auto_fetch": true,
     "note": "hash-pinned single file; fetched on first use, so it never returns exit 3"},
    {"package": "qwen3-forcedaligner", "environment": "torch", "kind": "weights",
     "bytes": null, "provisioned": false}
  ]
}
```

`speech_bounds` is the one derived capability with a real number behind it, so the
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
audio packages pull --stack qwen-0.6b --want speaker_attribution
audio transcribe run meeting.m4a --stack qwen-0.6b --want speaker_attribution \
  --format md
```

Measured on the 30-minute Cantonese SpiCE fixture, Apple M4 Max / 64 GiB, batch 1
with the MLX cache cleared after every batch: `qwen-0.6b` ran 29.90 s at 1.66 GiB RSS
versus `qwen-1.7b` at 53.77 s and 3.02 GiB, and scored 52.64% mixed-token error
versus 33.56%. Those are ASR-stage walls on identical diarized turns, both with the
`"Cantonese"` language hint, and the accuracy comparison is Cantonese-only.

`qwen-0.6b` is not simply a smaller `qwen-1.7b`. On the 139.284-second probe it
rendered `刷啥子` where 1.7B retained `耍啥子`, so its `verbatim` catalog entry carries
`quality: "refuted"` with that `observed_limit` while 1.7B's carries `unmeasured`.
Identical resolution, different recorded fidelity — which is why the two share a
column in the derivation table and not a catalog.

## 2. Product-demo editing — `vibevoice`

Verbatim-oriented text with native anonymous speaker structure and word
intervals for an editing agent.

```bash
audio transcribe plan demo.mp4 --stack vibevoice \
  --want verbatim,speaker_attribution,segment_bounds,word_bounds
```

`speaker_attribution` and `segment_bounds` are `native`, so this stack needs no
diarizer and no reconciler. Only `word_bounds` adds the aligner. **Abridged to the
fields that differ from §1.1** — the envelope, `request`, `packages`, and
`sample_output` take the same shape. `provenance_only` does not take the same shape:
it is `{}` here, because VibeVoice is handed whole media in a single call and has no
processing container to record.

```json
{
  "roles": {
    "decode":  {"backend": "ffmpeg",
                "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "asr":     {"backend": "vibevoice-asr-7b", "environment": "torch",
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
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "torch",
                "config": {"scope": "all_segments"},
                "selected_by": "add_on_required_by:word_bounds"}
  },
  "floors": ["punctuated_sentence_segmented_text", "punctuation_is_sentence_level",
             "canonical_timeline", "no_synthesized_bounds", "abstentions_survive",
             "adapter_normalization"],
  "provenance_only": {},
  "policy": {
    "policy_version": 1,
    "overlap": "abstain",
    "overlap_detection": "unavailable",
    "overlap_detection_note": "no backend in this plan detects overlap, so an empty abstention ledger means undetected, not absent",
    "abstention_reasons": []
  },
  "capabilities": {
    "verbatim":            {"satisfaction": "native",
                            "evidence": {"interface": "verified", "quality": "refuted"},
                            "observed_limit": "normalized 看哈 to 看一下 on the 27.8 s Sichuanese probe, where firered retained it; filler recall unmeasured",
                            "record": "model_tests/EXPERIMENT_RESULTS.md"},
    "speaker_attribution": {"satisfaction": "native",
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "measured_limit": "on the 149.9 s CantoMap conversation with 75 annotated speaker changes this stack matched 39; not validated for rapid backchannels, interruptions, or dense overlap",
                            "record": "model_tests/benchmark/DIARIZATION.md"},
    "segment_bounds":      {"satisfaction": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "word_bounds":         {"satisfaction": "derived",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "boundary MAE/P95 unlabeled; absent on segments with no speech to align"}
  },
  "unsized_packages": ["vibevoice-asr-7b", "qwen3-forcedaligner"],
  "measured": {
    "reference_run": "spice-30min-participant",
    "hardware": "apple-m4-max-64gib",
    "config": "mps, bfloat16, sdpa, seed 1234, logits_to_keep patch applied",
    "generation_seconds": 851.1,
    "peak_mps_live_bytes": 21770457600,
    "record": "model_tests/benchmark/results/2026-08-12-evidence.json"
  },
  "warnings": [
    {"code": "measured_peak_exceeds_target", "blocking": false,
     "detail": "measured 20.28 GiB live MPS allocation on spice-30min-participant; a strict 16 GiB MPS cap OOMs at model load, measured on a 27.8 s probe, while an 18 GiB cap passed that probe"}
  ]
}
```

```bash
audio packages pull --stack vibevoice \
  --want verbatim,speaker_attribution,segment_bounds,word_bounds
audio packages verify
audio transcribe run demo.mp4 --stack vibevoice \
  --want verbatim,speaker_attribution,segment_bounds,word_bounds \
  --format json -o demo.transcript.json
```

`verbatim` is the reason this stack's catalog is worth reading before choosing it.
It resolves `native` here exactly as it does on `firered`, and the difference is
entirely in the evidence: a recorded run *refuted* it. `quality: "refuted"` with an
`observed_limit` is not the same statement as `unmeasured`, and filing the
normalization as unmeasured would have made the stack that failed the probe read
like the stack that was never tested.

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
audio transcribe plan field.wav --stack firered \
  --want verbatim,word_bounds,speech_bounds,segment_bounds
```

Every requirement is `native`, so there are no add-ons at all — and this is the only
plan in this document that resolves `punctuator`, and the only one whose `vad` comes
from inside the stack rather than as an add-on. Abridged to `roles` and `policy`:

```json
{
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "vad":        {"backend": "firered-vad", "environment": "torch",
                   "selected_by": "stack"},
    "asr":        {"backend": "firered-asr2-aed", "environment": "torch",
                   "config": {"device": "cpu", "dtype": "float32", "batch_size": 4,
                              "return_timestamp": true},
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 1.0,
                   "determinism_basis": "exact-repeat 60-minute fixture: both halves' text sequences equal the standalone 30-minute run, maximum rebased timestamp drift 1.0 ms within a declared 2.0 ms tolerance; normalized segments are therefore not byte-equal"},
    "punctuator": {"backend": "firered-punc", "environment": "torch",
                   "config": {"batch_size": 4},
                   "selected_by": "floor:punctuated_sentence_segmented_text",
                   "recases_text": true}
  },
  "policy": {
    "policy_version": 1,
    "overlap": "abstain",
    "overlap_detection": "unavailable",
    "overlap_detection_note": "no backend in this plan detects overlap, so an empty abstention ledger means undetected, not absent",
    "abstention_reasons": []
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
audio packages pull --stack firered \
  --want verbatim,word_bounds,speech_bounds,segment_bounds
audio packages verify
audio transcribe run field.wav --stack firered \
  --want verbatim,word_bounds,speech_bounds,segment_bounds \
  --format json -o field.transcript.json
```

The LID weights are fetched only when `region_language` is in the plan, so this
request provisions less than the full package. Neither figure is recorded in a
tracked artifact — the only tracked source is a pre-harness "~9.2 GB" note that
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
audio transcribe plan field.wav --stack firered \
  --want verbatim,word_bounds,region_language
audio packages pull --stack firered --want verbatim,word_bounds,region_language
audio transcribe run field.wav --stack firered \
  --want verbatim,word_bounds,region_language --format json -o field.lid.json
```

That plan adds the eighth and last role, and it is the one case where a requirement
turns on a stage the stack already contains rather than adding a package:

```json
{
  "roles": {
    "lid": {"backend": "firered-lid", "environment": "torch",
            "config": {"batch_size": 4},
            "selected_by": "requirement:region_language",
            "granularity": "vad_region",
            "cost_note": "162.09 s with LID versus 84.24 s without, on the 139.284 s probe"}
  },
  "capabilities": {
    "region_language": {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "one label per VAD region, copied onto the sentences inside it; no per-sentence detection was measured"}
  }
}
```

Note `selected_by: "requirement:region_language"` rather than
`add_on_required_by:region_language`. Nothing was added — the `lid` role is declared
by the stack and was listed in step one's `roles_conditional`. The five
`selected_by` forms are `stack`, `requirement:<capability>`,
`add_on_required_by:<capability>`, `floor:<floor>`, and `pin:<flag>`. `decode` is the
one role that carries no `selected_by` at all, because it is unconditional; the field
exists to explain why a role that could have been absent is present.

FireRed has no speaker output, so speaker attribution here is an add-on like it
is on Qwen. Note the `pull` — entering at this section without it exits 3, since
nothing earlier in §3 provisioned a diarizer:

```bash
audio packages pull --stack firered \
  --want verbatim,word_bounds,speaker_attribution,overlap_intervals
audio transcribe run interview.wav --stack firered \
  --want verbatim,word_bounds,speaker_attribution,overlap_intervals \
  --format json -o interview.firered.json
```

## 4. Export

Deterministic post-processing. No stack, no packages, no `plan`/`run` split.

```bash
audio export meeting.timed.json --format srt -o meeting.srt
audio export meeting.timed.json --format vtt -o meeting.vtt
audio export meeting.transcript.json --format md
audio export meeting.transcript.json --format txt
audio export meeting.transcript.json --format jsonl
```

Subtitle formats require word timing and refuse without it.
`meeting.transcript.json` from §1.3 has none, so:

```bash
audio export meeting.transcript.json --format srt
```

Exit 2, stderr:

```json
{
  "code": "timing_required_for_format",
  "field": "--format",
  "provided": "srt",
  "requires_capability": "word_bounds",
  "found": ["container_bounds"],
  "note": "container bounds are processing extents, not cue timing",
  "fix": "audio transcribe run meeting.m4a --stack qwen-1.7b --want speaker_attribution,word_bounds --format json -o meeting.timed.json"
}
```

`md` and `txt` are for people. `jsonl` is one segment object per line, ordered by
start time — the same segment objects the JSON result carries, without the envelope
or the provenance — so a consumer can stream or `grep` a long transcript without
parsing the whole document. It has no timing requirement, and because it drops the
provenance it is an export for reading, not an artifact to audit against.

VTT carries speaker labels as voice tags when `speaker_attribution` is present,
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

```bash
audio transcribe run meeting.m4a --want speaker_attribution
```

Exit 2, `code: "stack_required"`, `field: "--stack"`, `allowed` listing the four
stack ids, and `stacks` mapping each to a one-line characterization plus a pointer to
the decision report.

```bash
audio transcribe plan meeting.m4a --stack qwen-1.7b --want segment_bounds
```

Exit 2, `code: "capability_unsatisfiable_on_stack"`, `capability: "segment_bounds"`,
`allowed: ["vibevoice", "firered"]`. Reported by `plan` as well as `run`, before
anything loads, and the fix is actionable: switch stacks. Qwen's only time-like
output is the processing container, and promoting that to a segment extent is the
fabrication this code exists to prevent.

```bash
audio transcribe plan meeting.m4a --stack qwen-1.7b --want word_timing
```

Exit 2, `code: "capability_unknown"`, `field: "--want"`, `provided: "word_timing"`,
`allowed` listing the nine requestable names. A misspelling and an unsatisfiable
requirement are different failures: this one has no `capability` field, because the
name is not one.

```bash
audio transcribe plan meeting.m4a --stack firered --want token_language
```

Exit 2, `code: "capability_unsupported"`, `capability: "token_language"`,
`allowed: []`, `reason: "no_backend_declares"`. An impossible request is an error
rather than a silently `unavailable` field, so an agent gets an explicit answer
instead of assuming code-switching support implies per-token labels. Step one's
catalog is where this is *discovered* without erroring.

```bash
audio transcribe plan meeting.m4a --stack qwen-1.7b --want container_bounds
```

Exit 2, `code: "capability_not_requestable"`, `capability: "container_bounds"`,
`allowed: []`, and `alternatives: ["word_bounds", "segment_bounds", "turn_bounds"]`
so the caller can pick the granularity it actually wanted. The pair is present in
the output as provenance; what it cannot be is a request.

```bash
audio transcribe plan meeting.m4a --stack vibevoice \
  --want speaker_attribution --diarizer fluidaudio
```

Exit 2, `code: "pin_conflicts_with_native_capability"`, `field: "--diarizer"`,
`provided: "fluidaudio"`, `allowed: []`, `capability: "speaker_attribution"`, because
`vibevoice` satisfies it natively and no diarizer role exists in this plan. Pins
select among implementations of a role the plan actually contains.

A backend crash is the one failure that is not a refusal:

```json
{
  "code": "backend_failed",
  "role": "asr",
  "backend": "vibevoice-asr-7b",
  "detail": "MPS backend out of memory during generate",
  "fix": "retry with --stack qwen-1.7b --want word_bounds, or free memory; the plan's measured_peak_exceeds_target warning applies"
}
```

Exit 1, and no result is written. This must stay distinguishable from an abstention,
which is a *successful* run that declines to assert something: exit 0, a result, and
a ledger entry. Collapsing the two would make a crash and a principled refusal look
identical to a caller.

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
| `speaker_attribution` | §1, §2, §3 | derived on Qwen and FireRed, native on VibeVoice |
| `turn_bounds` | §1.4 | derived |
| `overlap_intervals` | §1.4, §3 | derived |
| `speech_bounds` | §1.4, §3 | derived on Qwen via `silero-vad`, native on FireRed |
| `segment_bounds` | §2, §3, §5 | native on VibeVoice and FireRed; exit 2 `unsatisfiable_on_stack` on Qwen |
| `word_bounds` | §1.4, §2, §3 | derived on Qwen and VibeVoice, native on FireRed |
| `region_language` | §3 | native on FireRed only, with its inference cost and region granularity |
| `token_language` | step one, §5 | `impossible` in the catalog; exit 2 `unsupported` when requested |
| `container_bounds`, `container_language` | §1.1, §5 | provenance-only; exit 2 `not_requestable` when requested |
| `capture_role`, `*_candidates` | step one | `impossible` in the catalog; exit 2 `unsupported` with `reason: "not_implemented_v1"` when requested |

All eight roles appear in a resolved plan: `decode` and `asr` in §1.1, §2 and §3;
`diarizer` and `reconciler` in §1.1; `vad` in §1.4 (`silero-vad`) and §3
(`firered-vad`); `aligner` in §1.4 and §2; `punctuator` in §3; `lid` in §3's LID
variant. Four of the five `selected_by` forms appear in a resolved plan above —
`stack`, `requirement`, `add_on_required_by`, and `floor`; the fifth, `pin`, appears
only in §5 where a pin is rejected. Every error code in the table at the top of this
document is shown with its payload or its field list.
