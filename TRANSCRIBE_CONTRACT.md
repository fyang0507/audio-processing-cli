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

Every error payload carries `code` and `fix`. Remaining fields are per-code:
validation errors add `field`, `provided`, and `allowed`; provisioning errors add
`missing`; capability errors add `capability` and `allowed`.

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
  "roles_included": ["vad", "lid", "asr", "punctuator"],
  "environment": "torch",
  "floors": ["punctuated_sentence_segmented_text", "punctuation_attached_to_word",
             "canonical_timeline", "no_synthesized_bounds", "abstentions_survive",
             "adapter_normalization"],
  "capabilities": {
    "verbatim":            {"availability": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "retained 看哈 on the 27.8 s Sichuanese probe; two examples cannot rank varieties"},
    "word_bounds":         {"availability": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "word_confidence":     {"availability": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "speech_bounds":       {"availability": "native", "stage": "FireRedVAD",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "segment_bounds":      {"availability": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "region_language":     {"availability": "native", "stage": "FireRedLID",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "cost": "~2x inference; additional weights, size unrecorded"},
    "speaker_attribution": {"availability": "requires_add_on", "add_on": ["fluidaudio", "reconciler"],
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "measured_limit": "3 of 75 annotated speaker changes matched on a 149.9 s dense conversation"},
    "turn_bounds":         {"availability": "requires_add_on", "add_on": ["fluidaudio"],
                            "evidence": {"interface": "verified", "quality": "measured"}},
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

`provenance_only` is present in every plan and every catalog, empty here because
`container_bounds` and `container_language` are Qwen artifacts. Discovering that
`token_language` is `impossible` is what step one is for; *requesting* it is an
error, not a field (see Refusals).

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
turns; whether the abstention ledger is populated, since that depends on the
audio actually containing overlap; and output quality wherever a capability's
`evidence.quality` is `unmeasured`. The sample is the contract for a successful
run; a backend failure exits 1 and writes nothing.

Placeholder timing and text values are `null`, never `0.0` or a plausible
string. `0.0` is a legal timestamp and would violate the `no_synthesized_bounds`
floor the moment a consumer read it as measured. Real metadata that the plan
genuinely has — duration, path — is populated rather than stubbed.

Enum-valued fields are the one exception: they show a legal member rather than
`null`, because the member set is part of the shape a consumer needs. So
`"reason": "overlap"` is shape, while `"text": null` is content withheld. Free-text
and numeric fields are always `null`.

The binding test is that the sample's key set equals a real run's key set, and
that no key exists for a capability that was not requested. That is the
anti-fabrication guarantee, and it is parametrized over the derivation table:
each stack against each capability it satisfies natively, each capability
requiring an add-on, and one it cannot satisfy at all.

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
                   "config": {"batch_size": 1, "clear_mlx_cache_after_every_turn": true},
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_basis": "greedy decode; by construction, not a repeat-hash measurement"},
    "diarizer":   {"backend": "fluidaudio", "version": "0.15.5",
                   "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
                   "environment": "swift",
                   "selected_by": "add_on_required_by:speaker_attribution"},
    "reconciler": {"backend": "sample-exact-turn-partition",
                   "config": {"partition": "sample_exact"},
                   "selected_by": "add_on_required_by:speaker_attribution"}
  },
  "policy": {
    "policy_version": 1,
    "overlap": "abstain",
    "min_turn_ms": 250,
    "rendering": "clean",
    "synthesize_bounds": false,
    "protected_intervals": "fail_closed"
  },
  "capabilities": {
    "speaker_attribution": {"satisfaction": "derived", "backend": "fluidaudio",
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "note": "reconciled sample-exactly onto ASR text; anonymous labels only",
                            "measured_limit": "on a 149.9 s conversation with 75 annotated speaker changes this diarizer matched 3; not validated for rapid backchannels, interruptions, or dense overlap"}
  },
  "provenance_only": {
    "container_bounds":   {"backend": "qwen3-asr-1.7b-8bit",
                           "note": "processing container extents; not time evidence"},
    "container_language": {"backend": "qwen3-asr-1.7b-8bit",
                           "note": "single label; disagreed across Qwen sizes on one clip"}
  },
  "requires": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "kind": "weights",
     "bytes": 2463307541, "provisioned": false},
    {"package": "fluidaudio", "environment": "swift", "kind": "toolchain",
     "requires": ["swift"], "bytes": null, "provisioned": false},
    {"package": "speaker-diarization-coreml", "environment": "swift", "kind": "weights",
     "bytes": null, "license": "CC-BY-4.0", "provisioned": false}
  ],
  "total_known_download_bytes": 2463307541,
  "unsized_packages": ["fluidaudio", "speaker-diarization-coreml"],
  "measured": {
    "reference_run": "spice-30min-canonical-mix",
    "hardware": "apple-m4-max-64gib",
    "config": "qwen batch 1, cache cleared after every turn, 195 diarized turns",
    "asr_stage_wall_seconds": 53.77,
    "peak_rss_bytes": 3241689088,
    "end_to_end_wall_seconds": null,
    "end_to_end_note": "not separately measured for 1.7B; observed stages total about 69 s",
    "record": "model_tests/benchmark/results/2026-08-13-turn-attributed-fast-asr.json"
  },
  "warnings": [],
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "source": {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "start": null, "end": null, "speaker": null}
    ],
    "abstentions": [
      {"abstention_id": "ab_0", "reason": "overlap", "start": null, "end": null}
    ],
    "provenance": {"plan_version": 1, "stack": "qwen-1.7b"}
  }
}
```

Two things the sample deliberately does *not* contain. There is no `turns` array,
because `turn_bounds` was not requested — §1.4 adds it explicitly. And segments
carry no `words` array, because `word_bounds` was not requested. Keys are absent
rather than null-valued at the container level, so absence can never read as a
measured value.

The `abstentions` ledger *is* present, and not as an exception to that rule: it is
a floor artifact, governed by `policy.overlap` above rather than by any requested
capability, so the one-placeholder-per-request rule does not reach it. It is also
genuinely producible from this exact request, since FluidAudio emits
overlap-permitting output. Whether it fills depends on the audio.

`duration_seconds` is populated because `plan` reads container metadata rather
than stubbing what it already knows. The value `1794.2` is illustrative — this
document has no real `meeting.m4a` — and is deliberately *not* the 1800 s figure
in the `measured` block above it, which describes a fixture, not this input.

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
    {"package": "fluidaudio", "kind": "toolchain", "requires": ["swift"], "bytes": null},
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

### 1.5 Smaller stack, named directly

```bash
audio transcribe run meeting.m4a --stack qwen-0.6b --want speaker_attribution \
  --format md
```

Measured on the 30-minute Cantonese SpiCE fixture, Apple M4 Max / 64 GiB, batch 1
with the cache cleared after every turn: `qwen-0.6b` ran 29.90 s at 1.66 GiB RSS
versus `qwen-1.7b` at 53.77 s and 3.02 GiB, and scored 52.64% mixed-token error
versus 33.56%. Those are ASR-stage walls on identical diarized turns. The accuracy
comparison is Cantonese-only.

## 2. Product-demo editing — `vibevoice`

Verbatim-oriented text with native anonymous speaker structure and word
intervals for an editing agent.

```bash
audio transcribe plan demo.mp4 --stack vibevoice \
  --want verbatim,speaker_attribution,segment_bounds,word_bounds
```

`speaker_attribution` and `segment_bounds` are `native`, so this stack needs no
diarizer and no reconciler. Only `word_bounds` adds the aligner. **Abridged to the
fields that differ from §1.1** — the envelope, `request`, `provenance_only`,
`requires`, and `sample_output` all take the same shape:

```json
{
  "roles": {
    "decode":  {"backend": "ffmpeg",
                "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "asr":     {"backend": "vibevoice-asr-7b", "environment": "torch",
                "revision": "94da20d98b2fa7688e9cbfaf7692ddb4954f7600",
                "patch": "vibevoice-logits-to-keep",
                "config": {"device": "mps", "dtype": "bfloat16", "attention": "sdpa",
                           "seed": 1234},
                "deterministic": true,
                "determinism_basis": "three seeded repeats shared one normalized-output hash",
                "determinism_note": "acoustic tokenizer samples a Gaussian latent; fixed seed required",
                "selected_by": "stack"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "torch",
                "config": {"scope": "all_segments"},
                "selected_by": "add_on_required_by:word_bounds"}
  },
  "policy": {
    "policy_version": 1,
    "overlap": "abstain",
    "min_turn_ms": 250,
    "rendering": "verbatim",
    "synthesize_bounds": false,
    "protected_intervals": "fail_closed"
  },
  "capabilities": {
    "verbatim":            {"satisfaction": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "normalized 看哈→看一下 on the Sichuanese probe; filler recall unmeasured"},
    "speaker_attribution": {"satisfaction": "native",
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "measured_limit": "on a 149.9 s conversation with 75 annotated speaker changes this stack matched 39; not validated for rapid backchannels, interruptions, or dense overlap"},
    "segment_bounds":      {"satisfaction": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "word_bounds":         {"satisfaction": "derived",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "boundary MAE/P95 unlabeled"}
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

The memory warning is advisory by explicit product decision, and it is emitted
from the plan rather than as a mid-run OOM. Its reference run took roughly
fourteen minutes of generation for thirty minutes of audio — an RTF near 0.47,
which is the figure to scale by; the plan cannot know `demo.mp4`'s duration cost
in advance. Cut and rerender from the original media; this command only reads it.

No byte size is recorded anywhere for `vibevoice-asr-7b` or
`qwen3-forcedaligner`, so both appear in `unsized_packages`. The tracked figures
are a legacy "~17 GB" note and a 16.157 GiB BF16 weight floor, neither of which is
a provisioning measurement.

## 3. Dialect and audit — `firered`

The only stack with native word timing, word confidence, native speech bounds,
and a region language label.

```bash
audio transcribe plan field.wav --stack firered \
  --want verbatim,word_bounds,word_confidence,speech_bounds,segment_bounds
```

Every requirement is `native`, so there are no add-ons at all.

```bash
audio packages pull --stack firered \
  --want verbatim,word_bounds,word_confidence,speech_bounds,segment_bounds
audio transcribe run field.wav --stack firered \
  --want verbatim,word_bounds,word_confidence,speech_bounds,segment_bounds \
  --format json -o field.transcript.json
```

The LID weights are fetched only when `region_language` is in the plan, so this
request provisions less than the full package. Neither figure is recorded in a
tracked artifact — the only tracked source is a pre-harness "~9.2 GB" note that
its own document marks as history rather than decision evidence — so both appear
as `approximate, unrecorded` until per-artifact sizes are recorded the way the
MLX runs record `weight_bytes`.

This is the stack the `punctuation_attached_to_word` floor is aimed at. FireRedPunc
runs as a separate stage and emits punctuation with its own bounds, so the adapter
reattaches each mark to the preceding word token and drops the mark's bounds; the
word keeps its own `start` and `end` unchanged, since extending them over the
punctuation would synthesize a bound. Cue splitting reads the mark's presence in
the token, which is all it needs.

Adding the region language label pulls LID and roughly doubles inference: 162.09
versus 84.24 seconds on the 139.284-second probe, CPU float32 at batch size 4,
with identical ASR text and all 246 word texts and times in both runs.

```bash
audio transcribe plan field.wav --stack firered \
  --want verbatim,word_bounds,region_language
audio packages pull --stack firered --want verbatim,word_bounds,region_language
audio transcribe run field.wav --stack firered \
  --want verbatim,word_bounds,region_language --format json -o field.lid.json
```

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
  "requires": "word_bounds",
  "found": ["container_bounds"],
  "note": "container bounds are processing extents, not cue timing",
  "fix": "audio transcribe run meeting.m4a --stack qwen-1.7b --want speaker_attribution,word_bounds --format json -o meeting.timed.json"
}
```

VTT carries speaker labels as voice tags when `speaker_attribution` is present,
which is a commitment to VTT as a real format rather than SRT with dots. Cue
segmentation is deterministic and belongs here; its parameters, break-priority
order, and millisecond-quantization invariants are specified in issue #10, and v1
may ship them hard-coded. Timing quality is not yet validated: boundary MAE/P95
is unmeasured for both FireRed's native times and the aligner, so these files are
producible but not yet claimed broadcast-acceptable.

## 5. Refusals

```bash
audio transcribe run meeting.m4a --want speaker_attribution
```

Exit 2, `code: "stack_required"`, listing `qwen-1.7b`, `qwen-0.6b`, `vibevoice`,
`firered` with a one-line characterization of each and a pointer to the decision
report.

```bash
audio transcribe plan meeting.m4a --stack qwen-1.7b --want word_confidence
```

Exit 2, `code: "capability_unsatisfiable_on_stack"`, `capability:
"word_confidence"`, `allowed: ["firered"]`. Reported by `plan` as well as `run`,
before anything loads. The fix is actionable: switch stacks.

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

Exit 2, `code: "capability_not_requestable"`, pointing at `word_bounds`,
`segment_bounds`, or `turn_bounds` depending on the granularity wanted.

```bash
audio transcribe plan meeting.m4a --stack vibevoice \
  --want speaker_attribution --diarizer fluidaudio
```

Exit 2, `code: "pin_conflicts_with_native_capability"`, `field: "--diarizer"`,
`provided: "fluidaudio"`, `allowed: []`, because `vibevoice` satisfies
`speaker_attribution` natively and no diarizer role exists in this plan. Pins
select among implementations of a role the plan actually contains.

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
| `verbatim` | §2, §3 | native on both, with the normalization caveat on VibeVoice |
| `speaker_attribution` | §1, §2, §3 | derived on Qwen and FireRed, native on VibeVoice |
| `turn_bounds` | §1.4 | derived |
| `overlap_intervals` | §1.4, §3 | derived |
| `speech_bounds` | §1.4, §3 | derived on Qwen, native on FireRed |
| `segment_bounds` | §2, §3 | native; exit 2 on Qwen |
| `word_bounds` | §1.4, §2, §3 | derived on Qwen and VibeVoice, native on FireRed |
| `word_confidence` | §3, §5 | native on FireRed; exit 2 elsewhere |
| `region_language` | §3 | native on FireRed only, with its inference cost |
| `token_language` | step one, §5 | `impossible` in the catalog; exit 2 when requested |
| `container_bounds`, `container_language` | §1.1, §5 | provenance-only; exit 2 when requested |
| `capture_role`, `*_candidates` | step one | `impossible` in the catalog; exit 2 `capability_unsupported` with `reason: "not_implemented_v1"` when requested |
