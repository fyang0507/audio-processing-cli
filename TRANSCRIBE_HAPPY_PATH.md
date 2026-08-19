# `transcribe` happy paths — mocked, per use case

**Status: mock.** Nothing here has been executed; no command below exists yet. This is
the unabridged step-by-step an implementer can diff against and an agent can read as a
worked example. [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) is organized around
*why* the surface looks like this and abridges output to whatever differs; this document
does the opposite — it shows every command in order and the complete stdout of each, with no
design commentary. §4 covers the refusals, because what an agent actually sees includes being
told it was wrong, and a refusal that cannot be acted on is a worse outcome than a slow run.

Three use cases, one per recommended stack row in
[model_tests/DECISION_REPORT.md](model_tests/DECISION_REPORT.md):

| § | Use case | Stack | Deliverable |
| --- | --- | --- | --- |
| [1](#1-fast-long-form-transcription--interview) | Fast long-form transcription | `qwen-1.7b` | Speaker-attributed transcript + SRT |
| [2](#2-video-editing--product-demo) | Video editing | `vibevoice` | Verbatim segments with word timing + VTT |
| [3](#3-dialect-field-recording) | Dialect field recording | `firered` | Audited transcript with native word timing + region LID |

## How to read the output blocks

Every value below is one of three things, and they are never mixed:

- **Measured.** Traceable to a recorded artifact. All of these appear inside
  `cost.proved`, the `capabilities[].note` sentences, package byte counts, revisions, and
  configuration. Every one is checkable against `model_tests/`.
- **Illustrative.** Transcript text, speaker labels, timestamps, cue text, and input
  durations. These are made up. They are internally consistent — bounds are monotonic,
  words fall inside their segment, nothing exceeds the source duration — so the shapes
  are usable, but no number here is a claim about any real recording. Input durations
  borrow the real local fixture durations (1794.2 s is illustrative; 112.4 s and 27.8 s
  match actual local media) purely so they are plausible.
- **Structural.** Key sets, nesting, enum members, and which fields are absent. These
  are the part that must match a real run exactly, and they are the reason this document
  exists.

Arrays are printed short. A real 30-minute interview yields hundreds of segments, not
two. Cardinality is the one thing a plan's `sample_output` explicitly does not promise,
so it is not promised here either — but note that the arrays contain no ellipsis
placeholder, because a `"...": "..."` pseudo-key would corrupt the key-set comparison
this document is meant to support.

### Names this document introduces

The contract pins the plan document and the envelope of a result. It does not name every
result field, because it never printed a full result. The following names appear here
first and should be read as **proposed, pending confirmation**, not as already settled:

| Name | Carries | Requested by | Shown below |
| --- | --- | --- | --- |
| `segments[].words[]` with `word_id` | word text and bounds | `word_timestamps` | §1.4, §2.2, §3.1 |
| `vad_regions[]` | speech-activity regions | `vad` | §3.1 |
| `lid_regions[]` | region language labels | `lid` | §3.1 |
| `turns[]` with `turn_id` | speaker turn intervals | `diarization` | §1.2, §1.4, §2.1, §2.2 |
| `overlapped_speech[]` with `overlap_id` | cross-speaker overlap intervals | `overlapped_speech` | not exercised |

`overlapped_speech[]` is the one result key this document does not demonstrate: none of the
three use cases requests it. `TRANSCRIBE_CONTRACT.md` §1.4 is where it is requested.

`turns[]` arrives with `diarization` rather than as a separate request, because there is no
use for one without the other — labelling a transcript "three people spoke here" while
withholding who said what answers nothing, and intervals-without-text was never purchasable
anyway, since punctuated text is a floor rather than a capability. Both come out of the same
diarizer run, so splitting them would have priced one stage twice. Note that the turn bounds
in §1.4 are not the segments' word bounds: the diarizer measured them independently, and the
plan keeps both rather than deriving one from the other.

Ids are positional and **document-scoped** (`seg_0`, `w_0`, `turn_0`, `ab_0`), extending
the `segment_id`/`abstention_id` convention the contract's `sample_output` already shows.
They are deliberately *not* stable across runs, and nothing should be built on the
assumption that they are: the CLI offers no way to re-run the same input under the same
parameterization for a comparable result, so cross-run identity would be a guarantee with
no caller. The practical consequence is that merging two result documents — a partial run
plus its resumed remainder — means re-numbering, which is why `export` accepts several
transcripts in timeline order rather than expecting a consumer to splice JSON.

Everything a capability was not requested for is **absent**, not null. That is the
anti-fabrication guarantee, and it is why §1's segments carry no `start`/`end`.

## 0. Once per machine

```bash
brew install ffmpeg
uv tool install .
audio doctor
```

```json
{
  "tool": {"version": "0.1.0", "python": "3.12.12",
           "path": "/Users/you/.local/bin/audio"},
  "platform": {"system": "Darwin", "release": "25.5.0", "machine": "arm64"},
  "tools": {
    "ffmpeg": {"path": "/opt/homebrew/bin/ffmpeg", "present": true},
    "ffprobe": {"path": "/opt/homebrew/bin/ffprobe", "present": true},
    "git": {"path": "/opt/homebrew/bin/git", "present": true},
    "huggingface_hub": {"path": null, "present": true},
    "swift": {"path": "/usr/bin/swift", "present": true},
    "uv": {"path": "/Users/you/.local/bin/uv", "present": true}
  },
  "memory": {"total_bytes": 68719476736, "available_bytes": 30585126912,
             "note": "Host-wide counters; not process-attributable and not summable with per-stage peaks."},
  "disk": {"total_bytes": 994662584320, "free_bytes": 107730939904},
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "root_exists": true,
  "registry": "/Users/you/Library/Caches/audio-processing-cli/registry.json",
  "environments": {
    "mlx": {"state": "absent", "python": "3.13.9", "requires_tool": [],
            "provisional": false, "blocked_by_missing_tool": []},
    "swift": {"state": "absent", "python": null, "requires_tool": ["swift"],
              "provisional": false, "blocked_by_missing_tool": []},
    "torch-firered": {"state": "absent", "python": "3.12.12", "requires_tool": [],
                      "provisional": false, "blocked_by_missing_tool": []},
    "torch-vibevoice": {"state": "absent", "python": "3.12.12", "requires_tool": [],
                        "provisional": true, "blocked_by_missing_tool": []}
  },
  "packages": {
    "firered-asr2s": "absent", "fluidaudio": "absent",
    "qwen3-asr-0.6b-8bit": "absent", "qwen3-asr-1.7b-8bit": "absent",
    "qwen3-forcedaligner": "absent", "silero-vad": "absent",
    "speaker-diarization-coreml": "absent", "vibevoice-asr-7b": "absent"
  },
  "note": "An absent swift blocks only the packages that need it; it is reported rather than fatal."
}
```

Exit 0, and unlike every other block in this document this one is **real output**, not a mock —
`doctor` is implemented, so its shape is checked against a live run by
`tests/test_shipped_commands_match_the_document.py` rather than maintained by eye — as are the
`packages list` block in §5 and the `packages verify` block in §1.3, the other two commands that
ship. Values remain illustrative: paths, versions, and the memory and disk counters are this
machine's.

`swift` is present here. Absent, it would report `present: false` and appear in the
`blocked_by_missing_tool` list of the `swift` environment — reported rather than fatal, blocking
only the packages that need it. That per-environment list is why there is no top-level
`warnings` array: a blocked environment says so where a caller is already looking.

`environments` reports the four *provisioned* environments and not `core`, which is the CLI's own
and always present. `packages` is a state per package rather than a count, because a caller
deciding whether to `pull` needs to know *which* are absent, not how many.

## 1. Fast long-form transcription — interview

Goal: a speaker-attributed transcript of a 30-minute interview, plus subtitles. Word
timing is required for the subtitles, so it is requested up front rather than discovered
missing at export.

### 1.1 Ask what the stack can do with this file

```bash
audio transcribe capabilities --stack qwen-1.7b --input meeting.m4a
```

```json
{
  "stack": "qwen-1.7b",
  "family": "Qwen3-ASR",
  "environment": "mlx",
  "roles": "asr only",
  "input": {"path": "meeting.m4a", "duration_seconds": 1794.2, "container": "m4a",
            "sample_rate_hz": 44100, "channels": 2},
  "processing": {
    "unit": "fixed_chunk",
    "unit_count": 10,
    "note": "180-second chunks, one at a time. Requesting diarization replaces them with the diarizer's turns, and that count is not known until the diarizer runs."
  },
  "failure_recovery": {
    "partial_results": "per_unit",
    "note": "Units are independent, so a failure leaves the finished ones usable and --range addresses the rest. The global generation budget is what most often stops a long file early, and it stops between units rather than inside one."
  },
  "cost": {
    "proved": "54 s and 3.0 GiB peak to transcribe a 30-minute Cantonese interview on an M4 Max, batch 1 with the MLX cache cleared per batch and a Cantonese language hint.",
    "projected_seconds": 53.6
  },
  "capabilities": {
    "languages": {"availability": "native",
                  "note": "Advertises 30 language labels and has no dialect selector; only Mandarin, English and Cantonese have actually been run here. The one accuracy figure is Cantonese at 33.56% mixed-token error on a 30-minute interview. Accepts a --language hint, which every recorded figure above was produced with; the label it returns without one is not a detector to route on."
                  },
    "verbatim": {"availability": "native",
                 "note": "Does not clean disfluencies, but drops every spoken \"uh\" and sometimes fuses the neighbouring words. Filler recall is unmeasured on every stack."
                 },
    "diarization": {"availability": "requires_add_on",
                    "note": "Adds FluidAudio, which needs a Swift toolchain and a second environment: 15 s and 0.55 GiB peak on a 30-minute sample, and that same run also serves overlapped_speech. Produces speaker labels on the text and the turn intervals together, mapped onto the transcript by an exact partition of the timeline so no span is transcribed twice and no gap is invented. Measured 95.42% participant-interval F1 on a 30-minute interview but matched only 3 of 75 annotated speaker changes on a dense two-speaker conversation — strong on long turns, unsuitable where turns are short or overlapping. RSS excludes memory held by system Core ML services."
                    },
    "overlapped_speech": {"availability": "requires_add_on",
                          "note": "Comes out of the same FluidAudio run as diarization, so asking for both costs one stage. Unmeasured. Without it nothing in the plan detects overlap, so an empty abstention ledger means undetected rather than absent."},
    "vad": {"availability": "requires_add_on",
            "note": "Adds Silero, a hash-pinned file that fetches itself, so it never returns exit 3: 0.4 s and 0.11 GiB peak on a 150-second sample, about 5 s on this input. Measured 0.8505 frame-level F1 at 0.7655 precision and 0.9567 recall, so the gate over-includes — it bounds activity, not speakers."
            },
    "word_timestamps": {"availability": "requires_add_on",
                        "note": "Adds Qwen3-ForcedAligner in the torch environment, so this request spans three: 4.6 s of alignment on a 139-second sample excluding model load, about a minute on this input. Never scored against hand-labelled boundaries, and neither is FireRed's native timing, so switching stacks for accuracy would trade one unmeasured number for another. Absent on any segment with no speech to align."},
    "segment_timestamps": {"availability": "impossible", "reason": "no_native_segment_extents",
                           "note": "This stack emits no segment extents. The chunk boundaries it works in are not speech timing and are never published as any."},
    "lid": {"availability": "impossible", "reason": "no_backend_declares_on_stack",
            "note": "Available on firered, which runs a dedicated LID stage."},
    "token_lid": {"availability": "impossible", "reason": "no_backend_declares",
                  "note": "Named only so a request fails loudly. Code-switching support does not imply per-token labels, and no backend here produces them."}
  },
  "next": "audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want <capabilities>"
}
```

Exit 0. Nothing was downloaded; only the input's container metadata was read.

Every block above answers a question the caller has to decide, and answers it in prose
wherever prose will do. Three enums and two numbers carry everything a program branches on —
`availability`, `processing.unit`, `failure_recovery.partial_results`, the unit count, and
`projected_seconds`. The rest is sentences, because no caller dispatches on a nested
`evidence` object and a reader learns the same thing either way.

`cost.proved` is a run that actually happened, in units a reader can hold: seconds and
gibibytes against a named sample. `projected_seconds` applies that rate to this input so
nobody re-derives it. Where two capabilities come out of one stage the sentence says so —
`diarization` and `overlapped_speech` share a single FluidAudio run — which is the sort of
thing a caller summing per-capability costs would otherwise double-count.

### 1.2 Resolve the request

```bash
audio transcribe plan --input meeting.m4a \
  --stack qwen-1.7b \
  --want diarization,word_timestamps \
  --language Cantonese
```

```json
{
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "diarizer":   {"backend": "fluidaudio", "version": "0.15.5",
                   "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
                   "environment": "swift",
                   "config": {"preset": "quality", "step_ratio": 0.1,
                              "min_segment_duration": 0.0, "output": "regular",
                              "threshold": 0.6, "num_speakers": 2},
                   "config_note": "every cited diarization measurement used a known two-speaker prior; num_speakers must be supplied or the measured_limit figures do not apply",
                   "selected_by": "add_on_required_by:diarization"},
    "asr":        {"backend": "qwen3-asr-1.7b-8bit", "environment": "mlx",
                   "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
                   "config": {"batch_size": 1, "clear_mlx_cache_after_every_batch": true,
                              "language": "Cantonese",
                              "api_path": "_generate_chunks_batched"},
                   "adapter_strips": ["language <label><asr_text> scaffold"],
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 0.0,
                   "determinism_basis": "argmax decode; back-to-back calls in one process produced byte-identical text, cross-process repetition untested"},
    "aligner":    {"backend": "qwen3-forcedaligner", "environment": "mlx",
                   "config": {"scope": "all_segments"},
                   "selected_by": "add_on_required_by:word_timestamps"}
  },
  "execution": {
    "stage_order": ["decode", "diarizer", "asr", "aligner"],
    "residency": "one_model_stage_at_a_time",
    "environments_spanned": ["swift", "mlx"],
    "note": "stages run strictly sequentially and no two model stages are resident together; wall time adds across stages, peak memory does not, and the per-stage peaks below must not be summed"
  },
  "capabilities": {
    "diarization":     {"satisfaction": "derived", "backend": "fluidaudio",
                        "evidence": {"interface": "verified", "quality": "measured"},
                        "note": "Anonymous labels reconciled sample-exactly onto the ASR text, plus the diarizer's turn intervals. This preset matched 3 of 75 annotated speaker changes on a dense conversation and is not validated for rapid backchannels, interruptions, or dense overlap."
                        },
    "word_timestamps": {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "Boundary error against labels is unmeasured, and absent on any segment with no speech to align."}
  },
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "kind": "weights",
     "bytes": 2463307541, "provisioned": false},
    {"package": "fluidaudio", "environment": "swift", "kind": "toolchain",
     "requires_tool": ["swift"], "bytes": null, "provisioned": false},
    {"package": "speaker-diarization-coreml", "environment": "swift", "kind": "weights",
     "bytes": null, "provisioned": false},
    {"package": "qwen3-forcedaligner", "environment": "mlx", "kind": "weights",
     "bytes": null, "provisioned": false}
  ],
  "total_known_download_bytes": 2463307541,
  "unsized_packages": ["fluidaudio", "speaker-diarization-coreml", "qwen3-forcedaligner"],
  "warnings": [],
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "source": {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "speaker": null,
       "words": [{"word_id": "w_0", "text": null, "start": null, "end": null}]}
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

Exit 0. `segment_timestamps` was not requested and is impossible on this stack anyway, so
segments carry no `start`/`end` — in the sample above or in the real result below.

### 1.3 Provision

```bash
audio packages pull --stack qwen-1.7b
```

`pull` takes a stack or a list of package ids, and no `--want`: narrowing a stack to the
capabilities actually requested is the planner's job, so passing it is refused rather than
ignored — see the two refusals at the end of this section. A stack therefore provisions every
package it can use, `silero-vad` included.

Progress goes to stderr; stdout is the receipt:

```json
{
  "pulled": [
    {"package": "fluidaudio", "environment": "swift",
     "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
     "bytes": 471203840, "built": true, "product_runs": true},
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx",
     "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
     "bytes": 2467859030},
    {"package": "qwen3-forcedaligner", "environment": "mlx",
     "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
     "bytes": 1276475979},
    {"package": "silero-vad", "environment": "core",
     "bytes": 2327524, "digest_verified": true},
    {"package": "speaker-diarization-coreml", "environment": "swift",
     "revision": "1ed7a662fdc7109e36d822db793ee6eebdaf8594",
     "bytes": 129243647}
  ],
  "skipped": [],
  "environments_created": ["mlx", "swift"],
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "registry": "/Users/you/Library/Caches/audio-processing-cli/registry.json",
  "pulled_known_bytes": 4347110020,
  "unsized_packages": [],
  "warnings": [
    {"code": "license_unreviewed", "blocking": false,
     "packages": ["qwen3-asr-1.7b-8bit", "qwen3-forcedaligner"],
     "detail": "qwen3-asr-1.7b-8bit, qwen3-forcedaligner report a license their model card declares but nobody has reviewed. A declared license is evidence that one exists, not a redistribution clearance."}
  ]
}
```

Exit 0. `pulled_known_bytes` covers the packages in *this* pull and nothing else. It was called
`reclaimable_bytes`, which read as a running total and is not one — the figure legitimately goes
down on a second, smaller pull. `audio packages list` reports the cumulative
`total_known_bytes`.

`digest_verified` appears on exactly one entry, and that is the point of it. `silero-vad` is
pinned by content hash, so materializing it hashes the file against the manifest. The Hub packages
are pinned by `revision`; there is no hash in the manifest to compare a snapshot against, so they
report the revision they materialized and claim no digest. Every Hub entry used to carry
`digest_verified: true` for a check no code performed.

Run the same line twice and the second run does nothing: a package the registry already calls
`ready` is listed under `skipped`, contributes no bytes, and is not touched — in particular it is
not reopened as `pulling`, which an interrupt would leave behind as a downgraded install.
`pull --repair <package>` is how a caller asks for the work anyway; it re-materializes rather than
trusting what is on disk, which for a Hub snapshot means re-downloading it and for a checkout means
discarding and re-cloning it.

A package whose revision the shared Hugging Face cache already holds is not downloaded again. It
reports `hub_revisions_pre_existing` in place of a fetch, and teardown will not delete it: see
§5.

On a machine with no Swift toolchain this same command exits 0 having provisioned everything else,
and reports what it could not:

```json
{
  "warnings": [
    {"code": "toolchain_missing", "blocking": true,
     "packages": ["fluidaudio"], "requires_tool": ["swift"],
     "detail": "fluidaudio needs swift, which is not on PATH, so it was not provisioned; the rest of stack qwen-1.7b was. Install the toolchain and pull it by name."}
  ]
}
```

That is a `warnings` entry, not an error payload, which is what §0's promise about an absent
`swift` — "reported rather than fatal, blocking only the packages that need it" — actually
requires. `blocking: true` distinguishes it from `license_unreviewed`: something a caller asked
for is absent. Exit 3 is reserved for two cases: a stack where *nothing* was provisionable, and a
package named explicitly. Naming `fluidaudio` on the command line is an instruction, and skipping
an instruction quietly is worse than refusing it.

```bash
audio packages verify
```

```json
{
  "verified": [
    {"package": "fluidaudio", "product_runs": true, "patches_applied": []},
    {"package": "qwen3-asr-1.7b-8bit", "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"},
    {"package": "qwen3-forcedaligner", "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15"},
    {"package": "silero-vad", "digest": "ok"},
    {"package": "speaker-diarization-coreml", "revision": "1ed7a662fdc7109e36d822db793ee6eebdaf8594"}
  ],
  "environments": {"mlx": "ok", "swift": "ok", "torch-firered": "absent",
                   "torch-vibevoice": "absent"},
  "mlx_audio_private_api_expected_source_hash": "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250",
  "mlx_audio_private_api_source_hash": "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250",
  "mlx_audio_private_api_matches_expected": true,
  "failed": []
}
```

Exit 0. On the Swift-less machine above, the same command instead reports:

```json
{
  "environments": {"mlx": "ok", "swift": "blocked", "torch-firered": "absent",
                   "torch-vibevoice": "absent"}
}
```

`blocked` rather than `ok`, and still exit 0. `speaker-diarization-coreml` needs no toolchain of
its own, so it provisioned and left the `swift` environment `ready` in the registry — while
nothing in that environment can run, because the built product is launched through `swift run`.
`ok` there would tell a caller diarization is available on a machine that cannot do it. It is not
a `failed` entry either: nothing provisioned is broken, the gap is a package `list` already
reports as absent, and no `audio` command installs a toolchain for a `fix` to name. The registry
state stays `ready` in `doctor` and `list`, which report what the registry holds and publish
`blocked_by_missing_tool` beside it; only `verify` states a verdict, so only `verify` needed the
fourth word. VOCABULARY.md has the two enumerations.

Which key an entry carries *is* the claim, and the two are not the same claim. `digest: "ok"`
means the bytes on disk were hashed and match the manifest's pin. `revision` means that revision is
pinned and its snapshot is present — the pin is recorded and checkable, the contents were not
hashed, and there is nothing to hash them against. The absent key is the honest report; a
`digest_verified: false` beside a `revision` would confess a check that was never designed rather
than state the one that was. A four-repository package reports `revisions` for the same reason the
receipt does: no one of them is *the* revision.

`verify` reports every provisioned environment, not only the ones this pull touched, so the
two `torch` environments appear as `absent` rather than being omitted — an environment missing from
the map would be indistinguishable from one nobody has looked at. The private-API hash is published
alongside the value it is compared against, because `matches_expected: true` on its own is a claim
a reader cannot check.

Two ways of asking for a pull are refused, both exit 2, and both are refusals of a *silently
ignored argument* rather than of an unsupported idea:

```bash
audio packages pull --stack qwen-1.7b --want diarization,word_timestamps
```

```json
{
  "code": "want_not_implemented",
  "field": "--want",
  "provided": "diarization,word_timestamps",
  "detail": "capabilities cannot narrow a pull yet: resolving them to packages is the planner's job, so --stack qwen-1.7b provisions every package it can use",
  "fix": "audio packages pull --stack qwen-1.7b"
}
```

```bash
audio packages pull --stack qwen-1.7b silero-vad
```

```json
{
  "code": "stack_conflicts_with_named_packages",
  "stack": "qwen-1.7b",
  "packages": ["silero-vad"],
  "detail": "--stack qwen-1.7b was passed alongside named packages; a stack selects every package it can use and named ids select exactly those, so one of the two has to go",
  "fix": "audio packages pull silero-vad"
}
```

`--want` was accepted and dropped on the floor until this pass, and so was `--stack` beside a list
of ids. Both are the failure §4.6 names for `transcribe`: a caller would believe it had constrained
a 4 GiB download it never touched. The `fix` keeps the argument that was an instruction and drops
the one that was a guess.

### 1.4 Run

```bash
audio transcribe run --input meeting.m4a \
  --stack qwen-1.7b \
  --want diarization,word_timestamps \
  --language Cantonese \
  --format json -o meeting.timed.json
```

Exit 0. `meeting.timed.json`:

```json
{
  "schema_version": 1,
  "source": {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"},
  "segments": [
    {"segment_id": "seg_0", "speaker": "S1",
     "text": "好，我們今天想聊一下你的工作。",
     "words": [
       {"word_id": "w_0", "text": "好", "start": 2.31, "end": 2.48},
       {"word_id": "w_1", "text": "我們", "start": 2.62, "end": 2.94},
       {"word_id": "w_2", "text": "今天", "start": 2.94, "end": 3.26},
       {"word_id": "w_3", "text": "想", "start": 3.26, "end": 3.41},
       {"word_id": "w_4", "text": "聊一下", "start": 3.41, "end": 3.98},
       {"word_id": "w_5", "text": "你的", "start": 3.98, "end": 4.27},
       {"word_id": "w_6", "text": "工作", "start": 4.27, "end": 4.71}
     ]},
    {"segment_id": "seg_1", "speaker": "S2",
     "text": "嗯，好啊，我做咗五年設計。",
     "words": [
       {"word_id": "w_7", "text": "嗯", "start": 5.12, "end": 5.29},
       {"word_id": "w_8", "text": "好啊", "start": 5.44, "end": 5.81},
       {"word_id": "w_9", "text": "我", "start": 6.03, "end": 6.16},
       {"word_id": "w_10", "text": "做咗", "start": 6.16, "end": 6.52},
       {"word_id": "w_11", "text": "五年", "start": 6.52, "end": 6.95},
       {"word_id": "w_12", "text": "設計", "start": 6.95, "end": 7.44}
     ]}
  ],
  "turns": [
    {"turn_id": "turn_0", "speaker": "S1", "start": 2.28, "end": 4.79},
    {"turn_id": "turn_1", "speaker": "S2", "start": 5.06, "end": 7.51}
  ],
  "abstentions": [
    {"abstention_id": "ab_0", "reason": "overlap", "start": 41.86, "end": 42.73},
    {"abstention_id": "ab_1", "reason": "short_turn", "start": 118.44, "end": 118.79}
  ],
  "provenance": {
    "stack": "qwen-1.7b",
    "outcomes": {"diarization": "produced", "word_timestamps": "produced"},
    "observed": {
      "stage_wall_seconds": {"decode": 3.91, "diarizer": 14.68,
                             "asr": 54.02, "aligner": 46.77},
      "total_wall_seconds": 119.38,
      "peak_rss_bytes_by_stage": {"diarizer": 588251136, "asr": 3243020288,
                                  "aligner": 2104492032},
      "peak_rss_bytes": 3243020288,
      "segments": 2,
      "words": 13,
      "turns": 2,
      "abstentions": 2
    }
  }
}
```

`provenance` carries three things and embeds a fourth. `stack` names what ran, `outcomes`
says what became of each requested capability, and `observed` records what the run actually
cost. The fourth is `plan`: the executed plan verbatim, which these printouts omit because
§1.2 already shows it in full — a result does not restate what the plan said, it appends
what only running could tell you. The key-set test therefore compares against a real run,
not against these trimmed prints.

Note `peak_rss_bytes` is the **maximum** of the per-stage peaks, not their sum — that is
what `execution.residency` buys, and the per-stage numbers are kept so the claim is
checkable rather than asserted.

### 1.5 Export subtitles

```bash
audio export --input meeting.timed.json --format srt -o meeting.srt
```

```json
{
  "input": "meeting.timed.json",
  "output": "meeting.srt",
  "format": "srt",
  "cues": 2,
  "source_capability": "word_timestamps",
  "speaker_labels_rendered": false,
  "cue_policy": {"max_duration_s": 7.0, "max_lines": 2, "max_chars_per_line_cjk": 16,
                 "break_priority": ["sentence_end", "clause_punctuation", "word_gap"],
                 "never_spans_speaker_change": true, "quantization_ms": 1},
  "warnings": [
    {"code": "cue_timing_unvalidated", "blocking": false,
     "detail": "boundary MAE/P95 is unmeasured for the aligner that produced this timing, so cue placement is producible but not claimed broadcast-acceptable"}
  ]
}
```

Exit 0. `meeting.srt`:

```text
1
00:00:02,310 --> 00:00:04,710
好，我們今天想聊一下你的工作。

2
00:00:05,120 --> 00:00:07,440
嗯，好啊，我做咗五年設計。
```

Cue bounds come from the first and last word of each segment, not from the segment — this
stack has no `segment_timestamps` to use, which is exactly why `word_timestamps` was requested in
step 1.2.

## 2. Video editing — product demo

Goal: verbatim segments with native speaker structure and word timing, for an editing
agent that cuts on speaker changes and needs fillers preserved.

### 2.1 Resolve the request

The `capabilities` report is omitted here for length; it takes the same shape as §1.1 with
`vibevoice`'s own values.

```bash
audio transcribe plan --input demo.mp4 --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps
```

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
                "determinism_basis": "three seeded repeats shared one normalized-output hash; text decode is do_sample=False",
                "determinism_note": "acoustic tokenizer samples a Gaussian latent; fixed seed required",
                "selected_by": "stack"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "mlx",
                "config": {"scope": "all_segments"},
                "selected_by": "add_on_required_by:word_timestamps"}
  },
  "execution": {
    "stage_order": ["decode", "asr", "aligner"],
    "residency": "one_model_stage_at_a_time",
    "environments_spanned": ["mlx", "torch-vibevoice"],
    "note": "both model stages share the torch environment and are still not resident together; VibeVoice and the aligner have never been measured co-resident and this plan does not do so"
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
    "word_timestamps":         {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "boundary MAE/P95 unlabeled; absent on any segment with no speech to align"}
  },
  "packages": [
    {"package": "vibevoice-asr-7b", "environment": "torch-vibevoice", "kind": "weights",
     "bytes": null, "provisioned": false},
    {"package": "qwen3-forcedaligner", "environment": "mlx", "kind": "weights",
     "bytes": null, "provisioned": true}
  ],
  "total_known_download_bytes": 0,
  "unsized_packages": ["vibevoice-asr-7b"],
  "warnings": [
    {"code": "measured_peak_exceeds_target", "blocking": false,
     "detail": "measured 20.28 GiB live MPS allocation on spice-30min-participant; a strict 16 GiB MPS cap OOMs at model load, measured on a 27.8 s probe, while an 18 GiB cap passed that probe"}
  ],
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "source": {"path": "demo.mp4", "duration_seconds": 112.4, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "speaker": null,
       "start": null, "end": null,
       "words": [{"word_id": "w_0", "text": null, "start": null, "end": null}]}
    ],
    "turns": [
      {"turn_id": "turn_0", "speaker": null, "start": null, "end": null}
    ],
    "abstentions": [],
    "provenance": "<stack, outcomes, observed, and the executed plan; elided in print>"
  }
}
```

Exit 0. `qwen3-forcedaligner` is already `provisioned: true` from §1, so
`total_known_download_bytes` is 0 and only VibeVoice needs pulling. `abstentions` is an
empty array rather than absent: it is a floor artifact, and the plan's warning that nothing
here detects overlap is what says it cannot fill.

### 2.2 Provision and run

```bash
audio packages pull --stack vibevoice
audio transcribe run --input demo.mp4 --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps \
  --format json -o demo.transcript.json
```

Exit 0. `demo.transcript.json`:

```json
{
  "schema_version": 1,
  "source": {"path": "demo.mp4", "duration_seconds": 112.4, "timebase": "seconds"},
  "segments": [
    {"segment_id": "seg_0", "speaker": "0", "start": 0.0, "end": 4.52,
     "text": "So, um, this is the new editor. You can, like, drag a clip here.",
     "words": [
       {"word_id": "w_0", "text": "So", "start": 0.31, "end": 0.48},
       {"word_id": "w_1", "text": "um", "start": 0.62, "end": 0.83},
       {"word_id": "w_2", "text": "this", "start": 1.04, "end": 1.22},
       {"word_id": "w_3", "text": "is", "start": 1.22, "end": 1.33},
       {"word_id": "w_4", "text": "the", "start": 1.33, "end": 1.44},
       {"word_id": "w_5", "text": "new", "start": 1.44, "end": 1.69},
       {"word_id": "w_6", "text": "editor", "start": 1.69, "end": 2.21},
       {"word_id": "w_7", "text": "You", "start": 2.58, "end": 2.74},
       {"word_id": "w_8", "text": "can", "start": 2.74, "end": 2.93},
       {"word_id": "w_9", "text": "like", "start": 3.07, "end": 3.31},
       {"word_id": "w_10", "text": "drag", "start": 3.48, "end": 3.77},
       {"word_id": "w_11", "text": "a", "start": 3.77, "end": 3.84},
       {"word_id": "w_12", "text": "clip", "start": 3.84, "end": 4.19},
       {"word_id": "w_13", "text": "here", "start": 4.19, "end": 4.52}
     ]},
    {"segment_id": "seg_1", "start": 4.52, "end": 6.08,
     "text": "[Environmental Sounds]"},
    {"segment_id": "seg_2", "speaker": "1", "start": 6.08, "end": 9.41,
     "text": "And it renders straight away?",
     "words": [
       {"word_id": "w_14", "text": "And", "start": 6.22, "end": 6.39},
       {"word_id": "w_15", "text": "it", "start": 6.39, "end": 6.51},
       {"word_id": "w_16", "text": "renders", "start": 6.51, "end": 7.02},
       {"word_id": "w_17", "text": "straight", "start": 7.02, "end": 7.48},
       {"word_id": "w_18", "text": "away", "start": 7.48, "end": 7.86}
     ]}
  ],
  "turns": [
    {"turn_id": "turn_0", "speaker": "0", "start": 0.0, "end": 4.52},
    {"turn_id": "turn_1", "speaker": "1", "start": 6.08, "end": 9.41}
  ],
  "abstentions": [],
  "provenance": {
    "stack": "vibevoice",
    "outcomes": {"verbatim": "produced", "diarization": "produced", "segment_timestamps": "produced", "word_timestamps": "produced"},
    "observed": {
      "stage_wall_seconds": {"decode": 0.44, "asr": 53.16, "aligner": 3.72},
      "total_wall_seconds": 57.32,
      "peak_mps_live_bytes_by_stage": {"asr": 19983452160, "aligner": 2210398208},
      "peak_mps_live_bytes": 19983452160,
      "segments": 3,
      "words": 19,
      "turns": 2,
      "segments_without_words": 1,
      "abstentions": 0
    }
  }
}
```

Two things in `seg_1` are the recorded VibeVoice behaviours rather than invented shape,
and both are absences.

It carries **no `speaker` key**. VibeVoice emits `Speaker: "N/A"` on non-speech segments,
and the adapter-normalization floor requires that become an absent key rather than a
speaker whose id is the string `"N/A"`. A conforming result cannot contain `"N/A"` as an
attribution anywhere; that string appearing in output is the defect the floor exists to
catch.

It carries **no `words` array**, which is correct and is not an abstention: the aligner is
not run on a segment with no speech to align. So `words` is absent on some segments while
`word_timestamps` is `produced`, and `observed.segments_without_words` records how many.

### 2.3 Export subtitles with speaker voice tags

```bash
audio export --input demo.transcript.json --format vtt -o demo.vtt
```

Exit 0. `demo.vtt`:

```text
WEBVTT

1
00:00:00.310 --> 00:00:04.520
<v Speaker 0>So, um, this is the new editor. You can, like, drag a clip here.

2
00:00:06.220 --> 00:00:07.860
<v Speaker 1>And it renders straight away?
```

The `[Environmental Sounds]` segment produced no cue: it has no word stream, and cue
bounds come from words. Whether a non-speech event tag *should* render as an SDH cue is a
subtitle-convention question parked in issue #10, not a transcription one.

## 3. Dialect field recording

Goal: an audited transcript of a dialect recording, with native word timing, native
speech regions, and a region language label. Every requirement is native, so there are no
add-ons and the plan pulls one package.

### 3.1 Resolve and run

```bash
audio transcribe plan --input field.wav --stack firered \
  --want verbatim,word_timestamps,vad,segment_timestamps,lid
```

```json
{
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "vad":        {"backend": "firered-vad", "environment": "torch-firered",
                   "selected_by": "stack"},
    "lid":        {"backend": "firered-lid", "environment": "torch-firered",
                   "config": {"batch_size": 4},
                   "selected_by": "requirement:lid",
                   "granularity": "vad_region",
                   "cost_note": "162.09 s with LID versus 84.24 s without, on the 139.284 s probe"},
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
  },
  "execution": {
    "stage_order": ["decode", "vad", "lid", "asr", "punctuator"],
    "residency": "one_model_stage_at_a_time",
    "environments_spanned": ["torch-firered"],
    "note": "four model stages in one environment, none resident together"
  },
  "capabilities": {
    "verbatim":        {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "Emits disfluencies rather than cleaning them — 24 filler hits on the probe — and retained the dialect form on both probed clips; two lexemes cannot rank varieties."},
    "word_timestamps": {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "Monotonic across the 30- and 60-minute runs, but accuracy against hand-labelled boundaries is unmeasured."},
    "vad":             {"satisfaction": "native", "stage": "FireRedVAD",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "No measured accuracy for this stage; --vad silero-vad substitutes a path that has a measured 0.8505 frame-level F1."},
    "segment_timestamps":  {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "lid": {"satisfaction": "native", "stage": "FireRedLID",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "one label per VAD region, copied onto every sentence in that region; per-sentence variation would be fabricated"}
  },
  "packages": [
    {"package": "firered-asr2s", "environment": "torch-firered", "kind": "weights",
     "bytes": null, "provisioned": false, "includes_lid_weights": true}
  ],
  "total_known_download_bytes": 0,
  "unsized_packages": ["firered-asr2s"],
  "warnings": [
    {"code": "measured_config_differs_from_plan", "blocking": false,
     "detail": "the measured block above was recorded with lid off while this plan runs it; expect roughly double the inference time and no measured memory figure for the LID-on path"}
  ],
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "source": {"path": "field.wav", "duration_seconds": 27.8, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "start": null, "end": null,
       "words": [{"word_id": "w_0", "text": null, "start": null, "end": null}]}
    ],
    "vad_regions": [{"start": null, "end": null}],
    "lid_regions": [{"start": null, "end": null, "language": null, "confidence": null}],
    "abstentions": [],
    "provenance": "<stack, outcomes, observed, and the executed plan; elided in print>"
  }
}
```

Exit 0. Segments carry no `speaker` key in the sample or the result: FireRed has no
speaker output and `diarization` was not requested.

```bash
audio packages pull --stack firered
audio transcribe run --input field.wav --stack firered \
  --want verbatim,word_timestamps,vad,segment_timestamps,lid \
  --format json -o field.transcript.json
```

Exit 0. `field.transcript.json`:

```json
{
  "schema_version": 1,
  "source": {"path": "field.wav", "duration_seconds": 27.8, "timebase": "seconds"},
  "segments": [
    {"segment_id": "seg_0", "start": 0.38, "end": 1.62,
     "text": "This is a测试。",
     "words": [
       {"word_id": "w_0", "text": "this", "start": 0.41, "end": 0.62},
       {"word_id": "w_1", "text": "is", "start": 0.62, "end": 0.74},
       {"word_id": "w_2", "text": "a", "start": 0.74, "end": 0.81},
       {"word_id": "w_3", "text": "测", "start": 1.02, "end": 1.24},
       {"word_id": "w_4", "text": "试", "start": 1.24, "end": 1.48}
     ]},
    {"segment_id": "seg_1", "start": 3.28, "end": 4.49,
     "text": "我们要来看哈，",
     "words": [
       {"word_id": "w_5", "text": "我", "start": 3.31, "end": 3.44},
       {"word_id": "w_6", "text": "们", "start": 3.44, "end": 3.58},
       {"word_id": "w_7", "text": "要", "start": 3.58, "end": 3.72},
       {"word_id": "w_8", "text": "来", "start": 3.72, "end": 3.86},
       {"word_id": "w_9", "text": "看", "start": 3.86, "end": 4.03},
       {"word_id": "w_10", "text": "哈", "start": 4.03, "end": 4.29}
     ]}
  ],
  "vad_regions": [
    {"start": 0.38, "end": 1.66},
    {"start": 3.28, "end": 4.52}
  ],
  "lid_regions": [
    {"start": 0.38, "end": 1.66, "language": "en", "confidence": 0.724},
    {"start": 3.28, "end": 4.52, "language": "zh", "confidence": 0.961}
  ],
  "abstentions": [],
  "provenance": {
    "stack": "firered",
    "outcomes": {"verbatim": "produced", "word_timestamps": "produced", "vad": "produced", "segment_timestamps": "produced", "lid": "produced"},
    "observed": {
      "stage_wall_seconds": {"decode": 0.09, "vad": 0.61, "lid": 8.83, "asr": 9.14,
                             "punctuator": 1.07},
      "total_wall_seconds": 19.74,
      "peak_rss_bytes_by_stage": {"vad": 1284407296, "lid": 5871104000,
                                  "asr": 6903312384, "punctuator": 2415919104},
      "peak_rss_bytes": 6903312384,
      "segments": 2,
      "words": 11,
      "vad_regions": 2,
      "lid_regions": 2,
      "abstentions": 0,
      "punctuation_invariant_checked": true,
      "punctuation_invariant_note": "each segment's text, stripped of punctuation and whitespace, equalled the case-insensitive concatenation of its word texts"
    }
  }
}
```

Two things worth reading closely. `lid_regions` is region-granular and its bounds
match `vad_regions`, not the segments — the label is produced per VAD region, and the
two segments happen to sit one per region here. And `word_timestamps` covers only the first
six words of `seg_1` in this printout for length; a real result has one word object per
non-punctuation token of every segment, which is what the
`punctuation_invariant_checked` flag asserts.

### 3.2 Export

```bash
audio export --input field.transcript.json --format srt -o field.srt
```

Exit 0. `field.srt`:

```text
1
00:00:00,410 --> 00:00:01,480
This is a测试。

2
00:00:03,310 --> 00:00:04,290
我们要来看哈，
```

FireRed is the only stack whose timing needed no aligner, and its 1.0 ms repeat drift is
two orders of magnitude below the 41.7 ms of a single frame at 24 fps, so it does not
affect cue placement. What is still unvalidated is accuracy, not stability: boundary
MAE/P95 is unmeasured here exactly as it is for the aligner.

## 4. Paths that correct themselves

Every refusal carries a `fix`, and `fix` is a runnable command wherever a configuration
exists that would work. That is the point: an agent that misconfigures a request should be
able to copy one line and be right on the next attempt, without reading this document. The
seven request errors below are the ones where self-correction is the whole story; the
remaining five are in [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) §5 and summarised at
the end.

### 4.1 No stack

```bash
audio transcribe plan --input meeting.m4a --want diarization
```

Exit 2:

```json
{
  "code": "stack_required",
  "field": "--stack",
  "allowed": ["qwen-1.7b", "qwen-0.6b", "vibevoice", "firered"],
  "stacks": {
    "qwen-1.7b": "fast transcript, no native timing or speakers; the interview default",
    "qwen-0.6b": "same shape, smaller and faster, measurably worse text",
    "vibevoice": "native speakers and segment bounds, highest memory, cannot resume a failed run",
    "firered": "native word timing, speech regions and region language; no speakers"
  },
  "fix": "audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want diarization"
}
```

The `fix` names one stack rather than listing four again, because a fix a caller has to
choose between is not a fix. `stacks` is there so the choice can be revisited deliberately,
and the one-liners say what each stack costs as well as what it gives — including that
`vibevoice` cannot resume, which is the kind of thing nobody discovers until a long run dies.

### 4.2 No input

```bash
audio transcribe plan --stack qwen-1.7b --want diarization
```

Exit 2:

```json
{
  "code": "input_required",
  "field": "--input",
  "note": "a stack alone cannot be planned: how the audio is partitioned, how many units that is, what the run will cost, and whether a failure leaves anything usable are all properties of this file",
  "fix": "audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want diarization"
}
```

### 4.3 A capability name that does not exist

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want word_timing
```

Exit 2:

```json
{
  "code": "capability_unknown",
  "field": "--want",
  "provided": "word_timing",
  "did_you_mean": "word_timestamps",
  "available_on_stack": {
    "native": ["languages", "verbatim"],
    "requires_add_on": ["diarization", "overlapped_speech", "vad", "word_timestamps"],
    "impossible": ["segment_timestamps", "lid", "token_lid"]
  },
  "fix": "audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want word_timestamps"
}
```

Two fields do the correcting. `did_you_mean` handles the likeliest case — a misspelling, and
the nine names sit close enough together that reaching for the wrong one is not carelessness.
`available_on_stack` handles the rest: whatever the caller meant, this is the whole set it can
ask for on the stack it chose, split so the free ones are visible. Between them the error is
self-sufficient; correcting a request should not need a second command to find the menu.

### 4.4 A real capability this stack cannot satisfy

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want segment_timestamps
```

Exit 2:

```json
{
  "code": "capability_unsatisfiable_on_stack",
  "capability": "segment_timestamps",
  "allowed": ["vibevoice", "firered"],
  "available_on_stack": {
    "native": ["languages", "verbatim"],
    "requires_add_on": ["diarization", "overlapped_speech", "vad", "word_timestamps"],
    "impossible": ["segment_timestamps", "lid", "token_lid"]
  },
  "fix": "audio transcribe plan --input meeting.m4a --stack firered --want segment_timestamps"
}
```

Distinct from 4.3: the name is real, so `did_you_mean` would be wrong and misleading. Two
corrections are on offer and the payload does not choose for you — `allowed` names the stacks
that would satisfy this request, `available_on_stack` names what this stack would satisfy
instead, and `fix` takes the first reading because a caller who asked for segment timing
probably wants segment timing. The second reading is for the caller who cared more about the
stack.

### 4.5 A capability nothing provides

```bash
audio transcribe plan --input meeting.m4a --stack firered --want token_lid
```

Exit 2:

```json
{
  "code": "capability_unsupported",
  "capability": "token_lid",
  "allowed": [],
  "reason": "no_backend_declares",
  "fix": "no stack or add-on satisfies this; code-switching support does not imply per-token language labels, and the nearest available output is lid, which labels a whole speech region"
}
```

This is the one case where `fix` is a sentence rather than a command, because there is no
command. Emitting a plausible-looking one would be worse than admitting it: an agent that
retries a suggested fix and fails again learns nothing, while an agent told "nothing does
this, here is the nearest thing that does" can decide whether region-level labels are enough.

### 4.6 An option this stack does not take

```bash
audio transcribe plan --input demo.mp4 --stack vibevoice --want verbatim --language Cantonese
```

Exit 2:

```json
{
  "code": "option_unsupported_on_stack",
  "field": "--language",
  "provided": "Cantonese",
  "allowed": [],
  "stacks_accepting": ["qwen-1.7b", "qwen-0.6b"],
  "fix": "audio transcribe plan --input demo.mp4 --stack vibevoice --want verbatim"
}
```

The `fix` drops the flag rather than switching stacks, because the stack was the deliberate
choice and the flag was the accident. `stacks_accepting` is there for the caller who meant
the opposite. Accepting the flag silently would be the real failure: a caller would believe
it had constrained a decode it never touched.

### 4.7 A pin the plan has no role for

```bash
audio transcribe plan --input demo.mp4 --stack vibevoice --want diarization --diarizer fluidaudio
```

Exit 2:

```json
{
  "code": "pin_conflicts_with_native_capability",
  "field": "--diarizer",
  "provided": "fluidaudio",
  "allowed": [],
  "capability": "diarization",
  "fix": "audio transcribe plan --input demo.mp4 --stack vibevoice --want diarization"
}
```

`vibevoice` satisfies `diarization` natively, so this plan contains no diarizer role for a
pin to select among. Pins choose between implementations of a role that exists.

### 4.8 The rest

| Code | Exit | Trigger | What `fix` says |
| --- | --- | --- | --- |
| `packages_not_provisioned` | 3 | `run` before `pull` | The `audio packages pull --stack` line for this stack — every package it can use, since narrowing a pull to a want set is reserved for the planner |
| `package_integrity_failed` | 3 | a digest, size, or revision mismatch on something already provisioned | `audio packages pull --repair <package>` |
| `timing_required_for_format` | 2 | `export --format srt` on a transcript with no word timing | The `transcribe run` line that would produce timing, with `word_timestamps` added |
| `run_incomplete` | 4 | budget exhausted, or a stage died part-way on a partitioned stack | The `--range <watermark>:` line that transcribes only what is missing |
| `backend_failed` | 1 | a crash, most often out of memory | A suggestion — a smaller stack, or freeing memory — and it is a suggestion, not a guarantee |

The last row is the honest exception. Everything above it has a correction the tool can
compute; a crash does not, so `fix` proposes rather than instructs, and the payload says
which `role` and `backend` failed so a caller can tell a memory ceiling from a missing
toolchain.

Two further refusals belong to `audio packages pull` rather than to `transcribe`, and §1.3
publishes both: `want_not_implemented` and `stack_conflicts_with_named_packages`, each exit 2 and
each a refusal of an argument that used to be accepted and ignored.

## 5. Teardown

```bash
audio packages list
```

```json
{
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "bytes": 2467859030,
     "state": "ready", "license_declared": "apache-2.0", "license_reviewed": false,
     "used_by_stacks": ["qwen-1.7b"]},
    {"package": "fluidaudio", "environment": "swift", "bytes": null,
     "state": "ready", "license_declared": "apache-2.0", "license_reviewed": true,
     "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice", "firered"]},
    {"package": "speaker-diarization-coreml", "environment": "swift", "bytes": 129243647,
     "state": "ready", "license_declared": "cc-by-4.0", "license_reviewed": true,
     "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice", "firered"]},
    {"package": "qwen3-forcedaligner", "environment": "mlx", "bytes": 1276475979,
     "state": "ready", "license_declared": "apache-2.0", "license_reviewed": false,
     "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice"]},
    {"package": "vibevoice-asr-7b", "environment": "torch-vibevoice", "bytes": 17349559904,
     "state": "ready", "license_declared": "mit", "license_reviewed": false,
     "used_by_stacks": ["vibevoice"]},
    {"package": "firered-asr2s", "environment": "torch-firered", "bytes": 9583893873,
     "state": "ready", "license_declared": "apache-2.0", "license_reviewed": false,
     "used_by_stacks": ["firered"]}
  ],
  "environments": {"mlx": "ready", "torch-firered": "ready", "torch-vibevoice": "ready",
                   "swift": "ready"},
  "total_known_bytes": 30807032433,
  "unsized_packages": ["fluidaudio"]
}
```

Every byte count here is real, read from the Hub at the revision each package pins.
`fluidaudio` is the one that stays `null`: it is a build product, so its size depends on the
toolchain that produced it, and `pull` records the measured value afterwards. `total_known_bytes`
is named for what it is, because a total that silently omitted an unsized package would
understate the disk a `purge` reclaims.

`license` became two fields. `license_declared` is what the model card says at the pinned
revision; `license_reviewed` is whether anyone read the terms. Only FluidAudio and
`speaker-diarization-coreml` were read (`model_tests/benchmark/DIARIZATION.md`). One field
could not tell "nobody looked" apart from "the card says apache-2.0 and nobody checked what
that obliges", and a scraped string must not read as a clearance.

```bash
audio packages remove vibevoice-asr-7b
```

```json
{
  "removed": ["vibevoice-asr-7b"],
  "environments_removed": ["torch-vibevoice"],
  "environments_removed_reason": "no other provisioned package targets torch-vibevoice",
  "environments_kept": ["mlx", "swift", "torch-firered"],
  "environments_kept_reason": "qwen3-asr-1.7b-8bit, qwen3-forcedaligner still need mlx; fluidaudio, speaker-diarization-coreml still need swift; firered-asr2s still needs torch-firered",
  "hub_revisions_deleted": ["d0c9efdb8d614685062c04425d91e01b6f37d944"],
  "hub_revisions_not_found": [],
  "hub_cache_note": "only revisions this tool recorded as materialized here were deleted; the Hugging Face cache may be shared with other tools",
  "reclaimed_bytes": 17349559904
}
```

Reference counting cuts both ways here, which is the point of showing it: `torch-vibevoice`
held exactly one package and dies with it, while `mlx` survives because the aligner and the
ASR checkpoint are still provisioned there. The count is derived from the package table each
time rather than stored, so it cannot drift out of agreement with what is actually on disk.

`reclaimed_bytes` is measured, not projected. Most of it is a Hub revision rather than anything
under the root, so `remove` deletes exactly the commit hashes the registry recorded — by hash,
through the Hub cache's own revision-scoped deletion, leaving any sibling revision of the same
repository alone. A revision the cache no longer holds appears in `hub_revisions_not_found`
rather than being counted as freed.

```bash
audio packages purge --dry-run
```

```json
{
  "would_remove": {
    "packages": ["qwen3-asr-1.7b-8bit", "fluidaudio", "speaker-diarization-coreml",
                 "qwen3-forcedaligner", "firered-asr2s"],
    "environments": ["mlx", "torch-firered", "swift"],
    "root": "/Users/you/Library/Caches/audio-processing-cli"
  },
  "reclaimable_known_bytes": 13457472529,
  "unsized_packages": ["fluidaudio"],
  "untouched": ["user media", "transcript and subtitle outputs"]
}
```

`--dry-run` projects from the registry, so it reports `reclaimable_known_bytes` and names what
it could not size. The real `purge` reports `reclaimed_bytes`, which is what the filesystem and
the Hub cache actually gave back. The two are related and deliberately not the same field.

```json
{"note": "the real run's shape follows the same keys, with reclaimed_bytes in place of the projection"}
```

```bash
audio packages purge
uv tool uninstall audio-processing-cli
```

Purge before uninstalling: the resolved root otherwise outlives the only tool that knows
how to describe it. Neither `remove` nor `purge` touches `meeting.timed.json`,
`demo.transcript.json`, `field.transcript.json`, or any subtitle file.
