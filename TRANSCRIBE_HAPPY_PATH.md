# `transcribe` happy paths — mocked, per use case

**Status: mock.** Nothing here has been executed; no command below exists yet. This is
the unabridged step-by-step an implementer can diff against and an agent can read as a
worked example. [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) is organized around
*why* the surface looks like this and abridges output to whatever differs; this document
does the opposite — it shows every command in order and the complete stdout of each, with
no design commentary.

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
  `provenance.measured`, `capabilities[].measured_limit`, `capabilities[].observed_limit`,
  package byte counts, revisions, and configuration. Every one is checkable against
  `model_tests/`.
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
| `turns[]` with `turn_id` | diarization-grade turn intervals | `diarization_turns` | not exercised |
| `overlaps[]` with `overlap_id` | cross-speaker overlap intervals | `overlapped_speech` | not exercised |

The last two are named for completeness and are the two result keys this document does
*not* demonstrate: none of the three use cases needs them, since the interview path cuts
on segment speaker rather than on turn intervals. `TRANSCRIBE_CONTRACT.md` §1.4 is where
they are requested.

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
  "tool": {"name": "audio-processing-cli", "version": "0.4.0",
           "path": "/Users/you/.local/bin/audio"},
  "platform": {"os": "macOS 26.5.2", "arch": "arm64", "hw_model": "Mac16,5"},
  "external_tools": {"ffmpeg": "8.0", "ffprobe": "8.0", "uv": "0.9.7", "swift": "6.2"},
  "memory": {"physical_bytes": 68719476736, "available_bytes": 41234567168},
  "disk": {"root": "/Users/you/Library/Caches/audio-processing-cli",
           "available_bytes": 402653184000},
  "environments": {"core": "ready", "mlx": "absent", "torch": "absent", "swift": "absent"},
  "packages": {"provisioned": [], "count": 0},
  "warnings": []
}
```

Exit 0. `swift` present here; absent it would be reported and non-fatal, blocking only
the packages that need it.

## 1. Fast long-form transcription — interview

Goal: a speaker-attributed transcript of a 30-minute interview, plus subtitles. Word
timing is required for the subtitles, so it is requested up front rather than discovered
missing at export.

### 1.1 Ask what the stack can do with this file

```bash
audio transcribe plan --stack qwen-1.7b --input meeting.m4a
```

```json
{
  "catalog_version": 1,
  "stack": "qwen-1.7b",
  "family": "Qwen3-ASR",
  "roles_included": ["asr"],
  "roles_conditional": {},
  "environment": "mlx",
  "input": {"path": "meeting.m4a", "duration_seconds": 1794.2, "container": "m4a",
            "sample_rate_hz": 44100, "channels": 2},
  "processing": {
    "unit": "fixed_chunk",
    "unit_rule": "180-second chunks when timing is not supplied externally; diarized turns instead when diarization is requested, in which case the count is not known until the diarizer runs",
    "unit_count": 10,
    "unit_count_known_at_plan_time": true,
    "batch_size": 1
  },
  "failure_recovery": {
    "partial_results": "per_unit",
    "resumable": true,
    "resume_mechanism": "--range",
    "note": "units are independent, so a failure leaves the completed ones usable; the global generation budget is the mechanism most likely to stop a long file early and it stops between units, not inside one"
  },
  "cost": {
    "proved": "54 s and 3.0 GiB peak to transcribe a 30-minute Cantonese interview on an M4 Max, batch 1 with the MLX cache cleared per batch and a Cantonese language hint",
    "projected_seconds": 53.6,
    "record": "model_tests/benchmark/results/2026-08-13-turn-attributed-fast-asr.json"
  },
  "capabilities": {
    "languages":         {"availability": "native",
                          "advertised": "30 language labels; no dialect selector",
                          "verified_here": ["Mandarin", "English", "Cantonese"],
                          "hint_accepted": true,
                          "evidence": {"interface": "verified", "quality": "measured"},
                          "measured_limit": "the only accuracy figure is Cantonese, 33.56% mixed-token error on a 30-minute interview; the other advertised languages are untested here",
                          "record": "model_tests/benchmark/results/2026-08-13-turn-attributed-fast-asr.json"},
    "verbatim":          {"availability": "native",
                          "evidence": {"interface": "verified", "quality": "unmeasured"},
                          "note": "does not clean disfluencies, but drops every spoken \"uh\" and sometimes fuses the neighbouring words; filler recall unmeasured",
                          "record": "model_tests/benchmark/results/2026-08-17-qwen-verbatim-probe.json"},
    "diarization":       {"availability": "requires_add_on", "add_on": ["fluidaudio", "reconciler"],
                          "cost": {"proved": "15 s and 0.55 GiB peak on a 30-minute sample; RSS excludes memory held by system Core ML services",
                                   "projected_seconds": 14.7,
                                   "packages": ["fluidaudio", "speaker-diarization-coreml"],
                                   "environment": "swift", "requires_tool": ["swift"],
                                   "download_bytes": null},
                          "evidence": {"interface": "verified", "quality": "measured"},
                          "measured_limit": "matched 3 of 75 annotated speaker changes on a dense two-speaker conversation; unsuitable where turns are short or overlapping",
                          "record": "model_tests/benchmark/DIARIZATION.md"},
    "diarization_turns": {"availability": "requires_add_on", "add_on": ["fluidaudio"],
                          "cost": {"shares_stage_with": ["diarization", "overlapped_speech"]},
                          "evidence": {"interface": "verified", "quality": "measured"},
                          "measured_limit": "95.42% participant-interval F1 on a 30-minute interview but 5.50% speaker-change F1 on a dense conversation; strong on long turns, weak on rapid change",
                          "record": "model_tests/benchmark/DIARIZATION.md"},
    "overlapped_speech": {"availability": "requires_add_on", "add_on": ["fluidaudio"],
                          "cost": {"shares_stage_with": ["diarization", "diarization_turns"]},
                          "evidence": {"interface": "verified", "quality": "unmeasured"},
                          "note": "without this, policy.overlap_detection is \"unavailable\" and an empty abstention ledger means undetected rather than absent"},
    "vad":               {"availability": "requires_add_on", "add_on": ["silero-vad"],
                          "cost": {"proved": "0.4 s and 0.11 GiB peak on a 150-second sample",
                                   "projected_seconds": 4.5,
                                   "packages": ["silero-vad"], "environment": "core",
                                   "auto_fetch": true, "download_bytes": null,
                                   "note": "hash-pinned single file that fetches itself, so it never returns exit 3"},
                          "evidence": {"interface": "verified", "quality": "measured"},
                          "measured_limit": "0.8505 frame-level F1 at 0.7655 precision and 0.9567 recall; the gate over-includes, so it bounds activity and not speakers",
                          "record": "model_tests/benchmark/results/2026-08-15-silero-vad.json"},
    "word_timestamps":   {"availability": "requires_add_on", "add_on": ["qwen3-forcedaligner"],
                          "cost": {"proved": "4.6 s on a 139-second sample, alignment only and excluding model load; peak unrecorded",
                                   "projected_seconds": 59.7,
                                   "packages": ["qwen3-forcedaligner"], "environment": "torch",
                                   "download_bytes": null},
                          "evidence": {"interface": "verified", "quality": "unmeasured"},
                          "timing_precision": {"boundary_mae_ms": null,
                                               "note": "never scored against hand-labelled boundaries, and neither is FireRed's native timing, so switching stacks for timing accuracy trades one unmeasured number for another"}},
    "segment_timestamps": {"availability": "impossible", "reason": "no_native_segment_extents",
                          "note": "this stack emits no segment extents; the chunk boundaries it works in are not speech timing and are not published"},
    "lid":               {"availability": "impossible", "reason": "no_backend_declares_on_stack",
                          "note": "available on firered via its LID stage"},
    "token_lid":         {"availability": "impossible", "reason": "no_backend_declares"}
  },
  "next": "audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want <capabilities>"
}
```

Exit 0. Nothing was downloaded; only the input's container metadata was read.

Every block above answers a question the caller has to decide: which languages the stack
actually handles as against which it advertises, what it gives away free, what each
addition costs to install and to run, where a measured result says the addition is weak,
and what happens if the run dies. `cost.proved` is a run that actually happened, stated in
units a reader can hold — seconds and gibibytes against a named sample — and
`projected_seconds` is that rate applied to this input, so nobody re-derives it.
`shares_stage_with` marks the three capabilities that come out of one diarizer run, so a
caller adding costs does not count it three times.

### 1.2 Resolve the request

```bash
audio transcribe plan --input meeting.m4a \
  --stack qwen-1.7b \
  --want diarization,word_timestamps \
  --language Cantonese
```

```json
{
  "plan_version": 1,
  "request": {"input": "meeting.m4a", "stack": "qwen-1.7b",
              "want": ["diarization", "word_timestamps"],
              "language": "Cantonese"},
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
    "reconciler": {"backend": "sample-exact-turn-partition",
                   "config": {"partition": "sample_exact"},
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
    "aligner":    {"backend": "qwen3-forcedaligner", "environment": "torch",
                   "config": {"scope": "all_segments"},
                   "selected_by": "add_on_required_by:word_timestamps"}
  },
  "execution": {
    "stage_order": ["decode", "diarizer", "reconciler", "asr", "aligner"],
    "residency": "one_model_stage_at_a_time",
    "environments_spanned": ["swift", "mlx", "torch"],
    "note": "stages run strictly sequentially and no two model stages are resident together; wall time adds across stages, peak memory does not, and the per-stage peaks below must not be summed"
  },
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
    "diarization": {"satisfaction": "derived", "backend": "fluidaudio",
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "note": "reconciled sample-exactly onto ASR text; anonymous labels only",
                            "measured_limit": "on the 149.9 s CantoMap conversation with 75 annotated speaker changes this diarizer preset matched 3; not validated for rapid backchannels, interruptions, or dense overlap",
                            "record": "model_tests/benchmark/DIARIZATION.md"},
    "word_timestamps":         {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "boundary MAE/P95 unlabeled; absent on any segment with no speech to align"}
  },
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "kind": "weights",
     "bytes": 2463307541, "provisioned": false},
    {"package": "fluidaudio", "environment": "swift", "kind": "toolchain",
     "requires_tool": ["swift"], "bytes": null, "provisioned": false},
    {"package": "speaker-diarization-coreml", "environment": "swift", "kind": "weights",
     "bytes": null, "provisioned": false},
    {"package": "qwen3-forcedaligner", "environment": "torch", "kind": "weights",
     "bytes": null, "provisioned": false}
  ],
  "total_known_download_bytes": 2463307541,
  "unsized_packages": ["fluidaudio", "speaker-diarization-coreml", "qwen3-forcedaligner"],
  "measured": {
    "reference_run": "spice-30min-canonical-mix",
    "fixture_duration_seconds": 1800.0,
    "hardware": "apple-m4-max-64gib",
    "config": "batch 1; MLX cache cleared after every batch, 195 clear observations over 195 accepted turns; language hint \"Cantonese\"",
    "asr_stage_wall_seconds": 53.77,
    "diarizer_stage_wall_seconds": 14.74,
    "peak_rss_bytes": 3241689088,
    "end_to_end_wall_seconds": null,
    "end_to_end_note": "never measured for 1.7B with the aligner in the chain; the record's 69.16 s arithmetic sum covers diarizer plus ASR only, borrows its diarization stage from a separate 0.6B run, and carries measured_end_to_end: false",
    "record": "model_tests/benchmark/results/2026-08-13-turn-attributed-fast-asr.json"
  },
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
    "abstentions": [
      {"abstention_id": "ab_0", "reason": "overlap", "start": null, "end": null}
    ],
    "provenance": "<the full executed plan; elided in this printed example only>"
  }
}
```

Exit 0. `segment_timestamps` was not requested and is impossible on this stack anyway, so
segments carry no `start`/`end` — in the sample above or in the real result below.

### 1.3 Provision

```bash
audio packages pull --stack qwen-1.7b \
  --want diarization,word_timestamps
```

Progress goes to stderr; stdout is the receipt:

```json
{
  "pulled": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx",
     "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
     "bytes": 2463307541, "digest_verified": true},
    {"package": "fluidaudio", "environment": "swift", "version": "0.15.5",
     "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
     "built": true, "product_runs": true},
    {"package": "speaker-diarization-coreml", "environment": "swift",
     "bytes": 84279296, "digest_verified": true, "license": "CC-BY-4.0"},
    {"package": "qwen3-forcedaligner", "environment": "torch",
     "bytes": 1932735283, "digest_verified": true}
  ],
  "environments_created": ["mlx", "swift", "torch"],
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "registry": "/Users/you/Library/Caches/audio-processing-cli/registry.json",
  "reclaimable_bytes": 4480322120,
  "warnings": [
    {"code": "license_unreviewed", "blocking": false,
     "detail": "qwen3-asr-1.7b-8bit and qwen3-forcedaligner report license: \"unreviewed\"; only the FluidAudio SDK (Apache-2.0) and speaker-diarization-coreml (CC-BY-4.0) are recorded"}
  ]
}
```

Exit 0. The byte counts for the three previously unsized packages are **illustrative** —
they are the numbers a real `pull` would record and the reason `unsized_packages` exists
in the plan.

```bash
audio packages verify
```

```json
{
  "verified": [
    {"package": "qwen3-asr-1.7b-8bit", "digest": "ok"},
    {"package": "fluidaudio", "product_runs": true, "patches_applied": []},
    {"package": "speaker-diarization-coreml", "digest": "ok"},
    {"package": "qwen3-forcedaligner", "digest": "ok"}
  ],
  "environments": {"mlx": "ok", "swift": "ok", "torch": "ok"},
  "mlx_audio_private_api_source_hash": "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250",
  "mlx_audio_private_api_matches_expected": true,
  "failed": []
}
```

Exit 0.

### 1.4 Run

```bash
audio transcribe run --input meeting.m4a \
  --stack qwen-1.7b \
  --want diarization,word_timestamps \
  --language Cantonese \
  --format json -o meeting.timed.json
```

Exit 0. `meeting.timed.json`, with `provenance` shown collapsed here for length and
otherwise byte-identical to the plan above plus one `outcome` per capability:

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
  "abstentions": [
    {"abstention_id": "ab_0", "reason": "overlap", "start": 41.86, "end": 42.73},
    {"abstention_id": "ab_1", "reason": "short_turn", "start": 118.44, "end": 118.79}
  ],
  "provenance": {
    "plan_version": 1,
    "request": {"input": "meeting.m4a", "stack": "qwen-1.7b",
                "want": ["diarization", "word_timestamps"],
                "language": "Cantonese"},
    "capabilities": {
      "diarization": {"satisfaction": "derived", "outcome": "produced",
                              "backend": "fluidaudio",
                              "evidence": {"interface": "verified", "quality": "measured"}},
      "word_timestamps":         {"satisfaction": "derived", "outcome": "produced",
                              "backend": "qwen3-forcedaligner",
                              "evidence": {"interface": "verified", "quality": "unmeasured"}}
    },
    "observed": {
      "stage_wall_seconds": {"decode": 3.91, "diarizer": 14.68, "reconciler": 0.21,
                             "asr": 54.02, "aligner": 46.77},
      "total_wall_seconds": 119.59,
      "peak_rss_bytes_by_stage": {"diarizer": 588251136, "asr": 3243020288,
                                  "aligner": 2104492032},
      "peak_rss_bytes": 3243020288,
      "segments": 2,
      "words": 13,
      "abstentions": 2
    },
    "elided": "roles, floors, execution, policy, packages, measured, and warnings are byte-identical to the plan in 1.2"
  }
}
```

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

Step one is omitted here for length; it takes the same shape as §1.1 with `vibevoice`'s
own values.

```bash
audio transcribe plan --input demo.mp4 --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps
```

```json
{
  "plan_version": 1,
  "request": {"input": "demo.mp4", "stack": "vibevoice",
              "want": ["verbatim", "diarization", "segment_timestamps", "word_timestamps"],
              "language": null},
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
                "determinism_basis": "three seeded repeats shared one normalized-output hash; text decode is do_sample=False",
                "determinism_note": "acoustic tokenizer samples a Gaussian latent; fixed seed required",
                "selected_by": "stack"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "torch",
                "config": {"scope": "all_segments"},
                "selected_by": "add_on_required_by:word_timestamps"}
  },
  "execution": {
    "stage_order": ["decode", "asr", "aligner"],
    "residency": "one_model_stage_at_a_time",
    "environments_spanned": ["torch"],
    "note": "both model stages share the torch environment and are still not resident together; VibeVoice and the aligner have never been measured co-resident and this plan does not do so"
  },
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
                            "interface_basis": "emits disfluencies rather than cleaning them: 28 filler hits on the 139.284 s probe, the highest of the four stacks",
                            "observed_limit": "dialect form normalized twice: 看哈 to 看一下 on the 27.8 s probe and 刷啥子 for 耍啥子 on the 139.284 s probe, both retained by firered",
                            "record": "model_tests/benchmark/results/2026-08-17-qwen-verbatim-probe.json"},
    "diarization": {"satisfaction": "native",
                            "evidence": {"interface": "verified", "quality": "measured"},
                            "measured_limit": "on the 149.9 s CantoMap conversation with 75 annotated speaker changes this stack matched 39; not validated for rapid backchannels, interruptions, or dense overlap",
                            "record": "model_tests/benchmark/DIARIZATION.md"},
    "segment_timestamps":      {"satisfaction": "native",
                            "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "word_timestamps":         {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "boundary MAE/P95 unlabeled; absent on any segment with no speech to align"}
  },
  "packages": [
    {"package": "vibevoice-asr-7b", "environment": "torch", "kind": "weights",
     "bytes": null, "provisioned": false},
    {"package": "qwen3-forcedaligner", "environment": "torch", "kind": "weights",
     "bytes": null, "provisioned": true}
  ],
  "total_known_download_bytes": 0,
  "unsized_packages": ["vibevoice-asr-7b"],
  "measured": {
    "reference_run": "spice-30min-participant",
    "fixture_duration_seconds": 1800.0,
    "hardware": "apple-m4-max-64gib",
    "config": "mps, bfloat16, sdpa, seed 1234, logits_to_keep patch applied",
    "generation_seconds": 851.1,
    "rtf_generation": 0.4728,
    "peak_mps_live_bytes": 21770457600,
    "record": "model_tests/benchmark/results/2026-08-12-evidence.json"
  },
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
    "abstentions": [],
    "provenance": "<the full executed plan; elided in this printed example only>"
  }
}
```

Exit 0. `qwen3-forcedaligner` is already `provisioned: true` from §1, so
`total_known_download_bytes` is 0 and only VibeVoice needs pulling. `abstentions` is an
empty array rather than absent: it is a floor artifact, and `policy.abstention_reasons`
being empty is what says it cannot fill on this plan.

### 2.2 Provision and run

```bash
audio packages pull --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps
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
  "abstentions": [],
  "provenance": {
    "plan_version": 1,
    "request": {"input": "demo.mp4", "stack": "vibevoice",
                "want": ["verbatim", "diarization", "segment_timestamps", "word_timestamps"],
                "language": null},
    "capabilities": {
      "verbatim":            {"satisfaction": "native", "outcome": "produced",
                              "evidence": {"interface": "verified", "quality": "refuted"}},
      "diarization": {"satisfaction": "native", "outcome": "produced",
                              "evidence": {"interface": "verified", "quality": "measured"}},
      "segment_timestamps":      {"satisfaction": "native", "outcome": "produced",
                              "evidence": {"interface": "verified", "quality": "unmeasured"}},
      "word_timestamps":         {"satisfaction": "derived", "outcome": "produced",
                              "backend": "qwen3-forcedaligner",
                              "evidence": {"interface": "verified", "quality": "unmeasured"}}
    },
    "observed": {
      "stage_wall_seconds": {"decode": 0.44, "asr": 53.16, "aligner": 3.72},
      "total_wall_seconds": 57.32,
      "peak_mps_live_bytes_by_stage": {"asr": 19983452160, "aligner": 2210398208},
      "peak_mps_live_bytes": 19983452160,
      "segments": 3,
      "words": 19,
      "segments_without_words": 1,
      "abstentions": 0
    },
    "elided": "roles, floors, execution, policy, packages, measured, and warnings are byte-identical to the plan in 2.1"
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
  "plan_version": 1,
  "request": {"input": "field.wav", "stack": "firered",
              "want": ["verbatim", "word_timestamps", "vad", "segment_timestamps",
                       "lid"],
              "language": null},
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "vad":        {"backend": "firered-vad", "environment": "torch",
                   "selected_by": "stack"},
    "lid":        {"backend": "firered-lid", "environment": "torch",
                   "config": {"batch_size": 4},
                   "selected_by": "requirement:lid",
                   "granularity": "vad_region",
                   "cost_note": "162.09 s with LID versus 84.24 s without, on the 139.284 s probe"},
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
  "execution": {
    "stage_order": ["decode", "vad", "lid", "asr", "punctuator"],
    "residency": "one_model_stage_at_a_time",
    "environments_spanned": ["torch"],
    "note": "four model stages in one environment, none resident together"
  },
  "policy": {
    "policy_version": 1,
    "overlap": "abstain",
    "overlap_detection": "unavailable",
    "overlap_detection_note": "no backend in this plan detects overlap, so an empty abstention ledger means undetected, not absent",
    "abstention_reasons": []
  },
  "capabilities": {
    "verbatim":        {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "interface_basis": "emits disfluencies rather than cleaning them: 24 filler hits on the 139.284 s probe",
                        "note": "retained 看哈 on the 27.8 s probe and 耍啥子 on the 139.284 s probe; two lexemes cannot rank varieties"},
    "word_timestamps":     {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "timing_precision": {"repeat_drift_ms": 1.0, "boundary_mae_ms": null,
                                             "note": "monotonic in both the 30- and 60-minute runs; accuracy against hand-labelled boundaries is unmeasured"}},
    "vad":   {"satisfaction": "native", "stage": "FireRedVAD",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "alternative": {"add_on": ["silero-vad"],
                                        "note": "the Silero path has a measured 0.8505 frame-level F1 where this stage has none; --vad selects it"}},
    "segment_timestamps":  {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "lid": {"satisfaction": "native", "stage": "FireRedLID",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "one label per VAD region, copied onto every sentence in that region; per-sentence variation would be fabricated"}
  },
  "packages": [
    {"package": "firered-asr2s", "environment": "torch", "kind": "weights",
     "bytes": null, "provisioned": false, "includes_lid_weights": true}
  ],
  "total_known_download_bytes": 0,
  "unsized_packages": ["firered-asr2s"],
  "measured": {
    "reference_run": "spice-30min-participant",
    "fixture_duration_seconds": 1800.0,
    "hardware": "apple-m4-max-64gib",
    "config": "cpu, float32, lid off, asr and punc batch size 4",
    "end_to_end_seconds": 665.26,
    "rtf_end_to_end": 0.3696,
    "peak_rss_bytes": 9789620224,
    "note": "measured with LID off; this plan includes LID, which on a separate 139.284 s probe raised inference from 84.24 s to 162.09 s",
    "record": "model_tests/benchmark_runs/firered_lidoff_batch4_spice30m_participant.json"
  },
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
    "provenance": "<the full executed plan; elided in this printed example only>"
  }
}
```

Exit 0. Segments carry no `speaker` key in the sample or the result: FireRed has no
speaker output and `diarization` was not requested.

```bash
audio packages pull --stack firered \
  --want verbatim,word_timestamps,vad,segment_timestamps,lid
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
    "plan_version": 1,
    "request": {"input": "field.wav", "stack": "firered",
                "want": ["verbatim", "word_timestamps", "vad", "segment_timestamps",
                         "lid"],
                "language": null},
    "capabilities": {
      "verbatim":        {"satisfaction": "native", "outcome": "produced",
                          "evidence": {"interface": "verified", "quality": "unmeasured"}},
      "word_timestamps":     {"satisfaction": "native", "outcome": "produced",
                          "evidence": {"interface": "verified", "quality": "unmeasured"}},
      "vad":   {"satisfaction": "native", "outcome": "produced",
                          "evidence": {"interface": "verified", "quality": "unmeasured"}},
      "segment_timestamps":  {"satisfaction": "native", "outcome": "produced",
                          "evidence": {"interface": "verified", "quality": "unmeasured"}},
      "lid": {"satisfaction": "native", "outcome": "produced",
                          "evidence": {"interface": "verified", "quality": "unmeasured"}}
    },
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
    },
    "elided": "roles, floors, execution, policy, packages, measured, and warnings are byte-identical to the plan in 3.1"
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

## 4. Teardown

```bash
audio packages list
```

```json
{
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "bytes": 2463307541,
     "license": "unreviewed", "used_by_stacks": ["qwen-1.7b"]},
    {"package": "fluidaudio", "environment": "swift", "bytes": null,
     "license": "Apache-2.0", "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice", "firered"]},
    {"package": "speaker-diarization-coreml", "environment": "swift", "bytes": 84279296,
     "license": "CC-BY-4.0", "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice", "firered"]},
    {"package": "qwen3-forcedaligner", "environment": "torch", "bytes": 1932735283,
     "license": "unreviewed", "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice"]},
    {"package": "vibevoice-asr-7b", "environment": "torch", "bytes": 18253611008,
     "license": "unreviewed", "used_by_stacks": ["vibevoice"]},
    {"package": "firered-asr2s", "environment": "torch", "bytes": 9878424576,
     "license": "unreviewed", "used_by_stacks": ["firered"]}
  ],
  "environments": {"mlx": "ready", "torch": "ready", "swift": "ready"},
  "total_bytes": 32612357704
}
```

```bash
audio packages remove vibevoice-asr-7b
```

```json
{
  "removed": ["vibevoice-asr-7b"],
  "environments_kept": ["torch"],
  "environments_kept_reason": "qwen3-forcedaligner and firered-asr2s still need torch",
  "hub_revisions_deleted": ["d0c9efdb8d614685062c04425d91e01b6f37d944"],
  "hub_cache_note": "only revisions this tool recorded as materialized here were deleted; the Hugging Face cache may be shared with other tools",
  "reclaimed_bytes": 18253611008
}
```

```bash
audio packages purge --dry-run
```

```json
{
  "would_remove": {
    "packages": ["qwen3-asr-1.7b-8bit", "fluidaudio", "speaker-diarization-coreml",
                 "qwen3-forcedaligner", "firered-asr2s"],
    "environments": ["mlx", "torch", "swift"],
    "root": "/Users/you/Library/Caches/audio-processing-cli"
  },
  "reclaimable_bytes": 14358746696,
  "untouched": ["user media", "transcript and subtitle outputs"]
}
```

```bash
audio packages purge
uv tool uninstall audio-processing-cli
```

Purge before uninstalling: the resolved root otherwise outlives the only tool that knows
how to describe it. Neither `remove` nor `purge` touches `meeting.timed.json`,
`demo.transcript.json`, `field.transcript.json`, or any subtitle file.
