
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
                   "revision": "7990aaccc6b7aec1e527743bd30201f2c4a03b8c",
                   "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
                   "selected_by": "stack"},
    "lid":        {"backend": "firered-lid", "environment": "torch-firered",
                   "revision": "1bb4d285c8456429385d9c0810300df4297bc11b",
                   "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
                   "config": {"batch_size": 4},
                   "selected_by": "requirement:lid",
                   "granularity": "vad_region",
                   "cost_note": "162.09 s with LID versus 84.24 s without, on the 139.284 s probe"},
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
    "stage_order": ["decode", "vad", "lid", "asr", "punctuator"],
    "residency": "one_environment_process_at_a_time",
    "environments_spanned": ["torch-firered"],
    "note": "FireRed runs one process that loads VAD, LID, ASR and punctuator together. On the same 139.284-second probe, LID on measured 162.09 s and 13169377280 bytes (12.26 GiB) peak RSS; LID off measured 84.24 s and 9830449152 bytes (9.16 GiB)."
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
                        "note": "one of the 115 labels after the five special tokens in the pinned FireRedLID dict.txt, emitted once per VAD region and copied onto every sentence in that region; per-sentence variation would be fabricated"}
  },
  "packages": [
    {"package": "firered-asr2s", "environment": "torch-firered", "kind": "weights",
     "bytes": 9583893873, "provisioned": false, "includes_lid_weights": true}
  ],
  "total_known_download_bytes": 9583893873,
  "unsized_packages": [],
  "warnings": [],
  "next": "audio packages pull firered-asr2s",
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "complete": true,
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
  "complete": true,
  "source": {"path": "/Users/you/recordings/field.wav", "duration_seconds": 27.8, "timebase": "seconds"},
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
                             "punctuator": 1.07, "firered_process_overhead": 1.0},
      "total_wall_seconds": 20.74,
      "peak_rss_bytes_by_stage": {"firered_process": 13169377280},
      "peak_rss_bytes": 13169377280,
      "segments": 2,
      "words": 11,
      "segments_without_words": 0,
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

`firered_process_overhead` is the measured residual between the one FireRed process wall and its
explicitly timed VAD/LID/ASR/punctuator phases. It keeps imports, model loading, audio I/O, and
framework overhead visible so `total_wall_seconds` still equals the real non-overlapping wall
sum; it is not a sixth backend stage or an estimated allocation. A negative residual beyond the
measurement tolerance is invalid rather than silently clamped into a plausible total.

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
