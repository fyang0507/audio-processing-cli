
## 2. Video editing — product demo

Goal: verbatim segments with native speaker structure and word timing, for an editing agent that cuts on speaker changes and needs fillers preserved.

### 2.1 Resolve the request

The `capabilities` report is omitted here for length; it takes the same shape as §1.1 with `vibevoice`'s own values.

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
                "tokenizer": {"materialized_role": "tokenizer",
                              "repository": "Qwen/Qwen2.5-7B",
                              "revision": "d149729398750b98c0af14eb82c78cfe92750796"},
                "source_commit": "94da20d98b2fa7688e9cbfaf7692ddb4954f7600",
                "patch": "vibevoice-logits-to-keep",
                "config": {"device": "mps", "dtype": "bfloat16", "attention": "sdpa",
                           "seed": 1234, "max_new_tokens": 16384},
                "deterministic": true,
                "determinism_tolerance_ms": 0.0,
                "determinism_basis": "three seeded repeats shared one normalized-output hash; text decode is do_sample=False",
                "determinism_note": "acoustic tokenizer samples a Gaussian latent; fixed seed required",
                "selected_by": "stack"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "mlx",
                "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
                "config": {"scope": "all_segments", "max_overrun_ms": 0.501,
                           "language_rule": "Chinese when text matches [一-鿿], otherwise English; the ASR --language hint is never forwarded"},
                "selected_by": "add_on_required_by:word_timestamps"}
  },
  "execution": {
    "stage_order": ["decode", "asr", "aligner"],
    "residency": "one_model_stage_at_a_time",
    "environments_spanned": ["mlx", "torch-vibevoice"],
    "note": "VibeVoice runs in torch-vibevoice and the aligner runs later in mlx; the two environment processes are not resident together, so stage walls add and peaks do not"
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
    "word_timestamps":         {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                            "evidence": {"interface": "verified", "quality": "unmeasured"},
                            "note": "Boundary error against labels is unmeasured; non-speech events remain wordless, while unavailable ordinary-speech alignment records alignment_unavailable."}
  },
  "packages": [
    {"package": "vibevoice-asr-7b", "environment": "torch-vibevoice", "kind": "weights",
     "bytes": 17361048135, "provisioned": false},
    {"package": "qwen3-forcedaligner", "environment": "mlx", "kind": "weights",
     "bytes": 1276475979, "provisioned": true}
  ],
  "total_known_download_bytes": 17361048135,
  "unsized_packages": [],
  "warnings": [
    {"code": "measured_peak_exceeds_target", "blocking": false,
     "detail": "measured 20.28 GiB live MPS allocation on spice-30min-participant; a strict 16 GiB MPS cap OOMs at model load, measured on a 27.8 s probe, while an 18 GiB cap passed that probe"}
  ],
  "next": "audio packages pull vibevoice-asr-7b",
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "complete": true,
    "source": {"path": "demo.mp4", "duration_seconds": 112.4, "timebase": "seconds", "duration_basis": "probed_audio_stream"},
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

Exit 0. `qwen3-forcedaligner` is already `provisioned: true` from §1, so `total_known_download_bytes` is 0 and only VibeVoice needs pulling. `abstentions` is an empty array rather than absent: it is a floor artifact, and the plan's warning that nothing here detects overlap is what says it cannot fill.

The `vibevoice` capabilities payload carries the same source-scoped truncation risk in `failure_recovery.note`: on `spice-30min-participant`, 11,345 generated tokens over 1,800 seconds is 6.30 tokens/second, so at that observed rate the declared 16,384-token cap projects to about 43 minutes of comparable audio. This is a rate extrapolation, not an observed truncation.

### 2.2 Provision and run

```bash
audio packages pull --stack vibevoice
audio transcribe run --input demo.mp4 --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps \
  -o demo.transcript.json
```

Exit 0. `demo.transcript.json`:

```json
{
  "schema_version": 1,
  "complete": true,
  "source": {"path": "/Users/you/recordings/demo.mp4", "duration_seconds": 112.4, "timebase": "seconds", "duration_basis": "canonical_decoded_pcm"},
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

Two things in `seg_1` are the recorded VibeVoice behaviours rather than invented shape, and both are absences.

It carries **no `speaker` key**. VibeVoice emits `Speaker: "N/A"` on non-speech segments, and the adapter-normalization floor requires that become an absent key rather than a speaker whose id is the string `"N/A"`. A conforming result cannot contain `"N/A"` as an attribution anywhere; that string appearing in output is the defect the floor exists to catch.

It carries **no `words` array**, which is correct and is not an abstention: the aligner is not run on a segment with no speech to align. So `words` is absent on some segments while `word_timestamps` is `produced`, and `observed.segments_without_words` records how many segments carry no `words` key. A present empty array is a successful alignment with zero lexical tokens, not an abstention.

An ordinary speech segment is different. If its requested alignment result is absent or nonconforming, its text and native bounds remain, `words` is absent, and one `alignment_unavailable` abstention carries those exact bounds. The run-level `word_timestamps` outcome becomes `abstained` while valid word streams on other segments remain. VibeVoice turns likewise do not bridge silence or a wordless event: every bounded speech segment gets its own native turn, even when an adjacent segment carries the same anonymous label.

`--alignment-max-overrun-ms` has the same meaning here as in §1.2 because this request selects ForcedAligner. The default example uses 0.501 ms and records no beyond-allowance correction. If a separately chosen higher limit accepts clipped endpoints, `provenance.observed.alignment_corrections` records only corrections on words that survive the full unit's validation and text reconciliation. Rejected units retain their diagnostic abstentions; the limit cannot create word timing from the native segment interval.

### 2.3 Export subtitles with speaker voice tags

```bash
audio transcribe export --input demo.transcript.json --format vtt -o demo.vtt
```

Exit 0. `demo.vtt`:

```text
WEBVTT

1
00:00:00.310 --> 00:00:02.210
<v 0>So, um, this is the new editor.

2
00:00:02.580 --> 00:00:04.520
<v 0>You can, like, drag a clip here.

3
00:00:06.220 --> 00:00:07.860
<v 1>And it renders straight away?
```

The `[Environmental Sounds]` segment produced no cue: it has no word stream, and cue bounds come from words. Whether a non-speech event tag *should* render as an SDH cue is a subtitle-convention question parked in issue #10, not a transcription one. The same omission rule applies to a **bounded** ordinary speech segment carrying a same-bounds `alignment_unavailable` abstention when at least one other segment has real timed words; its ledger entry and abstained capability outcome preserve the warning. SRT/VTT refuse mixed results containing unbounded Qwen text; diagnostic segment links identify the absent timing but do not authorize silently dropping text or treating attempted unit intervals as word timing. With no real word stream anywhere, subtitle export refuses instead of inventing bounds, except for an all-bounded-event result whose timing provenance is already `produced`, which deliberately renders an empty subtitle.
