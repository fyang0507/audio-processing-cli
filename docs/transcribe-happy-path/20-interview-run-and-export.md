
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
  "complete": true,
  "source": {"path": "/Users/you/recordings/meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds", "duration_basis": "canonical_decoded_pcm"},
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
    {"abstention_id": "ab_0", "reason": "raw_fragment", "start": 41.86, "end": 42.07},
    {"abstention_id": "ab_1", "reason": "short_turn", "start": 118.44, "end": 118.79}
  ],
  "provenance": {
    "stack": "qwen-1.7b",
    "outcomes": {"diarization": "produced", "word_timestamps": "produced"},
    "observed": {
      "stage_wall_seconds": {"decode": 3.91, "diarizer": 14.68,
                             "asr": 54.02, "aligner": 46.77},
      "total_wall_seconds": 119.38,
      "peak_rss_bytes_by_stage": {"decode": 52363264,
                                  "diarizer": 588251136, "asr": 3243020288,
                                  "aligner": 2104492032},
      "peak_rss_bytes": 3243020288,
      "peak_mps_live_bytes_by_stage": {"asr": 3028287488,
                                        "aligner": 1987051520},
      "peak_mps_live_bytes": 3028287488,
      "segments": 2,
      "words": 13,
      "segments_without_words": 0,
      "turns": 2,
      "abstentions": 2
    }
  }
}
```

`provenance` carries three things and embeds a fourth. `stack` names what ran, `outcomes` says what became of each requested capability, and `observed` records what the run actually cost. The fourth is `plan`: the executed plan verbatim, which these printouts omit because §1.2 already shows it in full — a result does not restate what the plan said, it appends what only running could tell you. The key-set test therefore compares against a real run, not against these trimmed prints.

Note `peak_rss_bytes` is the **maximum** of the per-stage peaks, not their sum — that is what `execution.residency` buys, and the per-stage numbers are kept so the claim is checkable rather than asserted. A per-stage RSS value is the total footprint of the process executing that stage, runtime and orchestration included; it is not backend-owned allocation. For in-process VAD it is the core process high-water mark observed through the stage. `peak_mps_live_bytes_by_stage` is the MLX allocator's device-memory high-water mark, excludes ordinary RSS and non-MLX processes, and its aggregate is likewise a maximum rather than a sum.

The run recomputes `source.duration_seconds` from the canonical decode's PCM frame count so coverage uses the timeline actually processed. The path is the resolved absolute path of the original media, which lets a later export protect it across working directories; no temporary WAV format, sample-rate, or channel field enters the public `source` object.

A run invoked with `--range 1402.88:` adds this exact subtree to `provenance.plan.execution`; an open end resolves to the canonical WAV duration, and selected scope may expand to whole processing-unit bounds:

```json
{
  "range": {
    "requested": [1402.88, 1794.2],
    "selected_unit_scope": [1402.88, 1794.2]
  }
}
```

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
     "detail": "boundary MAE/P95 is unmeasured for the backend that produced this timing, so cue placement is producible but not claimed broadcast-acceptable"}
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

An on-disk untimed format reports segments rather than subtitle-only cue fields:

```bash
audio export --input meeting.transcript.json --format txt -o meeting.txt
audio export --input meeting.transcript.json --format md -o meeting.md
audio export --input meeting.transcript.json --format jsonl -o meeting.jsonl
```

```json
{
  "input": "meeting.transcript.json",
  "output": "meeting.txt",
  "format": "txt",
  "segments": 2
}
```

For a two-segment result, the three untimed files are exact deterministic projections of the normalized segment array. Exit 0, `meeting.txt`:

```text
First.
Second.
```

Exit 0, `meeting.md`:

```markdown
# Transcript

First.

Second.
```

Exit 0, `meeting.jsonl`:

```jsonl
{"segment_id":"seg_0","text":"First."}
{"segment_id":"seg_1","text":"Second."}
```

Cue bounds come from the first and last word of each segment, not from the segment — this stack has no `segment_timestamps` to use, which is exactly why `word_timestamps` was requested in step 1.2.

### 1.6 Export readable timestamps

```bash
audio export --input meeting.timed.json --format md --timestamps
```

Exit 0, stdout:

```markdown
# Transcript

[00:00:02.310 --> 00:00:04.710] [S1] 好，我們今天想聊一下你的工作。

[00:00:05.120 --> 00:00:07.440] [S2] 嗯，好啊，我做咗五年設計。
```

Here each range comes from the first and last supplied word bounds. With native segment timing, the range instead uses that segment's `start`/`end`, including bounded non-speech events. `--format txt --timestamps` emits the same segment content with single newlines, without the Markdown heading or paragraph separators. Omitting `--timestamps` preserves the default canonical text export. One untimed segment refuses the entire timestamp request; export never turns a processing interval or source duration into speech timing.
