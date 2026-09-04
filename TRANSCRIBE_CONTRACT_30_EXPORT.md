
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
