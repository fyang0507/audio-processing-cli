
## 4. Export

`audio transcribe export` is the canonical offline export command. `audio export` remains a compatibility alias with identical arguments, output, and exit status. Both dispatch directly to the export owner before media or stack resolution. No stack selection, probing, model execution, or package provisioning is required. Runnable export remediation uses the canonical spelling. The CLI usability follow-up is tracked in [issue #48](https://github.com/fyang0507/audio-processing-cli/issues/48).

```bash
audio transcribe export --input meeting.timed.json --format srt -o meeting.srt
audio transcribe export --input meeting.timed.json --format vtt -o meeting.vtt
audio transcribe export --input meeting.transcript.json --format md
audio transcribe export --input meeting.transcript.json --format txt
audio transcribe export --input meeting.transcript.json --format jsonl
audio transcribe export --input meeting.timed.json --format md --timestamps
```

Subtitle formats require word timing and refuse without it. `meeting.transcript.json` from §1.3 has none, so:

```bash
audio transcribe export --input meeting.transcript.json --format srt
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

A contradictory legacy or hand-edited result can record `word_timestamps: "produced"` while ordinary sentence text carries no real word stream. That document violates the result contract; export rejects the input before considering the requested format:

```bash
audio transcribe export --input ordinary.sentences.json --format srt
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

Changing formats cannot repair contradictory evidence. A valid requested-alignment failure uses `word_timestamps: "abstained"` and a same-bounds `alignment_unavailable` abstention; export may then preserve ordinary text in text formats and omit it from subtitle cues without mistaking the absence for successful timing.

A result written by an older CLI may instead lack a safe durable source identity. If word timing is missing and `source.path` is relative, missing, not a file, or cannot be resolved, export must not reinterpret it from the current working directory and prescribe a rerun against the wrong media:

```bash
audio transcribe export --input legacy.transcript.json --format srt
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

There is one narrow nonempty wordless exception. If `word_timestamps` is recorded as produced and **every** segment is a positive-duration, bracketed non-speech event such as `[Music]`, with source-relative `start`/`end` bounds but no `speaker` and no `words`, SRT/VTT export succeeds with an empty subtitle rather than turning the event's container bounds into a cue. Any ordinary sentence among those wordless segments restores the refusal above. Once at least one real timed word stream exists, export emits cues for timed segments and omits every wordless segment — both bounded events and bounded ordinary speech carrying an exact same-bounds `alignment_unavailable` abstention — rather than inventing bounds. A mixed result containing unbounded Qwen text refuses SRT/VTT; diagnostic segment links identify the absent timing but do not authorize dropping recognized text or reusing the attempted unit interval as word bounds. The ledger and capability outcome retain the incomplete-timing evidence.

`md` and `txt` are for people. Default Markdown starts with `# Transcript` and places a blank line between segments; TXT separates them with a single newline. Both preserve each segment's text, including embedded newlines, and supplied speaker labels. `jsonl` is one segment object per line, ordered by start time — the same segment objects the JSON result carries, without the envelope or the provenance — so a consumer can stream or `grep` a long transcript without parsing the whole document. It has no timing requirement, and because it drops the provenance it is an export for reading, not an artifact to audit against.

`--provenance` prepends a readable header only for TXT/Markdown. It copies the saved source path, stack, timebase, and `duration_basis` when present. It then lists every input in merge order with its own `complete`, owned source intervals, and saved `coverage` when present. A partial input remains identified as partial even beside a complete continuation; gaps between covered inputs are not turned into a claim of complete coverage. Existing merge compatibility checks still apply. Legacy inputs without a duration basis do not acquire one. Header values are JSON-quoted, and Markdown metacharacters are escaped. The header does not alter segment text, speaker absence, timing, export summaries, or canonical JSON.

```bash
audio transcribe export --input meeting.timed.json --format md --provenance --timestamps -o meeting.md
```

Before a timing-dependent export, inspect the run receipt's `outcomes` or the saved `provenance.outcomes` and abstentions. Processing `complete: true` and a positive word count do not establish complete timing delivery. A recorded `word_timestamps: abstained` means some timing was withheld; the canonical result identifies its scope, and the selected exporter validates its own timing requirements. The readable provenance header remains a source/completion projection; it does not replace these capability checks.

`--provenance` does not imply `--timestamps`; provenance can accompany untimed text. With SRT, VTT, or JSONL it refuses before reading inputs or overwriting a destination, including with `--force`. For example, JSONL with `--provenance` produces exit 2 and this stderr:

```json
{
  "code": "provenance_unsupported_for_format",
  "field": "--provenance",
  "provided": true,
  "format": "jsonl",
  "allowed_formats": ["txt", "md"],
  "fix": "use --provenance with --format txt or md, or remove --provenance"
}
```

`--timestamps` explicitly adds a source-relative range before each TXT or Markdown segment, for example `[00:00:02.310 --> 00:00:04.710] [S1] Hello.`. It uses the segment's supplied native `start`/`end` when present; otherwise it uses the first and last real word bounds. Ranges round to milliseconds for display, without extending them, filling gaps, cutting text, or changing the saved result. Sentence text, punctuation, speaker absence, and bounded event tags stay intact. Default TXT/Markdown content and export summaries are unchanged. These coarse readable ranges do not change the word-timing requirement for subtitles.

Before falling back to word bounds, export checks that the complete word stream reproduces the segment's lexical text under the punctuation invariant: casing, punctuation and whitespace may differ; missing, extra or reordered lexical text may not. A word containing only punctuation cannot supply timing. A contradictory word fallback refuses with `export_input_invalid`, while default text remains exportable. Native segment bounds remain independent of that word-stream check. Supplied bounds that overflow millisecond conversion also refuse with `export_input_invalid` before publication, rather than emitting a traceback.

Every segment must have supplied timing. A bounded VibeVoice segment can be shown even if its requested word alignment abstained. An untimed Qwen segment cannot borrow bounds from a diarizer turn, processing chunk, requested range, coverage, or source duration. If any segment lacks both native segment bounds and a nonempty word stream, the whole export refuses with exit 2 before writing output; it never drops untimed text. An empty transcript stays empty (Markdown retains its heading). For example, `audio transcribe export --input meeting.transcript.json --format md --timestamps` on an untimed segment returns this stderr:

```json
{
  "code": "timing_required_for_timestamps",
  "field": "--timestamps",
  "provided": true,
  "input": "meeting.transcript.json",
  "segment_id": "seg_0",
  "requires_any_capability": ["segment_timestamps", "word_timestamps"],
  "note": "every segment needs supplied bounds; processing intervals are never substituted",
  "fix": "remove --timestamps to preserve untimed text, or transcribe the original source with segment_timestamps on a native stack or word_timestamps"
}
```

With SRT, VTT, or JSONL, `--timestamps` refuses with exit 2 and code `timestamps_unsupported_for_format`: remove the option or select TXT/Markdown.

```bash
audio transcribe export --input meeting.transcript.json --format jsonl --timestamps
```

Exit 2, stderr:

```json
{
  "code": "timestamps_unsupported_for_format",
  "field": "--timestamps",
  "provided": true,
  "format": "jsonl",
  "allowed_formats": ["txt", "md"],
  "fix": "use --timestamps with --format txt or md, or remove --timestamps"
}
```

VTT carries speaker labels as voice tags when `diarization` is present, which is a commitment to VTT as a real format rather than SRT with dots. V1 ships a fixed deterministic cue policy: target at most 7 seconds and 2 lines, with 42 Latin or 16 CJK characters per line, sentence-end then clause-punctuation then word-gap break priority, no segment or speaker change inside a cue, and 1 ms quantization. Duration and line-width targets are soft: indivisible words or the grouping required to retain every mapped word can exceed them with `cue_duration_overlong` or `cue_line_overlong` warnings. Issue #10 tracks future tuning rather than an unimplemented prerequisite.

Every mapped word remains in the subtitles, including words whose supplied start and end are equal or collapse to the same millisecond. After candidate grouping, a collapsed candidate merges with the following candidate in its own segment, repeating until the first supplied word start and last supplied word end form a positive quantized interval. A remaining collapsed tail merges with the preceding candidate in that segment. Canonical word order, punctuation and speaker labels remain intact; no bound is extended or borrowed from another segment. If a whole segment's supplied word bounds still cannot form a positive subtitle interval, SRT/VTT refuse the entire export with `export_input_invalid` before publication, including forced replacement, rather than omit words or fabricate timing. Original word and cross-segment chronology are validated before regrouping, and cue overlap remains a refusal.

Three artifact-derived rules are already enforced. Breaking at punctuation maps marks in sentence text to word indexes under the punctuation floor's case-insensitive invariant, because FireRed's punctuator recases. And a segment may carry text with no word stream, so the splitter omits it rather than assuming every segment yields cues. Finally, result validation permits the recorded FireRed integer-millisecond seam where a word edge escapes its sentence by at most 1 ms, while refusing anything larger; the raw runner artifacts and exact affected rows are recorded in [HANDOFF.md](../../HANDOFF.md).

When `--output` is supplied, export serializes UTF-8 text and publishes it atomically. An existing regular file is refused unless `--force` is explicit; a directory is never replaceable. Neither mode may target any input transcript or the canonical source media, including aliases, and `--force` does not override that protection. The writer carries descriptor-captured device/inode identities through rendering. A forced replacement uses the platform's atomic entry-exchange primitive, inspects the displaced inode, and atomically rolls back when it is protected or non-regular; an absent destination is claimed with an exclusive hard link. A protected file renamed onto the output mid-command is therefore refused without an unaddressable or partial-output window. Without `--output`, the same serialized content is written to stdout. The boundary is accidental and public-destination retargeting, not a hostile same-credential process changing a random private sibling after its final identity check; POSIX has no portable conditional-unlink operation, and that process already has direct authority to remove the canonical file.

Timing quality is not yet validated: boundary MAE/P95 is unmeasured for both FireRed's native times and the aligner, so these files are producible but not yet claimed broadcast-acceptable.
