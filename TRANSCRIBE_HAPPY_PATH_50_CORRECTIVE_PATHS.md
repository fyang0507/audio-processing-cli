
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
    "vibevoice": "native speakers and segment bounds, highest memory, prefix-only recovery after generation truncation",
    "firered": "native word timing, speech regions and region language; no speakers"
  },
  "fix": "audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want diarization"
}
```

The `fix` names one stack rather than listing four again, because a fix a caller has to
choose between is not a fix. `stacks` is there so the choice can be revisited deliberately,
and the one-liners say what each stack costs as well as what it gives — including that
`vibevoice` can salvage only a complete decoded prefix after generation truncation, while a
model-load failure still leaves nothing.

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

### 4.7 An unsupported option value

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --language EN
```

Exit 2:

```json
{
  "code": "option_value_unsupported",
  "field": "--language",
  "provided": "EN",
  "allowed": ["Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"],
  "did_you_mean": "English",
  "fix": "audio transcribe plan --input meeting.m4a --stack qwen-1.7b --language English"
}
```

The value is matched case-insensitively but not by abbreviation: `english` resolves to the
canonical `English`, while `EN` is not one of the checkpoint's declared names. `allowed` is the
exact `support_languages` list published by both pinned Qwen configs; `did_you_mean` is optional
and appears only for a near value.

### 4.8 A pin the plan has no role for

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

### 4.9 The rest

Existing output is an actionable collision, so the refusal preserves the full request and adds
the explicit overwrite decision:

```json
{
  "code": "output_exists",
  "field": "--output",
  "provided": "meeting.timed.json",
  "existing": "meeting.timed.json",
  "fix": "audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization,word_timestamps --language Cantonese -o meeting.timed.json --force"
}
```

No force flag may target an input transcript or canonical source media, including through a
derived partial filename. The tool does not invent a replacement path owned by the caller:

```json
{
  "code": "output_is_canonical_input",
  "field": "--output",
  "provided": "meeting.json",
  "resolved_target": "meeting.partial.json",
  "fix": "choose an --output that does not resolve to an input transcript, its derived partial path, or canonical source media; --force cannot override this"
}
```

An output or derived partial path whose identity cannot be resolved fails before decode, and an
export destination whose parent or final file cannot be resolved and written safely reports the
same typed refusal rather than surfacing an untyped filesystem exception:

```json
{
  "code": "output_path_invalid",
  "field": "--output",
  "provided": "meeting.json",
  "target": "meeting.partial.json",
  "reason": "Symlink loop from 'meeting.partial.json'",
  "fix": "choose an --output whose destination and parent directory can be resolved and written safely"
}
```

Three export validation refusals are fixed-shape exit-2 payloads. `--force` has no effect when an
export is going to stdout, so it must be removed or paired with an explicit destination:

```json
{
  "code": "output_required_for_force",
  "field": "--force",
  "provided": true,
  "requires": "--output",
  "fix": "remove --force when writing to stdout, or add --output PATH"
}
```

An input that is not the exact normalized result schema is refused at the input boundary:

```json
{
  "code": "export_input_invalid",
  "field": "--input",
  "provided": "invalid.json",
  "reason": "document is not the exact current normalized result shape",
  "fix": "regenerate or repair the input transcript before exporting it"
}
```

Valid result documents still cannot be merged when doing so would guess across different source
identities or overlapping ownership. Two inputs with different canonical sources report:

```json
{
  "code": "export_inputs_incompatible",
  "field": "--input",
  "provided": ["meeting.part1.json", "other.json"],
  "reason": "source identity, duration, or timebase differs",
  "fix": "export these inputs separately, or select results with the same canonical source and non-overlapping ranges"
}
```

The same typed refusal protects an evidence boundary specific to VibeVoice. Native anonymous
speaker labels are local to one generation, so independently generated partial and resumed
documents that requested `diarization` cannot be merged without falsely equating their labels.
Its `reason` is `VibeVoice native speaker labels are local to each independent generation and
cannot be reconciled across multiple input documents`, and its actionable `fix` is `export these
VibeVoice documents separately, or rerun the desired ranges together as one generation`.

| Code | Exit | Trigger | What `fix` says |
| --- | --- | --- | --- |
| `packages_not_provisioned` | 3 | `run` before `pull` | The `audio packages pull --stack` line for this stack — every package it can use, since narrowing a pull to a want set is reserved for the planner |
| `package_integrity_failed` | 3 | a digest/revision/Hub cache identity/allowlist/recorded-size mismatch; a URL artifact that is not the contained non-symlink manifest file; a nonlaunching interpreter or redirected managed environment root; or a live source checkout whose HEAD, tracked-name set, manifest hashes, receipt shape/hashes, or ordinary/ignored untracked files differ | `audio packages pull --repair <package>`; for a redirected environment, first replace the path named by `actual` |
| `timing_required_for_format` | 2 | subtitle export with no recorded word timing and a safe absolute source identity | The `transcribe run` line that would produce timing, with `word_timestamps` added |
| `timing_required_for_format` | 2 | `word_timestamps` is already `produced`, but no segment contains usable timed words | A sentence choosing `md`/`txt`/`jsonl` or a different transcript with real timed words; rerunning the same request would be inert |
| `timing_required_for_format` | 2 | word timing is missing and a legacy result's `source.path` is relative, missing, non-file, or unresolvable | A sentence to regenerate the transcript with the current CLI; it cannot safely name an original-media rerun |
| `run_incomplete` | 4 | budget exhausted, or a stage died part-way on a partitioned stack | The `--range <watermark>:` line that transcribes only what is missing; a sentence instead when zero units completed and no range can make progress |
| `backend_failed` | 1 | a crash, most often out of memory | A suggestion — a smaller stack, or freeing memory — and it is a suggestion, not a guarantee |

The sentence remedies are the honest exceptions. Output safety and export validation require the
caller to choose a destination, remove an inert flag, repair a transcript, or separate incompatible
inputs; the CLI cannot invent those decisions. Replaying an already-produced timing outcome cannot
create words it did not contain; a legacy relative or vanished `source.path` cannot safely identify
the original media from another working directory; and a crash has no deterministic repair. In
each case `fix` says what is true rather than printing a command that only looks executable. A
backend failure also names the `role` and `backend` so a caller can tell a memory ceiling from a
missing toolchain.

Two further refusals belong to `audio packages pull` rather than to `transcribe`, and §1.3
publishes both: `want_not_implemented` and `stack_conflicts_with_named_packages`, each exit 2 and
each a refusal of an argument that used to be accepted and ignored.

A Qwen budget stop emits this exit-4 shape and writes the conforming partial named by `output`:

```json
{
  "code": "run_incomplete",
  "role": "asr",
  "backend": "qwen3-asr-1.7b-8bit",
  "detail": "global generation budget exhausted after 148 of 195 turns",
  "coverage": {
    "scope_intervals": [[0.0, 1794.2]],
    "covered_through_seconds": 1402.88,
    "covered_fraction": 0.782,
    "covered_intervals": [[0.0, 1402.88]],
    "missing_intervals": [[1402.88, 1794.2]],
    "units_total": 195,
    "units_completed": 148
  },
  "output": "meeting.timed.partial.json",
  "fix": "audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization,word_timestamps --language Cantonese --range 1402.88: -o meeting.timed.rest.json"
}
```

After the ranged resume finishes, multi-input export preserves the caller's source-timeline
order and reports that ordered input list rather than collapsing it to one path:

```bash
audio export --input meeting.timed.partial.json --input meeting.timed.rest.json \
  --format jsonl -o meeting.timed.merged.jsonl
```

```json
{
  "input": ["meeting.timed.partial.json", "meeting.timed.rest.json"],
  "output": "meeting.timed.merged.jsonl",
  "format": "jsonl",
  "segments": 4
}
```

Segment and word ids are document-scoped. The two inputs may both begin at `seg_0`; export
validates disjoint owned intervals, concatenates them in the supplied order, and re-ids the
merged stream once so the output has no duplicate ids. It never sorts away an incorrect input
order or guesses through overlapping coverage. This example applies to Qwen, FireRed, and
non-diarized VibeVoice results; independently generated VibeVoice results with native diarization
use the refusal above because their anonymous labels have no cross-generation identity.
