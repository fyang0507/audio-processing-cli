
## Floors

Floors are not capabilities. A caller never opts into them and never opts out, and a
backend that cannot meet one is not a conforming backend. They are therefore **not printed
in any payload**: they are constant for a schema version, a caller cannot act on them, and
an array of six invariant names in every plan is decoration. They live here, and in the
test suite as assertions. Encoding an invariant as a field would make it look like a setting.

- **Punctuated, sentence-segmented text.** Nothing downstream wants raw text —
  not a reader, not an LLM, not a cue splitter that needs sentence boundaries.
  FireRed ships punctuation as a separate stage, so its adapter always runs
  FireRedPunc.
- **Punctuation is sentence-level.** Marks live in sentence and segment `text`.
  They never form a parallel stream with their own bounds, they are never attached
  to a word token by this tool, and punctuation is not a request: floor one already
  puts it in the text. Both backends that produce word bounds already agree.
  FireRed builds its word stream from pre-punctuation AED timestamps
  (`fireredasr2system.py:181-184`), so word tokens carry only the intra-word marks
  the ASR itself produced — 20 of 12,370 recorded word tokens, every one an English
  contraction such as `it's`. Qwen3-ForcedAligner emits one token per
  non-punctuation character, so `好，现在开始。` aligns as five tokens.

  What cue splitting needs is not a mark on a token but a sound mapping from a
  mark's position in the sentence text to a word index, and that rests on one
  invariant: **stripping punctuation and whitespace from a sentence's `text` yields
  exactly the concatenation of its word `text` values, compared
  case-insensitively.** Verified on all five recorded FireRed artifacts (12,370
  words) and all 17 aligned segments of the forced-aligner artifact. Assert it at
  the adapter boundary per stack; it is the testable statement the old
  attach-to-word rule was reaching for, and unlike that rule it is not vacuous on
  the stack it names.

  Case-insensitively is not a hedge. FireRedPunc lowercases its input and then
  re-capitalizes sentence starts and standalone `i`
  (`fireredpunc/punc.py:349-382`), so sentence text and word text differ in case by
  construction — 234 characters across those artifacts. Neither stream is derivable
  from the other: the sentence text holds the marks and the casing, the word stream
  holds the bounds. Both are carried; the sentence text is canonical for reading and
  for subtitles. A segment may legitimately have no word stream at all — the
  forced-aligner artifact has two such segments, both VibeVoice non-speech event
  tags — so the invariant binds only where words exist.
- **The original source timeline is canonical.** Every bound refers to it. The
  transcription path never modifies the source.
- **No synthesized bounds.** A timing field absent from the backend stays absent.
- **Abstentions survive to the output.**
- **Known overlap has no sole-speaker attribution.** When `overlapped_speech` is requested,
  any transcript segment intersecting a detected overlap that intersects this document's scope
  keeps its text and bounds but omits `speaker`, even when the overlap began before the range.
  Publishing the overlap interval and its `overlap` abstention remains start-owned, independently
  of that intersection-based masking.
- **Normalization at the adapter boundary.** No model-specific object travels past
  it, no backend scaffolding survives it, and a backend's default-filled field is not a
  value. Three verified instances, one per stack. Qwen's private batched API returns the
  model's own scaffold inside its text — `language English<asr_text>` — which the public
  path strips and an adapter on the batched path must strip too. FireRed emits
  `lang: null, lang_confidence: 0` on every sentence whether or not LID ran
  (`fireredasr2system.py:149-150`), so with LID off the adapter drops both rather
  than publish a zero that reads as a measured confidence. VibeVoice emits
  `Speaker: "N/A"` on non-speech segments; that is the absence of a label and must
  not become a speaker id.

## Capabilities

Requestable, because a caller genuinely chooses them:

Names follow the field rather than this document. `diarization`, `vad`, `lid`, and
`word_timestamps` are what the literature, the model cards, and `DIARIZATION.md` already
call these things, and an internal dialect costs more than the precision it buys. Where a
standard term covers more than one output, it stays one capability rather than being split
into coinages: `diarization` delivers both the speaker label on text and the turn intervals,
because they come out of one stage and neither is useful alone.

| Capability | Meaning | Why it is its own name |
| --- | --- | --- |
| `languages` | Which languages the stack handles, separating what it advertises from what a recorded run here actually exercised, and whether it takes a `--language` hint. | The first stack-choice question and the one most easily answered with a marketing number. Advertised counts are 30, 50+, and 100+; the set verified locally is Mandarin, English, and Cantonese on every stack. |
| `verbatim` | The stack **can produce** verbatim text: it emits what it heard, disfluencies included, rather than a cleaned rendering. How faithfully it does so is the quality axis, not this one. | Interface verified on all four stacks — 24 to 28 filler hits on one probe, none of them cleaning and none of them complete. No backend exposes a verbatim switch and nothing in v1 cleans, so requesting this asserts an interface rather than selecting a mode, and the plan answers for fidelity separately: `quality: "refuted"` on the two stacks a recorded run caught normalizing a dialect form. |
| `diarization` | Anonymous speaker labels on transcript text **and** the speaker turn intervals, as `segments[].speaker` and `turns[]`. A known-overlap segment deliberately lacks a sole-speaker label. | One request, because there is no use for either half alone: intervals without text say "three people spoke" and never who said what, and text-without-intervals is not even purchasable, since punctuated text is a floor. Both fall out of one diarizer run, so splitting them would price one stage twice. Whether the labels were native or reconciled from a diarizer is provenance, not a separate request. |
| `overlapped_speech` | Cross-speaker overlap. | Feeds the abstention ledger; also the basis for refusing to attribute overlapping speech. |
| `vad` | Speech-activity regions. | Speech activity only; not turns and not events. |
| `segment_timestamps` | ASR segment extents. | Cheap coarse timing where a stack emits it natively. Cannot produce subtitle-grade cues. |
| `word_timestamps` | Word or character intervals. | The only timing that supports subtitle cues or word-level editing. Never inferred from a stack's internal chunk boundaries. It is absent without abstention on a non-speech event; on an ordinary speech segment it may be absent only with `alignment_unavailable`. |
| `lid` | Region language label and confidence. | Region-level; cannot locate a switch. Costs roughly double inference on FireRed. FireRed carries the label on each *sentence*, but it is produced once per VAD region and copied onto the sentences inside it (`fireredasr2system.py:129-155`), so per-sentence variation would be fabricated. |
| `token_lid` | Per-token language. | **No backend provides this.** Named so the catalog can report it `impossible` and a request for it can fail loudly with `capability_unsupported`, rather than the assumption being drawn silently from code-switching support. |

Not in the namespace at all, and deliberately not reserved in it: capture role, and
filler / repetition / false-start annotation. They are future work — capture role needs
the dedicated-channel merge and non-oracle role mapping, which are unmeasured, and
annotation needs a `DisfluencyAnnotator` role that belongs to `analyze` rather than
`transcribe`. Naming them in the catalog as `impossible` would advertise a surface that
does not exist, so requesting one is simply `capability_unknown`: the name is not a
capability yet. When they arrive they enter the namespace with the rest.

`token_lid` is the one exception, and it earns it: code-switching support invites the
assumption that per-token labels come with it, so the name exists purely to refuse loudly
with `no_backend_declares` instead of letting the assumption stand.

Speaker identity and semantic role are not capabilities either. They stay external to ASR
unless capture metadata establishes them.

## Resolution

The caller chooses a stack, then states requirements. The planner derives the
add-ons. There is no default stack, no preference scalar, and no tie-break
ordering, because a stack choice is a quality judgement the planner cannot make.

1. `--stack` and `--input` are both required, on `plan` as well as `run`. Omitting the
   stack fails with the stack list rather than guessing; omitting the input fails because
   the partition, the unit count, the projected cost, and the recoverability of a failure
   are all properties of the pair. There is no batch surface: one invocation, one input,
   with any internal segmentation disclosed in the plan rather than hidden.
2. Requirements the stack satisfies natively cost nothing.
3. Requirements the stack cannot satisfy natively force specific add-ons.
4. Requirements the chosen stack cannot satisfy fail the request with exit 2
   before any model loads: `capability_unsatisfiable_on_stack` with a non-empty
   `allowed` when another stack could serve it, `capability_unsupported` with
   `allowed: []` and a reason when none can. `capabilities` reports the same facts as
   `availability: impossible` without erroring, so discovery never costs a failed command.
5. A role with more than one implementation may be pinned. `--vad` is the case
   that exists today: `silero-vad` is the default and FluidAudio ships a second
   Core ML implementation. `--diarizer` has one implementation and is therefore
   forced; the pin is defined so that adding a second one is not a surface change.

This table is the planner's whole logic and the source the anti-fabrication test
is parametrized over, so it must distinguish both refusal codes rather than
render them alike.

| Requirement | `qwen-1.7b`, `qwen-0.6b` | `vibevoice` | `firered` |
| --- | --- | --- | --- |
| `languages` | native | native | native |
| `verbatim` | native | native | native |
| `diarization` | + `fluidaudio` | native | + `fluidaudio` |
| `overlapped_speech` | + `fluidaudio` | + `fluidaudio` | + `fluidaudio` |
| `vad` | + `silero-vad` | + `silero-vad` | native |
| `segment_timestamps` | exit 2: unsatisfiable_on_stack | native | native |
| `word_timestamps` | + `qwen3-forcedaligner` | + `qwen3-forcedaligner` | native |
| `lid` | exit 2: unsatisfiable_on_stack | exit 2: unsatisfiable_on_stack | native (FireRedLID stage) |
| `token_lid` | exit 2: unsupported | exit 2: unsupported | exit 2: unsupported |

Four cell forms, and a test parametrized over this table must accept exactly these:
`native`; `native (<stage>)`; `+ <package>`; and
`exit 2: <code>`. `native (<stage>)` is not an add-on. FireRedLID ships inside the
`firered` package and fills the `lid` role the stack already declares, so requesting
`lid` adds nothing to the plan the stack did not already contain — it
changes cost, not composition, which is why the add-on `+` notation would misreport
it.

The two refusal codes: `unsatisfiable_on_stack` carries a non-empty `allowed`, so the fix
is to switch stacks; `unsupported` carries `allowed: []` and a `reason`, which today is
`no_backend_declares` on `token_lid` alone. A name that is not in the namespace at all is
neither of these — it is `capability_unknown`.

Cells carry resolution only. Evidence lives in the per-stack catalog, because it
differs where resolution does not: `verbatim` resolves `native` on all four stacks
and has a recorded refutation on two of them. That is also why the two Qwen sizes
share one column and get separate catalogs — their resolution is identical for
every capability here and their observed text fidelity is not. Across two probes and
two dialect lexemes the split is consistent: `firered` and `qwen-1.7b` retained `看哈`
and `耍啥子`, while `vibevoice` normalized `看哈` to `看一下` and both it and
`qwen-0.6b` rendered `刷啥子`. Filler retention, by contrast, separates nothing — 24
to 28 hits across all four — so the differentiating half of text fidelity is dialect
form, and the catalog says which half a figure describes.
