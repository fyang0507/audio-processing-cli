# `transcribe` — implementation plan

**Status: plan, agreed 2026-08-19. None of it is built.** This answers
[TRANSCRIBE_DESIGN_HANDOFF.md](TRANSCRIBE_DESIGN_HANDOFF.md): the layer under the settled
agent-facing surface — modules, adapter boundaries, the stack table, orchestration, the test
strategy, and a sequence. It does not restate the surface. Where this document and
[TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) / [VOCABULARY.md](VOCABULARY.md) disagree,
those are authoritative and the disagreement is a defect in this one — except where a
correction is listed under [Document changes](#document-changes), which records what this pass
found wrong in them and why.

Every backend claim below cites a runner source, an installed source at its pinned commit, or a
recorded artifact. None cites a summary document. That rule is the reason nine of the findings
in the next section exist, and it is not optional here:
[SPEC_REVIEW_FINDINGS.md](SPEC_REVIEW_FINDINGS.md)'s process finding explains what happens
without it.

## What the evidence pass changed

Nine findings, worst first. Each changed a design decision or corrects a shipped document.

**1. FireRed's four models are co-resident, so `residency: one_model_stage_at_a_time` is false
on the only stack with four stages.** `run_firered.py:395-426` loads VAD, LID, ASR, and Punc,
assembles them onto one system object (`:435`), and then calls `system.process` once (`:463`).
Measured on the same 139.284 s clip: **9.16 GiB** peak RSS with LID off, **12.26 GiB** with LID
on — the LID checkpoint adds to the peak rather than replacing another stage's, which is what
co-residency looks like. The 30-minute channel peaks at 9.12 GiB with LID off, so the peak is
dominated by weights and not duration, as the catalog already says. HAPPY_PATH §3.1 publishes a
LID-**on** run at 6.43 GiB, below the measured LID-off floor and about half the measured
LID-on figure, because it was built as a maximum of isolated per-stage peaks. Artifacts:
`firered_lidoff_batch4_multispeaker_codeswitch_20260815.json`,
`firered_lidon_batch4_multispeaker_codeswitch_20260815.json`,
`firered_lidoff_batch4_spice30m_participant.json`.

**2. Qwen's `language English<asr_text>` scaffold tracks the `language` argument, not the API
path.** The specs attribute it to the private batched path. Measured across three runners:
public `generate()` with no language strips it (`run_mlx_asr.py:427-430`); the batched path with
`language=None` **leaks it** (`run_qwen_verbatim_probe.py:545,553-558`, all three runs); the
batched path with `language="Cantonese"` does not, in **0 of 195 segments**
(`run_turn_attributed_mlx_asr.py`, artifact
`mlx_qwen3_asr_1.7b_8bit_fluidaudio_turns_spice30m_mix_batch1_clearcache.json`). So the strip is
unconditional and its test must run the **no-hint** configuration, or it passes vacuously on the
configuration that never leaks. `mlx-audio` exposes `extract_language`
(`qwen3_asr.py:899`) — the adapter calls it rather than matching a prefix by hand.

**3. The aligner's `language` is a tokenizer selector with a Cantonese trap.**
`qwen3_forced_aligner.py:256-266` lowercases the argument and branches four ways —
`japanese`, `korean`, `chinese`, and everything else to `tokenize_space_lang`. Its checkpoint
declares **eleven** supported languages including `Cantonese`, and there is no Cantonese
branch, so `--language Cantonese` plumbed through would space-split CJK text into one "word"
per segment and silently degenerate `word_timestamps` — on the exact configuration every
recorded Cantonese figure used. This is why the recorded pipeline derives the aligner's
language from a CJK regex (`run_mlx_forced_aligner_probe.py:55,94`) instead of passing the
ASR's through, and why that rule is declared configuration here rather than an invisible
heuristic.

**4. A truncated VibeVoice decode loses the whole transcript upstream, not just its tail.**
`post_process_transcription` finds the closing bracket by counting (`:514-524`, commit
`94da20d`); on an unterminated array the count never returns to zero, `json_str` becomes empty,
`json.loads` raises, and the handler logs a warning and returns `[]` (`:559-562`). So writing a
prefix on exit 4 requires the adapter to parse the raw text itself and keep complete objects up
to the truncation. The cap is reachable: the 30-minute run generated 11,345 tokens for 1800 s =
**6.30 tok/s**, so the recorded 16,384-token cap exhausts near **43 minutes** of comparable
audio (`vibe_mps_bf16_logitskeep_spice30m_participant.json`; a rate extrapolation, not a
measured failure).

**5. The punctuation invariant is the word-partition algorithm, not only a check.** FireRed
emits one flat `words[]` for the whole file plus separate `sentences[]`, and nothing maps
between them; word bounds do not nest inside sentence bounds (word 0 starts at 650 ms inside a
sentence starting at 590 ms), so timestamps cannot do the partition. Greedy consumption of the
word stream against each sentence's punctuation-stripped, case-folded text matches **1,896 of
1,896 sentences and consumes 12,370 of 12,370 words with zero failures** across all five
recorded FireRed artifacts, with word bounds strictly monotonic and non-overlapping throughout.

**6. FireRed's LID output is a closed vocabulary and its region grouping is sound.** The label
is beam-decoded and detokenized (`lid.py:96-97`, commit `4e7d9aa`) from a **120-token
dictionary** — 5 special tokens and 115 labels — which is why it arrives as `en`,
`zh mandarin`, `zh xinan`, `zh xiang`: an ISO-ish code plus an optional Chinese-dialect token
(`pretrained_models/FireRedLID/dict.txt`). On the LID-on artifact all **58 of 58** VAD regions
contain their sentences under interval containment and **0** regions carry a non-constant
label, so `lid_regions[]` is derivable by grouping and label constancy is an assertion that can
fail.

**7. FluidAudio is deterministic across runs and sensitive to context, which is what makes
resume work.** Three separate runs of the 30-minute fixture, two of them weeks apart, produced
identical `normalized_output_sha256` (`3b6639…`, 589 segments); two runs of cantomap150s
likewise (`48c0011…`, 90 segments). But the same 30 minutes in two positions of a 60-minute
concat produced **588 against 549** intervals
(`fluidaudio_v0155_quality_regular_spice60m_canonical_mix_concat_diarization_repeat_score.json`).
So a `--range` resume must re-diarize the **whole file** — then the turn partition and the
anonymous labels reproduce exactly and the two results merge; diarizing only the range would
not.

**8. `packages_for()` cannot resolve FireRed.** It keys `roles.values()` on package ids
(`environments/__init__.py:141`), and `firered-vad`, `firered-lid`, `firered-asr2-aed`, and
`firered-punc` are backends of the single `firered-asr2s` package, so the call raises
`ManifestError`. The manifest needs a backend→(package, role) map; that is package data and
belongs beside the rest of it.

**9. The plans under-declare the configuration the measurements came from.** FireRed's AED runs
`beam_size=3, nbest=1, decode_max_len=0, softmax_smoothing=1.25, aed_length_penalty=0.6,
eos_penalty=1.0` (`run_firered.py:360-370`), none of it in any document. Qwen's
`max_tokens=16384` **is the exit-4 trigger** and appears in no plan. VibeVoice's
`max_new_tokens` likewise. The diarizer's `batch_size=32` and `--overlapping-segments` are
missing, and its offline mode has eleven further parameters the recorded runs left at the
product's defaults (`fluidaudiocli process --help` at commit `19600a4`). `config` is audited
provenance, so it declares what was passed and cites the pinned commit for the rest.

## Decisions taken

Seven, agreed with the repository owner this session. Each closes an item the design handoff
left open or corrects something the evidence pass found.

| Decision | Reason |
| --- | --- |
| **FireRed runs as one subprocess**, four Hub repositories in one package, three or four models loaded by request. | Provisioning and residency are different questions and only the second was wrong. One package, one download, one process is the measured configuration; splitting into four subprocesses would reimplement `FireRedAsr2System.process`'s glue, pay four model loads, and invalidate every recorded FireRed figure. `execution.residency` is corrected to the granularity that holds instead. |
| **No `--speakers` flag.** The diarizer estimates its own speaker count. | Simpler surface, and mechanically safe: `fluidaudiocli`'s offline options are "all optional" and `--num-speakers` merely overrides min/max. Cost, recorded rather than hidden: every cited diarization figure passed `--num-speakers 2`, so the shipped configuration is not the measured one. |
| **`--language` stays, as a closed enum on the Qwen stacks only.** 30 declared names, matched case-insensitively, passed through verbatim, refused otherwise. | `_build_prompt` (`qwen3_asr.py:926-929`) falls back to the raw string when a name misses, so `--language EN` would interpolate `language EN<asr_text>` into the prompt with no error — a flag that parses and silently changes the decode. The accepted set is closed and readable, so refusing is possible without any translation table. Dropping the flag instead would return every recorded Qwen figure to a configuration no caller can request. |
| **The language hint never reaches the aligner.** `roles.aligner.config.language_rule` declares the recorded CJK rule verbatim. | Finding 3. Two parameters answering one question can only disagree, and here one of the answers breaks CJK alignment. |
| **A truncated VibeVoice run exits 4 with the prefix it decoded.** `failure_recovery.partial_results` gains `prefix_only`; `none` is retired. | Finding 4: at 6.30 tok/s the cap lands inside the target use case, and losing a forty-minute transcript to an unterminated bracket is the worst available outcome. The cost is that the adapter owns a salvage parse rather than reading a field. `none` is retired because `vibevoice` was its only holder, and an enum member nothing reaches is the `satisfaction: unavailable` mistake. |
| **`--overlapping-segments` is passed only when `overlapped_speech` is requested.** | Owner's decision. It obliges an attribution fix rather than a note tweak: the 95.42 % participant-interval F1 *and* the 33.56 % MER both trace to a diarization run with the flag on, and that run shaped the turn set the MER was scored against — 589 raw segments to 195 accepted turns, 9 overlap abstentions, 33 short turns, 54 raw-fragment-only spans. The overlap-off configuration is unmeasured and the catalog says so. |
| **`complete` is always present in a result.** `coverage` appears only when it is `false`. | The one place where absence would be dangerous rather than meaningful: a saved document outlives its exit code, and `export` has to branch on it when merging. Every other absence rule is unchanged. |

## Architecture

```text
src/audio_cli/transcribe/
  stacks.json         the capability half of the stack table: 4 stacks x 9 capabilities
  stacks.py           reader; resolution -> availability, satisfaction, refusal code
  catalog.py          the `capabilities` report
  planner.py          pure: (stack, input_meta, wants, pins, language) -> Plan | Refusal
  plan.py             Plan and its serialization: roles, execution, capabilities, packages
  result.py           the normalized schema and the one serializer
  sample.py           a placeholder result through that same serializer
  refusals.py         one builder per error code, fields fixed by the contract's table
  orchestrator.py     stage order, observed accounting, abstention ledger, coverage, resume
  transport.py        per-stage execution: environment interpreter, product, or in-process
  adapters/           decode, silero, diarizer, qwen, firered, vibevoice, aligner
  stages/             standalone scripts that run inside provisioned environments
src/audio_cli/export/
  __init__.py         the `export` command
  cues.py             deterministic cue segmentation (issue #10 policy, hard-coded in v1)
  writers.py          srt, vtt, md, txt, jsonl
```

Three seams, and the middle one is the load-bearing choice.

**Provisioning.** `environments.packages_for(stack, roles)` stays the seam it already is, with
the backend→(package, role) map added to `manifest.json` per finding 8. The planner owns which
backend fills which role; the manifest owns which package supplies a backend. One table, two
owners.

**Transport.** One fresh subprocess per stage, chosen per environment, request JSON in and
result JSON out, exit code as the signal, progress on stderr —
[ENVIRONMENTS.md](ENVIRONMENTS.md) settles this and the recorded end-to-end measurement is what
it describes (`run_interview_pipeline.py:175`, "strictly sequential fresh subprocesses").

| Environment | How a stage runs | Residency |
| --- | --- | --- |
| `core` | in-process, `SileroOnnxVad` | **Declared exception.** No subprocess: it is a 2.3 MB hash-pinned ONNX file in the tool's own environment, already normalized to `SpeechRegion`, and there is no model object to leak. Normalization here is by code, not by construction. |
| `mlx` | `<root>/envs/mlx/bin/python <stage script> <request> <result>` | one process per stage, enforced by exit |
| `torch-firered` | same, one script for the whole stack | four models co-resident inside it (finding 1) |
| `torch-vibevoice` | same | one model |
| `swift` | the built product directly | one process |

Stage scripts live in the wheel, are passed by absolute path, and **import no `audio_cli`** —
installing it into a provisioned environment would drag onnxruntime and a conflicting numpy
into each one.

**Adapters run in `core`, over the stage's raw JSON.** The stage script emits the backend's own
JSON-safe shape, minus anything unbounded (FluidAudio's 256-float per-segment embeddings are
dropped at the boundary); normalization to the schema happens in the tool. Three reasons, and
the first is why this is not a detail: every adapter and every anti-fabrication assertion then
runs against the **recorded artifacts** with no provisioned environment at all, which is what
makes the schema's guarantees testable in the normal suite rather than only in a live run. The
stage script also stays thin and schema-free, and the schema can change without touching four
scripts in three runtimes. The normalization floor is still structural: no model-specific
object can cross a process boundary.

## The normalized result

```jsonc
{
  "schema_version": 1,
  "complete": true,
  "source": {"path": "...", "duration_seconds": 0.0, "timebase": "seconds"},
  "segments": [{"segment_id": "seg_0", "text": "...",
                "speaker": "...",          // iff diarization
                "start": 0.0, "end": 0.0,  // iff segment_timestamps
                "words": [{"word_id": "w_0", "text": "...", "start": 0.0, "end": 0.0}]}],
  "turns":              [],  // iff diarization
  "vad_regions":        [],  // iff vad
  "lid_regions":        [],  // iff lid
  "overlapped_speech":  [],  // iff overlapped_speech
  "abstentions":        [],  // floor artifact, always present
  "coverage":           {},  // iff complete is false
  "provenance": {"stack": "...", "outcomes": {}, "observed": {}, "plan": {}}
}
```

The gating rule is bidirectional and is the anti-fabrication guarantee: a key exists **iff**
its capability was requested. Two capabilities license no key at all — `languages` and
`verbatim` assert an interface and change no output shape — so their `outcome` is always
`produced` and the sample-output test must encode that they contribute nothing.

`provenance.plan` is the executed plan verbatim, which the spec documents elide for
readability; the key-set test compares against a real run, never against the elided print.
`total_wall_seconds` is defined as the **sum of the stage walls**, so the arithmetic stays
checkable; a caller timing the command externally sees more, because interpreter startup and
artifact writes are outside every stage — the same distinction
`run_interview_pipeline.py:201-206` draws.

`abstentions[].reason` is a three-member enum, one per cause the recorded runner distinguishes:
`overlap` (more than one speaker active), `short_turn` (an accepted turn below the 500 ms
minimum), and `raw_fragment` (a span whose only activity was a sub-250 ms diarizer fragment).
On the 30-minute interview those are 9, 33, and 54 entries covering 25 s of 1800 s. The
budget-unprocessed turns the runner files beside them are **coverage**, not abstention: nothing
was declined, the work was not reached.

`coverage` is over the source timeline: `missing_intervals` are the extents of units that did
not run, `covered_intervals` their complement, `covered_fraction` covered duration over source
duration, and `covered_through_seconds` the start of the first missing interval. That last
number is genuinely early when duration-bucketed ordering leaves the longest turns unprocessed
(`run_turn_attributed_mlx_asr.py:636`), which is why the explicit interval lists are carried
beside the watermark rather than instead of it.

## The stack table

`stacks.json` beside the module, read by `stacks.py`. It holds one cell per (stack,
capability), resolving to exactly one of five values, plus the per-stack columns the catalog
needs: family, environment, the `roles` sentence, unit and unit-count rule, failure recovery,
`cost.proved` with its rate and fixture, the language vocabulary, and the one-line
characterization `stack_required` prints.

| Cell | Catalog `availability` | Plan `satisfaction` | Requesting it |
| --- | --- | --- | --- |
| `native` | `native` | `native` | adds nothing |
| `native_stage:<stage>` | `native` | `native` with `stage` | adds cost, not composition |
| `add_on:<package>` | `requires_add_on` | `derived` with `backend` | adds a role and a package |
| `unsatisfiable_on_stack` | `impossible` + `reason` | — | exit 2, `allowed` non-empty |
| `unsupported` | `impossible` + `reason` | — | exit 2, `allowed: []` |

That is the whole of the planner's logic, and both refusal codes fall out of whether `allowed`
is empty, which is what keeps them from being rendered alike. VOCABULARY's derivation table
stays the source of truth and the test is parametrized over it.

**A separate file from `manifest.json`, deliberately.** The design handoff asks for "one table
with two owners, not two tables", and `environments/__init__.py` states in its own docstring
that the capability half is not there. Two files with a cross-check test — every stack named in
one appears in the other, every package a cell names exists — is two owners of one table; a
`stacks` key inside `manifest.json` would be one file with two audiences and would contradict a
boundary that was drawn on purpose.

**Language vocabularies are published per stack and never reconciled.** There are three, and
they disagree by design:

| Stack | `--language` | `lid` output |
| --- | --- | --- |
| `qwen-1.7b`, `qwen-0.6b` | 30 declared names, case-insensitive, verbatim | — (exit 2) |
| `vibevoice` | not accepted | — (exit 2) |
| `firered` | not accepted | FireRedLID's 115-label vocabulary (`en`, `zh yue`, `zh xinan`, …) |

Qwen calls it `Cantonese`; FireRed emits `yue`. Mapping between them is the translation layer
this surface exists without, so each is published as whose vocabulary it is. The declared/measured
line stays where the catalog already draws it: 30 names declared, three actually run here.

## Phases

Each phase ends with a green `uv run --extra dev pytest`, a shipped surface, and acceptance
stated as assertions that can fail. No phase depends on a later one.

### Phase 0 — the diff target

The documents are the acceptance criterion for everything after this, and three tests enforce
them, so the corrections in [Document changes](#document-changes) land first. No `src/` change
except the manifest's backend map (finding 8) and its test.

*Acceptance.* `tests/test_spec_docs.py` and `tests/test_environments.py` stay green with the
edits in place; `packages_for("firered", {...four backends...})` returns `[firered-asr2s]` and
still raises on a backend that does not fill its role.

### Phase A — the schema, the serializer, and the sample

`result.py` and `sample.py`. One serializer, no second rendering path. Placeholder timing and
text are `null`, never `0.0`; enum fields show one legal member.

*Acceptance.* A key exists iff its capability was requested, asserted over the derivation
table's every cell. `"N/A"` never appears as an attribution anywhere in a serialized document,
including on turns. A placeholder document contains no `0.0` bound. Adding a key for an
unrequested capability, or a `0.0` where `null` belongs, fails the suite — checked by
constructing both.

### Phase B — the table, the planner, and every refusal

`stacks.json`, `stacks.py`, `planner.py`, `refusals.py`, `catalog.py`, `plan.py`, and the two
commands. Pure functions over the table and a metadata probe: no media decoding beyond
`probe_media`, no provisioning, no network.

Ships `audio transcribe capabilities` and `audio transcribe plan`, and all thirteen refusals.
Refusal payloads print bare on stderr as the documents show; the shipped commands keep their
`{"error": …}` envelope, and the divergence is recorded rather than silently made a third
convention.

*Acceptance.* `capabilities` and `plan` output diffed key-for-key against HAPPY_PATH §1.1,
§1.2, §2.1, §3.1 by the `shape()` comparison `test_shipped_commands_match_the_document.py`
already defines, extended to the new commands. Every refusal in §4.1–4.7 and CONTRACT §5
reproduced field-for-field against the per-code table, with `capability_unknown`,
`capability_unsatisfiable_on_stack`, and `capability_unsupported` distinct. `--language EN`
returns `option_value_unsupported` with `did_you_mean: "English"`; `--language english`
resolves to `English`; the 30 names equal the provisioned checkpoint's `support_languages` when
the `mlx` environment exists, and the test skips when it does not. A plan carries no
`outcomes`.

### Phase C — transport, orchestration, and the Qwen stacks

`transport.py`, `orchestrator.py`, the `decode`, `silero`, `diarizer`, `qwen`, and `aligner`
adapters, their stage scripts, and `audio transcribe run` for `qwen-1.7b` and `qwen-0.6b`. Five
of seven roles, and the two exit codes that only `run` can return.

The ASR uses `_generate_chunks_batched` for both the diarized and the fixed-chunk case: one
declared `api_path`, and it is the only path that reports per-unit completion, without which
`failure_recovery: per_unit` and exit 4 are not implementable. `max_tokens` is declared
configuration. Resume decodes and re-diarizes the whole file (finding 7) and transcribes only
units intersecting `--range`.

*Acceptance.* A real run's `provenance` key set equals its plan's `sample_output` key set,
parametrized over the derivation table. Exit 3 before `pull` with the `missing` payload; exit 4
on a budget forced low enough to truncate, writing a document that satisfies every floor and
whose `covered_intervals` and `missing_intervals` partition the source exactly. A resumed run's
turn bounds equal the first run's for the units they share. The scaffold strip is asserted on
the **no-hint** configuration (finding 2). `peak_rss_bytes` is the maximum of the per-stage
peaks, never their sum.

### Phase D — FireRed

One stage script for the whole stack, three or four models by request. Carries the word
partition (finding 5), `lid_regions[]` by region grouping with the label-constancy assertion
(finding 6), the drop of `lang: null` / `lang_confidence: 0` when LID did not run, and the six
AED decode parameters declared.

*Acceptance.* The punctuation invariant holds per segment on real output and the assertion fails
when a word is dropped from the stream. `lid_regions[]` bounds equal `vad_regions[]` bounds, and
a region with two distinct labels fails. With LID off, no `lang` or `lang_confidence` key exists
anywhere in the document. `words` never carries a confidence field.

### Phase E — VibeVoice

Both forms of "no speaker" — the literal `"N/A"` and the absent key — become an absent key.
Non-speech event tags survive as segments with bounds and no words. The truncation salvage of
finding 4, `max_new_tokens` declared, and the 43-minute projection in the catalog.

*Acceptance.* A recorded segment carrying `Speaker: "N/A"` produces no `speaker` key, and the
test fails if the string reaches the document. `segments_without_words` counts the event tags
while `word_timestamps` still reports `produced`. A truncated `raw_text`, cut at an arbitrary
offset from a real artifact, yields every complete segment before the cut, exit 4, and a
coverage watermark at the last complete segment's end — and yields nothing at all through
`post_process_transcription`, which is the reason the salvage exists.

### Phase F — `export`

`srt`, `vtt`, `md`, `txt`, `jsonl`; several `--input` documents merged in timeline order with
re-numbered ids; `timing_required_for_format` when word timing is absent; cue policy hard-coded
per issue #10.

*Acceptance.* Subtitle output refuses a transcript with no `words`. Cue bounds come from the
first and last word of a segment, never from a segment extent on a stack that has none. A
segment with text and no word stream produces no cue. Merging a partial and its resumed
remainder yields contiguous ids and no duplicated span.

## Document changes

Corrections — each is a value or a claim the evidence refutes, in a document that is a diff
target:

| Where | Change |
| --- | --- |
| CONTRACT §1.1, §1.2 and HAPPY_PATH §1.2 (5 occurrences) | `bytes: 2463307541` → `2467859030`. The manifest carries the measured figure and `tests/test_environments.py:205` retires this exact number as illustrative. |
| HAPPY_PATH §1.1 `word_timestamps` note | "Adds Qwen3-ForcedAligner in the torch environment, so this request spans three" → the aligner runs in `mlx` beside the ASR and adds no runtime. |
| HAPPY_PATH §2.1 `execution.note` | "both model stages share the torch environment" contradicts `environments_spanned: ["mlx", "torch-vibevoice"]` one line above. |
| HAPPY_PATH §3.1 `execution` and `observed` | Residency corrected to environment granularity; illustrative peaks replaced by the measured 9.12 GiB (LID off) / 12.26 GiB (LID on) pair. |
| CONTRACT §1.1 prose, VOCABULARY "unit" | `unit_count_known_at_plan_time` is described but never emitted, and VOCABULARY says the count is "reported absent" where the payload prints `null`. Ruling: `null` in a fixed-shape structural block, and VOCABULARY's sentence changes. |
| `diarization` and `overlapped_speech` catalog notes | Both measured figures attributed to the overlap-on, two-speaker-prior configuration; the shipped configuration named as unmeasured. |

Additions — names and rows this plan needs that no document yet carries:

- `abstentions[].reason`: `overlap`, `short_turn`, `raw_fragment`, with the cause of each.
- The three turn-threshold **values** VOCABULARY names without publishing: `raw_fragment_min_ms`
  250, `accepted_turn_min_ms` 500, `same_label_merge_max_ms` 300
  (`run_turn_attributed_mlx_asr.py:78-84`).
- `failure_recovery.partial_results`: `prefix_only` added, `none` retired.
- `complete` in a result document, always present.
- One error code, `option_value_unsupported` (exit 2; `field`, `provided`, `allowed`,
  `did_you_mean` when a near value exists, `fix`) — for a stack that takes the option but not
  that value. `option_unsupported_on_stack` keeps its exact field list for the stack that takes
  no such option. Amending one row to carry conditional fields was considered and rejected: the
  three-way capability split exists precisely so two different failures are not rendered alike.
- The 30 accepted `--language` names, cited to `config.json:support_languages` at both pinned
  Qwen revisions, and FireRedLID's label vocabulary cited to `dict.txt`.
- `roles.aligner.config.language_rule`, and the Cantonese trap as its reason.

## Test strategy

Three kinds, and the repository has been bitten by the absence of each.

**Documents against each other** — `test_spec_docs.py`, unchanged in kind, extended to the new
names.

**Real output against the documents** — `test_shipped_commands_match_the_document.py`'s
`shape()` comparison, extended to `capabilities`, `plan`, and `run`. This is the comparison that
did not exist when three shipped commands had all drifted.

**Adapters against recorded artifacts** — new, and the reason adapters live in `core`.
`benchmark_runs/` is untracked, so the load-bearing fixtures are **excerpts** under
`tests/fixtures/`: two or three segments per stack, each carrying its source artifact path, that
artifact's sha256, and the excerpt rule in a header. Full-artifact tests run beside them and
skip when the untracked originals are absent, so the 12,370-word partition is checked locally
and the 30-word version is checked anywhere.

Two habits carried in from the provisioning pass. **Every invariant gets the assertion that
would fail if it were violated, and that failure is constructed** — the punctuation floor spent
four review passes being vacuous. And **a double must be able to represent the state the code
under test produces**: a fake stage that cannot return a truncated decode cannot test the
salvage, and a fake that cannot exhaust a budget cannot test exit 4. The flag-mutation sweep is
the cheap check for the rest: neutralize each `--range`, `--language`, `--want`, `--format`,
`-o` branch in turn and confirm the suite notices.

## Open evidence this plan does not close

New, from this pass:

- **The shipped diarizer configuration is unmeasured** in two ways: no `--num-speakers` prior,
  and no `--overlapping-segments` unless `overlapped_speech` is requested. Both cited figures
  were produced with the opposite settings. A cheap probe exists — the fixture, the product, and
  `score_diarization.py` are all present — and it is not run.
- **VibeVoice's 43-minute cap is a rate extrapolation**, not an observed truncation. No recorded
  run hit `hit_max_new_tokens`.
- **The aligner's language rule is unmeasured against alternatives.** The CJK rule is the one the
  equivalence probe used; whether a different rule would align better is untested, and the
  Cantonese branch's absence is read from source rather than from a failed run.

Carried from [HANDOFF.md](HANDOFF.md), unchanged: boundary MAE/P95 for FireRed's native word
times and for the aligner; filler recall on any stack; cross-process determinism for the Qwen
batched path; accuracy outside Cantonese on any stack; the license review; and no physical
16 GiB claim for the product-demo route.

## Out of scope

The Observation Store and its identity scheme; `analyze` and the disfluency annotator; capture
role and non-oracle role mapping; the cue-tuning surface of issue #10 beyond a hard-coded
policy; a persistent stage worker, which would leave what was measured and needs its own
evidence; and the VibeVoice MLX re-measurement, which stands between four provisioned
environments and three.
