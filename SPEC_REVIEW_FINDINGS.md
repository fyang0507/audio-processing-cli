# Spec review findings — status after triage

Adversarial review of [VOCABULARY.md](VOCABULARY.md) and
[TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md), 2026-08-16, triaged 2026-08-17.
Four review passes: three by a reviewer that had seen the drafts evolve, one by a
fresh reviewer with no prior context.

**Everything raised in that review is now closed.** O1 and O2 were decided by the
repository owner; O3–O9 are applied; O10–O13 were evidence questions and the evidence
now exists. What remains open is the N-series below — findings this triage pass
turned up while verifying the others, three of which are genuine decisions rather
than fixes.

Verification status is stated per item:

- **verified** — checked against the named runner source or recorded artifact.
- **reported** — a reviewer cited a source; not independently re-checked.

## Decisions taken

### O1. `word_confidence` removed — verified, high

FireRed emits no per-word confidence: every word object is exactly
`{start_ms, end_ms, text}` (`fireredasr2s/fireredasr2system.py:181-184`), confirmed
over 12,370 word tokens across all five recorded artifacts. Three research documents
claimed otherwise.

**Decision: remove the capability; not a strict requirement. Update all relevant
docs.** Applied. `word_confidence` is gone from the capability namespace, the
derivation table, the FireRed catalog, §3's request lines, the §5 refusal example
(now `segment_bounds` on Qwen, which is a real `unsatisfiable_on_stack` case with a
working fix), and the coverage matrix. It is recorded in VOCABULARY's retired words
so it cannot drift back, and specifically so it is not reintroduced as a name for the
sentence-level value. `DECISION_REPORT`, `EXPERIMENT_RESULTS`, `FINDINGS`, and
`HANDOFF` are corrected, and `FINDINGS` carries a dated corrections section, matching
the supersession convention it already uses.

What does exist is sentence-level `asr_confidence`, non-null on every recorded
sentence, range 0.158–0.997. It is documented as existing and is not requestable in
v1.

### O2. The punctuation floor rewritten — verified, high

The floor said FireRedPunc "emits punctuation with its own bounds" and told the
adapter to reattach each mark and drop the mark's bounds. FireRedPunc returns
punctuated *sentence* strings with *sentence* bounds
(`fireredpunc/punc.py:109-119`); `words` comes from the pre-punctuation AED
timestamps. There was no parallel stream to strip bounds from, so the rule could
never fire, and the contract test mandated for it passed vacuously.

**Decision: punctuation exists only at sentence level; it is not a CLI option and
should not appear as one.** Applied. The floor is now
`punctuation_is_sentence_level`, and what it requires is the invariant cue splitting
actually rests on: stripping punctuation and whitespace from a sentence's `text`
yields exactly the concatenation of its word `text` values, compared
case-insensitively. Verified on all five FireRed artifacts (12,370 words) and all 17
aligned segments of the forced-aligner artifact. The HANDOFF contract test is
rewritten to assert that instead of the vacuous condition.

Three things the verification turned up that were in no document:

- **The punctuation stage recases text.** `RuleBaedTxtFix.fix` lowercases its input,
  then re-capitalizes sentence starts and standalone `i`
  (`fireredpunc/punc.py:349-382`). 234 characters differ by case across the recorded
  artifacts. This is why the invariant is case-insensitive and why sentence text is
  not reconstructible from the word stream — each carries something the other does
  not, so both are kept and the sentence text is canonical for reading and subtitles.
- **Word tokens are not punctuation-free.** 20 of 12,370 carry an apostrophe, every
  one an English contraction (`it's`, `can't`, `you're`, `i've`, …) produced by the
  ASR itself. "No word token contains punctuation" would have been a false invariant.
- **The aligner agrees independently.** Qwen3-ForcedAligner emits one token per
  non-punctuation character — `好，现在开始。` aligns as five tokens — so both
  word-producing paths already treat punctuation as sentence-level.

## Applied without a decision

| ID | Finding | Resolution |
| --- | --- | --- |
| O3 | `region_language` was `native` in the catalog, add-on notation in the table, and an unconditional `lid` in `roles_included`. | One framing: `native (FireRedLID stage)` in the table, `roles_conditional: {"lid": "region_language"}` in the catalog, and `selected_by: "requirement:region_language"` in the plan. FireRedLID ships inside the package and fills a role the stack already declares, so it changes cost, not composition. |
| O4 | `provenance_only` documented as in every plan but `{}` for FireRed, with §2 claiming VibeVoice's "takes the same shape as §1.1". | Empty is now stated as the common case: Qwen is the only stack with a processing container. §2 shows `provenance_only: {}` explicitly and no longer claims §1.1's shape. Verified against `run_vibevoice.py`, which passes whole media in one call. |
| O5 | The per-code error-field contract matched none of the four payloads. | Replaced with a nine-row table of code → exit → exact fields, declared binding in both directions, plus a note that `allowed` means something different per code and is never free text. |
| O6 | A third exit-2 code existed but had no derivation-table row, and two passages said "one" unsatisfiable case. | The table has a `container_bounds, container_language` row; all three codes are defined together; the parametrization text says three. |
| O7 | `satisfaction` had an unreachable member (`failed`, which exits 1 and writes no result) and mixed plan-time with run-time values. | Split. `satisfaction` is plan-time `native`/`derived`; `outcome` is run-time `produced`/`abstained`; a crash is a `backend_failed` payload at exit 1, not an enum member. `outcome` is now the one field distinguishing a plan from a run's provenance, with its own assertion in HANDOFF. |
| O8 | `evidence` had no slot for a recorded **negative** observation, so a run that refuted dialect preservation filed as `unmeasured`. | `quality: "refuted"` added, with a required `observed_limit` companion. Two stacks now carry it: VibeVoice (看哈→看一下) and `qwen-0.6b` (刷啥子 where 1.7B retained 耍啥子). Evidence moved out of the derivation table into the per-stack catalogs, since resolution is identical across the two Qwen sizes and observed fidelity is not. |
| O9 | Eight line items: an unreferenced 1800 s note, no code for an unknown capability name, `--format jsonl` undescribed, `requires` carrying three meanings, no §3 plan JSON, `silero-vad` in no plan, `--vad` unpinnable, and a misleading 69 s sum. | All eight. `fixture_duration_seconds: 1800.0` is in the measured block; `capability_unknown` is defined; `jsonl` is specified as one segment object per line without provenance; `requires` split into `packages` / `requires_tool` / `requires_capability`; §3 shows plans exercising `vad`, `punctuator`, and `lid`; §1.4 shows `silero-vad` as the one auto-fetching package; `--vad` is the pin that has two implementations and `--diarizer` is stated as forced-with-a-pin-for-later; the 69 s figure now quotes the record's own `measured_end_to_end: false` and names the 0.6B stage it borrows. |

## Evidence questions, now answered

| ID | Question | Answer |
| --- | --- | --- |
| O10 | "Two Core ML bundles" with only one Core ML package named anywhere. | **One.** `FluidInference/speaker-diarization-coreml` is the only provisioned Core ML package. FluidAudio also ships a Core ML VAD, which its own `VadManager` resolves (`fluidaudio_vad_probe.swift:23`) and which this tool has never provisioned as a named package. Corrected to "one Core ML model package", with the VAD's status stated. |
| O11 | `determinism_basis: "greedy decode"` was unsourced. | **Sourced twice.** `run_turn_attributed_mlx_asr.py:661` passes `sampler=make_sampler(temp=0.0)`, and the capability record has `configuration/temperature = 0.0`. VibeVoice's text decode is likewise `do_sample=False` (`run_vibevoice.py:265`), which is now cited alongside its seeded-repeat hash. VOCABULARY now requires a `determinism_basis` to cite a repeat-hash measurement or a decode configuration; "by construction" alone is not a basis. |
| O12 | `license` asserted on one package only, which could read as "the others are cleared". | Licenses are a registry field reported by `audio packages list`, not a plan field — a plan resolves a pipeline, not a redistribution question. Two are recorded (FluidAudio SDK Apache-2.0, `speaker-diarization-coreml` CC-BY-4.0); every other package reports `license: "unreviewed"`. The field is dropped from the plan payload. |
| O13 | Boundary MAE/P95 unmeasured for FireRed native times and the aligner. | Still unmeasured — correctly, it is an open evidence item, not a spec defect. It stays in HANDOFF's open list and `export` states that its output is producible but not claimed broadcast-acceptable. |

## Open — from this triage pass

Found while verifying the above against artifacts and runner sources. N1–N3 are
decisions; N4–N6 are applied and listed so the changes are reviewable.

### N1. `verbatim`'s interface was never exercised on Qwen — verified, needs a decision

The capability record states plainly: *"No filler-specific or verbatim mode was
exercised. Filler preservation remains a transcript-quality measurement, not a native
output capability."* The only text-fidelity evidence for either Qwen size is one
Sichuanese lexeme. The spec currently declares `verbatim` `native` with
`interface: "verified"` on all four stacks.

Related, and the reason this is a decision rather than a fix: **`verbatim` changes no
plan composition in v1.** No backend exposes a verbatim switch and nothing in the
pipeline cleans, so the old claim "absent this, text is clean-rendered" described
machinery that does not exist — the same defect class as O2. I have redefined it as an
assertion the plan answers with evidence (and two stacks answer `refuted`), which
keeps it useful to an agent choosing a stack. The alternative is to demote it to
provenance-only until a rendering stage exists. Worth your call, because it is the
one requestable capability that resolves identically everywhere.

### N2. FireRed is text-deterministic but not timestamp-deterministic — verified, needs a decision

The recorded exact-repeat 60-minute run reproduced text exactly
(`all_text_sequences_equal: true`) with a maximum rebased timestamp drift of
**1.0 ms** against a declared 2.0 ms tolerance, so `all_normalized_segments_equal` is
**false**. A single `deterministic: true` boolean would have overclaimed this.

I added `determinism_tolerance_ms` (`1.0` for FireRed, `0.0` for the other two) so the
boolean means "reproducible within the declared tolerance". This matters downstream:
any `word_id` scheme keyed on word start times is not stable across FireRed runs, and
the Observation Store's identity scheme has not been specified yet. If `word_id` must
be run-stable, that is a constraint on the schema, not on the backend.

### N3. Every interview-route figure used a language hint — verified, needs a decision

The controlled A/B that produced 53.77 s, 3.02 GiB, and the 33.56%/52.64% MER pair ran
with `language = "Cantonese"` explicitly passed. The plan as specified passes
`language: null`. The measured block now says so rather than implying the figures
describe the no-hint path.

The open question is whether a language hint should be caller-settable. It is an
explicit input rather than an intent abstraction, so it does not conflict with the
"name the requirement" rule — but it is a new flag, so I have not added one. The
recorded no-hint probe shows both Qwen sizes still emit a language label and
**disagree with each other on the same recording**, which is an argument for allowing
the hint and against trusting the label.

### N4. VibeVoice's `Speaker: "N/A"` is the absence of a label — verified, applied

`test_multispeaker_pipeline.py:36` branches on `speaker == "N/A"`. Passed through
unchanged, `"N/A"` becomes a speaker id — a fabricated speaker of exactly the kind the
floors exist to prevent. The adapter-normalization floor now names it, alongside
FireRed's `lang: null, lang_confidence: 0`, which is emitted on every sentence whether
or not LID ran and would otherwise publish a zero confidence that reads as measured.

### N5. `word_bounds` can be absent on a segment even when satisfied — verified, applied

Two of the forced-aligner artifact's nineteen segments have `words: null`, both
VibeVoice non-speech event tags (`[Environmental Sounds]`) with real bounds and no
speech to align. This is neither a failure nor an abstention. The sample-output
guarantees now exclude per-segment word presence, the punctuation invariant binds only
where words exist, and issue #10 is told the cue splitter needs a rule for these
rather than assuming every segment yields cues.

### N6. Two figures were attributed to configurations that did not produce them — verified, applied

The MLX cache is cleared after every *batch*, not every turn — identical at
`batch_size: 1`, which is why it went unnoticed, and wrong at any other batch size.
And the `end_to_end_note`'s 69 s was an arithmetic sum whose 15.39 s diarization
component came from a separate **0.6B** end-to-end run; the record marks it
`measured_end_to_end: false` and calls it planning context. Both corrected here and in
`DECISION_REPORT`.

## The process finding

O1, O2, the already-fixed collar error, and N6 share one cause: **the spec was
verified against `DECISION_REPORT` and `EXPERIMENT_RESULTS` prose rather than against
the artifacts and runner sources those documents summarise.** O1 survived four review
passes because every summary asserts the capability and only the artifacts refute it.

Three operational consequences for the implementation phase:

1. A claim about backend output must cite a runner source or a recorded artifact, not
   a summary document. Summaries are the decision input; artifacts are the evidence.
2. Reviewer familiarity cut accuracy. The reviewer that had watched the drafts evolve
   declared convergence twice while O1, O2, and two other high-severity defects sat in
   material it had already read. The fresh reviewer found them by opening the source
   files. Restricting a later pass to the recent diff would have missed four of six.
3. Reading the artifacts is generative, not just corrective. This pass produced N1–N6
   — including a determinism limit, a fabricable speaker id, and a legitimately absent
   word stream — none of which any review pass over the prose could have found.

## Already applied in the review passes

Corrected across the four passes: five fabricated numbers (four byte/RSS/MPS counters
back-computed from prose GiB figures rather than read from the result JSONs, plus an
invented VibeVoice sample rate); a 0.6B full-pipeline wall time attributed to a 1.7B
plan under one `reference_run`; a backend crash that would have returned exit 0 with
an `abstained` capability, indistinguishable from a principled refusal; a sample that
emitted keys for capabilities the request never asked for, and separately segment
timing keys for a capability that is exit-2 on that stack; `satisfaction: unavailable`
surviving in a numbered rule list and eight cells of the derivation table after being
deleted from its definition; `policy.rendering` duplicating the `verbatim` capability;
a `quality: "measured"` claim with no number anywhere; the FluidAudio quality-preset
config dropped during a rewrite, leaving measured figures attributed to an undeclared
configuration; a single `min_turn_ms` collapsing three recorded thresholds and adopting
the one turns were *not* accepted at; `protected_intervals` emitted in machine-readable
output while defined nowhere in the repo; a `revision` field pinning VibeVoice's source
commit where the same field elsewhere means a checkpoint; a speaker-change F1
attributed to a 250 ms collar when its record specifies a one-second tolerance; and an
abstention ledger that read as "checked, none found" on stacks with no overlap detector.
