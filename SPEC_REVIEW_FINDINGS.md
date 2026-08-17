# Spec review findings — open items for triage

Adversarial review of [VOCABULARY.md](VOCABULARY.md) and
[TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md), 2026-08-16. Four passes: three
by a reviewer that had seen the drafts evolve, one by a fresh reviewer with no
prior context. Findings already applied are summarised at the bottom; everything
above the line is **open**.

Verification status is stated per item and matters for triage:

- **verified** — I independently checked it against the named source.
- **reported** — the reviewer cited a source; I did not re-check it.

## Needs a decision from you

### O1. FireRed does not emit per-word confidence, and three research documents say it does — verified, high

`fireredasr2s/fireredasr2system.py:182-184` builds every word as exactly
`{"start_ms", "end_ms", "text"}`. Confidence is attached only to *sentences*
(`asr_confidence`). Confirmed across all five recorded FireRed artifacts: word
key sets are exactly `['end_ms', 'start_ms', 'text']` over 12,370 words
(379 + 246 + 246 + 3833 + 7666).

The claim originates upstream, not in the spec:

| Document | Claim |
| --- | --- |
| `model_tests/DECISION_REPORT.md:10` | "word time/confidence" |
| `model_tests/EXPERIMENT_RESULTS.md:16` | "native word time/confidence" |
| `model_tests/FINDINGS.md:77` | "word-level timestamps + confidence" |
| `HANDOFF.md` | "native word time/confidence" |

**Decision needed:** correct all four, or only the two spec documents. Fixing only
the spec leaves it contradicting its own cited decision input in the opposite
direction. `DECISION_REPORT` and `FINDINGS` are the tracked research record of a
completed phase under draft PR #8, which is why I have not touched them.

**Product consequence, independent of the doc question:** `word_confidence` was
part of the case for the `firered` stack. It does not exist. FireRed's remaining
unique natives are word timing, native speech bounds, region language, and dialect
form retention — still a real case, but the audit-signal argument is gone. What
does exist is *sentence*-level confidence; if that is wanted it needs its own
capability name, since `word_confidence` cannot be satisfied by it.

**Blast radius in the spec if we re-declare it unavailable:** the `firered`
catalog entry, §3's "every requirement is native, so there are no add-ons at all",
§3's opening claim to be "the only stack with native word timing, word
confidence…", §5's refusal payload pointing callers at `allowed: ["firered"]`
(currently an error whose suggested fix does not work), the coverage matrix, and
the `VOCABULARY` derivation table.

### O2. The punctuation floor names the wrong risk and prescribes a no-op — verified, high

The floor currently says FireRedPunc "emits punctuation with its own bounds" and
instructs the adapter to reattach each mark and drop the mark's bounds.

What the code does: `fireredpunc/punc.py:109-119` returns `punc_sentences` —
punctuated **sentence strings** with **sentence** bounds. Individual marks have no
bounds at all. `fireredasr2system.py:182-184` builds `words` from the
pre-punctuation AED timestamps, so `words` contains no marks. Measured on the
recorded artifact: **0 of 246** word tokens contain any punctuation mark; word
`好` versus sentence `好，`.

So there is no parallel per-mark stream to drop bounds from, and the prescribed
rule is inert. The real, unspecified work is the inverse: aligning punctuated
sentence text back onto an unpunctuated word sequence to decide which word each
mark attaches to — which is exactly where a mark gets attached to the wrong word,
and where the tie-breaks live.

Consequence: the contract test mandated in `HANDOFF.md` — "no parallel punctuation
stream survives the adapter" — passes **vacuously** on FireRed while the actual
failure mode goes untested. This is HANDOFF's own "a floor with a named live risk
and no test is only prose", one level deeper: the named risk is the wrong risk.

Traceable origin: `DECISION_REPORT.md:10` says FireRed yields "punctuation
bounds", meaning punctuated sentences with bounds. Both spec documents read that
as per-mark bounds.

**Decision needed:** same upstream question as O1, plus whether specifying the
sentence-text-to-word-sequence attachment rule belongs in this spec or in the
implementation issue.

## Needs a spec fix — no decision required

| ID | Finding | Severity | Status |
| --- | --- | --- | --- |
| O3 | `region_language` is `native` in the catalog but written in add-on notation (`+ FireRedLID stage`) in the derivation table, and a third framing lists `lid` in `roles_included` unconditionally while the weights are fetched only on demand. A test parametrized over the table would classify it `requires_add_on`, contradicting the catalog enum. | medium | reported |
| O4 | `provenance_only` is documented as appearing in every plan but is `{}` for `firered`, and §2's abridgement note says it "takes the same shape as §1.1" when by the same rule VibeVoice's should be empty. | medium | reported |
| O5 | The stated per-code error-field contract matches none of the four payloads: `timing_required_for_format` has no `allowed` and adds three undeclared fields; `capability_unsupported` adds `reason`, which the contract does not permit but `VOCABULARY` requires; `packages_not_provisioned` adds two undeclared fields; two codes are prose-only with no payload shown. | medium | reported |
| O6 | A third exit-2 capability code exists (`capability_not_requestable`, for the provenance-only pair) but the derivation table — declared the single source of truth for exactly two codes — has no row for it, and two passages describe the parametrization as covering "one" unsatisfiable case, collapsing the codes back together. | medium | reported |
| O7 | `satisfaction` has an unreachable member and two tenses: `failed` "exits 1 and writes no result", so no artifact can ever carry it, while `native`/`derived` are resolved at plan time and `abstained` is a run-time outcome. Knock-on: the required-`evidence` rule binds only on `native`/`derived` and is silent on `abstained`. Suggested split: plan-time `satisfaction` versus run-time `outcome`. | medium | reported |
| O8 | The `evidence` model has no slot for a recorded **negative** observation. VibeVoice normalising `看哈→看一下` means the run *refuted* dialect preservation, yet it files as `interface: "verified"`, `quality: "unmeasured"` with the counter-example in a free-text note — the exact inversion the evidence object was added to prevent. `measured_limit` only covers the case where a quality *number* exists. Related: the merged `qwen-1.7b, qwen-0.6b` cell cannot express that 1.7B retained `耍啥子` where 0.6B rendered `刷啥子`. Suggested: `interface: "refuted"` or an `observed_limit` parallel to `measured_limit`; split the Qwen cell. | medium | reported |
| O9 | L-series line items: the "not the 1800 s figure" note refers to a value not present in the block it cites; no code is defined for an unrecognised capability name; `--format jsonl` appears for `export` but is described nowhere; `requires` carries three unrelated meanings; §3 shows no plan JSON at all, so `vad`, `lid` and `punctuator` never appear in any resolved plan and four of the eight roles are never exercised; `silero-vad` never appears in a `roles` or `requires` block despite §1.4 requesting `speech_bounds`; `--vad` has multiple candidate implementations but no pin, while `--diarizer` has one implementation and is the only pin defined; "observed stages total about 69 s" understates that the components came from different runs and cache states, one of them a 0.6B run. | low | reported |

## Needs evidence, not a fix

| ID | Question | Status |
| --- | --- | --- |
| O10 | `VOCABULARY` says "two Core ML bundles"; only one Core ML package is named anywhere in the repo. Plausible that the package contains segmentation plus embedding models, but nothing records the count. | reported |
| O11 | The Qwen role's `determinism_basis` says "greedy decode". No record states the MLX path decodes greedily, and the recorded private API signature takes a `sampler` argument whose configuration is unrecorded. The claim is honestly self-labelled as not a repeat-hash measurement, but "greedy decode" is itself unsourced. Contrast VibeVoice's, which is fully sourced to three seeded repeats sharing one normalised-output hash. | reported |
| O12 | `speaker-diarization-coreml` is the only package carrying a `license` field. Correct (CC-BY-4.0), but `HANDOFF` lists license review as open, so asserting one and omitting the rest may read as "the others are cleared". | reported |
| O13 | Boundary MAE/P95 remains unmeasured for both FireRed native word times and the forced aligner, so subtitle timing quality is unvalidated. Pre-existing open evidence item, restated here because `export` now depends on it. | verified |

## The process finding

O1, O2 and the already-fixed collar error share one cause: **the spec was verified
against `DECISION_REPORT` and `EXPERIMENT_RESULTS` prose rather than against the
artifacts and runner sources those documents summarise.** That is why O1 survived
four review passes — the summary documents all assert the capability, and only the
artifacts refute it.

Two operational consequences worth carrying into the implementation phase:

1. A claim about backend output must cite a runner source or a recorded artifact,
   not a summary document. Summaries are the decision input; artifacts are the
   evidence.
2. Reviewer familiarity cut accuracy. The reviewer that had watched the drafts
   evolve declared convergence twice while O1, O2 and two other high-severity
   defects sat in material it had already read. The fresh reviewer found them by
   going to the source files. Restricting a later pass to the recent diff would
   have missed four of six.

## Already applied

Corrected across the four passes: five fabricated numbers (four byte/RSS/MPS
counters back-computed from prose GiB figures rather than read from the result
JSONs, plus an invented VibeVoice sample rate); a 0.6B full-pipeline wall time
attributed to a 1.7B plan under one `reference_run`; a backend crash that would
have returned exit 0 with an `abstained` capability, indistinguishable from a
principled refusal; a sample that emitted keys for capabilities the request never
asked for, and separately segment timing keys for a capability that is exit-2 on
that stack; `satisfaction: unavailable` surviving in a numbered rule list and
eight cells of the derivation table after being deleted from its definition;
`policy.rendering` duplicating the `verbatim` capability; a `quality: "measured"`
claim with no number anywhere; the FluidAudio quality-preset config dropped during
a rewrite, leaving measured figures attributed to an undeclared configuration; a
single `min_turn_ms` collapsing three recorded thresholds and adopting the one
turns were *not* accepted at; `protected_intervals` emitted in machine-readable
output while defined nowhere in the repo; a `revision` field pinning VibeVoice's
source commit where the same field elsewhere means a checkpoint; a speaker-change
F1 attributed to a 250 ms collar when its record specifies a one-second tolerance;
and an abstention ledger that read as "checked, none found" on stacks with no
overlap detector.
