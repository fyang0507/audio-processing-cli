# Agent handoff

Updated 2026-08-16. This repository currently has a production-oriented audio
enhancement CLI and a completed ASR research/distillation phase. The next phase
is to turn the ASR decisions into a transcription CLI where the caller picks a
stack and states requirements, plus its associated agent skill.

## Current state

- `audio inspect` and `audio enhance` are implemented under `src/audio_cli/`.
  Their contract and usage are in [README.md](README.md); do not regress the
  deterministic, render-from-original enhancement flow.
- ASR research for [Issue #2](https://github.com/fyang0507/audio-processing-cli/issues/2)
  is on branch `codex/asr-benchmark-field-guide` in
  [draft PR #8](https://github.com/fyang0507/audio-processing-cli/pull/8).
- The benchmark harness, manifests, compact results, and controlled research
  record live under `model_tests/`. Raw media, downloaded model weights, and
  local run directories are intentionally ignored.
- No transcription command or ASR backend abstraction has been implemented yet.
- [VOCABULARY.md](VOCABULARY.md) settles the naming contract for that work:
  stack, role, backend, add-on, package, environment, capability, satisfaction,
  availability, evidence, plan, policy, and the floors that are never optional.
  It also records which words were retired so they do not drift back in.
- [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) is the command sequence that
  contract must produce, end to end for all four stack ids including teardown.

## Read this first

1. [VOCABULARY.md](VOCABULARY.md) — the settled naming contract. Read it before
   naming anything new; it resolves collisions between the PRD, this file, and
   the decision report.
2. [model_tests/DECISION_REPORT.md](model_tests/DECISION_REPORT.md) — short
   capability field guide and recommended stacks. Treat this as the product
   decision input.
3. [model_tests/EXPERIMENT_RESULTS.md](model_tests/EXPERIMENT_RESULTS.md) — short
   quantitative evidence and its limits. Use it to justify defaults and tests.
4. [model_tests/FINDINGS.md](model_tests/FINDINGS.md) — full trace only when a
   claim or experiment needs auditing.
5. [model_tests/benchmark/README.md](model_tests/benchmark/README.md) — benchmark
   commands, records, and artifact layout.

Keep three kinds of statement separate in code and documentation:

- **Capability:** an output/interface the tested backend exposes.
- **Measured:** a result tied to the recorded fixture, version, hardware, and
  configuration.
- **Unresolved:** a decision that still needs targeted evidence.

## Decisions to carry into the CLI

- Load the smallest sufficient set of backends for what was actually requested;
  do not hard-code a universal ASR → VAD → aligner → diarizer chain.
- Choose the stack, then state requirements. The caller names one of
  `qwen-1.7b`, `qwen-0.6b`, `vibevoice`, or `firered` — that choice fixes
  transcript quality, language and dialect behavior, and which capabilities
  arrive natively, none of which is derivable from a requirement list. The
  planner then derives add-ons mechanically: `word_bounds` on a stack without
  native word timing forces the aligner, `speaker_attribution` on a stack without
  native speaker structure forces a diarizer and the reconciler. No default
  stack, no preference scalar, no tie-break ordering — an earlier draft had the
  planner select the ASR from capabilities, which made it decide quality it has
  no basis to judge. See [VOCABULARY.md](VOCABULARY.md) and
  [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md).
- Make the plan answerable in two steps, and generate its sample output rather
  than writing one. Step one is the only place a caller can decide step two, so its
  catalog carries the context that decision needs, not just an `availability` enum:
  the stack's determinism tolerance and measured resource envelope, whether it accepts
  a language input, the packages/environment/tools/measured time and memory each
  add-on costs, which capabilities share one add-on stage so a caller does not
  triple-count a single diarizer run, and where a capability is native but its accuracy
  is unmeasured. A stack alone returns that catalog; a
  stack plus requirements returns the resolved plan and a `sample_output` built by
  serializing a placeholder result through the same serializer `run` uses. Four
  stacks against nine requestable capabilities is roughly 2,048 combinations, so a
  hand-maintained example set would both rot and be infeasible; one generated path
  covers all of them. A second, parallel sample-rendering path is the specific
  mistake to avoid here.
- Separate floors from choices. Six floors, none of them optional: punctuated and
  sentence-segmented text (so FireRed always runs FireRedPunc); punctuation at the
  sentence level only; the canonical source timeline; no synthesized bounds;
  abstention survival; and normalization at the adapter boundary. The punctuation
  floor was the one with a named live risk, and the named risk was wrong: FireRedPunc
  emits punctuated *sentences* with sentence bounds, not marks with their own bounds,
  so "reattach the mark and drop its bounds" could never fire. What the floor
  actually requires is the invariant cue splitting rests on — a sentence's text,
  stripped of punctuation and whitespace, equals the concatenation of its word texts,
  compared case-insensitively because FireRedPunc lowercases and then re-capitalizes.
  Verified on 12,370 recorded FireRed words and every aligned segment of the
  forced-aligner artifact.
- Provision explicitly. `audio packages pull` is the only thing that downloads
  weights, builds the FluidAudio Swift product, or applies the VibeVoice patch.
  A transcription request resolves its plan, then fails closed with the exact
  fix command when a package is missing. Only the small hash-pinned artifacts
  auto-fetch, as the Silero backend already does.
- For interviews, prefer **FluidAudio or trusted capture channels → reconciled
  external turns → persistent Qwen3-ASR 1.7B 8-bit**, with 0.6B as the explicit
  latency and memory tradeoff. This is a recommendation for the caller, not a
  default the planner applies.
- Use **VibeVoice 7B → Qwen3-ForcedAligner** when editing structure,
  speaker-labelled segments, code switches, and fillers matter. Word-level
  alignment is a required v1 output, so `transcribe` aligns the whole transcript
  when asked; selective per-segment alignment belongs to the Observation Store's
  lazy enrichment. VibeVoice needs a fixed internal seed because its acoustic
  tokenizer samples a Gaussian latent; unseeded runs disagree and no downstream
  `word_id` would be stable.
- Expose **FireRedASR2S as its full pipeline**. Prefer it for dialect form,
  region LID, native word timing, or tighter memory/latency; it has no native
  speaker output, and it emits no per-word confidence — only a sentence-level
  `asr_confidence`, which is not a v1 capability.
- Preserve external timestamps and speaker provenance. Qwen container bounds
  are processing containers, not speech or word timestamps. Language output is
  one capability dimension, not a routing oracle or token-level code-switch map.
- Abstain on ambiguous overlap, rapid backchannels, role identity, and semantic
  speaker claims instead of inventing certainty.
- Keep large runtimes and model families optional and lazy. Do not add them to
  the core enhancement dependency graph merely to expose transcription. Group
  them into as few provisioned environments as their dependencies permit —
  three is the floor — under one managed root, not beside a source checkout the
  way the `model_tests/` experiments were.

## What v1 covers

v1 ships all four stacks — `qwen-1.7b`, `qwen-0.6b`, `vibevoice`, and `firered` —
because the interface is now organized around choosing between them, so shipping
a subset would mean shipping a choice the caller cannot make. They differ exactly
where it matters, which is why carrying them together from the start is
deliberate rather than incidental scope: with a single backend there is no way to
distinguish a normalized schema from that backend's output shape. Qwen declares
container bounds and one container-level language label and no speakers;
VibeVoice declares native anonymous speaker structure and segment bounds and no
word timing; FireRed declares native word timing, native speech bounds, and a
region language label, and no speakers. The adapter boundary and the rule that
container bounds never become timestamps are under load immediately.

The two Qwen sizes are not a redundant pair. They resolve identically for every
capability, and their recorded text fidelity differs — on the same probe 1.7B
retained `耍啥子` where 0.6B rendered `刷啥子` — which is what forced the evidence
model to distinguish a refuted observation from an unmeasured one rather than filing
both as "no number".

Word-level timing is required output, not a nice-to-have, because the subtitle
artifact depends on it: container bounds give one cue per processing container and
diarizer turns averaged 9.2 seconds on the 30-minute fixture against a ≤7-second
subtitle convention. Only `word_bounds` supports real cues. Qwen and VibeVoice
need the aligner for it, so one request can span all three provisioned
environments; FireRed is the only stack that produces subtitle-grade timing
natively.

Subtitle generation itself lives in `export`, which is deterministic
post-processing with no packages and no `plan`/`run` split. It fails closed when
the transcript has no word timing rather than inventing cue bounds. Cue
segmentation is real work — duration and line limits, CJK versus Latin character
widths, breaking at punctuation, never spanning a speaker change — and its tuning
options are parked as a follow-up issue. Two constraints on it come from the
recorded artifacts rather than from convention: breaking at punctuation means
mapping a mark's position in the sentence text to a word index, which is sound only
case-insensitively; and a segment can carry text with no word stream at all, which
is how VibeVoice's non-speech event tags arrive. Timing quality is unvalidated:
boundary MAE/P95 is unmeasured for both FireRed's native times and the aligner.

Risk sits earliest in the schema, so it is worth settling before any heavyweight
runtime is wired: normalized transcription output with explicit provenance for
text, language, external and native bounds, word timing, anonymous speaker,
capture role, and abstentions. A capability no backend produced must stay missing
rather than be synthesized from an unrelated field — and neither must a backend's
default-filled field be published as a value, which is what FireRed's
`lang_confidence: 0` on LID-off runs and VibeVoice's `Speaker: "N/A"` on non-speech
segments would each become if passed straight through. Backend
capability declarations and the planner come next, then `audio packages`
provisioning and its registry, then the four stacks behind adapters, then
`export`. Model results are normalized at the adapter boundary; model-specific
objects do not travel past it. The associated agent skill is worth writing only once the
command contract is executable, and its references should be the decision report
and experiment digest rather than the full findings file.

## Acceptance and open evidence

- Add contract tests for capability negotiation, optional backend loading,
  provenance, absolute-bound reattachment, overlap abstention, and the absence
  of fabricated timestamps/speaker identity.
- Add a punctuation test per stack that can actually fail: a sentence's text,
  stripped of punctuation and whitespace, equals the concatenation of its word texts
  compared case-insensitively; no mark object carries bounds; and no word's bounds
  grew to cover a mark. Skip segments with no word stream rather than asserting they
  have one. The previous formulation — "no parallel punctuation stream survives the
  adapter" — passes vacuously on the one stack it was written for, which is how the
  wrong risk survived four review passes.
- Assert that a plan's `sample_output` key set equals a real run's key set, and
  that no key exists for a capability that was not requested. Parametrize it over
  the derivation table in [VOCABULARY.md](VOCABULARY.md) so the table stays the
  single source of truth, including its distinction between the three exit-2 codes
  and its four legal cell forms.
- Assert that a plan carries no `outcome` field and that a run's embedded provenance
  carries one per capability. That is the only difference between the two documents,
  so it is the only thing that can silently drift.
- Keep the existing suite green with `uv run --extra dev pytest`.
- Do not claim physical 16 GiB support yet. Current RSS, MLX, PyTorch, and Core
  ML counters have different scopes.
- The product-demo route knowingly exceeds the PRD's 16 GB target: it reached
  20.28 GiB live MPS allocation on the 30-minute run and OOMed under a strict
  16 GiB cap. Shipping it in v1 anyway is an accepted product decision. The plan
  therefore declares the route's measured peak and `transcribe` warns from it
  rather than blocking, and no 16 GiB claim is made for that route.
- FireRed and ForcedAligner word-boundary MAE/P95 remain unmeasured.
- Mandarin/English, broad dialect, switch-span, filler, and repair accuracy need
  frozen held-out labels; the Qwen auto-language probe verifies behavior, not
  broad accuracy.
- The complete dedicated-channel merge and non-oracle participant/interviewer
  role mapping remain unmeasured.
- Review checkpoint/runtime licenses before choosing production defaults or
  redistributing model integrations. Two are recorded: the FluidAudio SDK is
  Apache-2.0 and `speaker-diarization-coreml` is CC-BY-4.0. Every other package
  reports `license: "unreviewed"` rather than omitting the field, so an unreviewed
  package cannot read as a cleared one.
- Qwen's `verbatim` interface was never exercised: the capability record states that
  no verbatim or filler mode was run, so the only text-fidelity evidence for either
  Qwen size is a single Sichuanese lexeme. `verbatim` also changes no plan
  composition in v1, because nothing cleans — it is an assertion the plan answers
  with evidence.
- Every figure in the interview route's `measured` block comes from a run that passed
  the language hint `"Cantonese"`, so `--language` is exposed on the Qwen stacks: the
  measured configuration has to be one a caller can actually ask for. It is an input,
  not a capability — no role, no package, no output field — and the stacks that take no
  language argument refuse it rather than ignoring it. The no-hint path remains
  unmeasured for accuracy.
- FireRed reproduces text exactly but timestamps only to 1 ms, so
  `determinism_tolerance_ms` is declared per backend and surfaced in step one. That
  drift is a fraction of a video frame and irrelevant to subtitles; it is not
  irrelevant to a `word_id` keyed on a start time, so the Observation Store's identity
  scheme cannot key on one.

## Working commands

```bash
git switch codex/asr-benchmark-field-guide
uv sync --extra dev
uv run --extra dev pytest
```

Before new model experiments, reuse the tracked runners and compact result
format. Keep raw fixtures and local model/run artifacts outside Git.
