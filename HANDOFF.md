# Agent handoff

Updated 2026-09-04. This repository has a production-oriented audio enhancement CLI,
explicit model provisioning, transcription capability discovery and planning, four executable
transcription stacks, and deterministic transcript export. Issue #21 is merged; the candidate on
`codex/issues-22-24` completes FireRed (#22), VibeVoice (#23), and export (#24).

## Current state

- `audio inspect` and `audio enhance` are implemented under `src/audio_cli/`.
  Their contract and usage are in [README.md](README.md); do not regress the
  deterministic, render-from-original enhancement flow.
- `audio packages` and `audio doctor` are implemented, over the four provisioned
  environments in [ENVIRONMENTS.md](ENVIRONMENTS.md). The suite runs on a `FakeToolchain` and
  `FakeFetcher`, but the real paths are no longer unexercised: all four environments have been
  built from their hash-pinned locks and `verify` reports each `ok` against its own, a forced
  repair moved 1,101,647,163 measured bytes over the network for a 1,010,773,761-byte package,
  the Swift product builds and launches, and the `mlx-audio` private-API guard matches its pin
  with `signature_ok: true`. `torch-firered` resolves to transformers 5.1.0 and
  `torch-vibevoice` to 4.57.6, so the partition is installed rather than inferred.
- An audit of the CLI found **fifteen** defects and all fifteen are fixed on
  `fix/cli-provisioning-bugs` — ten in provisioning, five in the enhancement engine and its
  documents. The kinds of mistake worth remembering, because each survived review by looking
  finished:
  - **A claim nobody earned.** `pull` recorded `digest_verified: true` for every Hub package and
    hashed none of them; the manifest carries no `sha256` for a Hub source, so there was nothing
    to hash against. `verify` reported a check it never ran, and its honest `unverified` branch
    was unreachable.
  - **A flag that parses is not a flag that works.** `--repair` was declared, documented, and
    named in four `fix` strings while being read by no code at all. A mutation scan then
    neutralized all 20 flag branches and found six more the suite could not see.
  - **A parameter that decided nothing.** `vad_min_silence_ms` declared 300 ms while a second
    hard-coded merge required 540; two thresholds answering one question can only disagree.
  - **A bound the media does not have.** A resampler rounding up let a speech region end after
    the end of the file, which then made a treatment extension negative.
  - **A recorded failure that changed no outcome.** A Swift build whose product could not launch
    reported `product_runs: false` and exited 0 with an empty `warnings`.
  `pull` is now idempotent, tolerates a toolchain-blocked package when a *stack* selected it,
  refuses `--want` rather than ignoring it, and refuses a build whose product will not run.
- Two things that sweep surfaced and did **not** settle, both about `verify`. A moved
  `mlx-audio` private API is reported and never reaches `failed`, so `verify` exits 0 while
  saying `mlx_audio_private_api_matches_expected: false` — the guard the recorded Qwen timings
  depend on cannot fail the command. And a `blocked` environment is likewise exit 0, which is
  deliberate and documented. Whether the first should join the second or become an exit-3 check
  is a contract decision; there is no `audio` command that would fix it, so its `fix` would be a
  sentence.
- The suite's doubles are now held to one rule, learned twice: **a double must be able to
  represent the state a repair produces.** `FakeToolchain` could not stop being drifted and
  `FakeFetcher` would rewrite a snapshot unasked, and each made a repair test pass without the
  repair running. A flag-mutation scan over every `repair`/`force`/`dry_run`/`allow_*` branch in
  the command surfaces is the cheap way to find the rest; it found six unasserted branches on
  first run, all now covered.
- ASR research for [Issue #2](https://github.com/fyang0507/audio-processing-cli/issues/2)
  landed on `main` in [#8](https://github.com/fyang0507/audio-processing-cli/pull/8), and the
  `transcribe` specification plus the provisioning layer landed in
  [#15](https://github.com/fyang0507/audio-processing-cli/pull/15).
- The benchmark harness, manifests, compact results, and controlled research
  record live under `model_tests/`. Raw media, downloaded model weights, and
  local run directories are intentionally ignored.
- `audio transcribe capabilities`, `plan`, and `run` ship for all four stack ids. Every run uses
  one canonical temporary PCM WAV and keeps public bounds on the source timeline. Qwen uses
  strictly sequential fresh MLX/Swift stages; FireRed keeps native VAD, optional LID, ASR, and
  punctuation in one measured CPU process; VibeVoice makes one whole-media MPS generation call
  and invokes the MLX aligner only for requested word timing. Exit-3 preflight now verifies pinned
  weight revisions, Hub snapshot paths, every manifest allowlist pattern, and the pull receipt's
  recorded byte measurement, with repository/revision/path identity bound through the Hub cache
  index. For each live source checkout it also reads the full Git HEAD, derives the exact tracked set
  from the installed patch, checks manifest-owned post-patch hashes and matching receipt hashes,
  and rejects both ordinary and Git-ignored untracked artifacts before decode; ignored bytecode or
  extensions remain importable and therefore provenance-relevant. A legacy receipt may repeat the
  manifest's short commit alias or its full resolution, but the live HEAD must always be the full
  resolved commit. The two checkout-backed Python packages must also freeze under their
  manifest-pinned distribution names as direct `file://` installs of their exact managed checkout;
  verify repair validates those checkouts before syncing the lock, reinstalls them afterward, and
  freezes again. Managed Python interpreters and the contained, non-symlink
  FluidAudio product from its exact managed checkout are launched before decode rather than
  accepted from executable bits or receipt booleans. FluidAudio's pinned source patch requires
  the exact managed speaker-model directory and disables ModelHub downloads; pull records the
  product path and SHA256, and verify/run require the same live digest plus the manifest-owned
  post-patch source hash. The single URL artifact is accepted only at
  its contained, non-symlink manifest models path, both on pull's hash fast path and after download;
  Silero auto-fetch uses the same boundary. Pull refuses redirected `envs` parents or environment
  leaves before creation/install, while verify and run require every ready environment root to be
  the exact contained, non-symlink manifest directory before freeze or decode. A venv's inner
  `bin/python` symlink remains normal. Registry reads are descriptor-bound and a redirected
  provisioning-root leaf is `registry_unreadable` before `verify` probes subordinate state.
  Publication, downloads, and managed removal operate through
  already-opened non-symlink directory descriptors, so a concurrent parent-path swap cannot
  redirect an overwrite or deletion. URL and Silero downloads additionally bind publication and
  cleanup to the exact private temporary inode opened before transfer; managed leaf symlinks are
  replaced without following them, while directories are preserved. This guards accidental and
  public-destination retargeting; it does not claim protection from another same-credential
  process that discovers and changes a random private sibling between its final identity check
  and unlink, because that process can already unlink canonical user files directly and POSIX has
  no portable conditional-unlink primitive. Qwen and FireRed
  recover per-unit prefixes; VibeVoice recovers only complete decoded prefixes.
  Provider orchestration has one code owner under `src/audio_cli/transcribe/orchestrator/`:
  its dispatcher delegates directly to the Qwen, FireRed, or VibeVoice workflow. The previously
  importable `audio_cli.transcribe.native.run_native` path remains as a thin, frozen compatibility
  shim for FireRed and VibeVoice; new callers use `audio_cli.transcribe.orchestrator.run`. Here,
  “native” remains a capability or model-output property rather than a package boundary.
  The pre-release export-only builders formerly re-exported from
  `audio_cli.transcribe.refusals` remain as direct compatibility aliases, while their owner and
  preferred public import path is `audio_cli.export`. Shared refusal primitives retain one
  implementation under `audio_cli.command`, and the command-line payloads and exit codes are
  unchanged.
  `packages verify` also gates every ready package on a `ready` environment registry entry: a
  missing or non-ready entry keeps the environment verdict `absent`, emits
  `environment_not_ready`, and skips all checkout and runtime probes below that root.
- Issue #21 merged as [PR #29](https://github.com/fyang0507/audio-processing-cli/pull/29) at
  `1e8b979671273b04e677ef2e325c7796806fdcf5`. Its final
  verification collected and passed 408 tests; the targeted Ruff gate and `git diff --check`
  passed; `audio packages verify` reported all four environments `ok` with the pinned MLX private
  API hash and signature matching; a fresh floors-only Qwen 0.6B run produced 23 chronological
  segments over the canonical 139.284-second fixture timeline; and a fresh sdist/wheel build
  contained the environment stage scripts, core adapters, and shipped audio skill references.
  Claude reviewed the full staged diff adversarially through twelve numbered passes (one hung pass
  was discarded and rerun) and the terminal review returned exactly `CONVERGED`.
- [VOCABULARY.md](VOCABULARY.md) settles the naming contract for that work:
  stack, role, backend, add-on, package, environment, capability, satisfaction,
  availability, evidence, plan, policy, and the floors that are never optional.
  It also records which words were retired so they do not drift back in.
- [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) is the command surface that contract
  must produce, end to end for all four stack ids including teardown, and
  [TRANSCRIBE_HAPPY_PATH.md](TRANSCRIBE_HAPPY_PATH.md) is the unabridged expected output per
  use case plus all refusals. Both now describe the implemented v1 surface.
- [TRANSCRIBE_IMPLEMENTATION_PLAN.md](TRANSCRIBE_IMPLEMENTATION_PLAN.md) records the completed phase
  boundaries. [TRANSCRIBE_DESIGN_HANDOFF.md](TRANSCRIBE_DESIGN_HANDOFF.md) is the historical design
  record and still explains the risks and rejected alternatives.
- The `tests/test_spec_docs*.py` suite holds the spec documents' invariants and runs in the
  normal suite.

## Current candidate: issues #22–#24

- FireRed's adapter is asserted against recorded artifact excerpts and, when the local untracked
  artifacts exist, all five raw results: 1,896 sentences, 12,370 words, zero partition failures.
  Its stage loads the pinned system once and intentionally mirrors the pinned `process()` phases
  inside that same co-resident process because upstream exposes neither injected external VAD nor
  a per-region completion ledger. An executable test pins the exact `4e7d9aa` method body and proves
  the mirror matches its successful `sentences`, `words`, and `vad_segments_ms` on the same
  stateful components across multiple ASR and punctuation batches, including upstream blank
  filtering with LID off and the requested LID phase with no blank. A separate fail-closed case
  retains a conforming prefix when blank ASR would leave an unpublishable LID region. The stage
  preserves the global post-filter punctuation stream, emits only completed
  VAD-region prefixes on failure, and never manufactures per-word confidence.
  Result validation accepts the measured integer-millisecond edge where a final word ends at most
  1 ms after its sentence, but rejects a larger escape. The six observed seams are in raw runner
  artifacts `firered_lidoff_batch4_spice30m_participant.json` (sentence/word pairs 26/177 and
  359/2346) and `firered_lidoff_batch4_spice60m_participant_concat.json` (the same two plus
  597/4010 and 930/6179) under `model_tests/benchmark_runs/`; none of the five raw results has a
  larger end escape or a word beginning before its sentence.
- VibeVoice explicitly provisions both the ASR checkpoint and its pinned Qwen tokenizer subset.
  The adapter omits both `"N/A"` and absent speakers, preserves bounded event tags without words,
  and salvages complete JSON objects before a generation truncation. When requested alignment is
  absent or nonconforming for an ordinary speech segment, it preserves that segment without
  `words`, records an `alignment_unavailable` abstention over the segment's native bounds, and
  marks the run-level word-timing outcome `abstained`; valid segment word streams remain, while
  bracketed event tags are excluded from alignment and are not abstentions. Each bounded speech
  segment forms its own native turn, so no same-speaker grouping fills a gap or event. When a
  requested detected overlap intersects a segment and document scope, text/bounds remain but sole
  `speaker` attribution is omitted even across a range boundary; overlap and abstention rows remain
  start-owned. The stage samples live MPS allocation through load and
  inference and always stops its sampler on success or failure.
- `audio export` strictly validates and merges compatible v1 results, re-ids segments and words,
  derives subtitle cues only from word bounds, renders real VTT voice tags, and writes SRT, VTT,
  Markdown, text, and JSONL atomically. A bounded bracketed non-speech event may be omitted from
  subtitles; a bounded ordinary segment may be omitted beside real cues only when an exact
  same-bounds `alignment_unavailable` abstention records the failure. Unbounded ordinary text
  refuses SRT/VTT because schema v1 cannot associate it with a unit ledger row. A nonempty
  transcript with no real word stream fails closed unless every segment is a
  positive-duration bracketed event with bounds and no speaker, in which case the intentionally
  event-free subtitle is empty rather than fabricated from container extents. On-disk output is
  atomic UTF-8 and cannot target an input transcript or canonical source even with `--force`;
  publication carries descriptor-captured input device/inode identity across rendering or model
  work. Forced replacement atomically exchanges directory entries, inspects the displaced inode,
  and rolls back when it is protected or non-regular, so renaming a protected file onto the output
  does not evade the check or create a missing-output window.
  Independently generated VibeVoice documents that requested native diarization are intentionally
  not merge-compatible: their anonymous speaker labels are generation-local, so equal label text
  cannot be treated as a shared identity. Single-input and non-diarized ranged export remain
  supported.
- Every shipped run shape is checked against [TRANSCRIBE_HAPPY_PATH.md](TRANSCRIBE_HAPPY_PATH.md).
  The dead `stack_run_unavailable` refusal is retired. Final candidate verification passed all 903
  tests, the critical Ruff and compile gates, `git diff --check`, live verification of all four
  managed environments, skill validation, fresh sdist/wheel builds, and an isolated wheel-install
  smoke test. The established `plan -> implement -> adversarial review -> fix -> re-review` loop
  continued until both final audit streams returned exactly `CONVERGED` with no reproducible P0-P2
  finding. The implementation is commit
  `b5e6a55cd975779c06c300e590d57e6874cf8cad` in
  [PR #30](https://github.com/fyang0507/audio-processing-cli/pull/30), which closes issues #22,
  #23, and #24 when merged; the PR remains open for human review.

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
  planner then derives add-ons mechanically: `word_timestamps` on a stack without
  native word timing forces the aligner, `diarization` on a stack without
  native speaker structure forces a diarizer. No default
  stack, no preference scalar, no tie-break ordering — an earlier draft had the
  planner select the ASR from capabilities, which made it decide quality it has
  no basis to judge. See [VOCABULARY.md](VOCABULARY.md) and
  [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md).
- Split the two questions into two commands, and generate the plan's sample output rather
  than writing one. `audio transcribe capabilities --stack S --input F` answers what a stack
  can do with a file — the catalog, how the audio partitions, what a run costs, whether a
  failure leaves anything usable — and `audio transcribe plan` resolves one request. Nothing
  sequences them: an agent that knows what it needs goes straight to `plan`, and an omitted
  `--want` there means the floors-only request rather than a request for the menu. An earlier
  draft made both of these `plan` and numbered them, which had one command answering two
  questions depending on an absent argument and implied an order the tool never enforced.
  Because `capabilities` is where a request gets chosen, it carries what that choice needs and
  not just an `availability` enum: what each addition costs to install and to run, which
  capabilities share one add-on stage, where a capability is native but its accuracy is
  unmeasured, and what the languages actually are as against what is advertised. A plan carries
  a `sample_output` built by serializing a placeholder result through the same serializer `run`
  uses. Four stacks against nine capability names is roughly 2,048 combinations, so a
  hand-maintained example set would both rot and be infeasible; one generated path covers all
  of them. A second, parallel sample-rendering path is the specific mistake to avoid here.
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
  auto-fetch, as the Silero backend already does; its managed auto-fetch target is never accepted
  through a symlink even when the target bytes hash correctly.
- For interviews, prefer **FluidAudio or trusted capture channels → reconciled
  external turns → persistent Qwen3-ASR 1.7B 8-bit**, with 0.6B as the explicit
  latency and memory tradeoff. This is a recommendation for the caller, not a
  default the planner applies.
- Use **VibeVoice 7B → Qwen3-ForcedAligner** when editing structure,
  speaker-labelled segments, code switches, and fillers matter. When word timing is requested,
  `transcribe` attempts every alignable speech segment; event tags are excluded, and a missing or
  nonconforming per-segment stream becomes a bounded `alignment_unavailable` abstention rather
  than erasing otherwise valid text. Choosing only selected speech segments for alignment belongs
  to the Observation Store's lazy enrichment. VibeVoice needs a fixed internal seed because its acoustic
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
subtitle convention. Only `word_timestamps` supports real cues. Qwen and VibeVoice
need the aligner for it; on a Qwen stack that is free of any extra runtime, because the aligner
now runs in `mlx` alongside the ASR ([ENVIRONMENTS.md](ENVIRONMENTS.md)). FireRed is the only
stack that produces subtitle-grade timing natively.

Subtitle generation itself lives in `export`, which is deterministic
post-processing with no packages and no `plan`/`run` split. It fails closed when
the transcript has no real word stream rather than inventing cue bounds. The narrow exception is
a bounded-event-only result that already records produced word timing: bracketed non-speech events
carry no alignable words, so an empty subtitle is honest while a cue from their segment extents
would not be. Ordinary wordless sentences still refuse. Cue
segmentation is real work — duration and line limits, CJK versus Latin character
widths, breaking at punctuation, never spanning a speaker change — and its tuning
options are parked as a follow-up issue. Two constraints on it come from the
recorded artifacts rather than from convention: breaking at punctuation means
mapping a mark's position in the sentence text to a word index, which is sound only
case-insensitively; and a segment can carry text with no word stream at all, either because a
VibeVoice event is deliberately not alignable or because ordinary alignment abstained with
`alignment_unavailable`. Only bounded ordinary segments with a same-bounds abstention may be
omitted beside real subtitle cues; an unbounded Qwen failure refuses SRT/VTT because schema v1
cannot associate that segment with a processing-unit ledger row. Timing quality is unvalidated:
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
objects do not travel past it. The transcription know-how belongs in the existing
[`audio-cli`](.agents/skills/audio-cli/SKILL.md) skill rather than a second one, and it is worth
writing only once the command contract is executable; its references should be the decision report
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
- `verbatim` declares that a stack **can produce** verbatim text — that it emits
  disfluencies rather than cleaning them — and leaves accuracy to the quality axis.
  The interface half is now measured on all four stacks (24 to 28 filler hits on one
  probe, none cleaning, none complete) rather than assumed. Nothing selects it: four
  verbatim-requesting system prompts left Qwen's output byte-identical to its
  unprompted baseline, no backend exposes a switch, and nothing in v1 cleans, so it
  asserts an interface and changes no plan composition. The differentiating half is
  dialect form, where `firered` and `qwen-1.7b` retained both probed lexemes and
  `vibevoice` and `qwen-0.6b` did not. Filler recall remains unmeasured for every
  stack.
- Every figure in the interview route's `measured` block comes from a run that passed
  the language hint `"Cantonese"`, so `--language` is exposed on the Qwen stacks: the
  measured configuration has to be one a caller can actually ask for. It is an input,
  not a capability — no role, no package, no output field — and the stacks that take no
  language argument refuse it rather than ignoring it. The no-hint path remains
  unmeasured for accuracy.
- FireRed reproduces text exactly but timestamps only to 1 ms, so
  `determinism_tolerance_ms` is declared per backend and recorded in a plan's roles. That
  drift is a fraction of a video frame and irrelevant to subtitles; it is not
  irrelevant to a `word_id` keyed on a start time, so the Observation Store's identity
  scheme cannot key on one.
- Qwen's determinism is measured only *within* one process: back-to-back calls on the
  batched path were byte-identical, and cross-process repetition of that call shape has
  not been run. Worth closing before any repeat-hash assertion ships.
- Qwen has two decode entry points that do not agree. The public `generate()` and the
  private `_generate_chunks_batched` paths matched on every word and differed by two
  Chinese commas on the same input. Declare `api_path` in every plan and attribute each
  recorded figure to the path that produced it, or a punctuation-sensitive consumer will
  read evidence from the wrong one.

## Working commands

```bash
git switch main
uv sync --extra dev
uv run --extra dev pytest
```

Before new model experiments, reuse the tracked runners and compact result
format. Keep raw fixtures and local model/run artifacts outside Git.
