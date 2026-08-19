# `transcribe` — design handoff

**You are picking this up to produce a technical design and an implementation plan.** The
agent-facing surface is settled and signed off; none of it is built. Your job is the layer
underneath: modules, interfaces, adapter boundaries, provisioning mechanics, test strategy,
and a sequenced plan. This document tells you what is decided, what is deliberately open,
what will bite you, and what the evidence actually supports.

Nothing here restates the three spec documents. Read them; they are authoritative.

## What to read, and what each document decides

| Document | Authoritative for | How to treat it |
| --- | --- | --- |
| [VOCABULARY.md](VOCABULARY.md) | Every name, and the rules that generated them. Terms, floors, the capability namespace, the derivation table, environments, retired words. | The naming contract. Do not coin a term before checking whether it was already retired and why. |
| [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) | Command surface, exit codes, the per-code error-field table, plan and result structure, and the reasoning behind each. | The specification. Where it states a rule, implement the rule; where it explains a rejected alternative, that alternative stays rejected. |
| [TRANSCRIBE_HAPPY_PATH.md](TRANSCRIBE_HAPPY_PATH.md) | Exact command sequences and complete expected stdout for three use cases, plus all twelve refusals with their corrections. | The diff target. Your implementation's output should match these shapes key-for-key. |
| [model_tests/DECISION_REPORT.md](model_tests/DECISION_REPORT.md) | Which stack to prefer for what, and why. | Product decision input, not a source for backend behaviour claims — see the process rule below. |
| [model_tests/EXPERIMENT_RESULTS.md](model_tests/EXPERIMENT_RESULTS.md) | The measured numbers and their scope limits. | Cite it for figures; check the artifact when the figure matters. |
| [SPEC_REVIEW_FINDINGS.md](SPEC_REVIEW_FINDINGS.md) | What the review of the above found, decided, and rejected. | Read the process finding at minimum. It explains why a defect survived four review passes. |
| [HANDOFF.md](HANDOFF.md) | Repository-wide state, the shipped `inspect`/`enhance` commands, open evidence. | Onboarding. Its acceptance list is still live. |

`tests/test_spec_docs.py` enforces the spec documents' internal consistency and runs in the
normal suite. It is also a worked list of the invariants your implementation has to satisfy;
each assertion corresponds to a defect that review actually missed by eye.

## The one process rule that matters most

**A claim about what a backend produces must cite a runner source or a recorded artifact,
never a summary document.** Three research documents stated that FireRed emits per-word
confidence. It does not — every word object is exactly `{start_ms, end_ms, text}` across
12,370 recorded tokens — and the claim survived four review passes because every summary
repeated it and only the artifacts refuted it.

This will apply to you constantly, because the summaries are convenient and the artifacts are
not. Raw artifacts live in `model_tests/benchmark_runs/` (untracked, present locally); compact
results in `model_tests/benchmark/results/` (tracked); runners in `model_tests/benchmark/`.

The corollary bit twice more: verifying against artifacts is *generative*, not just corrective.
Reading them to check one claim turned up the punctuator's recasing, a fabricable speaker id, a
legitimately absent word stream, and two decode paths that disagree — none of which any amount
of prose review would have found.

## Decided — do not redesign

These were argued and settled. Each has a stated reason in the spec; the summary here is so
you recognise a settled question when you meet one.

- **Two commands, no ordering.** `capabilities` answers what a stack can do with a file;
  `plan` resolves one request; `run` executes. An omitted `--want` means the floors-only
  request, not the menu. `--stack` and `--input` are required on all three.
- **Stack first, add-ons derived.** The caller names the stack; the planner derives add-ons
  mechanically from the requirements. No default stack, no preference scalar, no tie-break
  ordering — an earlier draft had the planner choosing the ASR from a capability list, which
  made it judge quality it has no basis to judge.
- **Six floors, never optional, never printed.** They are invariants, not settings; they live
  in VOCABULARY and in tests.
- **Absence is meaningful.** A capability that was not requested has no key in the output. A
  capability that was requested may still be absent on a given segment where the backend
  supplied nothing. Those are different, and the second is one-directional: a key may appear
  only if its capability was requested.
- **Provisioning is explicit and fails closed.** `audio packages pull` is the only thing that
  downloads weights, builds the Swift product, or applies the VibeVoice patch. Only the small
  hash-pinned artifacts auto-fetch, as `src/audio_cli/vad.py` already does — copy that
  pattern, including the digest check and the atomic rename.
- **Payloads carry what a caller can act on.** Three enums and two numbers in a capabilities
  report; everything else is one sentence per capability. Plans keep their structure because
  they are dispatched on; catalogs do not because they are read.
- **`fix` is a runnable command** wherever a configuration exists that would work.

## Open — this is your design work

1. **Module and adapter boundaries.** The one hard constraint: model-specific objects do not
   travel past the adapter. Everything else — package layout, how a backend declares
   capabilities, whether the planner is a pure function over a stack table — is yours.
   `src/audio_cli/` currently has no abstraction for this; `vad.py` is the only backend-shaped
   file and it predates the vocabulary.
2. **Stage orchestration.** Stages run strictly sequentially with one model resident at a
   time. The transport is settled — one fresh subprocess per stage against the environment's
   own interpreter, JSON in and JSON out, residency enforced by process exit
   ([ENVIRONMENTS.md](ENVIRONMENTS.md)) — which also makes the adapter-normalization floor
   structural, since no model object can cross a process boundary. What remains is the
   orchestration above it: stage order, the observed-cost accounting, and the ledger.
3. ~~**The environment layout, concretely.**~~ **Done** in issue
   [#11](https://github.com/fyang0507/audio-processing-cli/issues/11): four provisioned
   environments derived by resolver rather than asserted, hash-pinned locks that ship in the
   wheel, and the registry schema with its crash-safety rule. `audio packages`
   (`list`/`path`/`pull`/`verify`/`remove`/`purge`) and `audio doctor` are implemented too, but
   only against fakes: every test in `tests/test_packages.py` runs on a `FakeToolchain` and a
   `FakeFetcher` in an isolated root, so no real Hub download, `uv venv` creation, digest check,
   or Swift build has been exercised. `pull --want` is accepted and inert until the planner
   lands, so `--stack` over-provisions ([#12](https://github.com/fyang0507/audio-processing-cli/issues/12)).
4. **Partial results and resume.** Exit 4 writes a result with a coverage ledger. How work
   units are tracked, and where the watermark comes from when completion is non-contiguous,
   is design. The recorded runner processes turns in duration-bucketed order, so completion is
   genuinely out of order.
5. **The cue segmenter** for `export`. Parked as issue #10, and it has two constraints from
   artifacts: mark-to-word-index mapping is sound only case-insensitively, and a segment may
   carry text with no word stream.
6. **Word and segment identity.** Ids are document-scoped and explicitly not stable across
   runs. If the Observation Store later needs run-stable identity it cannot key on a
   timestamp, because FireRed reproduces timestamps only to 1 ms.
7. **Where the stack table lives.** Half of it now exists:
   `src/audio_cli/environments/manifest.json` holds the package and environment columns —
   sources, revisions, real byte counts, declared licenses, roles, stacks — behind a reader
   whose `packages_for(stack, roles)` is the seam. The capability columns are still prose in
   two documents: availability, satisfaction, evidence, and the notes. Those are yours, and the
   rule is one table with two owners, not two tables.

## Risks, worst first

- ~~**The `torch` environment may not resolve.**~~ **Resolved, and it did not.** VibeVoice,
  `qwen-asr`, and FireRed cannot share an environment: ten of fifteen candidate groupings
  conflict and all ten conflict on `transformers`
  (`model_tests/benchmark/results/2026-08-17-environment-partition.json`). Four provisioned
  environments, and the partition is unique. The probe also turned up more than it was asked:
  the forced aligner runs in `mlx` token-identically, so a Qwen request needs no PyTorch at
  all, while FireRed cannot leave PyTorch because `mlx-audio` has no punctuator and
  punctuation is a floor. See [ENVIRONMENTS.md](ENVIRONMENTS.md); issue
  [#11](https://github.com/fyang0507/audio-processing-cli/issues/11).
- **`vibevoice` cannot resume.** It is handed whole media in a single `generate` call, so a
  failure at minute forty of a forty-one-minute run yields nothing. It is also the stack most
  likely to fail, having measured 20.28 GiB live MPS on thirty minutes and OOMed under a
  strict 16 GiB cap. Do not design a recovery story that quietly assumes partitioning.
- **A long Qwen run truncates silently today.** The recorded runner carries a global
  generation budget and stops between turns when it runs out. Every recorded run finished, but
  at the recorded token rate the budget exhausts near **1.6 hours** of comparable audio. Exit 4
  exists for this; make sure the budget is a declared configuration and not a constant nobody
  notices.
- **Two Qwen decode paths disagree.** Public `generate()` and private
  `_generate_chunks_batched` agree on every word and differ by two Chinese commas. The private
  path is the one that supports turn batching and the one the timing figures came from; it also
  returns the model's `language English<asr_text>` scaffold inside its text, which the adapter
  must strip. `api_path` is in the plan for this reason.
- **The `mlx-audio` private API is pinned by a source hash.** `0.4.5`, adapter sha256
  `c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250`. `verify` checks it.
  An upgrade is a breaking change, not a bump.
- **16 GiB is not validated.** RSS, MLX, PyTorch, and Core ML counters have different scopes
  and must not be summed. The product decision is to ship the product-demo route anyway and
  warn from the plan; do not turn that into a claim.

## Unmeasured — design around the gap, do not close it by assertion

Each of these is a number that does not exist. Where a design decision depends on one, say so
rather than assuming a value.

- Boundary MAE/P95 for FireRed's native word times **and** for the forced aligner. Neither is
  scored, so "switch stacks for better timing" is not a supported claim in either direction.
- Filler recall on any stack. Retention was counted on one scripted probe (24–28 hits); that
  is not recall.
- Cross-process determinism for the Qwen batched path. Within one process, back-to-back calls
  were byte-identical; across processes, untested.
- ~~Package byte sizes~~ — **closed.** Every package now carries a real count read from the
  Hub at its pinned revision; `fluidaudio` alone stays unsized, because it is a build product.
  Several of the figures these documents quoted were wrong, not merely illustrative.
- ~~Licenses~~ — **partly closed.** Every package now records the license its card declares at
  the pinned revision. `license_declared` and `license_reviewed` are separate fields, because a
  card is not a review, and only FluidAudio and `speaker-diarization-coreml` were actually
  read.
- Any accuracy figure outside Cantonese, on any stack.

## Suggested sequence, and why

Risk sits earliest in the schema, so the order is not arbitrary.

1. ~~**Resolve the `torch` environment question experimentally.**~~ **Done.** It does not
   resolve; the layout is in [ENVIRONMENTS.md](ENVIRONMENTS.md). Cost of skipping it would
   have been real: the aligner turned out to belong in a different environment than the spec
   assumed, which changes what half the plans in these documents print.
2. **The normalized result schema and the adapter contract.** Everything downstream is shaped
   by it, and it is where fabrication becomes possible. Build the anti-fabrication assertions
   at the same time, not after.
3. **The stack table and the planner.** A pure function from (stack, input metadata,
   requirements) to a plan, with `capabilities` and `plan` as two serialisations of it. Test it
   parametrized over VOCABULARY's derivation table so the table stays the single source of
   truth.
4. **`audio packages` and the registry**, including `verify` and `--repair`. Nothing above can
   be exercised end to end without it.
5. **One stack behind an adapter — `qwen-1.7b`.** It has the most recorded evidence, the
   cheapest runs, and no native anything, so it forces the derived-capability paths first.
6. **The remaining three stacks.** `firered` next, because it exercises `vad`, `punctuator`,
   `lid`, and the punctuation invariant. `vibevoice` last, because it is the slowest to
   iterate against and the least recoverable.
7. **`export`,** then the cue segmenter from issue #10.

## Acceptance

The spec-document tests already pass. Add, against real command output:

- The sample-output key-set test: a plan's `sample_output` key set equals a real run's, and no
  key exists for a capability that was not requested. Parametrize over VOCABULARY's derivation
  table, including both refusal codes.
- The punctuation invariant per stack: a sentence's text, stripped of punctuation and
  whitespace, equals the concatenation of its word texts compared case-insensitively; skip
  segments with no word stream rather than asserting they have one.
- Adapter normalization per stack, one case each: Qwen's `language <label><asr_text>` scaffold
  is stripped, FireRed's `lang: null, lang_confidence: 0` are dropped when LID did not run, and
  VibeVoice's `Speaker: "N/A"` becomes an absent key rather than a speaker id.
- A plan carries no `outcomes`; a run's provenance does.
- Exit codes end to end, including 3 before `pull`, 4 on a truncated run, and that a partial
  result is a conforming document rather than a debug dump.
- `uv run --extra dev pytest` stays green, and the deterministic render-from-original
  `enhance` flow is not regressed.

## Two things to be careful about, learned here

**Do not add structure nobody dispatches on.** The capabilities report went through an
`evidence` object, a seven-key `cost` object, a `timing_precision` object holding one null and
one string, and four sibling free-text fields, before all of it collapsed into one sentence per
capability. Every one of those seemed justified when added. The test is whether a caller
branches on it.

**Do not describe machinery that does not exist.** The punctuation floor spent four review
passes instructing the adapter to reattach marks and drop their bounds — for a stage that
emits no per-mark bounds at all. The rule was inert, and the test mandated for it passed
vacuously. When you write an invariant, write the assertion that would fail if it were
violated, and check that it can fail.
