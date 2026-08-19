# Agent handoff

Updated 2026-08-16. This repository currently has a production-oriented audio
enhancement CLI and a completed ASR research/distillation phase. The next phase
is to turn the ASR decisions into a capability-driven transcription CLI and its
associated agent skill.

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

## Read this first

1. [model_tests/DECISION_REPORT.md](model_tests/DECISION_REPORT.md) — short
   capability field guide and recommended stacks. Treat this as the product
   decision input.
2. [model_tests/EXPERIMENT_RESULTS.md](model_tests/EXPERIMENT_RESULTS.md) — short
   quantitative evidence and its limits. Use it to justify defaults and tests.
3. [model_tests/FINDINGS.md](model_tests/FINDINGS.md) — full trace only when a
   claim or experiment needs auditing.
4. [model_tests/benchmark/README.md](model_tests/benchmark/README.md) — benchmark
   commands, records, and artifact layout.

Keep three kinds of statement separate in code and documentation:

- **Capability:** an output/interface the tested backend exposes.
- **Measured:** a result tied to the recorded fixture, version, hardware, and
  configuration.
- **Unresolved:** a decision that still needs targeted evidence.

## Decisions to carry into the CLI

- Plan from requested capabilities and load the smallest sufficient stack; do
  not hard-code a universal ASR → VAD → aligner → diarizer chain.
- Use **FluidAudio or trusted capture channels → reconciled external turns →
  persistent Qwen3-ASR 1.7B 8-bit** as the default interview route. Keep 0.6B
  as the explicit latency/memory tradeoff.
- Use **VibeVoice 7B → selective Qwen3-ForcedAligner** when editing structure,
  speaker-labelled segments, code switches, and fillers matter and the machine
  has sufficient memory.
- Expose **FireRedASR2S as its full pipeline**. Prefer it for dialect form,
  region LID, native word time/confidence, or tighter memory/latency; it has no
  native speaker output.
- Preserve external timestamps and speaker provenance. Qwen container bounds
  are processing containers, not speech or word timestamps. Language output is
  one capability dimension, not a routing oracle or token-level code-switch map.
- Abstain on ambiguous overlap, rapid backchannels, role identity, and semantic
  speaker claims instead of inventing certainty.
- Keep large runtimes and model families optional and lazy. Do not add them to
  the core enhancement dependency graph merely to expose transcription.

## Recommended first implementation slice

1. Add a normalized transcription schema with explicit provenance for text,
   language, external/native bounds, word timing, confidence, anonymous speaker,
   capture role, and abstentions. Missing capabilities must remain missing—not
   synthesized from unrelated fields.
2. Define backend capability declarations and a planner that resolves a request
   such as transcript, speakers, word timing, LID, or verbatim structure into the
   smallest route. Make the resolved plan visible in JSON output.
3. Add an `audio transcribe` surface around the schema and planner before
   coupling it to heavyweight runtimes. Preserve structured errors and the
   existing CLI's machine-readable stdout conventions.
4. Implement one end-to-end interview route first, keeping diarization/capture
   reconciliation separate from the Qwen transcript worker. Normalize backend
   results at adapter boundaries rather than leaking model-specific objects.
5. Create an associated transcription agent skill only after the command
   contract is executable. Its reference docs should be the decision report and
   experiment digest, not the full findings file.

## Acceptance and open evidence

- Add contract tests for capability negotiation, optional component loading,
  provenance, absolute-bound reattachment, overlap abstention, and the absence
  of fabricated timestamps/speaker identity.
- Keep the existing suite green with `uv run --extra dev pytest`.
- Do not claim physical 16 GiB support yet. Current RSS, MLX, PyTorch, and Core
  ML counters have different scopes.
- FireRed and ForcedAligner word-boundary MAE/P95 remain unmeasured.
- Mandarin/English, broad dialect, switch-span, filler, and repair accuracy need
  frozen held-out labels; the Qwen auto-language probe verifies behavior, not
  broad accuracy.
- The complete dedicated-channel merge and non-oracle participant/interviewer
  role mapping remain unmeasured.
- Review checkpoint/runtime licenses before choosing production defaults or
  redistributing model integrations.

## Working commands

```bash
git switch codex/asr-benchmark-field-guide
uv sync --extra dev
uv run --extra dev pytest
```

Before new model experiments, reuse the tracked runners and compact result
format. Keep raw fixtures and local model/run artifacts outside Git.
