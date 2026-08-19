# Working in this repository

A local-first `audio` CLI for agent workflows: measure audio, enhance it deterministically, and
(in progress) transcribe it. Every command prints JSON on stdout for a caller to dispatch on;
prose interpretation belongs in your reply, not in the payload.

## The rule that matters most

**A claim about what a model or backend produces must cite a runner source or a recorded
artifact, never a summary document.** Three documents here agreed that FireRed emits per-word
confidence. It does not, and the claim survived four review passes because every summary
repeated it and only the artifacts refuted it. Summaries are convenient and artifacts are not;
read the artifact anyway. Doing so is also *generative* — checking one claim against recorded
output is how most of this repository's real findings arrived.

Two invariants that follow, and hold everywhere:

- **The original media is canonical.** Nothing modifies a source file. Renders and transcripts
  refer back to the source timeline.
- **Absence is meaningful.** A field that a backend did not supply stays absent rather than
  becoming a null, a zero, or a default that reads like a measurement.

## Where truth lives

| Document | Authoritative for |
| --- | --- |
| [VOCABULARY.md](VOCABULARY.md) | Every name in the transcription schema, the floors, and the retired words. Check it before coining a term. |
| [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) | The `transcribe` command surface, exit codes, and payloads. |
| [TRANSCRIBE_HAPPY_PATH.md](TRANSCRIBE_HAPPY_PATH.md) | Expected stdout, key for key. The diff target. |
| [ENVIRONMENTS.md](ENVIRONMENTS.md) | How model packages and their runtimes are provisioned, and why the layout is what it is. |
| [HANDOFF.md](HANDOFF.md) | Repository-wide state and open evidence. Start here when unsure. |
| [model_tests/](model_tests/) | Measured results and their scope limits. |

Specification documents are enforced, not decorative: `tests/test_spec_docs.py` and
`tests/test_environments.py` fail when a payload, a name, or an environment drifts from what
these documents publish.

## Evidence conventions

Runners live in `model_tests/benchmark/`, compact results are **tracked** under
`model_tests/benchmark/results/` as `YYYY-MM-DD-<topic>.json`, and raw artifacts stay
**untracked** in `model_tests/benchmark_runs/`. A measurement is reported with its fixture and
configuration, or not reported. Distinguish three things and never let them blur: a **declared**
interface, a **measured** result, and an **unresolved** question.

## Tooling

`uv` manages every environment — `uv venv` and `uv pip install`, never `python3 -m venv` with
`pip`. Tests are `uv run --extra dev pytest`. `ffmpeg` and `ffprobe` are machine runtime
dependencies. Python 3.11+.

Model weights and their runtimes are provisioned **only** by `audio packages pull`. Never
hand-download weights, hand-create a virtual environment, or edit a lock file to make an install
succeed — the pins are what make the recorded measurements mean anything.

## Two habits learned the hard way

- **Do not add structure nobody dispatches on.** A capability report here shed four nested
  objects that each seemed justified when added. The test is whether a caller branches on it.
- **Do not describe machinery that does not exist.** One rule instructed an adapter to handle
  output a stage never emits; it was inert and its test passed vacuously. When you write an
  invariant, write the assertion that would fail if it were violated, then check that it can
  fail.

## Skills

[`audio-cli`](.agents/skills/audio-cli/SKILL.md) is the one skill, and it ships in the source
distribution so it travels with the CLI. It is written for an agent *using* the tool on someone's
audio, not for someone developing it: a short router in `SKILL.md` sends the reader to one
task-shaped reference — diagnosing and fixing audio, a targeted fix, model provisioning, or
installation. Two rules keep it that way. Anything `--help` already states stays out of it, and
backend internals stay out too — how the models are partitioned into runtimes belongs in
[ENVIRONMENTS.md](ENVIRONMENTS.md), where a developer will look for it.
