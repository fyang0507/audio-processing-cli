# Working in this repository

A local-first `audio` CLI for agent workflows: measure audio, enhance it deterministically,
provision the model packages and runtimes transcription will need, and (in progress) transcribe it.
`inspect`, `enhance`, `doctor`, and `packages` ship; `transcribe` does not exist yet. Every command
prints JSON on stdout for a caller to dispatch on; prose interpretation belongs in your reply, not
in the payload.

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
  becoming a null, a zero, or a default that reads like a measurement. The sharpest instance:
  `pull` recorded `digest_verified: true` for every Hub package and hashed none of them — the
  manifest pins a *revision* and carries no `sha256` to hash a snapshot against — so `verify`
  printed `digest: "ok"` for a check no code performs. The repair is to claim what is true (the
  pinned `revision`) rather than to confess what is not with a `false`.

## Where truth lives

| Document | Authoritative for |
| --- | --- |
| [VOCABULARY.md](VOCABULARY.md) | Every name in the transcription schema, the provisioning states and `verify` verdicts, the floors, and the retired words. Check it before coining a term. |
| [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) | The `transcribe` command surface, exit codes, and payloads. |
| [TRANSCRIBE_HAPPY_PATH.md](TRANSCRIBE_HAPPY_PATH.md) | Expected stdout, key for key. The diff target. |
| [ENVIRONMENTS.md](ENVIRONMENTS.md) | How model packages and their runtimes are provisioned, and why the layout is what it is. |
| [HANDOFF.md](HANDOFF.md) | Repository-wide state and open evidence. Start here when unsure. |
| [model_tests/](model_tests/) | Measured results and their scope limits. |

Specification documents are enforced, not decorative: `tests/test_spec_docs.py` and
`tests/test_environments.py` fail when a payload, a name, or an environment drifts from what
these documents publish. Those two compare documents against documents, which cannot catch a
document the *code* disagrees with, so `tests/test_shipped_commands_match_the_document.py` runs
the commands that already exist — `doctor`, `packages list`, `packages verify` — and diffs their
real stdout against TRANSCRIBE_HAPPY_PATH.md, key set and nesting rather than values. All three
had drifted when it was written. Where the documents disagree with each other it abstains and
names the dispute instead of ratifying a side; the open one is the shape of `failed[]`, recorded
in HANDOFF.md. Add a shipped command to it when you ship one.

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

## Habits learned the hard way

- **Do not add structure nobody dispatches on.** A capability report here shed four nested
  objects that each seemed justified when added. The test is whether a caller branches on it.
- **Do not describe machinery that does not exist.** One rule instructed an adapter to handle
  output a stage never emits; it was inert and its test passed vacuously. When you write an
  invariant, write the assertion that would fail if it were violated, then check that it can
  fail.
- **A flag that parses is not a flag that works.** `--repair` was declared, documented in
  TRANSCRIBE_CONTRACT.md, and named in four `fix` strings `verify` emits, and no code read it.
  `--want` was accepted and ignored; `--stack` beside named packages dropped the stack. The cheap
  sweep that finds the rest: neutralize each `repair`/`force`/`dry_run`/`stack`/`allow_*` branch
  in turn and run the suite. Six of twenty were invisible. A flag the code cannot honour yet is
  refused at exit 2, never accepted quietly.
- **A double must be able to represent the state a repair produces.** `FakeToolchain` could not
  stop being drifted, so a repair test's final assertion ran against a second, undrifted toolchain
  that reports `ok` either way and the `if drift and repair` branch never executed. When a double
  cannot reach the post-repair state, the test passes without the repair running — and two payload
  shapes here turned out to be unreachable rather than untested for the same reason.
- **A parameter that decides nothing is worse than a missing one.** `vad_min_silence_ms` declared
  300 ms while a hard-coded merge downstream required 540, so the profile's number was inert
  across a 240 ms band. Two thresholds answering one question can only disagree; keep one.

## Skills

[`audio-cli`](.agents/skills/audio-cli/SKILL.md) is the one skill, and it ships in the source
distribution so it travels with the CLI. It is written for an agent *using* the tool on someone's
audio, not for someone developing it: a short router in `SKILL.md` sends the reader to one
task-shaped reference — diagnosing and fixing audio, a targeted fix, model provisioning, or
installation. Two rules keep it that way. Anything `--help` already states stays out of it, and
backend internals stay out too — how the models are partitioned into runtimes belongs in
[ENVIRONMENTS.md](ENVIRONMENTS.md), where a developer will look for it.
