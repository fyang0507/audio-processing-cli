# Working in this repository

A local-first `audio` CLI for agent workflows: measure audio, enhance it deterministically, provision model packages and runtimes, transcribe through four explicit stacks, and export saved results deterministically. `inspect`, `enhance`, `report summary`, `report compare`, `doctor`, `packages`, `transcribe stacks`, `transcribe capabilities`, `transcribe plan`, `transcribe run`, and `transcribe export` ship (`export` remains a compatibility alias). `transcribe run` produces reusable normalized JSON; `transcribe export` formats saved results offline. JSON is the machine-readable result format; prose interpretation belongs in your reply, not in the payload.

## Final feature acceptance

Use [agent acceptance](docs/agent-acceptance.md) as the final development gate for user-facing features. It requires fresh operator agents with only the shipped skill and public CLI guidance, original-media evidence, explicit capability/delivery verdicts and no fallback after an unexpected failure. Green unit/specification checks alone are not user acceptance. Follow its scope rule for documentation-only changes; keep this developer procedure out of the operator skill.

## The rule that matters most

**A claim about what a model or backend produces must cite a runner source or a recorded artifact, never a summary document.** Three documents here agreed that FireRed emits per-word confidence. The [recorded FireRed fixture](tests/fixtures/firered_lidon_first_vad_region.json) contains word timing and text, with confidence only at sentence level. The incorrect claim survived four review passes because every summary repeated it and only the artifacts refuted it. Summaries are convenient and artifacts are not; read the artifact anyway. Doing so is also *generative* — checking one claim against recorded output is how most of this repository's real findings arrived.

Two invariants that follow, and hold everywhere:

- **The original media is canonical.** Nothing modifies a source file. Renders and transcripts refer back to the source timeline.
- **Absence is meaningful.** A field that a backend did not supply stays absent rather than becoming a null, a zero, or a default that reads like a measurement. The sharpest instance: `pull` recorded `digest_verified: true` for every Hub package and hashed none of them — the manifest pins a *revision* and carries no `sha256` to hash a snapshot against — so `verify` printed `digest: "ok"` for a check no code performs. The repair is to claim what is true (the pinned `revision`) rather than to confess what is not with a `false`.

## Where truth lives

| Document | Authoritative for |
| --- | --- |
| [VOCABULARY.md](docs/VOCABULARY.md) | Every name in the transcription schema, the provisioning states and `verify` verdicts, the floors, and the retired words. Check it before coining a term. |
| [TRANSCRIBE_CONTRACT.md](docs/TRANSCRIBE_CONTRACT.md) | The `transcribe` command surface, exit codes, and payloads. |
| [TRANSCRIBE_HAPPY_PATH.md](docs/TRANSCRIBE_HAPPY_PATH.md) | Expected stdout, key for key. The diff target. |
| [ENVIRONMENTS.md](docs/ENVIRONMENTS.md) | How model packages and their runtimes are provisioned, and why the layout is what it is. |
| [Documentation index](docs/README.md) | Current contracts, enhancement guidance, research reports, and test layout. Start here when unsure; verify implementation state against source, tests, and the current PR. |
| [Enhancement](docs/enhancement.md) and [CLI feedback](docs/cli-feedback.md) | Processing methods, component outcomes, report navigation, and measured-evidence limits. |
| [model_tests/](model_tests/) | Measured results and their scope limits. |

## Source architecture

Treat a directory as an implementation-ownership boundary, not merely a filename namespace. An import or compatibility re-export does not transfer ownership. When a change appears to fit two owners, separate the lower-level mechanism from the feature policy instead of duplicating either.

### Direct packages under `audio_cli`

| Package | Owns | Must not own |
| --- | --- | --- |
| `command` | Shared presentation primitives for the newer command surfaces: shell-safe runnable-fix rendering, stage/elapsed progress, the bare-stderr refusal type, and generic output-refusal shapes. | Detecting domain failures, executing fixes, business policy, or the older commands' historical error envelope. |
| `dsp` | Pure, in-memory signal measurement and transformation over arrays, profiles, adjustments, and immutable speech-region values. | Files, subprocesses, model runtimes, publication, or workflow order. |
| `environments` | Bundled desired state: environment, package, and backend declarations plus validation of `manifest.json`. Its `locks/`, `requirements/`, and `patches/` directories are immutable provisioning inputs. | Network access, installed-state inspection, downloads, builds, repair, or registry mutation. |
| `media` | Trusted host-I/O and representation mechanics: FFmpeg/FFprobe command specifications, canonical PCM inspection/clipping, FFmpeg RNNoise execution, codec/container metadata, file identity and hashing, temporary paths, and atomic publication primitives. | Workflow order, profile/capability decisions, result interpretation, or package lifecycle. Callers choose the operation; `media` performs it safely. |
| `packages` | Mutable, explicitly requested provisioning lifecycle: selection, registry state, fetching, environment creation, builds, verification, repair, removal, and readiness diagnostics. | Capability resolution, transcription workflow, immutable declarations owned by `environments`, or the legacy on-demand Silero bootstrap owned by `vad.py`. |
| `pipeline` | The `inspect` and `enhance` workflow: stage ordering, profile application, DSP composition, reporting, enhanced-output validation/publication, and offline saved-report summary, navigation, and comparison under `pipeline/reports`. | Generic media mechanisms, package lifecycle, transcription, or export. The generic name is historical; this package means the enhancement pipeline. |
| `transcribe` | Transcription capability catalogs and plans, isolated model-stage execution, provider-result normalization, orchestration, and normalized result production. | Package installation, enhancement policy, or offline export rendering. |
| `export` | Offline consumption of one or more saved normalized results: strict loading, compatibility checks, merging, cue construction, deterministic rendering, and output publication. | Model execution, capability planning, or transcription-process control. It consumes the result contract without running transcription. |

These boundaries follow lifecycle cuts: `environments` declares desired state while `packages` reconciles mutable installed state; `media` performs trusted mechanisms while `pipeline` orders the enhancement feature; `transcribe` produces durable results while `export` only consumes them; and `command` formats shared presentation primitives while each feature detects its own conditions and owns its domain-specific codes and repair policy.

`media` deliberately includes the generic filesystem primitives used by provisioning as well as audio/video tooling: security-critical external-byte handling has one owner. It also writes and detects the enhancement marker because that marker is container metadata; `pipeline` decides when an enhanced input or output is permitted. The protected export writer and verified-media promotion remain feature-specific publication transactions that compose these mechanisms.

The allowed dependency direction is:

```text
__main__ -> cli
cli -> cli_parser | adjustments | profiles | vad
cli -> command | environments | export | media | packages | pipeline | transcribe
cli_parser -> profiles

export -> command | transcribe.result | media
transcribe -> command | packages | environments | media | vad | paths
pipeline -> dsp | media | profiles | adjustments | vad | vad_contract | __version__
packages -> environments | media | paths | __version__
dsp -> profiles | adjustments | vad_contract
vad -> vad_contract | media | paths
command, environments, media -> no feature package
```

Everything not listed is a boundary violation. In particular, `pipeline` and `transcribe` are peers; `packages` cannot know about transcription plans; `export` cannot call transcription execution; and no lower package imports `cli`. `tests/architecture/test_top_level_module_boundaries.py` enforces the package graph and the deliberately narrow `export -> transcribe.result` edge. One frozen source-compatibility bridge sits outside the ownership DAG: `transcribe.refusals -> export.refusals`, limited to four direct aliases whose implementation and ownership remain in `export`.

Root modules have one of two roles. `__main__.py`, `cli.py`, and `cli_parser.py` are composition roots. Small modules such as `adjustments.py`, `profiles.py`, `paths.py`, `vad_contract.py`, and `vad.py` hold shared value types or one cross-feature service; feature reports may also read the immutable root package version. The root is not a place for another multi-module subsystem. Likewise, `transcribe/catalog.py`, `plan.py`, `sample.py`, and `stacks.py` are small shared domain contracts, not an alternative home for one of the subpackages below. `transcribe/native.py` is a frozen compatibility module: it preserves only the old `run_native` import and its FireRed / VibeVoice restriction, then delegates to `transcribe.orchestrator`; it owns no provider workflow.

`vad_contract.py` owns only the lightweight speech-region value and detector protocol. `vad.py` owns the Silero ONNX implementation and the one legacy provisioning exception: first use may bootstrap that small, content-hash-pinned model into the managed cache. `packages` owns the explicit `audio packages pull silero-vad` lifecycle for the same declared artifact. No other model or runtime may be fetched implicitly.

The non-Python directories under `environments` are part of that package's declared state:

| Directory | Scope |
| --- | --- |
| `locks/` | Fully resolved, checked-in Python environment locks consumed by provisioning and verification. |
| `requirements/` | Human-authored resolver inputs from which the corresponding locks are produced. |
| `patches/` | Pinned adaptations applied to declared upstream source checkouts; `README.md` records why each exists. |

### Packages directly under `audio_cli/transcribe`

| Package | Owns | Must not own |
| --- | --- | --- |
| `adapters` | Pure, deterministic interpretation of a backend's raw payload into host-side values, including alignment and diarization reconciliation. | Filesystem/process I/O, model imports, planning, publication, or public-result serialization. |
| `execution` | Provider-neutral host run services after planning: mutable integrity/runtime probes, range validation, materialization receipt access, shared VAD invocation, metrics, completion, resume, publication, and published-result command receipts. | Media I/O mechanics, stack selection, provider workflow order, model-stage process lifecycle, or backend-payload interpretation. |
| `orchestrator` | The sole stack dispatcher and the Qwen, FireRed, and VibeVoice workflows. It decides which operations run and in what order by composing the other transcription packages. | Subprocess mechanics, model-framework imports, package provisioning, or schema ownership. |
| `planner` | Pure request validation and immutable plan construction from input metadata and declared stack/environment facts. | Installed-state inspection, model execution, normalization, or publication. |
| `refusals` | Fixed-shape, transcription-specific refusal payload builders that compose the shared runnable-command renderer. Callers detect the condition; this package only represents it. | Detecting failures, running remediation, export-specific failures, or non-transcription command errors. |
| `result` | Backend-independent durable result types, absence semantics, schema validation, and serialization. | Provider dialects, path selection, filesystem writes, lifecycle state, or orchestration. |
| `stages` | Thin child-process entry points inside managed model environments. They import provider frameworks, run inference, and write the raw stage response. | Imports from core `audio_cli`, cross-stage sequencing, normalization into the public schema, or final publication. |
| `transport` | The resource-accounted host/child process boundary: launching the media-owned canonical decode specification, provider wire-request encoding, result JSON framing, duplicate-key rejection, and exit translation. | Media command policy, interpreting provider responses, capability decisions, provider workflow order, model inference, or public results. |

The distinctions that most often prevent misplaced code are: planner reads declared facts while execution checks mutable installed state; orchestrator decides **what and when** while execution provides reusable host operations; transport encodes and moves provider wire requests while stages run inference; adapters interpret returned provider semantics while result defines the public schema. Stage scripts are reached by process invocation, never imported by host-side code. `stages/_firered_protocol.py` is the sole non-`__init__.py` underscore module in this tree: it is intentionally private and stdlib-only because the FireRed stage must load the same wire-protocol helper both as an installed package module and as a directly executed isolated script. The transcription boundary tests exact-inventory every module under these owners, so adding or moving a module requires an explicit owner decision rather than merely choosing a plausible name. They also require host-side transcription code to depend only on the standard library and other `audio_cli` owners: direct third-party model/runtime imports are exact-inventoried per isolated stage. Canonical WAV access is confined to `media.pcm`, including from standard-library callers. Host runtime modules use static imports. The sole exception is `packages/runtime_probe.py`, a stdlib-only child script launched under the managed MLX interpreter; its one dynamic import is the private-API target declared by the bundled environment manifest, and it never imports `audio_cli`. Boundary tests exact-inventory that exception and reject ordinary loader aliases, reflective `builtins` access, and `eval` elsewhere. Function-local static model imports in stages remain permitted and keep heavyweight frameworks out of the host process.

Direct packages import another owner's public facade except for deliberately narrow edges such as `export -> transcribe.result`. The composition roots may wire only the exact feature facades and contracts declared in `tests/architecture/test_top_level_module_boundaries.py`. Inside one package family, implementations import the concrete module that owns a behavior rather than reaching back through `__init__.py`. The broad exports in `pipeline.__init__`, the execution names exposed by `transcribe.orchestrator`, `transcribe.native.run_native`, and the four export-refusal names exposed by `transcribe.refusals` are frozen compatibility aliases from earlier layouts. They do not confer ownership and are not a pattern for new APIs. The inventoried `render_human` export and legacy `output_format` parameter retain Python compatibility. Public `transcribe run` produces JSON; callers create readable deliverables with the separate export command. New callers import the refusal builders from `audio_cli.export` and use `transcribe.orchestrator.run` for all stacks.

Specification documents are enforced, not decorative: `tests/docs/test_spec_docs*.py` checks contract and example consistency, while `tests/audio_cli/environments/test_environments.py` checks bundled declarations against specifications and recorded partition evidence. The `tests/docs/test_shipped_*.py` family compares real command payloads with the documented key sets and nesting, covering machine/package commands, Qwen, FireRed, VibeVoice, exports, and export refusals. These command tests use controlled fixtures and doubles; they do not establish live model quality. The unresolved `packages verify` failure-shape disagreement remains explicitly exempted in `VERIFY_CONDITIONAL` in [the shipped-command test](tests/docs/test_shipped_commands_match_the_document.py): the contract declares `package/check/expected/actual`, while implementation emits `package/code/detail/fix`. Keep these checks current when a command or result shape changes.

## Repository layout

Project documentation belongs under `docs/`; keep the root README, agent instructions, skill entry points, and colocated patch READMEs in their conventional locations. `CLAUDE.md` is a symlink to this file. Put implementation tests under `tests/audio_cli/` following source ownership, including its transcription and DSP subpackages. Cross-cutting repository checks live in `tests/architecture/`, specification and shipped-command checks in `tests/docs/`, and research-runner checks in `tests/benchmark/`. Shared helpers use explicit package imports from their owner. See [test layout](docs/testing.md).

Keep compact recorded fixtures in `tests/fixtures/`. Private `.m4a`/`.mp4` inputs belong in ignored `tests/artifacts/media/`; generated local outputs belong under `tests/artifacts/`, which is excluded from source distributions. Update active fixture discovery when moving media, but preserve historical result paths and hashes.

## Evidence conventions

Runners live in `model_tests/benchmark/`, compact results are **tracked** under `model_tests/benchmark/results/` as `YYYY-MM-DD-<topic>.json`, and raw artifacts stay **untracked** in `model_tests/benchmark_runs/`. Research prose lives under `docs/model-tests/`. A measurement is reported with its fixture and configuration, or not reported. Distinguish three things and never let them blur: a **declared** interface, a **measured** result, and an **unresolved** question.

## Tooling

Write Markdown paragraphs and list-item prose on single physical lines. Do not hard-wrap prose to a column width; use line breaks for Markdown structure such as headings, separate paragraphs, list items, tables, and code blocks.

`uv` manages every environment — `uv venv` and `uv pip install`, never `python3 -m venv` with `pip`. Tests are `uv run --extra dev pytest`. Install the repository hooks with `uv run --extra dev pre-commit install` and run the deterministic lint/format gate with `uv run --extra dev pre-commit run --all-files`; Ruff lint fixes run before Ruff formatting. `ffmpeg` and `ffprobe` are machine runtime dependencies. Python 3.11+.

Model weights and their runtimes are provisioned **only** by `audio packages pull`, except for the documented, hash-verified Silero VAD bootstrap owned by `vad.py`. Never hand-download weights, hand-create a virtual environment, or edit a lock file to make an install succeed — the pins are what make the recorded measurements mean anything.

Keep every tracked, authored file below 500 physical lines by splitting at responsibility boundaries. `tests/architecture/test_file_size_budget.py` enforces the limit and owns the narrow exception list for generated locks and indivisible recorded fixtures; do not add an exception merely to avoid decomposing maintained code, tests, or prose.

When a decomposition creates a family of modules, give that family a real subpackage instead of encoding the namespace in repeated root-level filename prefixes. Keep each subpackage's `__init__.py` as an explicit public facade, have implementation modules depend on the module that owns a behavior rather than reaching back through the facade, and split by responsibility rather than stopping just below the line limit.

## Habits learned the hard way

- **Do not add structure nobody dispatches on.** A capability report here shed four nested objects that each seemed justified when added. The test is whether a caller branches on it.
- **Do not describe machinery that does not exist.** One rule instructed an adapter to handle output a stage never emits; it was inert and its test passed vacuously. When you write an invariant, write the assertion that would fail if it were violated, then check that it can fail.
- **A flag that parses is not a flag that works.** `--repair` was declared, documented in TRANSCRIBE_CONTRACT.md, and named in four `fix` strings `verify` emits, and no code read it. `--want` was accepted and ignored; `--stack` beside named packages dropped the stack. The cheap sweep that finds the rest: neutralize each `repair`/`force`/`dry_run`/`stack`/`allow_*` branch in turn and run the suite. Six of twenty were invisible. A flag the code cannot honour yet is refused at exit 2, never accepted quietly.
- **A double must be able to represent the state a repair produces.** `FakeToolchain` could not stop being drifted, so a repair test's final assertion ran against a second, undrifted toolchain that reports `ok` either way and the `if drift and repair` branch never executed. When a double cannot reach the post-repair state, the test passes without the repair running — and two payload shapes here turned out to be unreachable rather than untested for the same reason.
- **A parameter that decides nothing is worse than a missing one.** `vad_min_silence_ms` declared 300 ms while a hard-coded merge downstream required 540, so the profile's number was inert across a 240 ms band. Two thresholds answering one question can only disagree; keep one.

## Skills

[`audio-cli`](skills/audio-cli/SKILL.md) is the one skill, developed and shipped under `skills/audio-cli/` in the source distribution so it travels with the CLI. Keep it outside development-agent auto-discovery: do not expose it through `.agents/skills/` or add a compatibility symlink there, and do not install it into another workspace as part of building this repository. It is written for an agent *using* the tool on someone's audio, not for someone developing it: a short router in `SKILL.md` sends the reader to the reference for enhancement, a targeted fix, original-source transcription and export, saved-report interpretation and comparison, model provisioning, readiness, or command failures. Two rules keep it that way. Anything `--help` already states stays out of it, and backend internals stay out too — how the models are partitioned into runtimes belongs in [ENVIRONMENTS.md](docs/ENVIRONMENTS.md), where a developer will look for it.
