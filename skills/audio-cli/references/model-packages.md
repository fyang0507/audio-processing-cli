# Provisioning and reclaiming model packages

Use this lane for preparing a stack, understanding download costs, readiness failures, and reclaiming managed disk space. For a transcript request, start with [transcribe.md](transcribe.md). Use the `audio` executable selected in [SKILL.md](../SKILL.md) throughout.

## Select only the packages the task needs

`doctor` reports host dependencies and the provisioning root. `packages list` reports registry state; `packages path` locates managed artifacts. These commands are read-only. A registry `ready` value records provisioning, not a fresh integrity or usability check.

For a transcription request, read the concrete plan's `packages[]`: `package` is the id, `provisioned` records availability known to planning, and `bytes`, `total_known_download_bytes`, and `unsized_packages` describe download estimates. The conditional top-level `next` is an exact missing-id pull command; it is absent when all packages are provisioned. On an older invocation without `next`, construct the named-id pull from missing `packages[].package` values and live help. Read warnings and retain any unsized cost.

Selecting by stack installs everything that stack can use, including optional additions. It can cost much more than the plan's request. Use that only when preparing the whole stack is intended. Do not combine named ids with a stack or transfer transcription's `--want` to `packages pull`; consult the selected invocation's help for accepted selection forms.

`audio packages pull` owns model and runtime provisioning. The sole first-use auto-fetch is the small hash-verified `silero-vad` model, which inspection and enhancement also use. It can exist without a pull receipt, so registry `absent` does not prove its file is missing, and a plan's `auto_fetch` entry does not prove a prior download. Explicit `packages pull silero-vad` is also available. No other model or runtime is fetched implicitly. Never download weights, build a runtime, alter its lock, or delete cache files by hand to work around a refusal.

## Follow progress and check readiness

Multi-gigabyte pulls can outlast a command timeout. Keep the running process and poll it; progress is on stderr and the completed receipt is on stdout. Silence alone is not evidence of a hang. An interrupted pull can resume; do not kill an active one just to start over.

`pulled_known_bytes` totals known sizes of owned artifacts materialized by this pull, including repairs; `skipped` packages contribute nothing. Read `pulled_known_bytes_note` for its accounting scope. Inspect successful receipts too: a stack pull can exit 0 with `warnings[].blocking: true` for packages it could not prepare. Naming such a package explicitly can instead fail at exit 3.

`packages verify` emits its check report on **stdout**, including at exit **3**. Read `failed[]` and each failure's `package` or `packages` and `environment`, alongside `verified[]` and `environments`. Match these against the current plan: an unrelated failed package is a separate problem, not proof that this request cannot run. A selected package's failed check or an unusable environment it needs does block it. Preserve that distinction in the reply.

| Environment verdict | Consequence |
| --- | --- |
| `ok` | the checks applicable to that environment passed |
| `drifted` | follow the prescribed repair; a redirected root requires resolving the named path first |
| `blocked` | an external tool is missing; no `audio` command installs that tool |
| `absent` | it is not ready; follow an `environment_not_ready` fix for its named dependents |

A zero verify exit does not cancel a `blocked` verdict or a false/unknown runtime diagnostic. Read the reported limitation before promising the selected task can proceed. Conversely, a provisioned built executable can be runnable even when the toolchain used to build it is absent.

Package-file failures generally prescribe `pull --repair` for named packages; environment lock drift prescribes `verify --repair`. The latter can repair across the managed root, so check its scope before running it on a shared installation. Use the actual failure's fix and the safeguards in [failures.md](failures.md), including stopping after an unchanged prescribed repair. Do not repair an unrelated package merely to obtain a globally green report.

Quote the check that passed. A `revision` or `revisions` entry establishes pinned cache identity and required-file checks, not a content digest. A `git_blob_sha1` with `bytes` records a checked single-file Git content identity and size; it is not a publisher-provided SHA256. `digest: "ok"` is the SHA256-pinned file check; `product_digest` is a separate built-product check. `license_declared` is not redistribution clearance; preserve `license_reviewed` and any `license_unreviewed` warning.

## Disk accounting and reclaiming

Trust the root and package `location` / `locations` returned by `path`. Hub weights can live in a cache shared with other tools outside the managed root, so directory size does not establish package completeness. `pulled_known_bytes` uses manifest-declared sizes for owned revisions and recorded local artifact sizes, excluding pre-existing shared revisions, skipped packages, and environment bytes. It is not network bytes or incremental disk usage. `list`'s `total_known_bytes` is cumulative, and teardown's `reclaimed_bytes` is measured after deletion.

Use `remove` for named packages and `purge` for everything this root provisioned, only when that reclamation is requested. Removing one unknown package refuses the whole named selection. Runtime environments remain while other packages need them, and media/transcripts are not teardown targets.

Preview a purge with its dry run. Read `would_remove.hub_revisions`, `would_keep.hub_revisions`, and `reclaimable_known_bytes`; the estimate excludes retained revisions and environment bytes. On a real teardown, read `hub_revisions_deleted`, `hub_revisions_not_found`, `hub_revisions_retained`, and measured `reclaimed_bytes`. Pre-existing or unowned shared weights may be retained, so reclaiming less than the manifest estimate is expected. Do not finish by deleting directories yourself. If uninstalling was requested, reclaim intended managed packages first because their root survives removal of the CLI.
