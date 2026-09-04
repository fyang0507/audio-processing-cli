# Provisioning the models transcription needs

The lane for *"what will this download?"*, *"prepare this transcription stack"*, and *"reclaim
that disk"*. For producing a transcript, read [transcribe.md](transcribe.md) first.

## Nothing downloads itself

`audio packages pull` is the only thing that fetches a model or prepares a runtime for it. No
measurement, render, or future transcription request will do it quietly in the background: a missing
model is a refusal with a `fix`, not a surprise download. That is deliberate — an agent that
silently pulls 17 GiB on someone's laptop has made a decision that was not its to make.

The one exception is the small speech-activity model, which is a couple of megabytes, verified by
hash, and fetched on first use. Everything else fails closed.

So: never hand-download weights, never hand-build an environment, and never edit a lock file to make
an install succeed. The pinned versions are what make this project's recorded measurements mean
anything, and working around them quietly invalidates them.

## Look before pulling

`audio doctor` is the first command on an unfamiliar machine and the only one that tells you whether
something missing from the machine itself — FFmpeg, a toolchain, disk, memory — will block the
request. `audio packages list` and `audio packages path` narrow that to what is provisioned and where
it lives. All three are read-only.

`doctor` also lists internal runtime environments with their states. You do not manage those: they
are created and removed with the packages that need them. A `provisional` marker on one is a note
about a possible future change, not a warning, and it changes nothing about any command.

Read two fields there, not one. The `state` is the registry's own record — `ready` means this root
provisioned the environment, not that anything in it can run — and `blocked_by_missing_tool` beside
it is the usability half: a non-empty list names a provisioning or repair tool that is off `PATH`
when no ready built runtime can take its place. In particular, Swift is needed to build or repair
FluidAudio, but a provisioned FluidAudio executable runs directly and remains usable if Swift later
leaves `PATH`. A stack pull can still leave the Swift environment `ready` but blocked when it
materializes only the model package and skips the missing build. `verify` is what turns the pair
into one verdict.

## Translate a request into package ids

Someone names a stack — `qwen-1.7b`, `qwen-0.6b`, `vibevoice`, or `firered` — and what they want
from it. Provision by id when you know them, or let the stack select everything it can use.

| Package id | Supplies | Download |
| --- | --- | --- |
| `qwen3-asr-1.7b-8bit` | speech recognition for `qwen-1.7b` | 2.30 GiB |
| `qwen3-asr-0.6b-8bit` | speech recognition for `qwen-0.6b` | 0.94 GiB |
| `qwen3-forcedaligner` | word-level timestamps, for stacks without their own | 1.19 GiB |
| `vibevoice-asr-7b` | recognition **and** native speaker structure for `vibevoice`, including its pinned tokenizer subset | 16.17 GiB |
| `firered-asr2s` | the entire `firered` pipeline | 8.93 GiB |
| `fluidaudio` | speaker diarization, for stacks without their own | unsized build |
| `speaker-diarization-coreml` | the diarizer's five required model artifacts; comes with `fluidaudio` | 0.02 GiB |
| `silero-vad` | speech-activity regions | 2.3 MB, fetches itself |

Three things to read off that table rather than guess:

- **`firered-asr2s` is one id for a whole pipeline** — speech activity, language identification,
  recognition, and punctuation. There is no separate punctuation or language package to look for, and
  punctuation is never optional.
- **Word timestamps and diarization are separate downloads** on the stacks that lack them natively,
  so "transcribe with word timings and speakers" can cost two or three ids rather than one.
- **`silero-vad` is only for stacks that need it.** A `firered` request covers speech activity
  itself, so pulling it alongside is harmless and pointless.

Selecting by stack deliberately over-provisions: it takes every package that stack *can* use,
diarizer and aligner included. Prefer explicit ids when the request is narrow and the difference is
gigabytes. Capability-based selection cannot trim a download today, and `pull` **refuses `--want`**
with exit 2 rather than accepting a flag it would ignore. Pass a stack or a list of ids, never both:
that is a conflict, also exit 2, because a stack is a guess about what might be needed and a list of
ids is an instruction.

## Expect a pull to be long, and silent

These are multi-gigabyte downloads plus a dependency install. On a fast connection the small packages
take a minute or two, the FireRed pipeline around five, and the largest one longer still — enough to
exceed a default command timeout. **Run a pull in the background and poll it.**

A long quiet stretch is a download in progress, not a hang. Do not kill and restart it, and do not
delete anything to tidy up: re-running the identical `pull` resumes a partial download and finishes
it. Progress goes to stderr; stdout is a JSON receipt whose `pulled_known_bytes` counts only what
*this* pull added, and whose `hub_revisions_pre_existing` names what was already cached and therefore
not downloaded again.

Re-running a finished `pull` is cheap and safe: anything already provisioned comes back under
`skipped` and is not touched, so the receipt for a second run lists no packages and no bytes. Read
`skipped` as "already there", never as "failed". Forcing the work anyway is `--repair PACKAGE`, and
that is not something to reach for routinely — it re-downloads.

Then confirm with `audio packages verify`. It exits **3** if any check fails and names a `fix` for
each failure. Two different repairs exist and the failure tells you which: a package whose files
changed or vanished is re-materialized with `pull --repair PACKAGE`, while a failure naming a runtime
environment lock is re-synced with `verify --repair`. A redirected environment root is not followed
or overwritten: replace the path named by its failure before running that repair. Read the `fix`
from the payload instead of choosing from memory.

Quote what a passing entry actually says. `digest: "ok"` means the bytes were hashed against a pin,
and only the speech-activity model has one. Hub packages instead report the manifest's `revision` or
`revisions` after the live Hub cache index binds each repository and revision to the snapshot path
recorded by pull, and after required allowlisted files and the recorded tree-byte total check out.
That is stronger than trusting the receipt and still is not a content hash. Never summarize a
`revision` entry as "digest verified".

The hash-pinned single-file package is also path-bound: a pass means the exact manifest models path
is a contained, non-symlink regular file, not merely that reachable bytes hash correctly. Pull and
Silero auto-fetch enforce the same rule before reusing or publishing that managed file. Publication
is bound to the exact private temporary inode written by the downloader; a managed leaf symlink is
replaced without following its target, but a directory at that path is preserved and refused.

`verify` also states one verdict per provisioned environment, and `drifted` is the only one this
command can repair:

| Verdict | Means | Your move |
| --- | --- | --- |
| `ok` | every check that applies to it passed | continue |
| `drifted` | its exact managed root is redirected, or its installed set no longer matches its lock | replace a redirected root first; otherwise use `verify --repair`; until then the drift sits in `failed` and the command exits 3 |
| `blocked` | a provisioning or repair tool is off `PATH` and no ready built runtime can substitute for it | report the missing tool; no `audio` command installs one |
| `absent` | its environment registry entry is missing or not `ready` | run the `environment_not_ready` pull-repair fix when it lists ready dependents; otherwise wait until a pull needs it |

`environment_not_ready` is the gate, not a subordinate integrity verdict. `verify` has not
inspected or launched those packages' checkouts, interpreters, or built products, so use its
`packages` and `fix` fields without describing an unprobed package as verified or corrupt.

`blocked` is the one to slow down on, because it does **not** fail the command: nothing provisioned
is necessarily corrupt and there is no `audio` fix that installs the external tool, so `verify`
exits 0 while reporting the limitation. Reading the exit code alone can therefore overstate what
the machine can provision or repair. A live runnable FluidAudio product is the deliberate exception:
its Swift environment stays `ok` because transcription executes the binary directly. The pinned private-API guard is the same shape — its
`mlx_audio_private_api_matches_expected` can come back `false`, or `null` where no verdict was
reachable, with the command still exiting 0, and a `null` is not a pass.

Pull, verify, and run all derive an environment root from the manifest. The `envs` parent and leaf
cannot redirect provisioning, and a ready root must be a contained, non-symlink directory before
freeze or runtime; run refuses before decode. An inner venv `bin/python` symlink is normal and does
not make the root drifted. The configured provisioning-root leaf is a separate ownership boundary:
if it is redirected, registry readers refuse it as `registry_unreadable` before `verify` probes
anything below it.

One package builds rather than downloads, and its build can fail late. A Swift product that compiles
but will not launch is refused at exit **3** with `package_build_unusable`; the registry entry stays
`pulling`, so `list` reports it not ready and nothing treats it as available. The `fix` is a
`pull --repair` on that package. Pull records `product_runs: true`, and `verify` does not trust that
history: it requires exactly one contained, non-symlink executable under the exact managed
FluidAudio checkout and launches it again. Do not report a diarizer as usable from the receipt alone.

Source-backed packages have the same live-evidence boundary. A pass requires the exact managed
checkout, the full resolved live Git HEAD, the manifest-owned post-patch hashes repeated exactly in
the receipt, the exact allowed tracked-file set, and no ordinary or ignored untracked files. A
legacy receipt's `checkout_commit` may be the short manifest alias or full resolved id; new pulls
write the full id, and that compatibility never relaxes the full live-HEAD check.

Expect `license_unreviewed` among a pull's warnings. It is non-blocking, and it means nobody has read
the license the model card declares. Report it, and never describe a declared license as a cleared
one.

## Disk accounting, which does not add up the way you expect

Everything provisioned lives under one root, printed by `path` and `doctor`, and
`AUDIO_PROCESSING_MODEL_CACHE` moves that root to another disk. Model weights are the exception: they
live in a cache shared with other tools, outside the root, so a root holding a couple of gigabytes
after a 17 GiB pull is normal. Trust each package's conditional `location` or repository-keyed
`locations` from `path`; checkout-backed packages also carry `checkout`. Never infer package size
or completeness from the root's size.

Three readings that trip people up:

- **`doctor` and `list` report what this tool provisioned, not what is on the disk.** Measured on a
  machine whose root held the speech-activity model with no registry at all, `doctor` reported it
  `absent` — it had been fetched automatically by a render, which records nothing. Read `absent` as
  "never pulled here", and let the tool decide whether it can use a file.
- **`path`'s `models.exists` is about a directory, nothing more.** It is not a statement that any
  package is ready.
- **Quote the right number.** `pulled_known_bytes` covers one pull, `total_known_bytes` from `list` is
  cumulative, and `reclaimed_bytes` from a teardown is measured afterwards and includes more than
  weights. None of the three will match. Say when a manifest estimate is unsized rather than
  omitting it — FluidAudio has no fixed pre-build size, although pull records the live build bytes.

## Reclaiming

`remove` takes named packages; `purge` takes everything this tool provisioned. Runtime environments
go when their last package does, and the report names what was kept and why. Neither touches media or
transcript output.

`remove` is all-or-nothing across the names you give it: one that was never provisioned refuses the
whole command with exit 2 and deletes nothing, so a typo costs a retry rather than gigabytes. Read
`removed` for what actually went; it never names a package the command left alone.

Because weights sit in that shared cache, teardown draws a narrow line: a revision is eligible only
when the receipt says **this root downloaded it**, the current manifest still pins it for that
package, and it was not **already there** before pull. Everything outside that intersection is
retained because it may belong to another tool or an earlier experiment. Local paths and
environment references also come from the installed manifest; an unknown registry package is
dropped without following its paths, and a failed local deletion keeps its owner and reclaims zero.
That is why a purge can legitimately free far less than the package sizes.

Run `purge --dry-run` first. Read deletion candidates from
`would_remove.hub_revisions`, retained ownership from `would_keep.hub_revisions`, and the package
projection from `reclaimable_known_bytes`; it excludes retained revisions and environment bytes.
On the real run, use `hub_revisions_deleted`, `hub_revisions_not_found`,
`hub_revisions_retained`, and measured `reclaimed_bytes` instead — the `would_*` and
`reclaimable_*` fields do not carry over.

One fact appears under three names — `hub_revisions_pre_existing` in a pull receipt,
`would_keep.hub_revisions` in a dry run, `hub_revisions_retained` in a teardown report. Alongside the
last, `hub_revisions_deleted` is what actually went and `hub_revisions_not_found` was already gone —
not reclaimable, and not an error.

To give a machine back, use `purge`: `remove` empties it only package by package and leaves state
behind. Do not finish either job by deleting directories yourself, and purge before uninstalling the
tool, or the provisioning root outlives the only thing that knows how to describe it.

## Exit codes

| Code | Meaning | What to do |
| --- | --- | --- |
| 0 | done | continue |
| 2 | the request was wrong — an unknown package or stack, a package that was never provisioned, or an argument the command cannot honour (`--want`; `--stack` beside package ids) | run the `fix` the payload names; for a bad name the error usually carries an `allowed` list |
| 3 | provisioning is incomplete or broken — a failed integrity check, a missing tool for a package named by id, a build whose product will not run, a drifted environment | run the `fix` the payload names, verbatim |

An absent toolchain blocks only the packages that need it — `doctor` says so, those packages report
`requires_tool`, and everything else still provisions. Report the blocked capability rather than
substituting something else for it.

That cuts two ways, deliberately. `pull --stack S` on a machine missing a toolchain exits **0**,
provisions everything it can, and names what it could not in a `warnings` entry carrying
`blocking: true` — so check `warnings` on a successful pull, not only the exit code. Naming that
package on the command line instead exits **3**, because skipping something asked for by name would
be worse than refusing it.
