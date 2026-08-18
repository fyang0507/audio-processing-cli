# Provisioning the models transcription needs

The lane for *"transcribe this"*, *"what will this download?"*, and *"reclaim that disk"*.

**The command that will use these models does not exist yet.** Provisioning shipped ahead of it, so
today this lane is for getting a machine ready, reporting what a request would cost, and cleaning up
— not for producing a transcript. Never describe `audio transcribe` as something someone can run.

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

## Translate a request into package ids

Someone names a stack — `qwen-1.7b`, `qwen-0.6b`, `vibevoice`, or `firered` — and what they want
from it. Provision by id when you know them, or let the stack select everything it can use.

| Package id | Supplies | Download |
| --- | --- | --- |
| `qwen3-asr-1.7b-8bit` | speech recognition for `qwen-1.7b` | 2.30 GiB |
| `qwen3-asr-0.6b-8bit` | speech recognition for `qwen-0.6b` | 0.94 GiB |
| `qwen3-forcedaligner` | word-level timestamps, for stacks without their own | 1.19 GiB |
| `vibevoice-asr-7b` | recognition **and** native speaker structure for `vibevoice` | 16.16 GiB |
| `firered-asr2s` | the entire `firered` pipeline | 8.93 GiB |
| `fluidaudio` | speaker diarization, for stacks without their own | unsized build |
| `speaker-diarization-coreml` | the diarizer's model; comes with `fluidaudio` | 0.12 GiB |
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
gigabytes. Capability-based selection is reserved and does not narrow anything yet, so it cannot be
used to trim a download today.

## Expect a pull to be long, and silent

These are multi-gigabyte downloads plus a dependency install. On a fast connection the small packages
take a minute or two, the FireRed pipeline around five, and the largest one longer still — enough to
exceed a default command timeout. **Run a pull in the background and poll it.**

A long quiet stretch is a download in progress, not a hang. Do not kill and restart it, and do not
delete anything to tidy up: re-running the identical `pull` resumes a partial download and finishes
it. Progress goes to stderr; stdout is a JSON receipt whose `pulled_known_bytes` counts only what
*this* pull added, and whose `hub_revisions_pre_existing` names what was already cached and therefore
not downloaded again.

Then confirm with `audio packages verify`. It exits **3** if any check fails and names a `fix` for
each failure. Two different repairs exist and the failure tells you which: a package whose files
changed or vanished is re-materialized with `pull --repair PACKAGE`, while a failure naming a runtime
environment is re-synced with `verify --repair`. Read the `fix` from the payload instead of choosing
from memory.

Expect `license_unreviewed` among a pull's warnings. It is non-blocking, and it means nobody has read
the license the model card declares. Report it, and never describe a declared license as a cleared
one.

## Disk accounting, which does not add up the way you expect

Everything provisioned lives under one root, printed by `path` and `doctor`, and
`AUDIO_PROCESSING_MODEL_CACHE` moves that root to another disk. Model weights are the exception: they
live in a cache shared with other tools, outside the root, so a root holding a couple of gigabytes
after a 17 GiB pull is normal. Trust the per-package `location` from `path`, never the root's size.

Three readings that trip people up:

- **`doctor` and `list` report what this tool provisioned, not what is on the disk.** Measured on a
  machine whose root held the speech-activity model with no registry at all, `doctor` reported it
  `absent` — it had been fetched automatically by a render, which records nothing. Read `absent` as
  "never pulled here", and let the tool decide whether it can use a file.
- **`path`'s `models.exists` is about a directory, nothing more.** It is not a statement that any
  package is ready.
- **Quote the right number.** `pulled_known_bytes` covers one pull, `total_known_bytes` from `list` is
  cumulative, and `reclaimed_bytes` from a teardown is measured afterwards and includes more than
  weights. None of the three will match. Say when something is unsized rather than omitting it —
  the diarizer build always is.

## Reclaiming

`remove` takes named packages; `purge` takes everything this tool provisioned. Runtime environments
go when their last package does, and the report names what was kept and why. Neither touches media or
transcript output.

Because weights sit in that shared cache, teardown draws one line: a revision **this machine's root
downloaded** is deleted, and a revision that was **already there** when it was pulled is retained,
since it may belong to another tool or an earlier experiment. That is why a purge can legitimately
free far less than the packages' sizes. Run `purge --dry-run` first and report the split rather than
promising the total.

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
| 2 | the request was wrong — an unknown package or stack, a package that was never provisioned | correct the name; the error usually carries an `allowed` list |
| 3 | provisioning is incomplete or broken — a failed integrity check, a missing tool, a failed verify | run the `fix` the payload names, verbatim |

An absent toolchain blocks only the packages that need it — `doctor` says so, those packages report
`requires_tool`, and everything else still provisions. Report the blocked capability rather than
substituting something else for it.
