
## 1. Fast long-form transcription — interview

Goal: a speaker-attributed transcript of a 30-minute interview, plus subtitles. Word
timing is required for the subtitles, so it is requested up front rather than discovered
missing at export.

### 1.1 Ask what the stack can do with this file

```bash
audio transcribe capabilities --stack qwen-1.7b --input meeting.m4a
```

```json
{
  "stack": "qwen-1.7b",
  "family": "Qwen3-ASR",
  "environment": "mlx",
  "roles": "asr only",
  "input": {"path": "meeting.m4a", "duration_seconds": 1794.2, "container": "m4a",
            "sample_rate_hz": 44100, "channels": 2},
  "processing": {
    "unit": "fixed_chunk",
    "unit_count": 10,
    "note": "180-second chunks, one at a time. Requesting diarization replaces them with the diarizer's turns, and that count is not known until the diarizer runs."
  },
  "failure_recovery": {
    "partial_results": "per_unit",
    "note": "Units are independent, so a failure leaves the finished ones usable and --range addresses the rest. The global generation budget is what most often stops a long file early, and it stops between units rather than inside one."
  },
  "cost": {
    "proved": "54 s and 3.0 GiB peak to transcribe a 30-minute Cantonese interview on an M4 Max, batch 1 with the MLX cache cleared per batch and a Cantonese language hint.",
    "projected_seconds": 53.6
  },
  "capabilities": {
    "languages": {"availability": "native",
                  "note": "Accepts the 30 names published by both pinned Qwen config.json support_languages arrays, case-insensitively and with canonical spelling; has no dialect selector. Only Mandarin, English and Cantonese have actually been run here. The one accuracy figure is Cantonese at 33.56% mixed-token error on a 30-minute interview, using diarization with --num-speakers 2 and --overlapping-segments. The label it returns without a hint is not a detector to route on."
                  },
    "verbatim": {"availability": "native",
                 "note": "Does not clean disfluencies, but drops every spoken \"uh\" and sometimes fuses the neighbouring words. Filler recall is unmeasured on every stack."
                 },
    "diarization": {"availability": "requires_add_on",
                    "note": "Adds FluidAudio, which needs a Swift toolchain and a second environment: 15 s and 0.55 GiB peak on a 30-minute sample. Produces speaker labels on the text and the turn intervals together, mapped onto the transcript by an exact partition of the timeline so no span is transcribed twice and no gap is invented. The published 95.42% participant-interval F1 and 3-of-75 speaker-change figures used --num-speakers 2 plus --overlapping-segments; the shipped default supplies no speaker-count prior and enables overlap detection only when overlapped_speech is requested, so its quality is unmeasured. RSS excludes memory held by system Core ML services."
                    },
    "overlapped_speech": {"availability": "requires_add_on",
                          "note": "Enables FluidAudio's overlapping-segments mode in the same stage as diarization. The recorded 95.42% participant-interval F1 and downstream 33.56% MER used this mode plus a two-speaker prior; the shipped no-prior configuration is unmeasured. Without this request nothing in the plan detects overlap, so an empty abstention ledger means undetected rather than absent."},
    "vad": {"availability": "requires_add_on",
            "note": "Adds Silero, a hash-pinned file that fetches itself, so it never returns exit 3: 0.4 s and 0.11 GiB peak on a 150-second sample, about 5 s on this input. Measured 0.8505 frame-level F1 at 0.7655 precision and 0.9567 recall, so the gate over-includes — it bounds activity, not speakers."
            },
    "word_timestamps": {"availability": "requires_add_on",
                        "note": "Adds Qwen3-ForcedAligner beside ASR in the existing mlx environment, so with FluidAudio this request spans two environments: 4.6 s of alignment on a 139-second sample excluding model load, about a minute on this input. Never scored against hand-labelled boundaries, and neither is FireRed's native timing, so switching stacks for accuracy would trade one unmeasured number for another. Absent on any segment with no speech to align."},
    "segment_timestamps": {"availability": "impossible", "reason": "no_native_segment_extents",
                           "note": "This stack emits no segment extents. The chunk boundaries it works in are not speech timing and are never published as any."},
    "lid": {"availability": "impossible", "reason": "no_backend_declares_on_stack",
            "note": "Available on firered, which runs a dedicated LID stage."},
    "token_lid": {"availability": "impossible", "reason": "no_backend_declares",
                  "note": "Named only so a request fails loudly. Code-switching support does not imply per-token labels, and no backend here produces them."}
  },
  "next": "audio transcribe plan --input meeting.m4a --stack qwen-1.7b"
}
```

Exit 0. Nothing was downloaded; only the input's container metadata was read.

Every block above answers a question the caller has to decide, and answers it in prose
wherever prose will do. Three enums and two numbers carry everything a program branches on —
`availability`, `processing.unit`, `failure_recovery.partial_results`, the unit count, and
`projected_seconds`. The rest is sentences, because no caller dispatches on a nested
`evidence` object and a reader learns the same thing either way.

`cost.proved` is a run that actually happened, in units a reader can hold: seconds and
gibibytes against a named sample. `projected_seconds` applies that rate to this input so
nobody re-derives it. Where two capabilities come out of one stage the sentence says so —
`diarization` and `overlapped_speech` share a single FluidAudio run — which is the sort of
thing a caller summing per-capability costs would otherwise double-count.

### 1.2 Resolve the request

```bash
audio transcribe plan --input meeting.m4a \
  --stack qwen-1.7b \
  --want diarization,word_timestamps \
  --language Cantonese
```

```json
{
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "diarizer":   {"backend": "fluidaudio", "version": "0.15.5",
                   "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
                   "environment": "swift",
                   "config": {"step_ratio": 0.1,
                              "min_segment_duration": 0.0,
                              "threshold": 0.6, "batch_size": 32},
                   "config_note": "the shipped default supplies no speaker-count prior and enables overlapping_segments only when overlapped_speech is requested; the cited quality figures used both --num-speakers 2 and --overlapping-segments, so they do not measure this request",
                   "selected_by": "add_on_required_by:diarization"},
    "asr":        {"backend": "qwen3-asr-1.7b-8bit", "environment": "mlx",
                   "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
                   "config": {"batch_size": 1, "clear_mlx_cache_after_every_batch": true,
                              "language": "Cantonese",
                              "api_path": "_generate_chunks_batched", "max_tokens": 16384},
                   "adapter_strips": ["language <label><asr_text> scaffold"],
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 0.0,
                   "determinism_basis": "argmax decode; back-to-back calls in one process produced byte-identical text, cross-process repetition untested"},
    "aligner":    {"backend": "qwen3-forcedaligner", "environment": "mlx",
                   "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
                   "config": {"scope": "all_segments",
                              "language_rule": "Chinese when text matches [一-鿿], otherwise English; the ASR --language hint is never forwarded"},
                   "selected_by": "add_on_required_by:word_timestamps"}
  },
  "execution": {
    "stage_order": ["decode", "diarizer", "asr", "aligner"],
    "residency": "one_model_stage_at_a_time",
    "environments_spanned": ["swift", "mlx"],
    "note": "stages run strictly sequentially and no two model stages are resident together; wall time adds across stages, peak memory does not, and the per-stage peaks below must not be summed"
  },
  "capabilities": {
    "diarization":     {"satisfaction": "derived", "backend": "fluidaudio",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "Anonymous labels are reconciled sample-exactly onto the ASR text, plus the diarizer's turn intervals. The shipped no-prior, overlap-off configuration is unmeasured; the 3-of-75 speaker-change result used a two-speaker prior with overlap detection enabled and does not apply to this request."
                        },
    "word_timestamps": {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "Boundary error against labels is unmeasured, and absent on any segment with no speech to align."}
  },
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "kind": "weights",
     "bytes": 2467859030, "provisioned": false},
    {"package": "fluidaudio", "environment": "swift", "kind": "toolchain",
     "requires_tool": ["swift"], "bytes": null, "provisioned": false},
    {"package": "speaker-diarization-coreml", "environment": "swift", "kind": "weights",
     "bytes": 21599417, "provisioned": false},
    {"package": "qwen3-forcedaligner", "environment": "mlx", "kind": "weights",
     "bytes": 1276475979, "provisioned": false}
  ],
  "total_known_download_bytes": 3765934426,
  "unsized_packages": [],
  "warnings": [],
  "next": "audio packages pull qwen3-asr-1.7b-8bit fluidaudio speaker-diarization-coreml qwen3-forcedaligner",
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "complete": true,
    "source": {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "speaker": null,
       "words": [{"word_id": "w_0", "text": null, "start": null, "end": null}]}
    ],
    "turns": [
      {"turn_id": "turn_0", "speaker": null, "start": null, "end": null}
    ],
    "abstentions": [
      {"abstention_id": "ab_0", "reason": "raw_fragment", "start": null, "end": null}
    ],
    "provenance": "<stack, outcomes, observed, and the executed plan; elided in print>"
  }
}
```

Exit 0. `segment_timestamps` was not requested and is impossible on this stack anyway, so
segments carry no `start`/`end` — in the sample above or in the real result below.

### 1.3 Provision

```bash
audio packages pull --stack qwen-1.7b
```

`pull` takes a stack or a list of package ids, and no `--want`: narrowing a stack to the
capabilities actually requested is the planner's job, so passing it is refused rather than
ignored — see the two refusals at the end of this section. A stack therefore provisions every
package it can use, `silero-vad` included.

Progress goes to stderr; stdout is the receipt:

```json
{
  "pulled": [
    {"package": "fluidaudio", "environment": "swift",
     "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
     "bytes": 471203840, "built": true, "product_runs": true},
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx",
     "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
     "bytes": 2467859030},
    {"package": "qwen3-forcedaligner", "environment": "mlx",
     "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
     "bytes": 1276475979},
    {"package": "silero-vad", "environment": "core",
     "bytes": 2327524, "digest_verified": true},
    {"package": "speaker-diarization-coreml", "environment": "swift",
     "revision": "1ed7a662fdc7109e36d822db793ee6eebdaf8594",
     "bytes": 21599417}
  ],
  "skipped": [],
  "environments_created": ["mlx", "swift"],
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "registry": "/Users/you/Library/Caches/audio-processing-cli/registry.json",
  "pulled_known_bytes": 4239465790,
  "pulled_known_bytes_note": "Known artifact sizes for packages materialized by this invocation, including repairs: manifest-declared sizes for Hub revisions owned by this root, and recorded local artifact sizes. Excludes pre-existing shared revisions, skipped packages, and environment bytes. Not measured network bytes or added disk usage.",
  "unsized_packages": [],
  "warnings": [
    {"code": "license_unreviewed", "blocking": false,
     "packages": ["qwen3-asr-1.7b-8bit", "qwen3-forcedaligner"],
     "detail": "qwen3-asr-1.7b-8bit, qwen3-forcedaligner report a license their model card declares but nobody has reviewed. A declared license is evidence that one exists, not a redistribution clearance."}
  ]
}
```

Exit 0. `pulled_known_bytes` covers the packages in *this* pull and nothing else. It was called
`reclaimable_bytes`, which read as a running total and is not one — the figure legitimately goes
down on a second, smaller pull. `audio packages list` reports the cumulative
`total_known_bytes`.

`digest_verified` appears on exactly one entry, and that is the point of it. `silero-vad` is
pinned by content hash, so materializing it hashes the file against the manifest. The Hub packages
are pinned by `revision`; there is no hash in the manifest to compare a snapshot against, so they
report the revision they materialized and claim no digest. Every Hub entry used to carry
`digest_verified: true` for a check no code performed.

Run the same line twice and the second run does nothing: a package the registry already calls
`ready` is listed under `skipped`, contributes no bytes, and is not touched — in particular it is
not reopened as `pulling`, which an interrupt would leave behind as a downgraded install.
`pull --repair <package>` is how a caller asks for the work anyway; it re-materializes rather than
trusting what is on disk, which for a Hub snapshot means re-downloading it and for a checkout means
discarding and re-cloning it.

Before a package is materialized, pull checks an existing interpreter environment against its lock
and required direct installs. A drifted environment is refused before model downloads on an ordinary
pull; the guidance asks for environment repair followed by the package retry. `pull --repair`
reconciles dependency drift in the selected environment before force-fetching its package. It does
not execute the selected package's old checkout while replacing it. Other ready checkouts retain
their integrity gate before reinstall, and a failed sync never promotes an environment to ready.

Native installation proves the pinned checkout before executing its build, removes generated
ordinary/ignored build residue within that install transaction, then proves integrity again.
The Git worktree and metadata must resolve to the managed clone. Cleanup keeps that directory
open through child execution and uses Git paths relative to it, so configuration changes or a
replaced pathname cannot redirect deletion elsewhere. Verification still
rejects arbitrary ordinary and ignored files outside that controlled build transaction.


`hub_revisions_pre_existing` lists only revisions this root found in the shared Hub cache before
materialization. `pre_existing_note` states the listed count out of the package's pinned revisions:
a multi-repository package may reuse only some of them. This is ownership evidence, not measured
transfer: missing files can still be fetched, and repair requests a fresh download. Teardown does
not delete those pre-existing revisions; see §5. `pulled_known_bytes_note` defines the accounting
scope, which excludes those revisions and is neither network bytes nor added disk usage.

On a machine with no Swift toolchain this same command exits 0 having provisioned everything else,
and reports what it could not:

```json
{
  "warnings": [
    {"code": "toolchain_missing", "blocking": true,
     "packages": ["fluidaudio"], "requires_tool": ["swift"],
     "detail": "fluidaudio needs swift, which is not on PATH, so it was not provisioned; the rest of stack qwen-1.7b was. Install the toolchain and pull it by name."}
  ]
}
```

That is a `warnings` entry, not an error payload, which is what §0's promise about an absent
`swift` — "reported rather than fatal, blocking only the packages that need it" — actually
requires. `blocking: true` distinguishes it from `license_unreviewed`: something a caller asked
for is absent. Exit 3 is reserved for two cases: a stack where *nothing* was provisionable, and a
package named explicitly. Naming `fluidaudio` on the command line is an instruction, and skipping
an instruction quietly is worse than refusing it.

```bash
audio packages verify
```

```json
{
  "verified": [
    {"package": "fluidaudio", "product_runs": true, "product_digest": "ok",
     "patches_applied": ["fluidaudio-pinned-model-dir.patch"]},
    {"package": "qwen3-asr-1.7b-8bit", "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"},
    {"package": "qwen3-forcedaligner", "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15"},
    {"package": "silero-vad", "digest": "ok"},
    {"package": "speaker-diarization-coreml", "revision": "1ed7a662fdc7109e36d822db793ee6eebdaf8594"}
  ],
  "environments": {"mlx": "ok", "swift": "ok", "torch-firered": "absent",
                   "torch-vibevoice": "absent"},
  "mlx_audio_private_api_target": "mlx_audio/stt/models/qwen3_asr/qwen3_asr.py",
  "mlx_audio_private_api_expected_source_hash": "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250",
  "mlx_audio_private_api_source_hash": "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250",
  "mlx_audio_private_api_matches_expected": true,
  "mlx_audio_private_api_signature_ok": true,
  "failed": []
}
```

Exit 0. After a partial stack pull on a Swift-less machine — where the Core ML weights
provisioned but FluidAudio could not be built — the command instead reports:

```json
{
  "environments": {"mlx": "ok", "swift": "blocked", "torch-firered": "absent",
                   "torch-vibevoice": "absent"}
}
```

`blocked` rather than `ok`, and still exit 0. `speaker-diarization-coreml` needs no toolchain of
its own, so it provisioned and left the `swift` environment `ready` in the registry, but no built
runtime exists and Swift is absent to create or repair one. It is not a `failed` entry either:
nothing provisioned is broken, the gap is a package `list` already reports as absent, and no
`audio` command installs a toolchain for a `fix` to name. If a ready FluidAudio executable does
exist, `verify` launches it directly; a successful launch makes the same Swift-less environment
`ok`, while a nonlaunching binary leaves it `blocked` and fails the package check. The registry
state stays `ready` in `doctor` and `list`, which report what the registry holds and publish
`blocked_by_missing_tool` beside it; only `verify` states a verdict, so only `verify` needed the
fourth word. VOCABULARY.md has the two enumerations.

A registry contradiction fails closed instead. In a minimal root where `fluidaudio` is the only
`ready` package while its `swift` environment entry is still `creating`, the relevant fields are:

```json
{
  "verified": [],
  "environments": {"swift": "absent"},
  "failed": [
    {
      "environment": "swift",
      "code": "environment_not_ready",
      "detail": "registry state is 'creating' while ready package(s) depend on it: fluidaudio",
      "packages": ["fluidaudio"],
      "fix": "audio packages pull --repair fluidaudio"
    }
  ]
}
```

A missing environment entry has the same verdict and code, with `absent` in `detail`. The
environment gate runs before package verification: no dependent package appears in `verified`,
and `verify` does not inspect its source checkout or launch its interpreter or built product. The
typed failure makes the command exit 3.

Which key an entry carries *is* the claim, and the two are not the same claim. `digest: "ok"`
means the bytes on disk were hashed and match the manifest's pin. `revision` means that revision is
pinned and its snapshot is present — the pin is recorded and checkable, the contents were not
hashed, and there is nothing to hash them against. The absent key is the honest report; a
`digest_verified: false` beside a `revision` would confess a check that was never designed rather
than state the one that was. A four-repository package reports `revisions` for the same reason the
receipt does: no one of them is *the* revision.

Revision-pinned does not mean path-only. For each Hub package, `verify` and the run preflight also
require the Hub cache index to bind each repository and revision to the recorded snapshot path,
every manifest-filtered `allow_patterns` match, and the same total tree bytes recorded by pull.
Those checks catch redirected or missing tokenizer files and size drift
without pretending to be a cryptographic digest. Source-backed native packages add a live Git
check before decode: HEAD, the exact tracked file set derived from the installed patch, every
manifest-owned post-patch SHA256, matching receipt hashes, and zero ordinary **or ignored**
untracked files. An unpatched checkout must have empty patch history, an empty hash map, and no
tracked changes. An ignored `.pyc` or extension can still be imported, so Git's default decision to hide it
is not a provenance exemption. A legacy receipt's `checkout_commit` may be the manifest's short
`commit` alias or its full `resolved_commit`; new pulls record the full value and the live HEAD
must always equal that full resolved commit. Selected managed Python interpreters are launched
before decode. A selected native source checkout must also be installed under its exact
manifest-pinned distribution name as a direct `file://` reference to that checkout; an
interpreter that launches after sync but no longer imports the checkout is not ready. FluidAudio
additionally requires exactly one contained, non-symlink executable
product under its exact managed checkout. Its live SHA256 must match the pull receipt before it
is launched; an executable bit or old receipt boolean is not liveness. The pinned source patch
makes offline mode require `--model-dir` and disables ModelHub downloads, so run binds the
product to the exact managed `speaker-diarization-coreml` directory rather than an implicit
user cache.

The hash-pinned URL package has no checkout or Hub index. Pull accepts its fast-path hit only when
the exact manifest `<root>/models` target is a contained, non-symlink regular file, then applies the
same check after download; a hash-matching symlink is replaced, not trusted. Silero's runtime
auto-fetch uses the same target rule.

An environment is trusted from its manifest-derived root outward, not from `registry.json` inward.
Pull refuses a redirected `envs` parent or environment leaf before creation or install. For a ready
root, `verify` checks that exact path is a non-symlink directory resolving under the provisioning
root before freeze; a redirect makes `environments.<name>` `"drifted"` and adds an
`environment_drifted` failure with `examples.environment_root` and a fix to replace the path before
`verify --repair`. Run performs the same root check before interpreter launch or decode and reports
`environment_<name>_managed_root` in `package_integrity_failed`. The normal venv `bin/python`
symlink remains allowed inside the trusted environment root.

`verify` reports every provisioned environment, not only the ones this pull touched, so the
two `torch` environments appear as `absent` rather than being omitted — an environment missing from
the map would be indistinguishable from one nobody has looked at. The private-API hash is published
alongside the value it is compared against, because `matches_expected: true` on its own is a claim
a reader cannot check.

Two ways of asking for a pull are refused, both exit 2, and both are refusals of a *silently
ignored argument* rather than of an unsupported idea:

```bash
audio packages pull --stack qwen-1.7b --want diarization,word_timestamps
```

```json
{
  "code": "want_not_implemented",
  "field": "--want",
  "provided": "diarization,word_timestamps",
  "detail": "--want belongs to transcribe plan; packages pull accepts explicit package ids or every package available to --stack",
  "fix": "use transcribe plan with the original --input, --stack, and --want; then run the plan next command for missing packages"
}
```

```bash
audio packages pull --stack qwen-1.7b silero-vad
```

```json
{
  "code": "stack_conflicts_with_named_packages",
  "stack": "qwen-1.7b",
  "packages": ["silero-vad"],
  "detail": "--stack qwen-1.7b was passed alongside named packages; a stack selects every package it can use and named ids select exactly those, so one of the two has to go",
  "fix": "audio packages pull silero-vad"
}
```

`--want` was accepted and dropped on the floor until this pass, and so was `--stack` beside a list
of ids. Both are the failure §4.6 names for `transcribe`: a caller would believe it had constrained
a 4 GiB download it never touched. The `fix` keeps the argument that was an instruction and drops
the one that was a guess.
