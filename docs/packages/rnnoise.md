# RNNoise artifact provisioning

`rnnoise-voice` is an optional enhancement-only model for the existing host FFmpeg `arnndn` filter. It belongs to `core`, has empty transcription `roles` and `stacks`, and adds no environment, build, lock, or implicit download. No transcription stack selects it. Only an explicit `audio packages pull rnnoise-voice` may fetch it; `--repair` retries failed or drifted provisioning. Runtime selection belongs to the enhancement CLI and pipeline, not package provisioning.

## Immutable source identity

| Field | Pinned value |
| --- | --- |
| Repository | `GregorR/rnnoise-models` |
| Revision | `3eee541a283fd3b8f81b85b1748e3b9ccbefa04d` |
| Repository path | `beguiling-drafter-2018-08-30/bd.rnnn` |
| Managed filename | `rnnoise-voice-bd-2018-08-30.rnnn` |
| Required bytes | `299693` |
| Git blob SHA-1 | `0173a18664f5905dbdc4543d3528741fb32cd69f` |

These pins come from [the upstream tree API](https://api.github.com/repos/GregorR/rnnoise-models/git/trees/3eee541a283fd3b8f81b85b1748e3b9ccbefa04d?recursive=1), inspected without downloading model bytes. The download URL is exactly the HTTPS `raw.githubusercontent.com` URL derived from this repository, revision, and path. No upstream SHA-256 was found or invented. Git SHA-1 here is content identity anchored by the trusted installed manifest and HTTPS, not a claim of modern cryptographic collision resistance or a signed upstream attestation. The publisher's [model description](https://github.com/GregorR/rnnoise-models/blob/3eee541a283fd3b8f81b85b1748e3b9ccbefa04d/beguiling-drafter-2018-08-30/info.txt) targets voice amid recording noise. Its [copyright statement](https://github.com/GregorR/rnnoise-models/blob/3eee541a283fd3b8f81b85b1748e3b9ccbefa04d/README.md) is recorded as a publisher declaration, with `license_reviewed: false`; it is not a redistribution clearance. Quality on user recordings is unmeasured by this provisioning work.

## Verification and lifecycle

The `git-blob` source validates the full revision/blob identifiers, exact immutable HTTPS URL, safe repository path and managed filename, positive declared byte count, and disabled auto-fetch. Pull uses the same descriptor-bound single-file download/publication transaction as URL SHA-256 artifacts. It refuses redirected parents, never accepts a symlink as a cache hit, preserves an old cache entry if verification fails, and publishes only the private temporary inode it created. Repair replaces an invalid leaf without following it; it does not replace redirected parent directories.

Git hashes `blob <actual byte count>\0` followed by the exact file bytes. The media hashing primitive derives the header from the opened regular file's size, checks that the number of bytes read equals that size, and checks that the file and directory bindings remain stable. Both the computed blob identity and actual count must equal the manifest. A correct cache entry avoids another transfer even during repair. Ordinary pull retains the existing lifecycle behavior of skipping ready registry entries; runtime resolution and `packages verify` perform the fresh checks.

On success, the Git artifact's pull receipt adds `revision` and `git_blob_sha1` alongside the standard package/environment/bytes fields. Its registry materialization contains only `path`, `revision`, `git_blob_sha1`, and `bytes`. Its entry under `packages verify`'s `verified` array has this shape:

```json
{
  "package": "rnnoise-voice",
  "revision": "3eee541a283fd3b8f81b85b1748e3b9ccbefa04d",
  "git_blob_sha1": "0173a18664f5905dbdc4543d3528741fb32cd69f",
  "bytes": 299693
}
```

This is the declared success contract, not a recorded model download. The Git artifact has no `digest`, `digest_verified`, or upstream `sha256` claim. Existing URL SHA-256 receipts retain `digest_verified: true` and verification retains `digest: "ok"`. Removal and purge derive the artifact path from the installed manifest under the selected `AUDIO_PROCESSING_MODEL_CACHE` root, never a mutable receipt path, and leave other roots and unrelated files alone. Git artifacts do not claim or delete Hub revisions.

## Composition API

```python
from audio_cli.packages import verified_artifact

path, provenance = verified_artifact("rnnoise-voice")
```

The return type is `tuple[pathlib.Path, dict]`. The provenance keys are `package`, `source` (a copy of the manifest source declaration), `bytes` (actual count), and `sha256` (local content digest). Git blob SHA-1 and this local SHA-256 are computed in one pass over the same open regular file. The helper requires a ready registry entry, resolves only the exact managed artifact, and freshly verifies bytes on every call without fetching or repairing. It raises `ProvisioningError` at exit 3 for an absent/incomplete package (`package_not_provisioned`, `fix: audio packages pull rnnoise-voice`) or missing, redirected, or changed artifact (`package_integrity_failed`, `fix: audio packages pull --repair rnnoise-voice`). Unknown or nonsingle-file package requests are exit 2. A redirected root or parent must first be restored to a real managed directory before retrying the fix.

The root CLI passes the path/provenance into enhancement. The consuming media mechanism opens the model without following symlinks, hashes the copied bytes against `provenance["sha256"]`, and uses its own verified private copy for FFmpeg. Returning a path alone cannot prevent a replacement after the helper returns; the local SHA-256 enables that subsequent verification without claiming any upstream SHA-256. Pipeline owns optional feature selection and ordering and never imports `packages`; `dsp` owns no model loading or filesystem I/O. Timing and speech-preservation behavior belong to the consuming filter workflow and its evidence.
