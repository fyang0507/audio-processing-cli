
## 5. Teardown

```bash
audio packages list
```

```json
{
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "bytes": 2467859030,
     "state": "ready", "license_declared": "apache-2.0", "license_reviewed": false,
     "used_by_stacks": ["qwen-1.7b"]},
    {"package": "fluidaudio", "environment": "swift", "bytes": 368187472,
     "state": "ready", "license_declared": "apache-2.0", "license_reviewed": true,
     "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice", "firered"]},
    {"package": "speaker-diarization-coreml", "environment": "swift", "bytes": 21599417,
     "state": "ready", "license_declared": "cc-by-4.0", "license_reviewed": true,
     "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice", "firered"]},
    {"package": "qwen3-forcedaligner", "environment": "mlx", "bytes": 1276475979,
     "state": "ready", "license_declared": "apache-2.0", "license_reviewed": false,
     "used_by_stacks": ["qwen-1.7b", "qwen-0.6b", "vibevoice"]},
    {"package": "vibevoice-asr-7b", "environment": "torch-vibevoice", "bytes": 17361048135,
     "state": "ready", "license_declared": "mixed: mit + apache-2.0", "license_reviewed": false,
     "used_by_stacks": ["vibevoice"]},
    {"package": "firered-asr2s", "environment": "torch-firered", "bytes": 9583893873,
     "state": "ready", "license_declared": "apache-2.0", "license_reviewed": false,
     "used_by_stacks": ["firered"]}
  ],
  "environments": {"mlx": "ready", "torch-firered": "ready", "torch-vibevoice": "ready",
                   "swift": "ready"},
  "total_known_bytes": 31186708136,
  "unsized_packages": []
}
```

The Hub byte counts are the recorded revision measurements. FluidAudio is different: the manifest
has no pre-build estimate, but pull records the live build bytes, so a ready `list` entry carries
an integer and `unsized_packages` is empty. Its illustrated value varies with the toolchain.
`total_known_bytes` is named for what it is because a total must not silently omit an unsized
package when one is actually present.

`license` became two fields. `license_declared` is what the model card says at the pinned
revision; `license_reviewed` is whether anyone read the terms. Only FluidAudio and
`speaker-diarization-coreml` were read (`model_tests/benchmark/DIARIZATION.md`). One field
could not tell "nobody looked" apart from "the card says apache-2.0 and nobody checked what
that obliges", and a scraped string must not read as a clearance.

```bash
audio packages remove vibevoice-asr-7b
```

```json
{
  "removed": ["vibevoice-asr-7b"],
  "environments_removed": ["torch-vibevoice"],
  "environments_removed_reason": "no other provisioned package targets torch-vibevoice",
  "environments_kept": ["mlx", "swift", "torch-firered"],
  "environments_kept_reason": "qwen3-asr-1.7b-8bit, qwen3-forcedaligner still need mlx; fluidaudio, speaker-diarization-coreml still need swift; firered-asr2s still needs torch-firered",
  "hub_revisions_deleted": ["d0c9efdb8d614685062c04425d91e01b6f37d944", "d149729398750b98c0af14eb82c78cfe92750796"],
  "hub_revisions_not_found": [],
  "hub_revisions_retained": [],
  "hub_cache_note": "weights live in the shared Hugging Face cache, not under this root. Only revisions this root recorded as downloaded and the current manifest still pins for that package are eligible for deletion; pre-existing and out-of-manifest revisions are retained because they may belong to another tool, another provisioning root, or an earlier experiment",
  "reclaimed_bytes": 18000000000
}
```

`remove` takes several names, and it is all-or-nothing: every name is resolved against the
registry before anything is deleted, so a typo among four ids costs nothing but the retry. It used
to delete as it went and then roll the registry back, which left a package whose bytes were gone
still reading as `ready`.

Reference counting cuts both ways here, which is the point of showing it: `torch-vibevoice`
held exactly one package and dies with it, while `mlx` survives because the aligner and the
ASR checkpoint are still provisioned there. Package-to-environment identity comes from the
installed manifest rather than a mutable registry field, so the count cannot be redirected.

`reclaimed_bytes` is measured, not projected. Most of it is a Hub revision rather than anything
under the root. A revision is deletion-eligible only when the receipt records it as downloaded by
this root, the installed manifest still pins it for that package, and it was not pre-existing;
everything else appears in `hub_revisions_retained` and, when nonempty, gets a
`hub_revisions_retained_reason`. Local deletion targets and environment references also come from
the installed manifest. An unknown registry package is dropped without following its paths, and a
failed local deletion leaves its registry owner intact and contributes no reclaimed bytes. A
revision the cache no longer holds appears in `hub_revisions_not_found`, not as deleted.

```bash
audio packages purge --dry-run
```

```json
{
  "would_remove": {
    "packages": ["firered-asr2s", "fluidaudio", "qwen3-asr-1.7b-8bit",
                 "qwen3-forcedaligner", "speaker-diarization-coreml"],
    "environments": ["mlx", "swift", "torch-firered"],
    "root": "/Users/you/Library/Caches/audio-processing-cli",
    "hub_revisions": ["0e1a68e91d815300c7c9754b2a7639378b23db15",
                      "1bb4d285c8456429385d9c0810300df4297bc11b",
                      "1ed7a662fdc7109e36d822db793ee6eebdaf8594",
                      "2304afed56eacfee6256dee5937ed22ffa0b64ec",
                      "7990aaccc6b7aec1e527743bd30201f2c4a03b8c",
                      "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
                      "e448fd967f44182a1c323cc30f5d89f2400c28da"]
  },
  "would_keep": {"hub_revisions": []},
  "hub_cache_note": "weights live in the shared Hugging Face cache, not under this root. Only revisions this root recorded as downloaded and the current manifest still pins for that package are eligible for deletion; pre-existing and out-of-manifest revisions are retained because they may belong to another tool, another provisioning root, or an earlier experiment",
  "reclaimable_known_bytes": 13825660001,
  "reclaimable_note": "projected from manifest-bounded registry receipts and package sizes; it counts deletion-eligible Hub weights and recorded local package artifacts, but excludes retained revisions and environment bytes",
  "unsized_packages": [],
  "untouched": ["user media", "transcript and subtitle outputs"]
}
```

`--dry-run` projects from manifest-bounded receipts, so it reports both the revisions eligible
for deletion and those it will retain, `reclaimable_known_bytes`, and what it could not size. The
real `purge` reports what actually happened:

```json
{
  "removed": {
    "packages": ["firered-asr2s", "fluidaudio", "qwen3-asr-1.7b-8bit",
                 "qwen3-forcedaligner", "speaker-diarization-coreml"],
    "environments": ["mlx", "swift", "torch-firered"]
  },
  "hub_revisions_deleted": ["0e1a68e91d815300c7c9754b2a7639378b23db15",
                            "1bb4d285c8456429385d9c0810300df4297bc11b",
                            "1ed7a662fdc7109e36d822db793ee6eebdaf8594",
                            "2304afed56eacfee6256dee5937ed22ffa0b64ec",
                            "7990aaccc6b7aec1e527743bd30201f2c4a03b8c",
                            "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
                            "e448fd967f44182a1c323cc30f5d89f2400c28da"],
  "hub_revisions_not_found": [],
  "hub_revisions_retained": [],
  "hub_cache_note": "weights live in the shared Hugging Face cache, not under this root. Only revisions this root recorded as downloaded and the current manifest still pins for that package are eligible for deletion; pre-existing and out-of-manifest revisions are retained because they may belong to another tool, another provisioning root, or an earlier experiment",
  "reclaimed_bytes": 14500000000,
  "unsized_packages": [],
  "untouched": ["user media", "transcript and subtitle outputs"]
}
```

The real response deliberately has no `would_remove`, `would_keep`, `reclaimable_known_bytes`,
or `reclaimable_note`. Its `reclaimed_bytes` integer is measured from the managed filesystem and
what the Hub cache actually returned, so its value can differ from the dry-run projection.

```bash
audio packages purge
uv tool uninstall audio-processing-cli
```

Purge before uninstalling: the resolved root otherwise outlives the only tool that knows
how to describe it. Neither `remove` nor `purge` touches `meeting.timed.json`,
`demo.transcript.json`, `field.transcript.json`, or any subtitle file.
