# `transcribe` happy paths — per use case

**Status: all four execution stacks and export ship.** Every `run` shape and the refusal blocks
are exercised against real serializer output. This is the unabridged
step-by-step an implementer can diff against and an agent can read as a
worked example. [TRANSCRIBE_CONTRACT.md](TRANSCRIBE_CONTRACT.md) is organized around
*why* the surface looks like this and abridges output to whatever differs; this document
does the opposite — it shows every command in order and the complete stdout of each, with no
design commentary. §4 covers the refusals, because what an agent actually sees includes being
told it was wrong, and a refusal that cannot be acted on is a worse outcome than a slow run.

Three use cases, one per recommended stack row in
[model_tests/DECISION_REPORT.md](model_tests/DECISION_REPORT.md):

| § | Use case | Stack | Deliverable |
| --- | --- | --- | --- |
| [1](#1-fast-long-form-transcription--interview) | Fast long-form transcription | `qwen-1.7b` | Speaker-attributed transcript + SRT |
| [2](#2-video-editing--product-demo) | Video editing | `vibevoice` | Verbatim segments with word timing + VTT |
| [3](#3-dialect-field-recording) | Dialect field recording | `firered` | Audited transcript with native word timing + region LID |

## How to read the output blocks

Every value below is one of three things, and they are never mixed:

- **Measured.** Traceable to a recorded artifact. All of these appear inside
  `cost.proved`, the `capabilities[].note` sentences, manifest-fixed package byte estimates,
  revisions, and configuration. Every one is checkable against `model_tests/`.
- **Illustrative.** Transcript text, speaker labels, timestamps, cue text, and input
  durations. These are made up. They are internally consistent — bounds are monotonic,
  words fall inside their segment, and nothing exceeds the source duration. The runtime
  validator additionally accepts the recorded FireRed integer-millisecond seam where a
  final word ends at most 1 ms after its sentence; none of these illustrative blocks use
  that allowance. The shapes are therefore usable, but no number here is a claim about any
  real recording. Input durations
  borrow the real local fixture durations (1794.2 s is illustrative; 112.4 s and 27.8 s
  match actual local media) purely so they are plausible. Machine-specific FluidAudio build and
  teardown-byte examples are also illustrative; their integer fields are structural, while a live
  pull or teardown measures the value for that root.
- **Structural.** Key sets, nesting, enum members, and which fields are absent. These
  are the part that must match a real run exactly, and they are the reason this document
  exists.

Arrays are printed short. A real 30-minute interview yields hundreds of segments, not
two. Cardinality is the one thing a plan's `sample_output` explicitly does not promise,
so it is not promised here either — but note that the arrays contain no ellipsis
placeholder, because a `"...": "..."` pseudo-key would corrupt the key-set comparison
this document is meant to support.

### Names this document introduces

The contract pins the plan document and the envelope of a result. It does not name every
result field, because it never printed a full result. The following settled names appear here
first:

| Name | Carries | Requested by | Shown below |
| --- | --- | --- | --- |
| `segments[].words[]` with `word_id` | word text and bounds | `word_timestamps` | §1.4, §2.2, §3.1 |
| `vad_regions[]` | speech-activity regions | `vad` | §3.1 |
| `lid_regions[]` | region language labels | `lid` | §3.1 |
| `turns[]` with `turn_id` | speaker turn intervals | `diarization` | §1.2, §1.4, §2.1, §2.2 |
| `overlapped_speech[]` with `overlap_id` | cross-speaker overlap intervals | `overlapped_speech` | not exercised |

`overlapped_speech[]` is the one result key this document does not demonstrate: none of the
three use cases requests it. `TRANSCRIBE_CONTRACT.md` §1.4 is where it is requested.
When it is requested, any transcript segment intersecting a detected overlap that intersects this
document's selected scope keeps its text and bounds but omits `speaker`, even when the overlap began
before the range. Masking is intersection-based, while the overlap interval and `overlap`
abstention themselves remain start-owned. The tool does not turn known multi-speaker activity into
one anonymous attribution.

`turns[]` arrives with `diarization` rather than as a separate request, because there is no
use for one without the other — labelling a transcript "three people spoke here" while
withholding who said what answers nothing, and intervals-without-text was never purchasable
anyway, since punctuated text is a floor rather than a capability. Both come out of the same
diarizer run, so splitting them would have priced one stage twice. Note that the turn bounds
in §1.4 are not the segments' word bounds: the diarizer measured them independently, and the
plan keeps both rather than deriving one from the other.

Ids are positional and **document-scoped** (`seg_0`, `w_0`, `turn_0`, `ab_0`), extending
the `segment_id`/`abstention_id` convention the contract's `sample_output` already shows.
They are deliberately *not* stable across runs, and nothing should be built on the
assumption that they are: the CLI offers no way to re-run the same input under the same
parameterization for a comparable result, so cross-run identity would be a guarantee with
no caller. The practical consequence is that merging two result documents — a partial run
plus its resumed remainder — means re-numbering, which is why `export` accepts several
transcripts in timeline order rather than expecting a consumer to splice JSON.

Everything a capability was not requested for is **absent**, not null. That is the
anti-fabrication guarantee, and it is why §1's segments carry no `start`/`end`.

## 0. Once per machine

```bash
brew install ffmpeg
uv tool install .
audio doctor
```

```json
{
  "tool": {"version": "0.1.0", "python": "3.12.12",
           "path": "/Users/you/.local/bin/audio"},
  "platform": {"system": "Darwin", "release": "25.5.0", "machine": "arm64"},
  "tools": {
    "ffmpeg": {"path": "/opt/homebrew/bin/ffmpeg", "present": true},
    "ffprobe": {"path": "/opt/homebrew/bin/ffprobe", "present": true},
    "git": {"path": "/opt/homebrew/bin/git", "present": true},
    "huggingface_hub": {"path": null, "present": true},
    "swift": {"path": "/usr/bin/swift", "present": true},
    "uv": {"path": "/Users/you/.local/bin/uv", "present": true}
  },
  "memory": {"total_bytes": 68719476736, "available_bytes": 30585126912,
             "note": "Host-wide counters; not process-attributable and not summable with per-stage peaks."},
  "disk": {"total_bytes": 994662584320, "free_bytes": 107730939904},
  "root": "/Users/you/Library/Caches/audio-processing-cli",
  "root_exists": true,
  "registry": "/Users/you/Library/Caches/audio-processing-cli/registry.json",
  "environments": {
    "mlx": {"state": "absent", "python": "3.13.9", "requires_tool": [],
            "provisional": false, "blocked_by_missing_tool": []},
    "swift": {"state": "absent", "python": null, "requires_tool": ["swift"],
              "provisional": false, "blocked_by_missing_tool": []},
    "torch-firered": {"state": "absent", "python": "3.12.12", "requires_tool": [],
                      "provisional": false, "blocked_by_missing_tool": []},
    "torch-vibevoice": {"state": "absent", "python": "3.12.12", "requires_tool": [],
                        "provisional": true, "blocked_by_missing_tool": []}
  },
  "packages": {
    "firered-asr2s": "absent", "fluidaudio": "absent",
    "qwen3-asr-0.6b-8bit": "absent", "qwen3-asr-1.7b-8bit": "absent",
    "qwen3-forcedaligner": "absent", "silero-vad": "absent",
    "speaker-diarization-coreml": "absent", "vibevoice-asr-7b": "absent"
  },
  "note": "Swift is required to build or repair FluidAudio; a ready built product runs directly without Swift. Missing provisioning tools are reported rather than fatal."
}
```

Exit 0, and unlike every other block in this document this one is **real output**, not a mock —
`doctor` is implemented, so its shape is checked against a live run by
`tests/test_shipped_commands_match_the_document.py` rather than maintained by eye — as are the
`packages list` block in §5 and the `packages verify` block in §1.3, the other two commands that
ship. Values remain illustrative: paths, versions, and the memory and disk counters are this
machine's.

`swift` is present here. Absent before FluidAudio has been built, it would report `present: false`
and appear in the `blocked_by_missing_tool` list of the `swift` environment — reported rather
than fatal, blocking provisioning or repair of the package that needs it. Once a ready built
product exists, runtime executes that binary directly and the environment is no longer blocked
by Swift leaving `PATH`. That per-environment list is why there is no top-level `warnings` array:
a blocked environment says so where a caller is already looking.

`environments` reports the four *provisioned* environments and not `core`, which is the CLI's own
and always present. `packages` is a state per package rather than a count, because a caller
deciding whether to `pull` needs to know *which* are absent, not how many.
