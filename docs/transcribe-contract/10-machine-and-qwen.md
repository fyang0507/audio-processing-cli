
## 0. Once per machine

```bash
brew install ffmpeg
uv tool install .
audio --help
audio doctor
```

`audio doctor` reports tool version and path, ffmpeg/ffprobe/swift/uv presence,
platform, total and available memory, available disk, the resolved root, and
per-environment and per-package state. Absent `swift` is reported, not fatal: it
blocks only the packages that need it.

## 1. Interview — `qwen-1.7b`

Fast long-form transcript with anonymous speaker attribution.

### 1.1 Plan before committing to anything

```bash
audio transcribe plan --input meeting.m4a \
  --stack qwen-1.7b \
  --want diarization
```

Exits 0 whether or not anything is provisioned:

```json
{
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "asr":        {"backend": "qwen3-asr-1.7b-8bit", "environment": "mlx",
                   "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
                   "config": {"batch_size": 1, "clear_mlx_cache_after_every_batch": true,
                              "language": null,
                              "api_path": "_generate_chunks_batched", "max_tokens": 16384},
                   "adapter_strips": ["language <label><asr_text> scaffold"],
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 0.0,
                   "determinism_basis": "sampler=make_sampler(temp=0.0), i.e. argmax decode (model_tests/benchmark/turn_attributed_mlx_asr/inference.py:133-136); back-to-back calls in one process produced byte-identical text, cross-process repetition untested"
                   },
    "diarizer":   {"backend": "fluidaudio", "version": "0.15.5",
                   "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
                   "environment": "swift",
                   "config": {"step_ratio": 0.1,
                              "min_segment_duration": 0.0,
                              "threshold": 0.6, "batch_size": 32},
                   "config_note": "the shipped default supplies no speaker-count prior and enables overlapping_segments only when overlapped_speech is requested; the cited quality figures used both --num-speakers 2 and --overlapping-segments, so they do not measure this request",
                   "selected_by": "add_on_required_by:diarization"}
  },
  "execution": {
    "stage_order": ["decode", "diarizer", "asr"],
    "residency": "one_model_stage_at_a_time",
    "note": "stages run strictly sequentially and no two model stages are resident together; wall time adds across stages, peak memory does not, and the per-stage peaks below must not be summed"
  },
  "capabilities": {
    "diarization": {"satisfaction": "derived", "backend": "fluidaudio",
                    "evidence": {"interface": "verified", "quality": "unmeasured"},
                    "note": "Anonymous labels are reconciled sample-exactly onto the ASR text, plus the diarizer's turn intervals. The shipped no-prior, overlap-off configuration is unmeasured; the 3-of-75 speaker-change result used a two-speaker prior with overlap detection enabled and does not apply to this request."
                    }
  },
  "packages": [
    {"package": "qwen3-asr-1.7b-8bit", "environment": "mlx", "kind": "weights",
     "bytes": 2467859030, "provisioned": false},
    {"package": "fluidaudio", "environment": "swift", "kind": "toolchain",
     "requires_tool": ["swift"], "bytes": null, "provisioned": false},
    {"package": "speaker-diarization-coreml", "environment": "swift", "kind": "weights",
     "bytes": 21599417, "provisioned": false}
  ],
  "total_known_download_bytes": 2489458447,
  "unsized_packages": [],
  "warnings": [],
  "sample_output": {
    "sample": true,
    "note": "shape only; values are placeholders and cardinality is unknown until run",
    "schema_version": 1,
    "complete": true,
    "source": {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"},
    "segments": [
      {"segment_id": "seg_0", "text": null, "speaker": null}
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

`turns` is present because `diarization` was requested, and the two arrive together: the
speaker labels land on `segments[].speaker` and the intervals the diarizer measured land in
`turns`. They are separate arrays because they are separate measurements at separate
granularities — a turn can span several sentences — and neither is derived from the other.
When `overlapped_speech` is requested, a segment intersecting any detected overlap that intersects
the selected document scope keeps its text and any bounds but omits `speaker`, even if that overlap
starts before the range. This attribution mask is intersection-based; publication in
`overlapped_speech[]` and the `overlap` abstention ledger remains start-owned. A known
multi-speaker interval is never collapsed into one anonymous identity merely because diarization
was also requested.

Two things the sample deliberately does *not* contain. Segments carry no `words` array,
because `word_timestamps` was not requested. And they carry no `start` or `end`: those are
`segment_timestamps`, which is exit-2 on this stack, since Qwen emits no segment extents and
the chunk boundaries it works in are not speech timing. A segment exists as a floor artifact;
its time extents do not come free with it. Keys are absent rather than null-valued at the
container level, so absence can never read as a measured value.

`provenance` is shown as a string here purely to keep the example readable. In a
real result it is the full executed plan, and the key-set test must compare against
that, not against this placeholder.

The `abstentions` ledger *is* present, and not as an exception to that rule: it is
a floor artifact rather than a requested capability, so the one-placeholder-per-request
rule does not reach it. It is also
genuinely producible from this exact request, since FluidAudio emits
overlap-permitting output. Whether it fills depends on the audio.

`duration_seconds` is populated because `plan` reads container metadata rather than
stubbing what it already knows. The value `1794.2` is illustrative — this document
has no real `meeting.m4a` — and is deliberately not the 30-minute reference fixture behind
the `capabilities` report's `cost.proved`, which describes a recorded run rather than this input.
On `run`, the same source-audio duration is recomputed from the canonical decode's PCM frame
count so range and coverage arithmetic use the timeline actually processed. `source.path` is the
resolved absolute path of the original media, so a later `export` invocation can protect that
canonical input even from a different working directory. `source` publishes no temporary format,
sample-rate, or channel fields; the working WAV remains an unpublished transport artifact.

There is no `measured` block here, and there was one. It restated the `capabilities` report's timing and
memory figures inside every plan, which duplicated the one place those figures belong now
that a plan cannot be reached without a stack and an input. The ASR stage dominates the cost
on every stack, and the `capabilities` report's `cost.proved` already says what that stage did on a named
sample.

`api_path` is in the config for a reason that is not a detail. Qwen has two decode entry
points and they do not produce the same text: on identical input, weights, and greedy
settings, the public `generate()` path and the private `_generate_chunks_batched` path agreed
on **every word** and differed by **two Chinese commas**, at 242 versus 240 generated tokens.
Lexical evidence therefore transfers between the paths and punctuation does not — and
punctuation is what cue splitting breaks on, so a plan has to say which path ran.

`adapter_strips` records a live instance of the adapter-normalization floor. The private
batched API returns the model's own scaffold inside its text — the observed prefix is
literally `language English<asr_text>` — where the public path strips it. An adapter on the
batched path that forgets to do the same produces a transcript beginning with the scaffold.
The scaffold also carries the model's own language guess, which is where the `languages`
entry's warning against trusting it comes from: on one Mandarin-majority clip 1.7B read
English and 0.6B read Chinese. It is stripped with the rest and never published.

`execution` states what every recorded figure in this document already assumed and no
earlier draft declared. The orchestrator that produced the end-to-end interview
measurement ran *strictly sequential fresh subprocesses*, and its record says in as many
words that the stages did not overlap. Publishing per-stage memory peaks without
declaring that is how a reader ends up summing them. Two consequences worth stating
outright: a request spanning several provisioned environments costs the sum of the
stage walls and the maximum of the stage peaks, not the sum of both; and strict
sequencing is load-bearing for the memory story rather than an implementation detail,
because `vibevoice` at 20.28 GiB and the aligner have never been measured resident at
the same time and nothing here should imply they can be.

Each RSS cell is the total resident footprint of the process executing that stage, including
its runtime and orchestration overhead; it is not allocation attributed only to the backend.
The in-process VAD cell is the core process high-water mark observed through that stage, while
fresh model and Swift stage cells measure those child processes. `peak_mps_live_bytes_by_stage`
is different: it is the MLX allocator's live-device-memory high-water mark inside each MLX
stage, excludes ordinary RSS and non-MLX processes, and is also aggregated with `max`, never sum.

### 1.2 What `run` does with packages absent

```bash
audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization
```

Exit 3. Nothing computed, nothing downloaded, stderr:

```json
{
  "code": "packages_not_provisioned",
  "missing": [
    {"package": "qwen3-asr-1.7b-8bit", "kind": "weights", "bytes": 2467859030},
    {"package": "fluidaudio", "kind": "toolchain", "requires_tool": ["swift"], "bytes": null},
    {"package": "speaker-diarization-coreml", "kind": "weights", "bytes": 21599417}
  ],
  "total_known_download_bytes": 2489458447,
  "unsized_packages": [],
  "fix": "audio packages pull --stack qwen-1.7b"
}
```

### 1.3 Provision, verify, execute

```bash
audio packages pull --stack qwen-1.7b
audio packages verify
audio packages list
audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization \
  --format md
audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization \
  --format json -o meeting.transcript.json
```

The JSON result carries the executed plan as `provenance`, anonymous speaker
labels, and the overlap abstention ledger. Anonymous labels are never mapped to a
person or a role.

### 1.4 Adding timing, and what it costs

Qwen has no native word timing, so `word_timestamps` forces the aligner. It does **not** force
another environment: the aligner runs in `mlx`, the same environment the ASR is already in, so
the cost is 1.19 GiB of weights and one more stage rather than a second runtime. The request
spans `mlx` and `swift`.

That was not true when this document was written, and it is worth saying why it changed. The
aligner was measured through `qwen-asr`, a PyTorch package, and the same alignment is available
from `mlx-audio` — the library the Qwen stacks already pin. Rerunning the recorded case through
it produced identical token text on all 246 aligned tokens
(`model_tests/benchmark/results/2026-08-17-mlx-collapse-probes.json`), so the move costs no
transcript change; bounds agree to a median of 0 s with a P95 of 80 ms, and boundary accuracy
is unmeasured against labels on both paths, which is the same caveat `evidence.quality` already
carries. [ENVIRONMENTS.md](../../ENVIRONMENTS.md) holds the layout.

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b \
  --want diarization,word_timestamps,overlapped_speech,vad
audio packages pull --stack qwen-1.7b
audio transcribe run --input meeting.m4a --stack qwen-1.7b \
  --want diarization,word_timestamps,overlapped_speech,vad \
  --format json -o meeting.timed.json
```

`word_timestamps` arrives `derived` with `evidence.quality: "unmeasured"`, because
boundary MAE/P95 has no labels.

The roles and packages this adds to §1.1 — two of the four roles not yet shown, plus
the one package that auto-fetches:

```json
{
  "roles": {
    "vad":     {"backend": "silero-vad", "environment": "core",
                "version": "silero-vad-6.2.1",
                "config": {"threshold": 0.5, "exit_threshold": 0.35,
                           "min_speech_ms": 100, "min_silence_ms": 300,
                           "speech_pad_ms": 120},
                "selected_by": "add_on_required_by:vad"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "mlx",
                "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
                "config": {"scope": "all_segments",
                           "language_rule": "Chinese when text matches [一-鿿], otherwise English; the ASR --language hint is never forwarded"},
                "selected_by": "add_on_required_by:word_timestamps"}
  },
  "capabilities": {
    "vad": {"satisfaction": "derived", "backend": "silero-vad",
            "evidence": {"interface": "verified", "quality": "measured"},
            "note": "0.8505 frame-level F1 at 0.7655 precision and 0.9567 recall, against the union of 83 annotated utterance intervals on one 150-second fixture. That is an activity gate for this exact five-value configuration and says nothing about language coverage, turns, overlap, or chained VAD-plus-ASR behaviour."
            }
  },
  "packages": [
    {"package": "silero-vad", "environment": "core", "kind": "weights",
     "bytes": null, "provisioned": true, "auto_fetch": true,
     "note": "hash-pinned single file; fetched on first use, so it never returns exit 3"},
    {"package": "qwen3-forcedaligner", "environment": "mlx", "kind": "weights",
     "bytes": null, "provisioned": false}
  ]
}
```

`vad` is the one derived capability with a real number behind it, so the
whole recorded configuration is declared rather than just the threshold: precision is
0.77 against a union-of-speakers reference, which means the gate over-includes, and
that is a property of these five values together. Declaring one of them and citing the
F1 of all five is how a measured figure ends up attributed to a configuration that
never produced it.

`silero-vad` is the one package that can be `provisioned: true` on a machine that
never ran `pull`, because it is the small hash-pinned artifact the existing Silero
backend already fetches on demand. Everything else fails closed.

`--vad` selects among the implementations offered as add-ons: `silero-vad` today, plus
FluidAudio's Core ML VAD once that ships as a package. FireRed's own VAD is inside its
stack and is not offered to other stacks. That makes `vad` the only role with a real
choice, which is why the pin is defined there while `--diarizer` remains a forced
single implementation with a pin reserved for later ones.

### 1.5 Smaller stack, named directly

`qwen-0.6b` is a separate package from `qwen-1.7b`, so it needs its own pull —
without it this exits 3:

```bash
audio packages pull --stack qwen-0.6b
audio transcribe run --input meeting.m4a --stack qwen-0.6b --want diarization \
  --language Cantonese --format md
```

Measured on the 30-minute Cantonese SpiCE fixture, Apple M4 Max / 64 GiB, batch 1
with the MLX cache cleared after every batch: `qwen-0.6b` ran 29.90 s at 1.66 GiB RSS
versus `qwen-1.7b` at 53.77 s and 3.02 GiB, and scored 52.64% mixed-token error
versus 33.56%. Those are ASR-stage walls on identical diarized turns, and the accuracy
comparison is Cantonese-only.

`--language Cantonese` is on that command deliberately: it is the configuration those
figures were measured under, and the flag exists so a caller who knows the language
can state it instead of leaving the model to guess. The plan echoes it as
`roles.asr.config.language`, so the executed provenance records which of the two
configurations ran. Nothing else in the plan changes — a hint is an input to the ASR,
not a capability, so it adds no role, no package, and no output field.

`qwen-0.6b` is not simply a smaller `qwen-1.7b`. On the 139.284-second probe it
rendered `刷啥子` where 1.7B retained `耍啥子`, so its `verbatim` catalog entry carries
`quality: "refuted"` with that `observed_limit` while 1.7B's carries `unmeasured`.
Identical resolution, different recorded fidelity — which is why the two share a
column in the derivation table and not a catalog. Their filler retention is identical
at 26 hits, so the interface half of `verbatim` is the same on both; only the dialect
half separates them.
