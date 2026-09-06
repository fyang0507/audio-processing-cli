
## 5. Refusals

Each of these is shown as a caller sees it, with the corrected command, in [TRANSCRIBE_HAPPY_PATH.md](../TRANSCRIBE_HAPPY_PATH.md) §4.

```bash
audio transcribe run --input meeting.m4a --want diarization
```

Exit 2, `code: "stack_required"`, `field: "--stack"`, `allowed` listing the four stack ids, and `stacks` mapping each to a one-line characterization. `fix` asks the caller to choose the stack and repeat the original command; it does not insert a default stack or change `run` to `plan`. Missing-input guidance likewise names the missing original media path without inventing a filename. These semantic refusals remain bare JSON on stderr for all three commands.

```bash
audio transcribe plan --stack qwen-1.7b --want diarization
```

Exit 2, `code: "input_required"`, `field: "--input"`, and a `note` saying why a stack alone cannot be planned: the partition, the unit count, the projected cost, and whether a failure is recoverable are all properties of the input. Reported by `plan`, not only by `run`, because `plan` is the command whose whole job is to answer those questions.

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want segment_timestamps
```

Exit 2, `code: "capability_unsatisfiable_on_stack"`, `capability: "segment_timestamps"`, `allowed: ["vibevoice", "firered"]`, plus `available_on_stack` listing everything Qwen would accept instead. Reported by `plan` as well as `run`, before anything loads, and the fix is actionable: switch stacks. Qwen's only time-like output is the processing container, and promoting that to a segment extent is the fabrication this code exists to prevent.

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --want word_timing
```

Exit 2, `code: "capability_unknown"`, `field: "--want"`, `provided: "word_timing"`, `did_you_mean: "word_timestamps"`, and `available_on_stack` listing every name this stack accepts. A misspelling and an unsatisfiable requirement are different failures: this one has no `capability` field, because the name is not one.

```bash
audio transcribe plan --input demo.mp4 --stack vibevoice --want verbatim --language Cantonese
```

Exit 2, `code: "option_unsupported_on_stack"`, `field: "--language"`, `provided: "Cantonese"`, `allowed: []`, `stacks_accepting: ["qwen-1.7b", "qwen-0.6b"]`. VibeVoice takes no language argument, so the alternative to refusing is accepting a flag that does nothing — which would let a caller believe it had constrained a decode it had not touched.

```bash
audio transcribe plan --input meeting.m4a --stack qwen-1.7b --language EN
```

Exit 2, `code: "option_value_unsupported"`, `field: "--language"`, `provided: "EN"`, `allowed` containing the exact 30-name Qwen vocabulary, and `did_you_mean: "English"`. This is distinct from `option_unsupported_on_stack`: Qwen accepts the option, but not that value. `english` is accepted case-insensitively and normalized to `English` in the plan.

```bash
audio transcribe plan --input meeting.m4a --stack firered --want token_lid
```

Exit 2, `code: "capability_unsupported"`, `capability: "token_lid"`, `allowed: []`, `reason: "no_backend_declares"`. An impossible request is an error rather than a silently `unavailable` field, so an agent gets an explicit answer instead of assuming code-switching support implies per-token labels. `capabilities` is where this is *discovered* without erroring.

```bash
audio transcribe plan --input meeting.m4a --stack vibevoice \
  --want diarization --diarizer fluidaudio
```

Exit 2, `code: "pin_conflicts_with_native_capability"`, `field: "--diarizer"`, `provided: "fluidaudio"`, `allowed: []`, `capability: "diarization"`, because `vibevoice` satisfies it natively and no diarizer role exists in this plan. Pins select among implementations of a role the plan actually contains.

A malformed or nonintersecting resume range is a request error, not an FFmpeg failure:

```json
{
  "code": "range_invalid",
  "field": "--range",
  "provided": "400:",
  "reason": "range does not intersect the source duration",
  "fix": "correct --range using the reported reason; keep the intended source interval; repeat the original command, preserving every other argument"
}
```

A backend crash is the one failure that is not a refusal:

```json
{
  "code": "backend_failed",
  "role": "asr",
  "backend": "vibevoice-asr-7b",
  "detail": "MPS backend out of memory during generate",
  "fix": "retry with --stack qwen-1.7b --want word_timestamps, or free memory; the plan's measured_peak_exceeds_target warning applies"
}
```

Exit 1, and no result is written. This must stay distinguishable from an abstention, which is a *successful* run that declines to assert something: exit 0, a result, and a ledger entry. Collapsing the two would make a crash and a principled refusal look identical to a caller.

A corrupt package is a third thing again — provisioned, but not usable:

```json
{
  "code": "package_integrity_failed",
  "failed": [
    {"package": "vibevoice-asr-7b", "check": "hub_snapshot_integrity",
     "expected": "snapshot directories, manifest-filtered files, and recorded byte size",
     "actual": ["Qwen/Qwen2.5-7B is missing allow_pattern tokenizer.json"]}
  ],
  "fix": "audio packages pull --repair vibevoice-asr-7b"
}
```

Exit 3, and nothing loads. For Hub materializations, both explicit `verify` and run preflight require the cache index to bind each repository and pinned revision to the recorded snapshot path, all manifest `allow_patterns`, and the tree-byte total recorded at pull. These are cheap live checks, not a fabricated weight digest: a same-size content mutation can still pass and then fail at model load as `backend_failed` exit 1. The hash-pinned URL artifact has a different boundary: pull accepts it only at the exact manifest-derived models path as a contained, non-symlink regular file, including after download; Silero's runtime auto-fetch applies the same rule. Source-backed native packages additionally inspect the live Git HEAD, derive the exact tracked file set from the installed patch, require every manifest-owned post-patch SHA256 and the same exact values in the receipt, and reject ordinary and Git-ignored untracked artifacts. Ignored bytecode and extensions remain importable, so they cannot hide behind `.gitignore`. A legacy receipt may record `checkout_commit` as the manifest's short `commit` alias or its full `resolved_commit`; new pulls record the full value, and the live HEAD must always equal the full resolved commit. Run preflight also launches every selected managed interpreter and requires exactly one contained, non-symlink FluidAudio product from its exact managed checkout before decode. Before provisioning, freeze, interpreter launch, or decode, every managed environment root is independently derived from the manifest and must be the exact non-symlink directory resolving under the provisioning root; a normal venv `bin/python` symlink is allowed inside that trusted root. `pull` also refuses a redirected `envs` parent before creation or installation. `verify` reports a redirected ready root as `drifted`, and run refuses it before decode rather than following the registry or filesystem redirect. A runtime abort's `fix` is deliberately a sentence directing the caller to the reported condition; a package verification command can legitimately print `ok` after an OOM and therefore cannot be advertised as its repair.

A built package has a fourth failure of its own: the build succeeded and the executable it produced does not launch.

```json
{
  "code": "package_build_unusable",
  "package": "fluidaudio",
  "product": "fluidaudiocli",
  "built": true,
  "fix": "audio packages pull --repair fluidaudio"
}
```

Exit 3, and the registry entry stays `pulling`, so `list` and `run` both report the package absent rather than ready. `built: true` is kept because it is the useful half of the diagnosis — a compile failure and a product that will not start need different responses, and this is the second. The condition was real: the product name was hard-coded as `fluidaudio` while `Package.swift` at the pinned commit declares `fluidaudiocli`, so `swift run` refused a perfectly good build. It is pinned in the manifest beside the commit now, and `pull` refuses instead of recording `product_runs: false` and exiting 0.

### 5.1 Incomplete runs

Long-form transcription has partial-completion mechanisms that are real and recorded, not hypothetical. Two are visible in the harness today:

- **The Qwen path carries a global generation budget** and stops when it runs out. The stage records every turn it finished; the partial document retains the longest chronological prefix so its resume is disjoint. The recorded runner tracks `input_turns`, `processed_turns`, and `unprocessed_turns`, and every recorded run has `unprocessed_turns: []` — but the 60-minute stress run consumed 10,169 of 16,384 tokens, so at that token rate the budget exhausts somewhere near **1.6 hours** of comparable material. A three-hour interview runs into this before it runs into anything else.
- **VibeVoice has a single generation cap** and a `hit_max_new_tokens` detector, applied to one `generate` call over the whole file. Recorded runs set it between 1,024 and 16,384 depending on fixture. Hitting it truncates the transcript.

Add out-of-memory to those — the product-demo route measured 20.28 GiB live MPS on thirty minutes and OOMs at model load under a strict 16 GiB cap — and interruption, and a failure in any one stage of a five-stage chain.

**What a partial run must leave behind.** On a stack whose work is partitioned, an incomplete run writes its result and exits 4 rather than throwing the work away:

```json
{
  "code": "run_incomplete",
  "role": "asr",
  "backend": "qwen3-asr-1.7b-8bit",
  "detail": "global generation budget exhausted after 148 of 195 turns",
  "coverage": {
    "scope_intervals": [[0.0, 1794.2]],
    "covered_through_seconds": 1402.88,
    "covered_fraction": 0.782,
    "covered_intervals": [[0.0, 1402.88]],
    "missing_intervals": [[1402.88, 1794.2]],
    "units_total": 195,
    "units_completed": 148
  },
  "output": "meeting.partial.json",
  "fix": "audio transcribe run --input meeting.m4a --stack qwen-1.7b --want diarization,word_timestamps --language Cantonese --range 1402.88: -o meeting.rest.json"
}
```

Four properties that matter more than the shape.

`scope_intervals` records the complete source-timeline span of the processing units this invocation selected. A fixed-unit ordinary Qwen run covers the whole source; FireRed can begin at its first native VAD region and end at its last, leaving leading or trailing silence outside the processing ledger. For a ranged run it exactly matches the recorded `selected_unit_scope`. Covered and missing intervals partition that scope exactly; material outside it is neither claimed nor counted in `covered_fraction`.

`covered_through_seconds` is the end of the longest **contiguous prefix** that is fully transcribed, which is the number an agent can act on without reasoning about gaps. It is not the same as "the last unit that finished": the runner processes turns in duration-bucketed order and restores chronological order afterwards. On an incomplete Qwen run, completed units after the first gap are deliberately omitted from the partial document and counted as missing; the emitted resume therefore produces a disjoint suffix. `covered_intervals` and `missing_intervals` still carry the exact artifact truth, and the schema can represent non-contiguous coverage for later partitioned backends without pretending it is a prefix.

`--range <start>[:<end>]` is the resume mechanism, and it exists so the caller or downstream consumer does **not** pre-clip the audio. A user-supplied clip would shift the timeline and force every bound in the second result to be re-offset by hand before merge — consumer timestamp arithmetic that the canonical-timeline floor exists to prevent. The CLI may still make an unpublished internal clip when a whole-media backend requires one: ranged VibeVoice generation runs against that transport clip, then the adapter adds the validated range start exactly once to every relative segment/event bound before anything becomes public. Its aligner receives those restored source-timeline bounds, the result's `source.path` still names the original canonical media, and no temporary clip path escapes. With `--range`, the consumer therefore receives bounds already on the original timeline and merging remains concatenation.

A ranged run also records the selection in the embedded executed plan:

```json
{
  "range": {
    "requested": [1402.88, 1794.2],
    "selected_unit_scope": [1402.88, 1794.2]
  }
}
```

`requested` is the validated interval (an open end resolves to the canonical WAV duration), while `selected_unit_scope` can expand to whole processing-unit boundaries. On an incomplete Qwen run, only the longest chronological prefix is published; later duration-bucketed successes are rerun so the `--range` continuation is disjoint rather than asking export to guess duplicates. Auxiliary VAD, overlap, and abstention spans use the same expanded scope and are owned by their start. An open range reaches the canonical duration even when the last diarized turn ends earlier, so tail evidence belongs to the continuation instead of disappearing between partial documents. For an incomplete hand-written range that begins in a diarization gap, coverage remains bounded to the selected processing units while auxiliary ownership begins at the requested bound; the gap's evidence is preserved without claiming it was transcribed.

The partial result is a **conforming result document** with `complete: false` and the same `coverage` block, so every floor still holds inside it: no synthesized bounds, abstentions survive, punctuation invariant intact for the units that ran. It is not a debug dump. If `units_completed` is zero, `fix` is deliberately a sentence rather than a `--range` command: the deterministic run has no later unit to skip to, so replaying the same request cannot be presented as a remedy.

Ids are document-scoped, so merging two results means re-numbering. `export` accepts several transcripts in timeline order and re-ids as it goes, which covers the subtitle case without anyone hand-editing JSON:

```bash
audio export --input meeting.partial.json --input meeting.rest.json \
  --format srt -o meeting.srt
```

One stack has a deliberate exception. VibeVoice's anonymous native speaker labels are local to each independent generation: `Speaker 0` in a resumed range is not evidence for the same person as `Speaker 0` in the partial run. Multi-input VibeVoice results that requested `diarization` are therefore refused as `export_inputs_incompatible`; export them separately or rerun the desired ranges together as one generation. Single-input export and multi-input VibeVoice results without native diarization remain supported.

**On `vibevoice`, recovery is `prefix_only`.** The upstream parser yields no structured result when generation stops inside an unterminated segment, so the adapter must parse the raw output and retain every complete segment before the cut. That prefix is a conforming result with `complete: false`, a coverage watermark at its end, and a `--range` fix for the remainder. This does not turn every failure into a partial result: an OOM at model load has decoded no prefix, writes no result, and remains exit 1.
