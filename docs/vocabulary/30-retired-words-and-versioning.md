
## Retired and reserved words

- **route**, **recipe**, **preset** — not CLI concepts. `route` remains prose in
  DECISION_REPORT for describing recommendations to people. `stack` was
  previously retired alongside them and is now a defined term above; use it only
  in that sense, not for a runtime-footprint claim.
- **feature** — not used on the request axis; it stays available for acoustic
  features. Issue #1 §2.2's `analyze --features` should become a capability list.
- **component** — Issue #1 §4.3 and §5 use it for what this document calls a
  role. Prefer `role`, `backend`, or `add-on` in new work.
- **profile** — reserved for `enhance`'s conformance target, which declares
  loudness and true-peak bounds to conform to. It is not a stack or a backend
  selection and does not migrate to `transcribe`.
- **view** — reserved for Observation Store projections (`inspect --view`).
- **model** — too coarse for the schema and wrong for the command surface: the
  provisioned set includes a Swift build product, one Core ML model package, and
  two dependency environments containing no weights. Prefer `package`, `stack`, or
  `backend`. The command group is `audio packages`.
- **speaker_attribution**, **turn_bounds**, **overlap_intervals**, **speech_bounds**,
  **word_bounds**, **segment_bounds**, **region_language**, **token_language** — the first
  draft's coinages, replaced by the field's own terms. `speaker_attribution` and
  `turn_bounds` both became `diarization`, which produces both; the rest map to
  `overlapped_speech`, `vad`, `word_timestamps`, `segment_timestamps`, `lid`, `token_lid`. The old names drew a method/outcome
  distinction that reads well in a design document and creates a private dialect
  everywhere else. Where the standard term is ambiguous, qualify it rather than invent.
- **language_input** — folded into the `languages` capability. A block saying a stack accepts
  a language hint, without saying which languages it handles, gave a caller nothing to act on.
- **add_on_cost**, **stage_cost**, **measured_envelope**, **timing_precision**,
  **alternative**, **interface_basis**, **measured_limit**, **observed_limit**,
  **shares_stage_with**, **produces**, **roles_included**, **roles_conditional** — catalog
  fields collapsed into one `note` sentence per capability. Each existed to hold a string or a
  null that no caller dispatched on. `measured_limit` and `observed_limit` survive as a
  *writing rule* rather than as keys: the sentence must distinguish a refuted property from an
  unmeasured one.
- **container_bounds**, **container_language**, **provenance_only** — removed outright, not
  renamed. They published a stack's internal chunk extents and its own language guess on
  the theory that auditability justified the space. A caller can act on neither, and their
  removal took a refusal code with it: a capability that does not exist cannot be one that
  exists but may not be asked for.
- **punctuation** — not a capability, and not a flag. See Floors.
- **word_confidence** — retired as a capability. FireRed was the only claimed
  source and it emits no per-word confidence: every word is exactly
  `{start_ms, end_ms, text}` across 12,370 recorded tokens
  (`fireredasr2system.py:181-184`). What it does emit is `asr_confidence` per
  *sentence*, which is a different granularity and is not requestable in v1. Do
  not reintroduce the name for the sentence value.
- **unit_count_known_at_plan_time** — retired because `processing` already has the fixed
  `unit_count` field. Its unknown value is `null`; a second boolean can only disagree with it.
- **none** under `failure_recovery.partial_results` — retired when VibeVoice gained
  `prefix_only` salvage. Failures that produce no complete prefix still write no result; they
  do not need an enum member in a catalog of what partial results can contain.

## Versioning

Only the durable artifact is versioned. A result document carries `schema_version`, because it
is saved, re-read, and passed to `export` long after the run. A plan and a catalog carry no
version: they are answers to a command, and the tool version that printed them is what a
consumer would need anyway. Three version fields on three documents implied three independent
compatibility surfaces where there is one.

A plan also does not echo the request back. The caller just typed it, `source.path` records the
input, and the requested capabilities are the keys of the `capabilities` block — so a `request`
field restated three things the document already contained. Catalog and plan paths preserve the
caller's spelling; a durable run result resolves `source.path` to an absolute identity so
`export` can refuse to overwrite canonical media even from another working directory. What a
saved result does need, and could not otherwise state plainly, is which stack produced it, so
`provenance.stack` carries that one value.

Stacks are named, not versioned: a different model size is a different stack id.
`enhance --profile transcription@3` keeps its own versioning.
