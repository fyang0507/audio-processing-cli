# Offline report details and comparison

`audio report summary REPORT --metrics --evidence-limits` adds recorded measurements and a flat index of limits to the existing summary. `audio report compare LEFT RIGHT` compares two saved reports by recorded identity and time overlap. Both read report files only; they never open recorded source/output media paths, run detection, change processing decisions, or rewrite reports. The public callables `audio_cli.pipeline.summarize_report` and `audio_cli.pipeline.compare_reports` own this offline enhancement-report policy in [`pipeline/reports`](../src/audio_cli/pipeline/reports/).

## Optional summary details

```python
summarize_report(path: Path, *, include_metrics: bool = False, include_evidence_limits: bool = False) -> dict[str, Any]
```

The default `audio_enhancement_summary` version `"1"` projection remains unchanged. Each keyword independently adds the following fields to the existing summary:

| Keyword | Additional fields |
| --- | --- |
| `include_metrics=True` | `measurements` with the recorded `before`, `predicted`, and `after` blocks; each contains its block pointer and any recorded `program_actual`, `regional`, `duration_delta_ms`, and `duration_basis` values. `regions` is included only when recorded, providing the original intervals for regional measurements. |
| `include_evidence_limits=True` | `evidence_limits`, a flat index of recorded limits under stages, unresolved rows, rule evaluations, timeline verification, and peak validation. |

A copied metric block or region uses `{"value": <recorded value>, "report_pointer": "/exact/location"}`. Pointers refer to the original file named by `report`, including escaped JSON Pointer segments. Values are copied without rounding or remeasurement. Raw loudnorm `program.output_*` fields describe a hypothetical normalization pass and are never substituted for actual measurements. Legacy reports missing `program_actual` retain only their available blocks and pointers; no actual measurement is inferred from their raw diagnostics. Absent phases, regions, or metrics stay absent, and prediction never becomes encoded-output evidence.

Each evidence-limit row has `report_pointer` and the complete recorded `evidence` object. The flat index includes `abstained`, `abstained_overlap`, `outside_target`, `bounded_outside_target`, `failed`, `rejected`, `not_run`, `skipped`, and `unmeasured` statuses, plus a recorded `target_attained: false`. It includes nested limits beneath applied parents, such as RNNoise speech-preservation abstention and rejected noise-reference intervals. It does not change the parent's operation status. Stage operations can repeat component evidence; their distinct pointers are retained rather than silently deduplicated.

Where available, `context` retains the nearest recorded stage/component, time/frequency scope, start/end, measurement phase, reference basis, and speech reference, each with its own pointer. These pointers distinguish inherited treatment scope from a candidate reference interval or a child-specific scope. `region_basis` and `region_scopes` link to the recorded basis and manifest intervals when a report-local region ID resolves. No missing scope, measurement phase, or reference is guessed. An empty index means no indexed recorded limits, not an overall quality verdict.

## Compare two reports

```python
compare_reports(left: Path, right: Path) -> dict[str, Any]
```

Inputs may be two saved `audio_inspection` or `audio_enhancement_report` documents, schema version `"1"`, in either order. The result is `audio_report_comparison`, schema version `"1"`, with `compatibility`, `left`, `right`, and, when both reports supply region manifests, `region_comparison`. The function raises `PipelineError` for invalid reports or unestablished/incompatible identity; callers can use the existing enhancement error envelope and exit 2.

Compatibility requires recorded SHA256 identity and decoded timeline metadata: `duration_basis: "decoded_pcm"`, `time_origin: "first_decoded_sample"`, nonnegative integer `sample_count`, and positive integer `sample_rate_hz`. Recorded duration seconds, when present, must agree with the sample ratio to its published microsecond precision. Same-source reports must have the same source digest and decoded duration. Historical or moved media paths do not affect compatibility. Legacy reports without sufficient identity/timeline evidence refuse comparison; they remain eligible for the existing summary projection.

An enhancement report also bridges to an inspection of its delivered output when `enhancement.output.sha256 == inspection.source.sha256`. That bridge requires a rendered, non-dry-run report, `timeline_preserved: true`, matching decoded output/inspection durations, and a recorded passing `decoded_audio_duration_only` verification. The source/output duration difference must satisfy that report's recorded tolerance. The comparator consumes this saved check; it does not set a new media policy. `compatibility.identities` and `compatibility.timeline_verification` carry side-qualified exact evidence pointers. The latter preserves any recorded content-alignment and A/V-sync abstentions. A digest link and duration check establish neither content alignment nor A/V synchronization.

Each side retains its report path, source identity, profile and region basis when present, actual metric views, evidence limits, and recorded rule evaluations. Enhancement measurements retain their `before`/`predicted`/`after` names. Inspection measurements use `inspected`, with `program_actual` and pointed `observations`; inspection regional evidence is in observations rather than an invented enhancement-style `regional` block. Recorded relative-to-speech values and their reference names are copied exactly. Each side's `speech_reference` independently retains its speech intervals, available aggregate speech measurements, and detection basis.

`region_comparison.basis` is `positive_time_interval_overlap`. `overlaps` lists every positive intersection with its original left/right region, available measurements, intersection interval, `same_interval`, `same_kind`, and `ambiguous`. Ambiguity means either interval overlaps multiple intervals on the other side; no arbitrary best match is chosen. `left_without_overlap` and `right_without_overlap` retain regions without any positive intersection. Touching endpoints alone do not overlap. Region IDs join a measurement to its own report's manifest only; IDs never match regions across reports. Cross-kind overlaps remain visible, so changed classification does not disappear from the comparison.

These are comparisons of recorded measurements and detection scopes. A changed interval or classification is not a semantic regression or proof that an event appeared/disappeared. Regional values over different intervals, or relative to different aggregate speech scopes, are retained separately without a fabricated metric delta or quality verdict. Missing region manifests suppress `region_comparison` rather than manufacturing an empty match result.

Both functions reject malformed JSON, duplicate keys at any depth, nonfinite numbers (including exponent overflow), unencodable Unicode, unsupported kinds/versions, and nonregular input files. Requested metric views additionally validate actual metric/region structures. Validation completes before returning machine output.

## RNNoise guide calibration metadata

[`calibrate_guide_input`](../src/audio_cli/dsp/denoise/guide.py) retains its exact existing gain calculation: `min(target_rms_dbfs - level, peak_limit_dbfs - peak, maximum_gain_db)`. It adds `target_attained`, determined from the unrounded resolved gain, and `limiting_reasons`, listing every bound that equals the chosen gain and is strictly below the requested gain. The reason names are `peak_headroom` and `maximum_gain`, in that deterministic order. Tied limiting bounds both appear; bounds tied at the target do not prevent attainment and yield an empty list. Metadata neither feeds back into processing nor claims delivered-media quality.

Attenuation and a no-op can attain the target. A no-op constrained below the target reports the applicable limit. An unattained target can round to the displayed target at six decimals; attainment still reflects the original calculation. The no-energy path retains its existing `status: "abstained", reason: "no_speech_energy"` with no fabricated RMS, target-attainment result, or limiting-bound list. Calibration concerns only the model guide; it is not delivered speech leveling or measured denoising effectiveness.
