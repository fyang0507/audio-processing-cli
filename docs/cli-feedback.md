# CLI discovery, progress, and saved-report navigation

## Discover declared stacks

Run `audio transcribe stacks` from any directory, without an input or installed models. It prints JSON with one `stacks` array. Each row contains `id`, `characterization`, `native`, `requires_add_on`, and `impossible`, projected from the same declared stack table used by request resolution. No runtime probe or readiness claim is part of discovery. `audio transcribe capabilities --help` leads to this list and describes how to query one choice for an original input. Missing-stack and unsupported-capability requests retain their existing semantic JSON refusals.

## Follow a run and retain its diagnostics

`audio transcribe run` prints host stage start/finish notices on stderr and a child elapsed-time notice every ten seconds while a child remains running. Elapsed time is liveness information; it is neither a percentage nor a prediction of remaining time. A child exit notice reports the process exit code, while the host still validates its result protocol before reporting stage completion. An exit-zero child with malformed result JSON therefore remains a failure.

Before each decode or model child launches, the host announces absolute paths to `stdout.log` and `stderr.log` in a fresh private `audio-transcribe-*` temporary directory. The child writes its original bytes directly to those files, including carriage returns, escape sequences, and non-UTF-8 bytes. Logs are available during execution and retained after success, failure, or handled interruption until the operating system or user removes them. Copy these files with the evidence when durable retention is needed. There is no automatic CLI retention policy, no new diagnostic-directory flag, and no transcript-schema change for log locations.

The host does not classify or filter backend warning text. Backend stdout is preserved alongside stderr; neither stream is mixed into canonical transcript text or machine stdout. Treat warnings as evidence to investigate, not proof of harmlessness, recognition quality, or failure. Use exit status, semantic refusals, and result coverage for command outcomes. The decoded diagnostic tail may still appear in a failure's existing detail field; the retained files are the byte-exact evidence.

Keyboard interruption terminates and reaps the direct child before closing the log handles; an unresponsive child is killed after five seconds. Bytes already written survive; bytes a backend never flushed from its own userspace buffers cannot be recovered. Abrupt host termination such as SIGKILL cannot run Python cleanup, though bytes already written to the retained files remain. A host error creating or reading diagnostic files becomes a transport failure.

## Navigate an enhancement report offline

```bash
audio report summary /absolute/path/output.wav.report.json
```

This stdout-only JSON projection reads the saved `audio_enhancement_report` schema version `"1"`. It does not open the source media, invoke detection, run models, remeasure audio, or rewrite either the report or transcript. Invalid JSON, duplicate keys, nonfinite numbers, wrong report kinds, unknown schema versions, and malformed required structures refuse at exit 2 using the enhancement commands' existing JSON error envelope.

| Summary field | Meaning |
| --- | --- |
| `kind`, `schema_version` | `audio_enhancement_summary`, `"1"`; identifies the projection |
| `report`, `source`, `profile` | Absolute report path, recorded source path/hash when present, and profile name/version |
| `rendered`, `dry_run` | Copied run state; a render is not proof that every requested goal was met |
| `stages` | Recorded stage name/status/reason, full component and final-region evaluations when present, and report pointers; bulky operations remain in the original report |
| `unresolved` | Full recorded unresolved rows with nested scope/evidence intact, plus pointers; absent in the summary when absent in the source report |
| `measurement_scopes` | Pointers to available `before`, `predicted`, and `after` blocks and their `program_actual`/`regional` blocks when present; no copies of those measurement blocks |
| `region_basis`, `timeline_preserved` | Copied when present, retaining the original report's scope and limits |
| `final_peak_validation` | Recorded status when present, plus a pointer to full validation details |

`report_pointer` is a JSON Pointer into the file named by `report`; for example, `/unresolved/0` identifies one complete unresolved record and `/measurements/after/program_actual` identifies delivered-file program measurements. Before and predicted blocks never acquire an `after` label. Per-decision `measured_at` stays absent when the report did not supply it; a rendered report does not retroactively make every stage decision a post-encode measurement. Older reports without `program_actual` retain a pointer to the containing measurement block and acquire no replacement measurement.

Unknown additional evidence keys inside components and unresolved rows are retained, including structured `noise_reference_scopes`; required known structures are validated, and unrelated top-level report extensions remain accessible through the original file. `applied` remains the parent's operation status even when a child abstains. Skipped/no-op stages remain visible. There is no overall success enum: an empty unresolved ledger describes recorded checks and cannot establish perceptual quality, completeness of detection, or achievement of unmeasured goals.
