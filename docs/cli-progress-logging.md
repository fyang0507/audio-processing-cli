# Progress and retained-log integration

These APIs provide issue #46's implementation hooks. The CLI composition and parser owners wire their options and contexts; this change does not edit `cli.py` or `cli_parser.py`. Default machine JSON, canonical media and results, absence semantics, model settings, and processing decisions are unchanged.

## Enhancement

`audio_cli.pipeline.EnhancementPipeline(profile, *, skipped_stages=None, adjustments=None, detector=None, denoiser_model=None, progress=None)` accepts an optional synchronous callable `progress(stage: str, status: Literal["started", "finished", "failed"]) -> None`. Existing arguments and `run(...)` are unchanged. `None` emits nothing. The pipeline's internal `progress_stage` context brackets domain work and emits `failed` before reraising exceptions, including `KeyboardInterrupt`. Callbacks must return promptly and should not raise; a custom callback's exception propagates to its caller.

`audio_cli.command.ProgressReporter(label: str, *, stream: TextIO | None = None, heartbeat_seconds: float = 10.0)` is a context-managed callable compatible with that sink. It defaults to the current `sys.stderr`; elapsed notices use a monotonic clock and the currently active nested stage. Keep its context open around the whole pipeline call so the timer is stopped and joined on success or exception. `close()` is idempotent. Closed or broken presentation streams disable further notices without changing processing results. Context instances cannot be reused.

```python
from audio_cli.command import ProgressReporter
from audio_cli.pipeline import EnhancementPipeline

with ProgressReporter("enhance") as progress:
    report = EnhancementPipeline(
        profile,
        detector=detector,
        progress=progress,
    ).run(source, output=output, dry_run=dry_run)
# Serialize the unchanged report to stdout after the progress context closes.
```

The outer event is `enhancement` or `dry-run`. Inner events are `preparation`, `decode`, `inspection`, `channel-balance`, `environment-denoise`, `frequency-adjustments`, `voice-enhance`, `source-balance`, `fullband-adjustments`, and `normalization`. Rendered runs additionally emit `encode`, `encoded-output-inspection`, `publication`, and `published-output-inspection`; codec correction can nest another `encode` inside the encoded-output inspection. A completed stage event reports execution, including evaluation of skipped/disabled stages; the report alone carries processing decisions. No percentages, ETA, measured quality, or per-model completion claims are synthesized. Dry runs still perform their existing processing and normalization predictions.

The pipeline imports no command presentation code. Preparation, stage processing, and publication also accept keyword-only `progress=None` for their existing internal callers. No progress data enters a signal, report, hash, or result schema.

## Transcription logs

`audio_cli.transcribe.transport.StageTransport(runner=None, *, progress=None, log_root: Path | None = None)` creates and validates requested log storage at construction, before decode or model execution. Create one transport per transcription run and pass it through the existing `audio_cli.transcribe.orchestrator.run(..., transport=transport)` argument. The CLI should resolve its log option to a `Path`, construct the transport inside its existing error boundary, and represent constructor `OSError`/`ValueError` as an argument/path failure rather than starting the run. The CLI's exact architecture import inventory needs the `audio_cli.transcribe.transport` facade edge when wiring this import; no lower owner gains a CLI dependency.

```python
from audio_cli.transcribe.transport import StageTransport

transport = StageTransport(progress=sys.stderr, log_root=selected_log_root)
product = run_transcription(request, metadata, transport=transport)
```

`audio_cli.transcribe.transport.SubprocessRunner(progress, *, heartbeat_seconds=10.0, log_root: Path | None = None)` offers the same selection directly. Its `run(command: list[str], *, stage: str = "child")` accepts a path-safe diagnostic label using lowercase ASCII letters, digits, underscores, and hyphens. Transport supplies `decode`, `asr`, `aligner`, `firered_process`, or `diarizer`. Injected custom runners retain their existing `run(command)` protocol; supplying both a custom runner and `StageTransport(log_root=...)` raises `ValueError` rather than ignoring the root. Configure a supplied `SubprocessRunner` directly when also changing its heartbeat interval.

With a root, each runner creates `ROOT/audio-transcribe-<unique>/`, exposed as `runner.log_directory`, containing `001-<stage>-<unique>/stdout.log` and `stderr.log`, then a new directory for every subsequent invocation. Existing directories/files are never reused. Run and stage directories are private (0700). Nonexistent root parents are created; a successful private-directory creation checks actual filesystem access. Root failures propagate from construction; later stage-directory/file failures retain the existing transport-failure behavior and prevent that child launch. Neither failure falls back elsewhere. `CompletedProcess.diagnostics_directory` still identifies the per-child directory on completed executions. Stderr announces absolute file paths before launch, including when the child later fails or is interrupted.

Omitting `log_root` preserves the existing per-child retained temporary directories and cleanup policy; `runner.log_directory` is then `None`. Selecting a durable root changes storage location only. Both streams still receive exact original bytes directly from the child, including warnings, carriage returns, escape sequences, invalid UTF-8, and binary zero bytes. Decoded failure tails remain diagnostic summaries of those files. The host does not automatically expire logs or promise recovery of unflushed child buffers. See [warning interpretation and log access](cli-feedback.md#follow-a-run-and-retain-its-diagnostics).
