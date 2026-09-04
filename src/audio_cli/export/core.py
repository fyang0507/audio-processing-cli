"""Export orchestration and safe publication."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from audio_cli.media import ProtectedFileIdentity, capture_file_identity, resolve_path_identity

from .cues import Cue, CueError, build_cues
from .errors import IncompatibleResultsError, InvalidResultError, OutputWriteError
from .loading import load_result_document
from .merge import merge_documents
from .models import _TIMED_FORMATS, EXPORT_FORMATS, ExportProduct
from .timing import _require_word_timing, _validate_word_timing_ledger
from .writers import (
    normalize_voice_annotation,
    render_jsonl,
    render_markdown,
    render_srt,
    render_text,
    render_vtt,
    write_text_atomic,
)


def export_documents(
    inputs: Sequence[Path],
    output_format: str,
    output: Path | None = None,
    force: bool = False,
) -> ExportProduct:
    """Read, merge, render, and optionally atomically publish transcript exports."""
    if output_format not in EXPORT_FORMATS:
        raise ValueError(
            f"output_format must be one of {sorted(EXPORT_FORMATS)}, got {output_format!r}"
        )
    loaded = tuple(load_result_document(Path(path)) for path in inputs)
    merged = merge_documents(loaded)
    source_path: Path | None = None
    source_file_identity: ProtectedFileIdentity | None = None
    if output is not None:
        source_path = Path(merged.source["path"])
        if not source_path.is_absolute():
            raise InvalidResultError(
                merged.inputs[0],
                "source.path must be absolute before writing an export; regenerate "
                "the transcript with the current audio CLI",
            )
        try:
            resolve_path_identity(source_path)
            source_file_identity = capture_file_identity(source_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidResultError(
                merged.inputs[0],
                f"source.path cannot be resolved safely: {exc}",
            ) from exc
    cues: tuple[Cue, ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    speaker_labels_rendered = False

    if output_format in _TIMED_FORMATS:
        for document in merged.documents:
            _validate_word_timing_ledger(document)
        _require_word_timing(merged)
        for document in merged.documents:
            try:
                build_cues(
                    document.payload["segments"],
                    duration=float(document.payload["source"]["duration_seconds"]),
                )
            except CueError as exc:
                raise InvalidResultError(document.path, str(exc)) from exc
        try:
            cue_build = build_cues(
                merged.segments,
                duration=float(merged.source["duration_seconds"]),
            )
        except CueError as exc:
            raise IncompatibleResultsError(
                merged.inputs, f"merged cue sequence is invalid: {exc}"
            ) from exc
        cues = cue_build.cues
        warnings = (
            {
                "code": "cue_timing_unvalidated",
                "blocking": False,
                "detail": (
                    "boundary MAE/P95 is unmeasured for the backend that produced this "
                    "timing, so cue placement is producible but not claimed "
                    "broadcast-acceptable"
                ),
            },
            *cue_build.warnings,
        )
        if output_format == "srt":
            content = render_srt(cues)
        else:
            content = render_vtt(cues)
            speaker_labels_rendered = any(
                cue.speaker is not None and normalize_voice_annotation(cue.speaker) is not None
                for cue in cues
            )
    elif output_format == "md":
        content = render_markdown(merged.segments)
    elif output_format == "txt":
        content = render_text(merged.segments)
    else:
        content = render_jsonl(merged.segments)

    product = ExportProduct(
        inputs=merged.inputs,
        output_format=output_format,
        content=content,
        segments=merged.segments,
        cues=cues,
        speaker_labels_rendered=speaker_labels_rendered,
        warnings=warnings,
    )
    if output is not None:
        assert source_path is not None
        protected = [*merged.inputs, source_path]
        protected_identities = [
            identity
            for identity in (
                *(document.file_identity for document in merged.documents),
                source_file_identity,
            )
            if identity is not None
        ]
        try:
            write_text_atomic(
                Path(output),
                content,
                force=force,
                protected_paths=protected,
                protected_identities=protected_identities,
            )
        except OSError as exc:
            raise OutputWriteError(Path(output), str(exc)) from exc
    return product
