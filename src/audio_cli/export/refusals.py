"""Fixed-shape command refusals specific to deterministic transcript export."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.command import (
    Refusal,
    build_refusal,
    export_command,
    transcribe_run_command,
)
from audio_cli.command import output_exists as _output_exists
from audio_cli.command import output_is_canonical_input as output_is_canonical_input
from audio_cli.command import output_path_invalid as output_path_invalid
from audio_cli.media import resolve_path_identity


def output_required_for_force() -> Refusal:
    return build_refusal(
        "output_required_for_force",
        2,
        "remove --force when writing to stdout, or add --output PATH",
        field="--force",
        provided=True,
        requires="--output",
    )


def timestamps_unsupported_for_format(output_format: str) -> Refusal:
    return build_refusal(
        "timestamps_unsupported_for_format",
        2,
        "use --timestamps with --format txt or md, or remove --timestamps",
        field="--timestamps",
        provided=True,
        format=output_format,
        allowed_formats=["txt", "md"],
    )


def _boundary_recovery_fix(alignment_rejections: list[dict[str, Any]] | None) -> str | None:
    for entry in alignment_rejections or ():
        alignment = entry.get("alignment")
        if (
            isinstance(alignment, Mapping)
            and alignment.get("code") == "out_of_unit_bounds"
            and isinstance(alignment.get("boundary"), Mapping)
        ):
            return (
                "inspect alignment_rejections for the observed original_bounds, unit_bounds, "
                "start_overrun_ms, end_overrun_ms and max_overrun_ms. An agent may choose an "
                "explicit --alignment-max-overrun-ms or another supported stack on the original "
                "source, preserving the required capabilities and scope and writing a new "
                "canonical output. Inspect the new result before exporting; the CLI does not "
                "switch models automatically and neither choice guarantees usable timing"
            )
    return None


def timing_required_for_timestamps(
    input_path: str | Path,
    segment_id: str,
    *,
    word_timing_outcome: str | None = None,
    alignment_rejections: list[dict[str, Any]] | None = None,
) -> Refusal:
    boundary_fix = _boundary_recovery_fix(alignment_rejections)
    if boundary_fix is not None:
        fix = boundary_fix
    elif word_timing_outcome == "abstained":
        fix = (
            "remove --timestamps to preserve untimed text; word_timestamps was already "
            "requested but abstained. Inspect this segment's alignment_unavailable entry "
            "and retained structured stage diagnostics; rerunning the same request is not "
            "an established timing repair"
        )
    elif word_timing_outcome is not None:
        fix = (
            "remove --timestamps to preserve untimed text; this result already records "
            "word_timestamps but this segment has no usable timing"
        )
    else:
        fix = (
            "remove --timestamps to preserve untimed text, or transcribe the original "
            "source with segment_timestamps on a native stack or word_timestamps"
        )
    return build_refusal(
        "timing_required_for_timestamps",
        2,
        fix,
        field="--timestamps",
        provided=True,
        input=str(input_path),
        segment_id=segment_id,
        requires_any_capability=["segment_timestamps", "word_timestamps"],
        note="every segment needs supplied bounds; processing intervals are never substituted",
        **(
            {"alignment_rejections": alignment_rejections}
            if alignment_rejections is not None
            else {}
        ),
    )


def provenance_unsupported_for_format(output_format: str) -> Refusal:
    return build_refusal(
        "provenance_unsupported_for_format",
        2,
        "use --provenance with --format txt or md, or remove --provenance",
        field="--provenance",
        provided=True,
        format=output_format,
        allowed_formats=["txt", "md"],
    )


def output_exists(
    input_paths: Sequence[str | Path],
    output_format: str,
    output: str | Path,
    *,
    replaceable: bool,
    timestamps: bool = False,
    provenance: bool = False,
) -> Refusal:
    if replaceable:
        fix = export_command(
            input_paths,
            output_format,
            output,
            force=True,
            timestamps=timestamps,
            provenance=provenance,
        )
    else:
        fix = (
            "choose a regular-file --output path; an existing directory cannot "
            "be replaced by --force, and neither can a symlink or special file"
        )
    return _output_exists(output, output, fix)


def export_input_invalid(input_path: str | Path, reason: str) -> Refusal:
    return build_refusal(
        "export_input_invalid",
        2,
        "regenerate or repair the input transcript before exporting it",
        field="--input",
        provided=str(input_path),
        reason=reason,
    )


def export_inputs_incompatible(
    input_paths: Sequence[str | Path],
    reason: str,
    *,
    fix: str | None = None,
) -> Refusal:
    return build_refusal(
        "export_inputs_incompatible",
        2,
        fix
        or (
            "export these inputs separately, or select results with the same canonical "
            "source and non-overlapping ranges"
        ),
        field="--input",
        provided=[str(path) for path in input_paths],
        reason=reason,
    )


def _unused_timed_output(transcript: Path, source_path: Path | None) -> Path:
    name = transcript.name
    for suffix in (".transcript.json", ".partial.json", ".rest.json", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    source_resolved: Path | None = None
    if source_path is not None:
        try:
            source_resolved = source_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("source.path is not a resolvable regular file") from exc
    try:
        name_limit = int(os.pathconf(transcript.parent, "PC_NAME_MAX"))
    except (OSError, ValueError):
        name_limit = 255
    fallback_name = "transcript-" + hashlib.sha256(os.fsencode(transcript.name)).hexdigest()[:16]
    index = 1
    while True:
        marker = "" if index == 1 else f".{index}"
        candidate_name = f"{name}.timed{marker}.json"
        partial_name = f"{Path(candidate_name).with_suffix('').name}.partial.json"
        if max(len(os.fsencode(candidate_name)), len(os.fsencode(partial_name))) > name_limit:
            candidate_name = f"{fallback_name}.timed{marker}.json"
            partial_name = f"{Path(candidate_name).with_suffix('').name}.partial.json"
        candidate = transcript.with_name(candidate_name)
        partial = candidate.with_name(partial_name)
        try:
            occupied = any(path.exists() or path.is_symlink() for path in (candidate, partial))
        except OSError as exc:
            raise ValueError("no safe sibling result path fits beside this transcript") from exc
        if not occupied:
            try:
                protected = {
                    resolve_path_identity(candidate),
                    resolve_path_identity(partial),
                }
            except (OSError, RuntimeError) as exc:
                raise ValueError(
                    "no safe sibling result path can be resolved beside this transcript"
                ) from exc
            if source_resolved is None or source_resolved not in protected:
                return candidate
        index += 1


def timing_required_for_format(
    input_path: str | Path,
    output_format: str,
    found: Sequence[str],
    stack: str,
    wants: Sequence[str],
    *,
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
    language: str | None = None,
    vad: str | None = None,
    run_range: str | None = None,
    word_timing_outcome: str | None = None,
    alignment_rejections: list[dict[str, Any]] | None = None,
    alignment_max_overrun_ms: float | None = None,
) -> Refusal:
    requested = list(dict.fromkeys([*wants, "word_timestamps"]))
    transcript = Path(input_path)
    unsafe_legacy_source = False
    if source_path is not None:
        source = Path(source_path)
        if not source.is_absolute() or not source.is_file():
            unsafe_legacy_source = True
        else:
            try:
                source.resolve(strict=True)
            except (OSError, RuntimeError):
                unsafe_legacy_source = True
    boundary_fix = _boundary_recovery_fix(alignment_rejections)
    if boundary_fix is not None:
        fix = boundary_fix
    elif word_timing_outcome == "abstained":
        fix = (
            "choose md, txt, or jsonl, provide another result with produced timed words, "
            "or choose a different stack; this result already attempted word_timestamps "
            "and abstained, so repeating the same command is not a repair"
        )
    elif "word_timestamps" in found:
        fix = (
            "choose md, txt, or jsonl, or provide a transcript containing timed "
            "words; this result already records word_timestamps but has no word "
            "stream from which subtitle cues can be built"
        )
    elif unsafe_legacy_source:
        fix = (
            "regenerate this transcript with the current audio CLI before rerunning "
            "or exporting it; its relative, missing, non-file, or unresolvable "
            "source.path cannot safely identify the original media"
        )
    else:
        if output_path is None:
            try:
                output_path = _unused_timed_output(
                    transcript,
                    Path(source_path) if source_path is not None else None,
                )
            except ValueError:
                fix = (
                    "choose md, txt, or jsonl, or rerun transcription from a "
                    "shorter output directory; no safe sibling timed-result path "
                    "fits beside this transcript"
                )
        if output_path is not None:
            fix = transcribe_run_command(
                source_path or input_path,
                stack,
                requested,
                language=language,
                vad=vad,
                alignment_max_overrun_ms=alignment_max_overrun_ms,
                run_range=run_range,
                output=output_path,
            )
    return build_refusal(
        "timing_required_for_format",
        2,
        fix,
        field="--format",
        provided=output_format,
        requires_capability="word_timestamps",
        found=list(found),
        note="subtitle cue bounds come from word timestamps and are never synthesized",
        **(
            {"alignment_rejections": alignment_rejections}
            if alignment_rejections is not None
            else {}
        ),
    )
