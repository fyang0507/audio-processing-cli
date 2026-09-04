"""Fixed-shape refusals specific to deterministic transcript export."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

from .refusal_request import Refusal, _refusal, _run_command


def export_input_invalid(input_path: str | Path, reason: str) -> Refusal:
    return _refusal(
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
    return _refusal(
        "export_inputs_incompatible",
        2,
        fix or (
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
    fallback_name = (
        "transcript-" + hashlib.sha256(os.fsencode(transcript.name)).hexdigest()[:16]
    )
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
            raise ValueError(
                "no safe sibling result path fits beside this transcript"
            ) from exc
        if not occupied:
            try:
                protected = {
                    candidate.resolve(strict=False),
                    partial.resolve(strict=False),
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
    if word_timing_outcome == "abstained":
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
            fix = _run_command(
                source_path or input_path,
                stack,
                requested,
                language=language,
                vad=vad,
                run_range=run_range,
                output=output_path,
            )
    return _refusal(
        "timing_required_for_format",
        2,
        fix,
        field="--format",
        provided=output_format,
        requires_capability="word_timestamps",
        found=list(found),
        note="subtitle cue bounds come from word timestamps and are never synthesized",
    )
