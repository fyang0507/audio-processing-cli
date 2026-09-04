"""Normalize, publish, and resume transcription execution results."""

from __future__ import annotations

import os
import shlex
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.media import (
    ProtectedFileIdentity,
    ProtectedOutputError,
    atomic_write_json,
    atomic_write_text,
)

from . import refusals
from ._orchestrator_runtime import RunRange
from .planner import ResolvedRequest
from .result import ABSENT, NormalizedResult
from .transport import StageOutcome


def _record_metrics(outcomes: Sequence[StageOutcome], result: NormalizedResult) -> dict[str, Any]:
    walls: dict[str, float] = {}
    for item in outcomes:
        if item.wall_seconds_by_stage:
            reported = dict(item.wall_seconds_by_stage)
            internal_wall = sum(float(value) for value in reported.values())
            residual = float(item.wall_seconds) - internal_wall
            if residual < -0.005:
                raise ValueError(
                    f"{item.role} internal wall metrics exceed its process wall"
                )
            # A co-resident stage's full process wall includes imports, model loading,
            # audio I/O, and framework overhead outside its explicitly timed phases.
            # Keep that measured residual visible so total_wall_seconds remains the
            # actual sum of non-overlapping stage walls instead of silently dropping it.
            reported[f"{item.role}_overhead"] = max(0.0, residual)
        else:
            reported = {item.role: item.wall_seconds}
        overlap = set(walls) & set(reported)
        if overlap:
            raise ValueError(f"stage wall metrics repeat roles: {sorted(overlap)}")
        walls.update({name: round(float(value), 6) for name, value in reported.items()})
    observed: dict[str, Any] = {
        "stage_wall_seconds": walls,
        "total_wall_seconds": round(sum(walls.values()), 6),
        "segments": len(result.segments),
        "words": sum(len(item.get("words", [])) for item in result.segments),
        "abstentions": len(result.abstentions),
    }
    if result.turns is not ABSENT:
        observed["turns"] = len(result.turns)
    if result.vad_regions is not ABSENT:
        observed["vad_regions"] = len(result.vad_regions)
    if result.lid_regions is not ABSENT:
        observed["lid_regions"] = len(result.lid_regions)
    if result.overlapped_speech is not ABSENT:
        observed["overlapped_speech"] = len(result.overlapped_speech)
    if "word_timestamps" in result.requested_capabilities:
        observed["segments_without_words"] = sum(
            1 for item in result.segments if "words" not in item
        )
    rss = {item.role: item.peak_rss_bytes for item in outcomes if item.peak_rss_bytes is not None}
    if rss:
        observed["peak_rss_bytes_by_stage"] = rss
        observed["peak_rss_bytes"] = max(rss.values())
    mps = {
        item.role: item.peak_mps_live_bytes
        for item in outcomes if item.peak_mps_live_bytes is not None
    }
    if mps:
        observed["peak_mps_live_bytes_by_stage"] = mps
        observed["peak_mps_live_bytes"] = max(mps.values())
    return observed


def _coverage(
    unfinished: Sequence[Mapping[str, Any]], *, total_units: int,
    completed_units: int, scope_start: float, scope_end: float,
) -> dict[str, Any]:
    watermark = min(float(item["start"]) for item in unfinished)
    missing = [[round(watermark, 6), round(scope_end, 6)]]
    covered = [] if watermark <= scope_start else [[
        round(scope_start, 6), round(watermark, 6)
    ]]
    covered_seconds = sum(end - start for start, end in covered)
    return {
        "scope_intervals": [[round(scope_start, 6), round(scope_end, 6)]],
        "covered_through_seconds": round(watermark, 6),
        "covered_fraction": round(covered_seconds / (scope_end - scope_start), 6),
        "covered_intervals": covered,
        "missing_intervals": missing,
        "units_total": total_units,
        "units_completed": completed_units,
    }


def _span_owned(
    span: Mapping[str, Any], *, scope: tuple[float, float] | None,
) -> bool:
    """Assign each whole-file auxiliary observation to exactly one ranged document."""
    start = float(span["start"])
    return scope is None or scope[0] <= start < scope[1]


def _partial_path(source: Path, output: Path | None) -> Path:
    if output is None:
        return source.with_name(f"{source.stem}.partial.json")
    return output.with_name(f"{output.with_suffix('').name}.partial.json")


def _unused_partial_path(source: Path) -> Path:
    candidate = _partial_path(source, None)
    index = 2
    while os.path.lexists(candidate):
        candidate = source.with_name(f"{source.stem}.partial.{index}.json")
        index += 1
    return candidate


def validate_output_targets(
    request: ResolvedRequest,
    output: Path | None,
    *,
    output_format: str,
    run_range: RunRange | None,
    force: bool,
) -> None:
    """Refuse every destination collision before decoding or model work begins."""
    source = request.input_path
    targets = (("Output", output),)
    if output is not None:
        targets += (("Partial output", _partial_path(source, output)),)
    for _, target in targets:
        if (
            target is not None
            and target.is_dir()
            and not target.is_symlink()
        ):
            raise refusals.output_path_invalid(
                output or target,
                target,
                "destination is a directory and cannot be replaced",
            )
        if target is not None and os.path.lexists(target) and not force:
            raise refusals.output_exists(
                source,
                request.stack.id,
                request.wants,
                output,
                target,
                language=request.language,
                vad=request.vad,
                diarizer=request.diarizer,
                run_range=run_range.provided if run_range is not None else None,
                output_format=output_format,
            )
    if output is None:
        return
    try:
        source_identity = source.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise refusals.output_path_invalid(output, source, str(exc)) from exc
    for _, target in targets:
        if target is None:
            continue
        try:
            target_identity = target.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise refusals.output_path_invalid(output, target, str(exc)) from exc
        if source_identity == target_identity:
            raise refusals.output_is_canonical_input(output, target)


def _resume_command(
    request: ResolvedRequest,
    coverage: Mapping[str, Any],
    output: Path,
    run_range: RunRange | None,
) -> str:
    prefix_only = request.stack.failure_recovery.get("partial_results") == "prefix_only"
    has_prefix = bool(coverage.get("covered_intervals"))
    if int(coverage["units_completed"]) == 0 and not (prefix_only and has_prefix):
        return (
            "no processing unit completed; --range would repeat the same deterministic "
            "work, so inspect the first unit or backend budget before retrying"
        )
    stem = output.stem
    if stem.endswith(".partial"):
        stem = stem.removesuffix(".partial")
    rest = output.with_name(f"{stem}.rest.json")
    parts = [
        "audio", "transcribe", "run", "--input",
        refusals.command_path_argument(request.input_path),
        "--stack", request.stack.id,
    ]
    if request.wants:
        parts.extend(("--want", ",".join(request.wants)))
    if request.language:
        if request.language.startswith("-"):
            parts.append(f"--language={request.language}")
        else:
            parts.extend(("--language", request.language))
    if request.vad:
        parts.extend(("--vad", request.vad))
    if request.diarizer:
        parts.extend(("--diarizer", request.diarizer))
    watermark = coverage["covered_through_seconds"]
    explicit_end = bool(
        run_range is not None and run_range.provided.partition(":")[2]
    )
    range_value = (
        f"{watermark}:{run_range.end}"
        if explicit_end and run_range is not None
        else f"{watermark}:"
    )
    parts.extend((
        "--range", range_value, "-o", refusals.command_path_argument(rest),
    ))
    return shlex.join(parts)


def render_human(payload: Mapping[str, Any], output_format: str) -> str:
    lines = []
    for item in payload["segments"]:
        prefix = f"[{item['speaker']}] " if "speaker" in item else ""
        lines.append(prefix + item["text"])
    body = "\n".join(lines)
    if output_format == "md":
        return "# Transcript\n\n" + (body + "\n" if body else "")
    return body + ("\n" if body else "")


def _write_result_file(
    path: Path,
    payload: dict[str, Any],
    *,
    output_format: str,
    force: bool,
    protected_path: Path,
    protected_identity: ProtectedFileIdentity | None,
) -> None:
    protected_identities = (
        (protected_identity,) if protected_identity is not None else ()
    )
    if output_format == "json":
        atomic_write_json(
            path,
            payload,
            force=force,
            protected_paths=(protected_path,),
            protected_identities=protected_identities,
        )
    else:
        atomic_write_text(
            path,
            render_human(payload, output_format),
            force=force,
            protected_paths=(protected_path,),
            protected_identities=protected_identities,
        )


def _publish_result(
    request: ResolvedRequest,
    payload: dict[str, Any],
    target: Path,
    *,
    output: Path | None,
    output_format: str,
    run_range: RunRange | None,
    force: bool,
    protected_source_identity: ProtectedFileIdentity | None,
    storage_format: str | None = None,
) -> None:
    try:
        _write_result_file(
            target,
            payload,
            output_format=storage_format or output_format,
            force=force,
            protected_path=request.input_path,
            protected_identity=protected_source_identity,
        )
    except ProtectedOutputError as exc:
        raise refusals.output_is_canonical_input(
            output or target, target
        ) from exc
    except (FileExistsError, IsADirectoryError) as exc:
        if target.is_dir() and not target.is_symlink():
            raise refusals.output_path_invalid(
                output or target,
                target,
                "destination is a directory and cannot be replaced",
            ) from exc
        raise refusals.output_exists(
            request.input_path,
            request.stack.id,
            request.wants,
            output or target,
            target,
            language=request.language,
            vad=request.vad,
            diarizer=request.diarizer,
            run_range=run_range.provided if run_range is not None else None,
            output_format=output_format,
        ) from exc
    except OSError as exc:
        raise refusals.output_path_invalid(
            output or target,
            target,
            str(exc),
        ) from exc


def _publish_partial(
    request: ResolvedRequest,
    payload: dict[str, Any],
    *,
    output: Path | None,
    output_format: str,
    run_range: RunRange | None,
    force: bool,
    protected_source_identity: ProtectedFileIdentity | None,
) -> Path:
    if output is None and not force:
        while True:
            target = _unused_partial_path(request.input_path)
            try:
                atomic_write_json(
                    target,
                    payload,
                    force=False,
                    protected_paths=(request.input_path,),
                    protected_identities=(
                        (protected_source_identity,)
                        if protected_source_identity is not None
                        else ()
                    ),
                )
                return target
            except ProtectedOutputError as exc:
                raise refusals.output_is_canonical_input(target, target) from exc
            except FileExistsError:
                # Another writer claimed the candidate after selection.  Find a
                # new sibling instead of overwriting either result.
                continue
            except OSError as exc:
                raise refusals.output_path_invalid(
                    target,
                    target,
                    str(exc),
                ) from exc
    target = _partial_path(request.input_path, output)
    _publish_result(
        request,
        payload,
        target,
        output=output,
        output_format=output_format,
        run_range=run_range,
        force=force,
        protected_source_identity=protected_source_identity,
        storage_format="json",
    )
    return target


def _backend_fix(role: str, backend: str) -> str:
    return (
        f"inspect the {role} failure from {backend} and correct the reported runtime "
        "condition before retrying"
    )


def _outcomes(
    wants: Sequence[str], *, segments: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    outcomes = {name: "produced" for name in wants}
    wordless_events = {"[Environmental Sounds]", "[Silence]", "[Human Sounds]"}
    if "word_timestamps" in outcomes and any(
        item.get("text")
        and item.get("text") not in wordless_events
        and "words" not in item
        for item in segments
    ):
        outcomes["word_timestamps"] = "abstained"
    return outcomes


def _has_lexical_text(text: str) -> bool:
    return any(
        not character.isspace()
        and not unicodedata.category(character).startswith("P")
        for character in text
    )
