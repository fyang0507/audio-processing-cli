"""Fixed-shape transcription refusals.

These payloads print bare on stderr.  The older shipped commands retain their historical
``{"error": ...}`` envelope; sharing their exception type would silently erase that boundary.
"""

from __future__ import annotations

import hashlib
import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import stacks


class Refusal(RuntimeError):
    """A request that can be rejected without loading or provisioning a model."""

    def __init__(self, payload: Mapping[str, Any], *, exit_code: int) -> None:
        self.payload = dict(payload)
        self.exit_code = exit_code
        super().__init__(str(self.payload["code"]))


def _refusal(code: str, exit_code: int, fix: str, **fields: Any) -> Refusal:
    return Refusal({"code": code, **fields, "fix": fix}, exit_code=exit_code)


def command_path_argument(value: str | Path) -> str:
    rendered = str(value)
    return f"./{rendered}" if rendered.startswith("-") else rendered


def _plan_command(
    input_path: str | Path,
    stack: str,
    wants: Sequence[str] = (),
    *,
    language: str | None = None,
    vad: str | None = None,
    diarizer: str | None = None,
) -> str:
    parts = [
        "audio", "transcribe", "plan", "--input", command_path_argument(input_path),
    ]
    if stack.startswith("-"):
        parts.append(f"--stack={stack}")
    else:
        parts.extend(("--stack", stack))
    if wants:
        parts.extend(("--want", ",".join(wants)))
    if language is not None:
        if language.startswith("-"):
            parts.append(f"--language={language}")
        else:
            parts.extend(("--language", language))
    if vad is not None:
        parts.extend(("--vad", vad))
    if diarizer is not None:
        parts.extend(("--diarizer", diarizer))
    return shlex.join(parts)


def _run_command(
    input_path: str | Path,
    stack: str,
    wants: Sequence[str] = (),
    *,
    language: str | None = None,
    vad: str | None = None,
    diarizer: str | None = None,
    run_range: str | None = None,
    output_format: str = "json",
    output: str | Path | None = None,
    force: bool = False,
) -> str:
    parts = [
        "audio", "transcribe", "run", "--input", command_path_argument(input_path),
    ]
    if stack.startswith("-"):
        parts.append(f"--stack={stack}")
    else:
        parts.extend(("--stack", stack))
    if wants:
        parts.extend(("--want", ",".join(wants)))
    if language is not None:
        if language.startswith("-"):
            parts.append(f"--language={language}")
        else:
            parts.extend(("--language", language))
    if vad is not None:
        parts.extend(("--vad", vad))
    if diarizer is not None:
        parts.extend(("--diarizer", diarizer))
    if run_range is not None:
        parts.extend(("--range", run_range))
    if output_format != "json":
        parts.extend(("--format", output_format))
    if output is not None:
        parts.extend(("-o", command_path_argument(output)))
    if force:
        parts.append("--force")
    return shlex.join(parts)


def stack_required(input_path: str | Path | None, wants: Sequence[str]) -> Refusal:
    chosen_input = input_path or "meeting.m4a"
    return _refusal(
        "stack_required",
        2,
        _plan_command(chosen_input, "qwen-1.7b", wants),
        field="--stack",
        allowed=list(stacks.stack_ids()),
        stacks={
            identifier: definition.characterization
            for identifier, definition in stacks.stack_definitions().items()
        },
    )


def input_required(stack: str, wants: Sequence[str]) -> Refusal:
    return _refusal(
        "input_required",
        2,
        _plan_command("meeting.m4a", stack, wants),
        field="--input",
        note=(
            "a stack alone cannot be planned: how the audio is partitioned, how many units "
            "that is, what the run will cost, and whether a failure leaves anything usable "
            "are all properties of this file"
        ),
    )


def capability_unknown(
    stack: stacks.StackDefinition,
    input_path: str | Path,
    provided: str,
    wants: Sequence[str],
    suggestion: str | None,
) -> Refusal:
    fixed = [
        suggestion if item == provided else item
        for item in wants
        if item != provided or suggestion is not None
    ]
    fields: dict[str, Any] = {
        "field": "--want",
        "provided": provided,
    }
    if suggestion is not None:
        fields["did_you_mean"] = suggestion
    fields["available_on_stack"] = stacks.availability_groups(stack)
    return _refusal(
        "capability_unknown",
        2,
        _plan_command(input_path, stack.id, fixed),
        **fields,
    )


def capability_unsatisfiable_on_stack(
    stack: stacks.StackDefinition,
    input_path: str | Path,
    capability: str,
    wants: Sequence[str],
) -> Refusal:
    allowed = stacks.allowed_stacks(capability)
    if not allowed:
        raise stacks.StackTableError(
            f"{capability!r} is marked unsatisfiable_on_stack but has no alternative"
        )
    preferred = stacks.recommended_stack(capability) or allowed[0]
    return _refusal(
        "capability_unsatisfiable_on_stack",
        2,
        _plan_command(input_path, preferred, wants),
        capability=capability,
        allowed=allowed,
        available_on_stack=stacks.availability_groups(stack),
    )


def capability_unsupported(capability: str, reason: str, fix: str) -> Refusal:
    return _refusal(
        "capability_unsupported",
        2,
        fix,
        capability=capability,
        allowed=[],
        reason=reason,
    )


def option_unsupported_on_stack(
    stack: stacks.StackDefinition,
    input_path: str | Path,
    field: str,
    provided: str,
    wants: Sequence[str],
) -> Refusal:
    return _refusal(
        "option_unsupported_on_stack",
        2,
        _plan_command(input_path, stack.id, wants),
        field=field,
        provided=provided,
        allowed=[],
        stacks_accepting=[
            item.id
            for item in stacks.stack_definitions().values()
            if item.language_vocabulary is not None
        ],
    )


def option_value_unsupported(
    stack: stacks.StackDefinition,
    input_path: str | Path,
    field: str,
    provided: str,
    allowed: Sequence[str],
    wants: Sequence[str],
    suggestion: str | None,
) -> Refusal:
    fixed_value = suggestion or allowed[0]
    kwargs: dict[str, str | None] = {"language": None, "vad": None, "diarizer": None}
    if field == "--language":
        kwargs["language"] = fixed_value
    elif field == "--vad":
        kwargs["vad"] = fixed_value
    elif field == "--diarizer":
        kwargs["diarizer"] = fixed_value
    fields: dict[str, Any] = {
        "field": field,
        "provided": provided,
        "allowed": list(allowed),
    }
    if suggestion is not None:
        fields["did_you_mean"] = suggestion
    return _refusal(
        "option_value_unsupported",
        2,
        _plan_command(input_path, stack.id, wants, **kwargs),
        **fields,
    )


def pin_conflicts_with_native_capability(
    stack: stacks.StackDefinition,
    input_path: str | Path,
    field: str,
    provided: str,
    capability: str,
    wants: Sequence[str],
) -> Refusal:
    return _refusal(
        "pin_conflicts_with_native_capability",
        2,
        _plan_command(input_path, stack.id, wants),
        field=field,
        provided=provided,
        allowed=[],
        capability=capability,
    )


def range_invalid(
    input_path: str | Path,
    stack: str,
    wants: Sequence[str],
    provided: str,
    reason: str,
    *,
    language: str | None = None,
    vad: str | None = None,
    diarizer: str | None = None,
) -> Refusal:
    return _refusal(
        "range_invalid",
        2,
        _run_command(
            input_path, stack, wants, language=language, vad=vad, diarizer=diarizer
        ),
        field="--range",
        provided=provided,
        reason=reason,
    )


def output_exists(
    input_path: str | Path,
    stack: str,
    wants: Sequence[str],
    output: str | Path,
    existing: str | Path,
    *,
    language: str | None = None,
    vad: str | None = None,
    diarizer: str | None = None,
    run_range: str | None = None,
    output_format: str = "json",
) -> Refusal:
    return _refusal(
        "output_exists",
        2,
        _run_command(
            input_path,
            stack,
            wants,
            language=language,
            vad=vad,
            diarizer=diarizer,
            run_range=run_range,
            output_format=output_format,
            output=output,
            force=True,
        ),
        field="--output",
        provided=str(output),
        existing=str(existing),
    )


def output_is_canonical_input(
    output: str | Path, resolved_target: str | Path,
) -> Refusal:
    return _refusal(
        "output_is_canonical_input",
        2,
        (
            "choose an --output that does not resolve to an input transcript, its derived "
            "partial path, or canonical source media; --force cannot override this"
        ),
        field="--output",
        provided=str(output),
        resolved_target=str(resolved_target),
    )


def output_path_invalid(
    output: str | Path, target: str | Path, reason: str,
) -> Refusal:
    return _refusal(
        "output_path_invalid",
        2,
        (
            "choose an --output whose destination and parent directory can be "
            "resolved and written safely"
        ),
        field="--output",
        provided=str(output),
        target=str(target),
        reason=reason,
    )


def output_required_for_force() -> Refusal:
    return _refusal(
        "output_required_for_force",
        2,
        "remove --force when writing to stdout, or add --output PATH",
        field="--force",
        provided=True,
        requires="--output",
    )


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


def _unused_timed_output(
    transcript: Path, source_path: Path | None,
) -> Path:
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
        "transcript-"
        + hashlib.sha256(os.fsencode(transcript.name)).hexdigest()[:16]
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
            occupied = any(
                path.exists() or path.is_symlink()
                for path in (candidate, partial)
            )
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


def packages_not_provisioned(
    stack: str,
    missing: Sequence[Mapping[str, Any]],
    total_known_download_bytes: int,
    unsized_packages: Sequence[str],
) -> Refusal:
    return _refusal(
        "packages_not_provisioned",
        3,
        f"audio packages pull --stack {stack}",
        missing=list(missing),
        total_known_download_bytes=total_known_download_bytes,
        unsized_packages=list(unsized_packages),
    )


def package_integrity_failed(failed: Sequence[Mapping[str, Any]]) -> Refusal:
    first = failed[0]["package"] if failed else "<package>"
    return _refusal(
        "package_integrity_failed",
        3,
        f"audio packages pull --repair {first}",
        failed=[dict(item) for item in failed],
    )


def package_build_unusable(package: str, product: str) -> Refusal:
    return _refusal(
        "package_build_unusable",
        3,
        f"audio packages pull --repair {package}",
        package=package,
        product=product,
        built=True,
    )


def backend_failed(role: str, backend: str, detail: str, fix: str) -> Refusal:
    return _refusal(
        "backend_failed", 1, fix, role=role, backend=backend, detail=detail
    )


def run_incomplete(
    role: str,
    backend: str,
    detail: str,
    coverage: Mapping[str, Any],
    output: str | Path,
    fix: str,
) -> Refusal:
    return _refusal(
        "run_incomplete",
        4,
        fix,
        role=role,
        backend=backend,
        detail=detail,
        coverage=dict(coverage),
        output=str(output),
    )
