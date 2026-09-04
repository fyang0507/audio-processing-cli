"""Fixed-shape request and output refusals.

These payloads print bare on stderr.  The older shipped commands retain their historical
``{"error": ...}`` envelope; sharing their exception type would silently erase that boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.command import (
    Refusal,
    build_refusal,
    transcribe_plan_command,
    transcribe_run_command,
)
from audio_cli.command import command_path_argument as command_path_argument
from audio_cli.command import output_exists as _output_exists
from audio_cli.command import output_is_canonical_input as output_is_canonical_input
from audio_cli.command import output_path_invalid as output_path_invalid

from .. import stacks


def stack_required(input_path: str | Path | None, wants: Sequence[str]) -> Refusal:
    chosen_input = input_path or "meeting.m4a"
    return build_refusal(
        "stack_required",
        2,
        transcribe_plan_command(chosen_input, "qwen-1.7b", wants),
        field="--stack",
        allowed=list(stacks.stack_ids()),
        stacks={
            identifier: definition.characterization
            for identifier, definition in stacks.stack_definitions().items()
        },
    )


def input_required(stack: str, wants: Sequence[str]) -> Refusal:
    return build_refusal(
        "input_required",
        2,
        transcribe_plan_command("meeting.m4a", stack, wants),
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
    return build_refusal(
        "capability_unknown",
        2,
        transcribe_plan_command(input_path, stack.id, fixed),
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
    return build_refusal(
        "capability_unsatisfiable_on_stack",
        2,
        transcribe_plan_command(input_path, preferred, wants),
        capability=capability,
        allowed=allowed,
        available_on_stack=stacks.availability_groups(stack),
    )


def capability_unsupported(capability: str, reason: str, fix: str) -> Refusal:
    return build_refusal(
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
    return build_refusal(
        "option_unsupported_on_stack",
        2,
        transcribe_plan_command(input_path, stack.id, wants),
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
    return build_refusal(
        "option_value_unsupported",
        2,
        transcribe_plan_command(input_path, stack.id, wants, **kwargs),
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
    return build_refusal(
        "pin_conflicts_with_native_capability",
        2,
        transcribe_plan_command(input_path, stack.id, wants),
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
    return build_refusal(
        "range_invalid",
        2,
        transcribe_run_command(
            input_path,
            stack,
            wants,
            language=language,
            vad=vad,
            diarizer=diarizer,
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
    return _output_exists(
        output,
        existing,
        transcribe_run_command(
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
    )


def packages_not_provisioned(
    stack: str,
    missing: Sequence[Mapping[str, Any]],
    total_known_download_bytes: int,
    unsized_packages: Sequence[str],
) -> Refusal:
    return build_refusal(
        "packages_not_provisioned",
        3,
        f"audio packages pull --stack {stack}",
        missing=list(missing),
        total_known_download_bytes=total_known_download_bytes,
        unsized_packages=list(unsized_packages),
    )


def package_integrity_failed(failed: Sequence[Mapping[str, Any]]) -> Refusal:
    first = failed[0]["package"] if failed else "<package>"
    return build_refusal(
        "package_integrity_failed",
        3,
        f"audio packages pull --repair {first}",
        failed=[dict(item) for item in failed],
    )


def package_build_unusable(package: str, product: str) -> Refusal:
    return build_refusal(
        "package_build_unusable",
        3,
        f"audio packages pull --repair {package}",
        package=package,
        product=product,
        built=True,
    )


def backend_failed(role: str, backend: str, detail: str, fix: str) -> Refusal:
    return build_refusal("backend_failed", 1, fix, role=role, backend=backend, detail=detail)


def run_incomplete(
    role: str,
    backend: str,
    detail: str,
    coverage: Mapping[str, Any],
    output: str | Path,
    fix: str,
) -> Refusal:
    return build_refusal(
        "run_incomplete",
        4,
        fix,
        role=role,
        backend=backend,
        detail=detail,
        coverage=dict(coverage),
        output=str(output),
    )
