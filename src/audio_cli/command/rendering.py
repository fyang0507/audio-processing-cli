"""Shell-safe rendering for runnable fix commands emitted by the CLI."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path


def command_path_argument(value: str | Path) -> str:
    """Keep an option-like path positional when a repair command is shell-split."""
    rendered = str(value)
    return f"./{rendered}" if rendered.startswith("-") else rendered


def export_command(
    input_paths: Sequence[str | Path],
    output_format: str,
    output: str | Path,
    *,
    force: bool = False,
    timestamps: bool = False,
    provenance: bool = False,
) -> str:
    parts = ["audio", "transcribe", "export"]
    for input_path in input_paths:
        parts.extend(("--input", command_path_argument(input_path)))
    parts.extend(("--format", output_format, "-o", command_path_argument(output)))
    if timestamps:
        parts.append("--timestamps")
    if provenance:
        parts.append("--provenance")
    if force:
        parts.append("--force")
    return shlex.join(parts)


def packages_pull_command(package_ids: Sequence[str], *, repair: bool = False) -> str:
    """Render the exact package selection chosen by the calling feature."""
    if not package_ids:
        raise ValueError("a pull command requires at least one package")
    parts = ["audio", "packages", "pull"]
    if repair:
        parts.append("--repair")
    if any(identifier.startswith("-") for identifier in package_ids):
        parts.append("--")
    return shlex.join([*parts, *package_ids])


def transcribe_plan_command(
    input_path: str | Path,
    stack: str,
    wants: Sequence[str] = (),
    *,
    language: str | None = None,
    vad: str | None = None,
    diarizer: str | None = None,
) -> str:
    parts = [
        "audio",
        "transcribe",
        "plan",
        "--input",
        command_path_argument(input_path),
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


def transcribe_run_command(
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
        "audio",
        "transcribe",
        "run",
        "--input",
        command_path_argument(input_path),
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
