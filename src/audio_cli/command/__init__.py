"""Shared command presentation primitives for the newer CLI surfaces."""

from .progress import ProgressReporter
from .refusals import (
    Refusal,
    build_refusal,
    output_exists,
    output_is_canonical_input,
    output_path_invalid,
)
from .rendering import (
    command_path_argument,
    export_command,
    packages_pull_command,
    transcribe_plan_command,
    transcribe_run_command,
    with_transcribe_run_options,
)

__all__ = [
    "ProgressReporter",
    "Refusal",
    "build_refusal",
    "command_path_argument",
    "export_command",
    "output_exists",
    "output_is_canonical_input",
    "output_path_invalid",
    "packages_pull_command",
    "transcribe_plan_command",
    "transcribe_run_command",
    "with_transcribe_run_options",
]
