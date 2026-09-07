"""Runnable commands preserve explicit alignment normalization choices."""

import shlex
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli.command import (
    transcribe_plan_command,
    transcribe_run_command,
    with_transcribe_run_options,
)


@pytest.mark.parametrize("render", [transcribe_plan_command, transcribe_run_command])
@pytest.mark.parametrize("limit", [None, 0.501, 150.25])
def test_alignment_option_round_trips_shell_safe_commands(render, limit):
    command = render(
        "-original ' $(no-command).wav",
        "qwen-1.7b",
        ("word_timestamps",),
        language="English",
        alignment_max_overrun_ms=limit,
    )
    parts = shlex.split(command)
    parsed = cli._parser().parse_args(parts[1:])
    assert parsed.input == Path("./-original ' $(no-command).wav")
    assert parsed.language == "English"
    assert parsed.want == "word_timestamps"
    assert parsed.alignment_max_overrun_ms == (str(limit) if limit is not None else None)


@pytest.mark.parametrize(
    "command",
    [
        "inspect the reported stage before retrying",
        "audio packages pull qwen3-forcedaligner",
        "audio transcribe plan --input original.wav --stack qwen-1.7b",
    ],
)
def test_run_option_decoration_leaves_other_remedies_unchanged(command):
    assert with_transcribe_run_options(command, alignment_max_overrun_ms=150) == command


@pytest.mark.parametrize(
    "option", ["--alignment-max-overrun-ms 150.25", "--alignment-max-overrun-ms=150.25"]
)
def test_run_option_decoration_does_not_duplicate_existing_argument(option):
    command = f"audio transcribe run --input original.wav --stack qwen-1.7b --want word_timestamps {option}"
    decorated = with_transcribe_run_options(command, receipt=True, alignment_max_overrun_ms=150.25)
    parts = shlex.split(decorated)
    parsed = cli._parser().parse_args(parts[1:])
    assert parsed.receipt
    assert float(parsed.alignment_max_overrun_ms) == 150.25
    assert sum(value.startswith("--alignment-max-overrun-ms") for value in parts) == 1
