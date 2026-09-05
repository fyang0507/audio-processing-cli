"""Keep repository-owned text in reviewable, task-shaped files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAX_LINES_EXCLUSIVE = 500

# These files are indivisible generated or recorded artifacts. Their exact bytes are the
# useful unit; hand-splitting them would either be overwritten or destroy the evidence.
EXCEPTIONS = {
    "uv.lock": "uv-generated dependency lock",
    "src/audio_cli/environments/locks/mlx.txt": "uv-generated hashed environment lock",
    "src/audio_cli/environments/locks/torch-firered.txt": ("uv-generated hashed environment lock"),
    "src/audio_cli/environments/locks/torch-vibevoice.txt": (
        "uv-generated hashed environment lock"
    ),
    "model_tests/benchmark/manifests/spice_vf19a_cantonese_interview30m.json": (
        "canonical recorded corpus manifest"
    ),
    "tests/fixtures/fluidaudio/ProcessCommand.swift": (
        "exact pinned upstream source fixture used to verify patch applicability"
    ),
}


def _tracked_files() -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 and not (REPO / ".git").exists():
        pytest.skip("the tracked-file line budget is a Git-checkout-only repository gate")
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    relative_paths = tuple(entry.decode() for entry in completed.stdout.split(b"\0") if entry)
    return tuple(
        relative_path for relative_path in relative_paths if (REPO / relative_path).is_file()
    )


def _line_count(relative_path: str) -> int:
    return len((REPO / relative_path).read_bytes().splitlines())


def test_tracked_files_stay_below_the_line_budget() -> None:
    tracked = set(_tracked_files())
    stale_exceptions = sorted(set(EXCEPTIONS) - tracked)
    assert not stale_exceptions, f"remove stale line-budget exceptions: {stale_exceptions}"

    unexplained = {
        relative_path: _line_count(relative_path)
        for relative_path in sorted(tracked - set(EXCEPTIONS))
        if _line_count(relative_path) >= MAX_LINES_EXCLUSIVE
    }
    assert not unexplained, (
        f"tracked files must stay below {MAX_LINES_EXCLUSIVE} lines; "
        f"split these files or document a narrow exception: {unexplained}"
    )


def test_line_budget_exceptions_remain_narrow_and_explained() -> None:
    assert all(reason.strip() for reason in EXCEPTIONS.values())
    assert all(_line_count(relative_path) >= MAX_LINES_EXCLUSIVE for relative_path in EXCEPTIONS), (
        "remove an exception once its file falls below the budget"
    )
