"""The descriptor-cwd child process uses its held directory and preserves command results."""

from __future__ import annotations

import os
import sys

import pytest

from audio_cli.media import bound_directory, run_in_directory


def test_command_changes_only_held_directory_after_original_path_is_replaced(tmp_path):
    target = tmp_path / "work"
    target.mkdir()
    (target / "remove.txt").write_text("ours")
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "remove.txt"
    sentinel.write_text("external")
    parent_cwd = os.getcwd()
    with (
        pytest.raises(OSError, match="directory identity changed"),
        bound_directory(target, root=tmp_path, create=False) as descriptor,
    ):
        target.rename(tmp_path / "moved")
        target.symlink_to(external, target_is_directory=True)
        result = run_in_directory(
            [
                sys.executable,
                "-B",
                "-c",
                "from pathlib import Path; Path('remove.txt').unlink()",
            ],
            descriptor,
        )
        assert result.returncode == 0, result.stderr
        assert os.fstat(descriptor), "the helper closed its caller's descriptor"
    assert os.getcwd() == parent_cwd
    assert sentinel.read_text() == "external"
    assert not (tmp_path / "moved/remove.txt").exists()


def test_descriptor_cwd_preserves_target_exit_status_and_output(tmp_path):
    with bound_directory(tmp_path, root=tmp_path, create=False) as descriptor:
        result = run_in_directory(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys; print('out'); print('err',file=sys.stderr); sys.exit(7)",
            ],
            descriptor,
        )
    assert result.returncode == 7
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
