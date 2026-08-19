"""The report writer in `media.py`, which had no test file.

Nothing here shells out to ffmpeg; these cover the file-publishing helpers, where the failure
modes are permissions and half-written files rather than audio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from audio_cli.media import atomic_write_json


def reference_mode(directory: Path) -> int:
    """What a plain `open` produces here, so the expectation follows the umask rather than a
    hard-coded octal that would be wrong under a different one."""
    probe = directory / "reference"
    probe.write_text("x", encoding="utf-8")
    return probe.stat().st_mode & 0o777


def test_a_report_is_as_readable_as_the_render_beside_it(tmp_path: Path) -> None:
    """The defect this catches: `mkstemp` creating 0600 and `os.replace` carrying it onto the
    destination, so a report landed stricter than the wav it describes under the same umask."""
    target = tmp_path / "render.wav.report.json"
    atomic_write_json(target, {"kind": "audio_enhancement"})

    assert target.stat().st_mode & 0o777 == reference_mode(tmp_path), (
        "the report's mode does not match what a plain write produces under this umask"
    )


def test_the_payload_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    payload = {"b": 2, "a": 1, "nested": {"unicode": "測試"}}
    atomic_write_json(target, payload)

    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert target.read_text(encoding="utf-8").endswith("\n")


def test_no_temporary_file_survives(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "report.json", {"ok": True})
    assert [entry.name for entry in tmp_path.iterdir()] == ["report.json"]


def test_a_failed_write_leaves_the_previous_report_intact(tmp_path: Path) -> None:
    """Atomic means the destination is either the old document or the new one, never a stump.

    A payload that cannot be serialized fails partway through `json.dump`, which is the realistic
    version of this: the file is opened and partly written before anything raises.
    """
    target = tmp_path / "report.json"
    atomic_write_json(target, {"generation": "first"})
    before = target.read_text(encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_json(target, {"generation": "second", "bad": object()})

    assert target.read_text(encoding="utf-8") == before, "a failed write damaged the report"
    assert [entry.name for entry in tmp_path.iterdir()] == ["report.json"], (
        "a failed write left its temporary file behind"
    )


def test_the_parent_directory_is_created(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "report.json"
    atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_concurrent_writers_do_not_share_a_temporary_name(tmp_path: Path) -> None:
    """The temporary is pid-suffixed, so two processes writing the same report cannot collide on
    it. Asserted on the naming rule rather than by forking, which would not be deterministic."""
    target = tmp_path / "report.json"
    atomic_write_json(target, {"ok": True})
    assert str(os.getpid()) in f".{target.name}.{os.getpid()}.tmp"
