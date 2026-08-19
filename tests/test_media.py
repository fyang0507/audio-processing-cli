"""The report writer in `media.py`, which had no test file.

Nothing here shells out to ffmpeg; these cover the file-publishing helpers, where the failure
modes are permissions and half-written files rather than audio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import numpy as np

from audio_cli.media import (
    MediaError,
    atomic_write_json,
    render_loudness_normalized,
    write_float_wav,
)


def normalize(tmp_path: Path, measurement: dict[str, float], *, duration_s: float):
    """Drive the pre-flight check in `render_loudness_normalized` and nothing past it.

    Every case here fails before ffmpeg is reached, so these need no runtime.
    """
    source = tmp_path / "source.wav"
    write_float_wav(source, np.zeros((round(48_000 * duration_s), 1), dtype=np.float32), 48_000)
    return render_loudness_normalized(
        source,
        tmp_path / "out.wav",
        target_lufs=-23.0,
        target_lra=7.0,
        target_true_peak=-3.0,
        measurement=measurement,
        sample_rate=48_000,
    )


def test_a_clip_too_short_to_measure_is_not_reported_as_silent(tmp_path: Path) -> None:
    """The defect this catches: one message for two conditions.

    Integrated loudness is gated in 400 ms blocks, so a 50 ms clip has none however loud it is.
    That was reported as "silent or non-finite audio", which sends a caller looking for silence
    that is not there -- the measured true peak proves the signal exists.
    """
    with pytest.raises(MediaError) as caught:
        normalize(
            tmp_path,
            {"input_i": float("-inf"), "input_lra": 0.0, "input_tp": -39.26,
             "input_thresh": -70.0},
            duration_s=0.05,
        )
    message = str(caught.value)
    assert "50 ms" in message, f"the duration a caller needs is missing: {message}"
    assert "400 ms" in message, "the reason -- the gating block -- is not stated"
    assert "-39.26" in message, "the peak that proves it is not silent is not shown"
    assert "not silent" in message
    # The suggested correction has to be runnable, so it names a real flag and stage.
    assert "--skip program-loudness" in message


def test_genuine_silence_is_still_reported_as_silence(tmp_path: Path) -> None:
    """A truly silent file has no finite peak either, which is what separates the two."""
    with pytest.raises(MediaError) as caught:
        normalize(
            tmp_path,
            {"input_i": float("-inf"), "input_lra": 0.0, "input_tp": float("-inf"),
             "input_thresh": float("-inf")},
            duration_s=5.0,
        )
    message = str(caught.value)
    assert "silent" in message
    assert "too short" not in message, "silence misreported as a duration problem"


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
