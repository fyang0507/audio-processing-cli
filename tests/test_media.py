"""The report writer in `media.py`, which had no test file.

Nothing here shells out to ffmpeg; these cover the file-publishing helpers, where the failure
modes are permissions and half-written files rather than audio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpy as np

from audio_cli import media as media_module
from audio_cli.media import (
    MediaError,
    ProtectedOutputError,
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


def test_nonforce_json_publication_never_clobbers_an_existing_entry(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_text("owner\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        atomic_write_json(target, {"ok": True}, force=False)

    assert target.read_text(encoding="utf-8") == "owner\n"


def test_exclusive_temporary_does_not_follow_a_precreated_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = tmp_path / "canonical.wav"
    canonical.write_bytes(b"canonical")
    target = tmp_path / "report.json"
    monkeypatch.setattr(
        "audio_cli.media.uuid.uuid4", lambda: SimpleNamespace(hex="fixed")
    )
    temporary = tmp_path / f".audio-write-{os.getpid()}-fixed.tmp"
    temporary.symlink_to(canonical)

    with pytest.raises(FileExistsError):
        atomic_write_json(target, {"ok": True})

    assert canonical.read_bytes() == b"canonical"
    assert temporary.is_symlink()
    assert not target.exists()


def test_atomic_writer_closes_descriptor_when_temporary_identity_capture_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "report.json"
    captured_descriptor: int | None = None

    def fail_identity(descriptor: int, _path: Path):
        nonlocal captured_descriptor
        captured_descriptor = descriptor
        raise OSError("identity unavailable")

    monkeypatch.setattr(media_module, "file_identity_from_descriptor", fail_identity)

    with pytest.raises(OSError, match="identity unavailable"):
        atomic_write_json(target, {"ok": True})

    assert captured_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(captured_descriptor)
    assert not target.exists()


def test_atomic_writer_cannot_follow_a_parent_swapped_after_open(
    tmp_path: Path, monkeypatch,
) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "report.json"
    external.write_bytes(b"canonical")

    def swap_parent():
        safe.rename(tmp_path / "safe-old")
        safe.symlink_to(outside, target_is_directory=True)
        return SimpleNamespace(hex="fixed")

    monkeypatch.setattr("audio_cli.media.uuid.uuid4", swap_parent)
    with pytest.raises(OSError, match="directory identity changed"):
        atomic_write_json(safe / "report.json", {"destroyed": True})

    assert external.read_bytes() == b"canonical"


def test_force_writer_does_not_have_an_identity_check_replace_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"canonical")
    output = tmp_path / "report.json"
    output.write_bytes(b"replaceable")
    real_exchange = media_module._rename_exchange
    raced = False

    def race_at_exchange(directory_descriptor, left_name, right_name):
        nonlocal raced
        if not raced:
            raced = True
            os.replace(source, output)
        return real_exchange(directory_descriptor, left_name, right_name)

    monkeypatch.setattr(media_module, "_rename_exchange", race_at_exchange)
    with pytest.raises(ProtectedOutputError):
        atomic_write_json(
            output,
            {"destroyed": True},
            force=True,
            protected_paths=(source,),
        )

    assert raced is True
    assert not source.exists()
    assert output.read_bytes() == b"canonical"
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["report.json"]


def test_force_writer_keeps_an_existing_destination_continuously_addressable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    output.write_bytes(b"previous")
    real_exchange = media_module._rename_exchange
    observations: list[bool] = []

    def observe_exchange(directory_descriptor, left_name, right_name):
        observations.append(output.exists())
        real_exchange(directory_descriptor, left_name, right_name)
        observations.append(output.exists())

    monkeypatch.setattr(media_module, "_rename_exchange", observe_exchange)
    atomic_write_json(output, {"generation": "next"}, force=True)

    assert observations == [True, True]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "generation": "next"
    }


def test_force_writer_rejects_a_substituted_private_temporary_at_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "report.json"
    output.write_bytes(b"previous")
    real_exchange = media_module._rename_exchange
    raced = False
    substituted_name: str | None = None

    def substitute_temporary(directory_descriptor, left_name, right_name):
        nonlocal raced, substituted_name
        if not raced:
            raced = True
            substituted_name = left_name
            os.rename(
                left_name,
                "held-legitimate.tmp",
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            descriptor = os.open(
                left_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_descriptor,
            )
            try:
                os.write(descriptor, b"substituted")
            finally:
                os.close(descriptor)
        return real_exchange(directory_descriptor, left_name, right_name)

    monkeypatch.setattr(media_module, "_rename_exchange", substitute_temporary)
    with pytest.raises(OSError, match="temporary changed identity during publication"):
        atomic_write_json(output, {"generation": "next"}, force=True)

    assert raced is True
    assert output.read_bytes() == b"previous"
    assert (tmp_path / "held-legitimate.tmp").read_bytes().startswith(b"{")
    assert substituted_name is not None
    assert (tmp_path / substituted_name).read_bytes() == b"substituted"


def test_failed_atomic_rollback_preserves_the_observed_protected_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"canonical")
    output = tmp_path / "report.json"
    output.write_bytes(b"replaceable")
    real_exchange = media_module._rename_exchange
    calls = 0

    def fail_rollback(directory_descriptor, left_name, right_name):
        nonlocal calls
        calls += 1
        if calls == 1:
            os.replace(source, output)
            return real_exchange(directory_descriptor, left_name, right_name)
        raise OSError("synthetic rollback failure")

    monkeypatch.setattr(media_module, "_rename_exchange", fail_rollback)
    with pytest.raises(ProtectedOutputError) as caught:
        atomic_write_json(
            output,
            {"generation": "new"},
            force=True,
            protected_paths=(source,),
        )

    assert calls == 2
    assert caught.value.preserved_at is not None
    assert caught.value.preserved_at.read_bytes() == b"canonical"
