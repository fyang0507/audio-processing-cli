"""Structured diagnostic copies preserve exact bytes and never follow file aliases."""

import os

import pytest

from audio_cli.media import retain_diagnostic_file


def test_structured_diagnostic_copy_is_exact_private_and_exclusive(tmp_path):
    source = tmp_path / "stage.json"
    source.write_bytes(b'{"malformed":\xff\r\n')
    target = tmp_path / "retained.json"
    retain_diagnostic_file(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_mode & 0o777 == 0o600
    source.write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        retain_diagnostic_file(source, target)
    assert target.read_bytes() == b'{"malformed":\xff\r\n'


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_nonregular_response_refuses_without_following_or_blocking(tmp_path, kind):
    source = tmp_path / "response"
    protected = tmp_path / "source.media"
    protected.write_bytes(b"original")
    if kind == "symlink":
        source.symlink_to(protected)
    elif kind == "directory":
        source.mkdir()
    else:
        os.mkfifo(source)
    target = tmp_path / "retained.json"
    with pytest.raises(OSError):
        retain_diagnostic_file(source, target)
    assert not target.exists()
    assert protected.read_bytes() == b"original"
