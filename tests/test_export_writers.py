"""Descriptor-safe and atomic export writer behavior."""

from __future__ import annotations

# ruff: noqa: F403, F405
from export_test_support import *

def test_export_writer_closes_descriptor_when_temporary_identity_capture_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "transcript.txt"
    captured_descriptor: int | None = None

    def fail_identity(descriptor: int, _path: Path):
        nonlocal captured_descriptor
        captured_descriptor = descriptor
        raise OSError("identity unavailable")

    monkeypatch.setattr(
        "audio_cli.export.writers.file_identity_from_descriptor",
        fail_identity,
    )

    with pytest.raises(OutputWriteError, match="identity unavailable"):
        write_text_atomic(output, "hello\n")

    assert captured_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(captured_descriptor)
    assert not output.exists()

def test_atomic_writer_refuses_collisions_and_protected_paths(tmp_path: Path) -> None:
    destination = tmp_path / "out.txt"
    write_text_atomic(destination, "one\n")
    assert destination.read_bytes() == b"one\n"
    with pytest.raises(OutputExistsError):
        write_text_atomic(destination, "two\n")
    assert destination.read_text(encoding="utf-8") == "one\n"
    write_text_atomic(destination, "two\n", force=True)
    assert destination.read_text(encoding="utf-8") == "two\n"

    protected = tmp_path / "source.wav"
    protected.write_bytes(b"source")
    with pytest.raises(UnsafeOutputError):
        write_text_atomic(protected, "destroyed", force=True, protected_paths=[protected])
    assert protected.read_bytes() == b"source"

    directory = tmp_path / "directory-output"
    directory.mkdir()
    with pytest.raises(OutputExistsError) as directory_error:
        write_text_atomic(directory, "destroyed", force=True)
    assert directory_error.value.replaceable is False
    assert list(tmp_path.glob(f".{directory.name}.*.tmp")) == []

    loop = tmp_path / "loop"
    loop.symlink_to(loop.name)
    with pytest.raises(OutputWriteError, match="Symlink loop"):
        write_text_atomic(loop, "destroyed", force=True)
    assert loop.is_symlink()

    symlink_target = tmp_path / "symlink-target.txt"
    symlink_target.write_text("keep\n", encoding="utf-8")
    symlink_output = tmp_path / "symlink-output.txt"
    symlink_output.symlink_to(symlink_target)
    with pytest.raises(OutputExistsError) as symlink_error:
        write_text_atomic(symlink_output, "destroyed", force=True)
    assert symlink_error.value.replaceable is False
    assert symlink_output.is_symlink()
    assert symlink_target.read_text(encoding="utf-8") == "keep\n"

    protected_loop = tmp_path / "protected-loop"
    protected_loop.symlink_to(protected_loop.name)
    separate_output = tmp_path / "separate.txt"
    with pytest.raises(OutputWriteError) as caught:
        write_text_atomic(
            separate_output,
            "destroyed",
            protected_paths=[protected_loop],
        )
    assert caught.value.output == separate_output
    assert f"protected path {protected_loop}" in caught.value.reason
    assert not separate_output.exists()

    alias = tmp_path / "source-alias.wav"
    os.link(protected, alias)
    with pytest.raises(UnsafeOutputError):
        write_text_atomic(alias, "destroyed", force=True, protected_paths=[protected])
    assert protected.read_bytes() == b"source"


def test_atomic_writer_accepts_a_name_max_destination(tmp_path: Path) -> None:
    limit = os.pathconf(tmp_path, "PC_NAME_MAX")
    suffix = ".txt"
    destination = tmp_path / ("a" * (limit - len(suffix)) + suffix)

    write_text_atomic(destination, "published\n")

    assert destination.read_text(encoding="utf-8") == "published\n"


def test_export_writer_cannot_follow_a_parent_swapped_after_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "out.txt"
    external.write_text("KEEP\n", encoding="utf-8")

    def swap_parent():
        safe.rename(tmp_path / "safe-old")
        safe.symlink_to(outside, target_is_directory=True)
        return type("Uuid", (), {"hex": "fixed"})()

    monkeypatch.setattr("audio_cli.export.writers.uuid.uuid4", swap_parent)
    with pytest.raises(OutputWriteError, match="directory identity changed"):
        write_text_atomic(
            safe / "out.txt",
            "DESTROYED\n",
            force=True,
            protected_paths=[external],
        )

    assert external.read_text(encoding="utf-8") == "KEEP\n"
    assert not list(tmp_path.glob(".audio-export-*.tmp"))


def test_export_writer_preserves_a_protected_file_renamed_to_output_mid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"canonical")
    output = tmp_path / "out.txt"
    output.write_bytes(b"replaceable")

    def move_source_to_output():
        os.replace(source, output)
        return type("Uuid", (), {"hex": "fixed"})()

    monkeypatch.setattr(
        "audio_cli.export.writers.uuid.uuid4", move_source_to_output
    )
    with pytest.raises(UnsafeOutputError):
        write_text_atomic(
            output,
            "replacement\n",
            force=True,
            protected_paths=[source],
        )

    assert not source.exists()
    assert output.read_bytes() == b"canonical"
    assert not list(tmp_path.glob(".audio-export-*.tmp"))


def test_export_writer_rejects_a_substituted_private_temporary_at_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out.txt"
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
    with pytest.raises(OutputWriteError, match="temporary changed identity"):
        write_text_atomic(output, "replacement\n", force=True)

    assert raced is True
    assert output.read_bytes() == b"previous"
    assert (tmp_path / "held-legitimate.tmp").read_bytes() == b"replacement\n"
    assert substituted_name is not None
    assert (tmp_path / substituted_name).read_bytes() == b"substituted"
