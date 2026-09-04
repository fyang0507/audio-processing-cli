"""High-level export orchestration and canonical-source safety."""

from __future__ import annotations

from export_test_support import (
    InputMetadata,
    InvalidResultError,
    Path,
    UnsafeOutputError,
    _payload,
    _write,
    export_documents,
    os,
    pwd,
    pytest,
    result_source,
)


def test_high_level_write_is_atomic_and_never_overwrites_input_or_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"canonical")
    payload = _payload(
        [{"segment_id": "seg_0", "text": "Hello."}],
        source_path=str(source),
    )
    transcript = _write(tmp_path / "transcript.json", payload)
    output = tmp_path / "transcript.txt"
    product = export_documents([transcript], "txt", output)
    assert output.read_text(encoding="utf-8") == product.content

    with pytest.raises(UnsafeOutputError):
        export_documents([transcript], "txt", transcript, force=True)
    with pytest.raises(UnsafeOutputError):
        export_documents([transcript], "txt", source, force=True)
    assert source.read_bytes() == b"canonical"


def test_producer_and_export_preserve_literal_existing_user_tilde_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    username = pwd.getpwuid(os.getuid()).pw_name
    relative_source = Path(f"~{username}") / "source.wav"
    relative_source.parent.mkdir()
    relative_source.write_bytes(b"canonical")
    metadata = InputMetadata(str(relative_source), 1.0, "wav", 16_000, 1)
    recorded_source = result_source(metadata, 1.0)
    assert recorded_source["path"] == str((tmp_path / relative_source).resolve())

    transcript = _write(
        tmp_path / "transcript.json",
        _payload(
            [{"segment_id": "seg_0", "text": "Hello."}],
            source_path=recorded_source["path"],
        ),
    )
    with pytest.raises(UnsafeOutputError):
        export_documents(
            [transcript], "txt", output=relative_source, force=True,
        )
    assert relative_source.read_bytes() == b"canonical"


def test_high_level_write_refuses_ambiguous_relative_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_directory = tmp_path / "original"
    original_directory.mkdir()
    source = original_directory / "source.wav"
    source.write_bytes(b"canonical")
    transcript = _write(
        original_directory / "transcript.json",
        _payload(
            [{"segment_id": "seg_0", "text": "Hello."}],
            source_path="source.wav",
        ),
    )

    other_directory = tmp_path / "later"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    with pytest.raises(InvalidResultError, match="source.path must be absolute"):
        export_documents([transcript], "txt", source, force=True)
    assert source.read_bytes() == b"canonical"
