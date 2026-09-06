"""Shipped export refusal shapes against the transcription contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_cli import cli
from tests.docs.shipped_command_test_support import (
    CONTRACT,
    assert_documented_shape,
    configure_isolated_root,
    documented_block,
    write_export_result,
)


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    configure_isolated_root(tmp_path, monkeypatch)


def test_export_timing_refusal_matches_the_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "recordings" / "meeting.m4a"
    source.parent.mkdir()
    source.write_bytes(b"source")
    write_export_result(
        Path("meeting.transcript.json"),
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"diarization": "produced"},
        language="Cantonese",
    )

    assert (
        cli.main(
            [
                "transcribe",
                "export",
                "--input",
                "meeting.transcript.json",
                "--format",
                "srt",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "Subtitle formats require word timing",
        document=CONTRACT,
    )
    documented["fix"] = documented["fix"].replace("/Users/you/recordings/meeting.m4a", str(source))

    assert_documented_shape(actual, documented, "audio transcribe export timing refusal")
    for field in (
        "code",
        "field",
        "provided",
        "requires_capability",
        "note",
        "fix",
    ):
        assert actual[field] == documented[field]


def test_export_recorded_timing_without_words_matches_the_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    write_export_result(
        Path("ordinary.sentences.json"),
        source=Path("/Users/you/recordings/meeting.m4a"),
        segments=[
            {
                "segment_id": "seg_0",
                "text": "Ordinary sentence text with no word stream.",
            }
        ],
        outcomes={"word_timestamps": "produced"},
        language="Cantonese",
    )

    assert (
        cli.main(
            [
                "transcribe",
                "export",
                "--input",
                "ordinary.sentences.json",
                "--format",
                "srt",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "A contradictory legacy or hand-edited result",
        document=CONTRACT,
    )

    assert actual == documented


def test_export_legacy_source_timing_refusal_matches_the_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    write_export_result(
        Path("legacy.transcript.json"),
        source=Path("recordings/meeting.m4a"),
        segments=[{"segment_id": "seg_0", "text": "Legacy sentence."}],
        outcomes={"verbatim": "produced"},
        language="Cantonese",
    )

    assert (
        cli.main(
            [
                "transcribe",
                "export",
                "--input",
                "legacy.transcript.json",
                "--format",
                "srt",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "A result written by an older CLI",
        document=CONTRACT,
    )
    assert actual == documented
