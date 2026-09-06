"""Exercise the public readable-timestamp flag, refusals, and executable retry."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
from export_cli_test_support import _write_result
from shipped_command_test_support import documented_block, documented_fenced_block

from audio_cli import cli

EXPORT_CONTRACT = Path(__file__).resolve().parents[1] / "TRANSCRIBE_CONTRACT.md"


@pytest.mark.parametrize("output_format", ["txt", "md"])
@pytest.mark.parametrize("timing", ["segment_timestamps", "word_timestamps"])
def test_cli_readable_timestamps_and_default_text(
    tmp_path: Path, capsys, output_format: str, timing: str
) -> None:
    segment = {"segment_id": "seg_0", "speaker": "S1", "text": "Hello, WORLD!"}
    if timing == "segment_timestamps":
        segment.update(start=0.125, end=1.875)
    else:
        segment["words"] = [
            {"word_id": "w_0", "text": "hello", "start": 0.125, "end": 0.25},
            {"word_id": "w_1", "text": "world", "start": 1.125, "end": 1.875},
        ]
    path = _write_result(
        tmp_path / "transcript.json",
        source=tmp_path / "original.wav",
        segments=[segment],
        outcomes={timing: "produced", "diarization": "produced"},
        stack="vibevoice" if timing == "segment_timestamps" else "qwen-0.6b",
    )
    before = path.read_bytes()
    args = ["export", "--input", str(path), "--format", output_format]
    assert cli.main([*args, "--timestamps"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    header = "# Transcript\n\n" if output_format == "md" else ""
    assert captured.out == header + "[00:00:00.125 --> 00:00:01.875] [S1] Hello, WORLD!\n"
    assert cli.main(args) == 0
    assert capsys.readouterr().out == header + "[S1] Hello, WORLD!\n"
    assert path.read_bytes() == before


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_cli_untimed_qwen_refusal_matches_contract_and_writes_nothing(
    tmp_path: Path, monkeypatch, capsys, output_format: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_result(
        Path("meeting.transcript.json"),
        source=tmp_path / "original.wav",
        segments=[{"segment_id": "seg_0", "text": "Untimed Qwen speech."}],
        outcomes={},
    )
    assert (
        cli.main(
            [
                "export",
                "--input",
                "meeting.transcript.json",
                "--format",
                output_format,
                "--timestamps",
                "-o",
                "result.txt",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not Path("result.txt").exists()
    refusal = json.loads(captured.err)
    assert refusal == documented_block("For example, `audio export", document=EXPORT_CONTRACT)


@pytest.mark.parametrize("output_format", ["jsonl", "srt", "vtt"])
def test_cli_rejects_readable_flag_for_other_formats(tmp_path: Path, capsys, output_format) -> None:
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(tmp_path / "missing.json"),
                "--format",
                output_format,
                "--timestamps",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    refusal = json.loads(captured.err)
    expected = documented_block("With SRT, VTT, or JSONL", document=EXPORT_CONTRACT)
    expected["format"] = output_format
    assert refusal == expected


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_existing_output_retry_preserves_timestamps_and_shell_safe_paths(
    tmp_path: Path, monkeypatch, capsys, output_format: str
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"canonical original")
    transcript = Path("-meeting's transcript.json")
    _write_result(
        transcript,
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello.", "start": 0.2, "end": 0.8}],
        outcomes={"segment_timestamps": "produced"},
        stack="vibevoice",
    )
    original = transcript.read_bytes()
    output = Path("-out's transcript.txt")
    output.write_text("old")
    assert (
        cli.main(
            [
                "export",
                f"--input={transcript}",
                "--format",
                output_format,
                "--timestamps",
                f"--output={output}",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    refusal = json.loads(captured.err)
    assert refusal["code"] == "output_exists"
    retry = shlex.split(refusal["fix"])
    assert "--timestamps" in retry and "--force" in retry
    assert cli.main(retry[1:]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["segments"] == 1
    assert set(summary) == {"input", "output", "format", "segments"}
    header = "# Transcript\n\n" if output_format == "md" else ""
    assert output.read_text() == header + "[00:00:00.200 --> 00:00:00.800] Hello.\n"
    assert source.read_bytes() == b"canonical original"
    assert transcript.read_bytes() == original


def test_readable_markdown_matches_documented_happy_path(tmp_path: Path, capsys) -> None:
    # Use the documented saved result (including its full word stream), not a second
    # set of prose-derived time constants which can drift independently.
    fragment = (
        Path(__file__).resolve().parents[1]
        / "docs/transcribe-happy-path/20-interview-run-and-export.md"
    )
    payload = json.loads(fragment.read_text().split("```json\n", 1)[1].split("```", 1)[0])
    payload["provenance"]["plan"] = {"execution": {"partition": "fixture"}}
    path = tmp_path / "meeting.timed.json"
    path.write_text(json.dumps(payload))
    assert cli.main(["export", "--input", str(path), "--format", "md", "--timestamps"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == documented_fenced_block("### 1.6 Export readable timestamps", "markdown")
