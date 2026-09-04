"""CLI event-only output, file rendering, and summaries."""

from __future__ import annotations

from pathlib import Path

from export_cli_test_support import (
    _write_result,
    cli,
    json,
)


def test_export_cli_event_only_timing_produces_an_empty_subtitle_file(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "events.json",
        source=source,
        segments=[
            {
                "segment_id": "seg_0",
                "text": "[Music]",
                "start": 0.0,
                "end": 1.0,
            }
        ],
        outcomes={"word_timestamps": "produced", "segment_timestamps": "produced"},
    )
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "srt",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_export_cli_does_not_prescribe_an_inert_timing_rerun_for_ordinary_text(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "ordinary.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Ordinary untimed speech."}],
        outcomes={"word_timestamps": "produced"},
    )
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "srt",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "export_input_invalid"
    assert "alignment_unavailable" in error["reason"]


def test_export_cli_writes_subtitle_and_prints_a_summary(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "timed.json",
        source=source,
        segments=[
            {
                "segment_id": "seg_0",
                "text": "Hello.",
                "words": [
                    {
                        "word_id": "w_0",
                        "text": "Hello",
                        "start": 0.2,
                        "end": 0.8,
                    }
                ],
            }
        ],
        outcomes={"word_timestamps": "produced"},
    )
    output = tmp_path / "meeting.srt"
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "srt",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text(encoding="utf-8") == ("1\n00:00:00,200 --> 00:00:00,800\nHello.\n")
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(output)
    assert summary["cues"] == 1
    assert summary["warnings"][0]["code"] == "cue_timing_unvalidated"


def test_export_cli_summary_reports_only_emitted_vtt_voice_tags(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "timed.json",
        source=source,
        segments=[
            {
                "segment_id": "seg_0",
                "text": "Hello.",
                "speaker": " \n\t ",
                "words": [
                    {
                        "word_id": "w_0",
                        "text": "Hello",
                        "start": 0.2,
                        "end": 0.8,
                    }
                ],
            }
        ],
        outcomes={"word_timestamps": "produced", "diarization": "produced"},
    )
    output = tmp_path / "meeting.vtt"
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "vtt",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["speaker_labels_rendered"] is False
    assert "<v " not in output.read_text(encoding="utf-8")
