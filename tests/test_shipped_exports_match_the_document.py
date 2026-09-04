"""Shipped export output and refusal shapes against the specification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shipped_command_test_support import (
    CONTRACT,
    assert_documented_shape,
    configure_isolated_root,
    documented_block,
    documented_fenced_block,
    documented_text_block,
)

from audio_cli import cli
from audio_cli.transcribe.result import NormalizedResult, serialize_result


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    configure_isolated_root(tmp_path, monkeypatch)


def _write_export_result(
    path: Path,
    *,
    source: str | Path,
    segments: list[dict],
    outcomes: dict[str, str],
    language: str | None,
) -> None:
    optional_arrays = {"turns": []} if "diarization" in outcomes else {}
    payload = serialize_result(NormalizedResult(
        source={
            "path": str(source),
            "duration_seconds": 10.0,
            "timebase": "seconds",
        },
        segments=segments,
        abstentions=[],
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": outcomes,
            "observed": {},
            "plan": {
                "roles": {"asr": {"config": {"language": language}}},
            },
        },
        requested_capabilities=frozenset(outcomes),
        **optional_arrays,
    ))
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_success_summary_matches_the_happy_path(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"source")
    _write_export_result(
        Path("meeting.timed.json"),
        source=source,
        segments=[
            {
                "segment_id": "seg_0",
                "text": "好，我們今天想聊一下你的工作。",
                "words": [{
                    "word_id": "w_0",
                    "text": "好我們今天想聊一下你的工作",
                    "start": 2.31,
                    "end": 4.71,
                }],
            },
            {
                "segment_id": "seg_1",
                "text": "嗯，好啊，我做咗五年設計。",
                "words": [{
                    "word_id": "w_1",
                    "text": "嗯好啊我做咗五年設計",
                    "start": 5.12,
                    "end": 7.44,
                }],
            },
        ],
        outcomes={"word_timestamps": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "meeting.timed.json", "--format", "srt",
        "-o", "meeting.srt",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    actual = json.loads(captured.out)
    documented = documented_block("### 1.5 Export subtitles")

    assert_documented_shape(actual, documented, "audio export --format srt")
    assert actual["warnings"][0]["detail"] == documented["warnings"][0]["detail"]
    assert Path("meeting.srt").read_text(encoding="utf-8") == documented_text_block(
        "Exit 0. `meeting.srt`:"
    )


def test_export_vtt_matches_the_documented_voice_tag_render(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"source")
    _write_export_result(
        Path("demo.transcript.json"),
        source=source,
        segments=[
            {
                "segment_id": "seg_0",
                "speaker": "0",
                "start": 0.0,
                "end": 4.52,
                "text": (
                    "So, um, this is the new editor. "
                    "You can, like, drag a clip here."
                ),
                "words": [
                    {
                        "word_id": "w_0",
                        "text": "So um this is the new editor",
                        "start": 0.31,
                        "end": 2.21,
                    },
                    {
                        "word_id": "w_1",
                        "text": "You can like drag a clip here",
                        "start": 2.58,
                        "end": 4.52,
                    },
                ],
            },
            {
                "segment_id": "seg_1",
                "start": 4.52,
                "end": 6.08,
                "text": "[Environmental Sounds]",
            },
            {
                "segment_id": "seg_2",
                "speaker": "1",
                "start": 6.08,
                "end": 9.41,
                "text": "And it renders straight away?",
                "words": [{
                    "word_id": "w_2",
                    "text": "And it renders straight away",
                    "start": 6.22,
                    "end": 7.86,
                }],
            },
        ],
        outcomes={
            "verbatim": "produced",
            "diarization": "produced",
            "segment_timestamps": "produced",
            "word_timestamps": "produced",
        },
        language=None,
    )

    assert cli.main([
        "export", "--input", "demo.transcript.json", "--format", "vtt",
        "-o", "demo.vtt",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["speaker_labels_rendered"] is True
    assert summary["cues"] == 3
    assert Path("demo.vtt").read_text(encoding="utf-8") == documented_text_block(
        "Exit 0. `demo.vtt`:"
    )


@pytest.mark.parametrize(
    ("output_format", "output_name", "document_anchor", "fence_language"),
    [
        ("txt", "meeting.txt", "Exit 0, `meeting.txt`:", "text"),
        ("md", "meeting.md", "Exit 0, `meeting.md`:", "markdown"),
        ("jsonl", "meeting.jsonl", "Exit 0, `meeting.jsonl`:", "jsonl"),
    ],
)
def test_untimed_file_exports_match_the_happy_path(
    output_format: str,
    output_name: str,
    document_anchor: str,
    fence_language: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"source")
    _write_export_result(
        Path("meeting.transcript.json"),
        source=source,
        segments=[
            {"segment_id": "seg_0", "text": "First."},
            {"segment_id": "seg_1", "text": "Second."},
        ],
        outcomes={"verbatim": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "meeting.transcript.json", "--format", output_format,
        "-o", output_name,
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    actual = json.loads(captured.out)
    documented = documented_block(
        "An on-disk untimed format reports segments"
    )

    assert set(actual) == set(documented)
    assert actual["input"] == documented["input"]
    assert actual["segments"] == documented["segments"]
    assert actual["format"] == output_format
    assert actual["output"] == output_name
    assert Path(output_name).read_text(encoding="utf-8") == documented_fenced_block(
        document_anchor, fence_language
    )


def test_multi_input_export_summary_and_merged_order_match_the_happy_path(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"source")
    base_plan = {
        "roles": {"asr": {"config": {"language": "Cantonese"}}},
        "execution": {},
    }
    outcomes = {"segment_timestamps": "produced"}

    partial = serialize_result(NormalizedResult(
        source={
            "path": str(source),
            "duration_seconds": 10.0,
            "timebase": "seconds",
        },
        segments=[
            {"segment_id": "seg_0", "text": "First.", "start": 1.0, "end": 2.0},
            {"segment_id": "seg_1", "text": "Second.", "start": 3.0, "end": 4.0},
        ],
        abstentions=[],
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": outcomes,
            "observed": {},
            "plan": base_plan,
        },
        requested_capabilities=frozenset(outcomes),
        complete=False,
        coverage={
            "scope_intervals": [[0.0, 10.0]],
            "covered_through_seconds": 5.0,
            "covered_fraction": 0.5,
            "covered_intervals": [[0.0, 5.0]],
            "missing_intervals": [[5.0, 10.0]],
            "units_total": 4,
            "units_completed": 2,
        },
    ))
    Path("meeting.timed.partial.json").write_text(
        json.dumps(partial), encoding="utf-8"
    )

    rest_plan = json.loads(json.dumps(base_plan))
    rest_plan["execution"]["range"] = {
        "requested": [5.0, 10.0],
        "selected_unit_scope": [5.0, 10.0],
    }
    rest = serialize_result(NormalizedResult(
        source=partial["source"],
        segments=[
            {"segment_id": "seg_0", "text": "Third.", "start": 6.0, "end": 7.0},
            {"segment_id": "seg_1", "text": "Fourth.", "start": 8.0, "end": 9.0},
        ],
        abstentions=[],
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": outcomes,
            "observed": {},
            "plan": rest_plan,
        },
        requested_capabilities=frozenset(outcomes),
    ))
    Path("meeting.timed.rest.json").write_text(json.dumps(rest), encoding="utf-8")

    assert cli.main([
        "export",
        "--input", "meeting.timed.partial.json",
        "--input", "meeting.timed.rest.json",
        "--format", "jsonl",
        "-o", "meeting.timed.merged.jsonl",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    actual = json.loads(captured.out)
    documented = documented_block("After the ranged resume finishes")
    assert actual == documented

    merged = [
        json.loads(line)
        for line in Path("meeting.timed.merged.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [segment["start"] for segment in merged] == [1.0, 3.0, 6.0, 8.0]
    segment_ids = [segment["segment_id"] for segment in merged]
    assert segment_ids == ["seg_0", "seg_1", "seg_2", "seg_3"]
    assert len(segment_ids) == len(set(segment_ids))


def test_export_timing_refusal_matches_the_contract(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "recordings" / "meeting.m4a"
    source.parent.mkdir()
    source.write_bytes(b"source")
    _write_export_result(
        Path("meeting.transcript.json"),
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"diarization": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "meeting.transcript.json", "--format", "srt",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "Subtitle formats require word timing", document=CONTRACT,
    )
    documented["fix"] = documented["fix"].replace(
        "/Users/you/recordings/meeting.m4a", str(source)
    )

    assert_documented_shape(actual, documented, "audio export timing refusal")
    for field in (
        "code", "field", "provided", "requires_capability", "note", "fix",
    ):
        assert actual[field] == documented[field]


def test_export_recorded_timing_without_words_matches_the_contract(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_export_result(
        Path("ordinary.sentences.json"),
        source=Path("/Users/you/recordings/meeting.m4a"),
        segments=[{
            "segment_id": "seg_0",
            "text": "Ordinary sentence text with no word stream.",
        }],
        outcomes={"word_timestamps": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "ordinary.sentences.json", "--format", "srt",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "A contradictory legacy or hand-edited result", document=CONTRACT,
    )

    assert actual == documented


def test_export_legacy_source_timing_refusal_matches_the_contract(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_export_result(
        Path("legacy.transcript.json"),
        source=Path("recordings/meeting.m4a"),
        segments=[{"segment_id": "seg_0", "text": "Legacy sentence."}],
        outcomes={"verbatim": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "legacy.transcript.json", "--format", "srt",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "A result written by an older CLI", document=CONTRACT,
    )
    assert actual == documented
