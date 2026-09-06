"""Readable timestamps preserve text and require supplied segment or word bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_cli.export import (
    ReadableTimingRequiredError,
    TimestampsUnsupportedError,
    export_documents,
    timestamps_unsupported_for_format,
    timing_required_for_timestamps,
)
from audio_cli.export.writers import render_text
from tests.audio_cli.export.export_test_support import _payload, _timed_segment, _write


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_recorded_vibevoice_segments_keep_text_speakers_events_and_native_bounds(
    tmp_path: Path, output_format: str
) -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[3] / "tests/fixtures/vibevoice_multispeaker_excerpt.json"
        ).read_text()
    )
    segments = []
    for index, raw in enumerate(fixture["segments"]):
        segment = {
            "segment_id": f"seg_{index}",
            "text": raw["Content"],
            "start": raw["Start"],
            "end": raw["End"],
        }
        if raw["Speaker"] != "N/A":
            segment["speaker"] = str(raw["Speaker"])
        segments.append(segment)
    payload = _payload(
        segments,
        duration=60,
        stack="vibevoice",
        outcomes={"segment_timestamps": "produced", "diarization": "produced"},
    )
    path = _write(tmp_path / "vibevoice.json", payload)
    original_bytes = path.read_bytes()
    default = export_documents([path], output_format)
    timed = export_documents([path], output_format, timestamps=True)
    separator = "\n\n" if output_format == "md" else "\n"
    lines = timed.content.removeprefix("# Transcript\n\n").removesuffix("\n").split(separator)
    expected_prefixes = [
        "[00:00:16.830 --> 00:00:32.050] ",
        "[00:00:33.640 --> 00:00:37.840] ",
        "[00:00:37.840 --> 00:00:40.280] ",
        "[00:00:40.280 --> 00:00:52.840] ",
    ]
    canonical_lines = (
        default.content.removeprefix("# Transcript\n\n").removesuffix("\n").split(separator)
    )
    assert lines == [
        prefix + text for prefix, text in zip(expected_prefixes, canonical_lines, strict=True)
    ]
    assert lines[2] == "[00:00:37.840 --> 00:00:40.280] [Environmental Sounds]"
    assert all("words" not in segment for segment in timed.segments)
    assert timed.segments == default.segments
    assert path.read_bytes() == original_bytes
    assert export_documents([path], output_format).content == default.content
    assert timed.summary(None) == default.summary(None)


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_qwen_words_time_canonical_punctuation_without_filling_word_gaps(
    tmp_path: Path, output_format: str
) -> None:
    payload = _payload(
        [_timed_segment("Hello, WORLD!", [("hello", 60.125, 60.5), ("world", 63, 64.875)])],
        outcomes={"word_timestamps": "produced"},
        duration=70,
    )
    path = _write(tmp_path / "qwen.json", payload)
    product = export_documents([path], output_format, timestamps=True)
    header = "# Transcript\n\n" if output_format == "md" else ""
    assert product.content == header + "[00:01:00.125 --> 00:01:04.875] Hello, WORLD!\n"
    assert "start" not in product.segments[0]
    assert "end" not in product.segments[0]


def test_native_bounds_take_precedence_over_narrower_aligned_word_bounds(tmp_path: Path) -> None:
    segment = _timed_segment("Hello.", [("hello", 0.25, 0.75)])
    segment.update(start=0, end=1.5)
    path = _write(
        tmp_path / "both.json",
        _payload(
            [segment],
            stack="vibevoice",
            outcomes={"word_timestamps": "produced", "segment_timestamps": "produced"},
        ),
    )
    assert export_documents([path], "txt", timestamps=True).content == (
        "[00:00:00.000 --> 00:00:01.500] Hello.\n"
    )


@pytest.mark.parametrize("output_format", ["txt", "md"])
@pytest.mark.parametrize("has_timed_peer", [False, True])
def test_untimed_qwen_refuses_every_text_segment_without_using_turns_or_processing_scope(
    tmp_path: Path, output_format: str, has_timed_peer: bool
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"canonical original")
    segments = [_timed_segment("Timed.", [("timed", 1, 2)])] if has_timed_peer else []
    segments.append({"segment_id": "untimed_original_id", "text": "Still untimed."})
    payload = _payload(
        segments,
        source_path=str(source),
        duration=20,
        outcomes={"diarization": "produced", "word_timestamps": "abstained"},
        run_range=[0, 10],
    )
    payload["abstentions"] = [
        {"abstention_id": "ab_0", "reason": "alignment_unavailable", "start": 3, "end": 5}
    ]
    payload["turns"] = [{"turn_id": "t_0", "speaker": "S1", "start": 3, "end": 5}]
    path = _write(tmp_path / "qwen.json", payload)
    output = tmp_path / "existing.txt"
    output.write_text("keep existing output")
    with pytest.raises(ReadableTimingRequiredError) as error:
        export_documents([path], output_format, output, force=True, timestamps=True)
    assert error.value.input_path == path
    assert error.value.segment_id == "untimed_original_id"
    assert output.read_text() == "keep existing output"
    assert source.read_bytes() == b"canonical original"
    assert "Still untimed." in export_documents([path], output_format).content


def test_empty_word_stream_is_not_timing_and_never_gets_zero_bounds(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "punctuation.json",
        _payload(
            [{"segment_id": "seg_0", "text": "...", "words": []}],
            outcomes={"word_timestamps": "produced"},
        ),
    )
    with pytest.raises(ReadableTimingRequiredError):
        export_documents([path], "txt", timestamps=True)
    assert export_documents([path], "txt").content == "...\n"


def test_readable_native_timing_survives_an_alignment_abstention(tmp_path: Path) -> None:
    payload = _payload(
        [{"segment_id": "seg_0", "text": "Hello.", "start": 0, "end": 1}],
        stack="vibevoice",
        outcomes={"segment_timestamps": "produced", "word_timestamps": "abstained"},
    )
    payload["abstentions"] = [
        {"abstention_id": "ab_0", "reason": "alignment_unavailable", "start": 0, "end": 1}
    ]
    path = _write(tmp_path / "abstained.json", payload)
    assert export_documents([path], "txt", timestamps=True).content == (
        "[00:00:00.000 --> 00:00:01.000] Hello.\n"
    )


def test_readable_rendering_keeps_hours_and_zero_length_bounds_without_nudging() -> None:
    assert (
        render_text([{"text": "Hello.", "start": 3600.125, "end": 3600.125}], timestamps=True)
        == "[01:00:00.125 --> 01:00:00.125] Hello.\n"
    )


def test_ranged_exports_preserve_absolute_source_bounds_and_gaps(tmp_path: Path) -> None:
    paths = []
    for index, start in enumerate((10.125, 30.125)):
        paths.append(
            _write(
                tmp_path / f"range-{index}.json",
                _payload(
                    [_timed_segment("Hello.", [("hello", start, start + 1)])],
                    outcomes={"word_timestamps": "produced"},
                    duration=40,
                    run_range=[start - 0.125, start + 2],
                ),
            )
        )
    product = export_documents(paths, "txt", timestamps=True)
    assert product.content == (
        "[00:00:10.125 --> 00:00:11.125] Hello.\n[00:00:30.125 --> 00:00:31.125] Hello.\n"
    )
    assert [segment["segment_id"] for segment in product.segments] == ["seg_0", "seg_1"]


def test_empty_transcript_does_not_invent_a_timed_segment(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.json", _payload([]))
    assert export_documents([path], "txt", timestamps=True).content == ""
    assert export_documents([path], "md", timestamps=True).content == "# Transcript\n\n"


@pytest.mark.parametrize("output_format", ["srt", "vtt", "jsonl"])
def test_timestamps_refuse_other_formats_before_opening_inputs(output_format: str) -> None:
    with pytest.raises(TimestampsUnsupportedError) as error:
        export_documents([Path("missing.json")], output_format, timestamps=True)
    assert error.value.output_format == output_format
    refusal = timestamps_unsupported_for_format(output_format)
    assert refusal.exit_code == 2
    assert refusal.payload["code"] == "timestamps_unsupported_for_format"
    assert refusal.payload["allowed_formats"] == ["txt", "md"]


def test_absent_timing_refusal_identifies_original_input_segment() -> None:
    refusal = timing_required_for_timestamps("qwen.json", "untimed_original_id")
    assert refusal.exit_code == 2
    assert refusal.payload["code"] == "timing_required_for_timestamps"
    assert refusal.payload["input"] == "qwen.json"
    assert refusal.payload["segment_id"] == "untimed_original_id"
    assert refusal.payload["requires_any_capability"] == ["segment_timestamps", "word_timestamps"]


@pytest.mark.parametrize("outcome", [None, "abstained", "produced"])
def test_readable_remedy_uses_saved_timing_outcome(tmp_path, capsys, outcome):
    from audio_cli.cli import main

    segment = {"segment_id": "seg_0", "text": "..." if outcome == "produced" else "Like."}
    if outcome == "produced":
        segment["words"] = []
    payload = _payload([segment], outcomes={"word_timestamps": outcome} if outcome else {})
    if outcome == "abstained":
        payload["abstentions"] = [
            {
                "abstention_id": "ab_0",
                "reason": "alignment_unavailable",
                "start": 0,
                "end": 1,
                "alignment": {
                    "unit_id": "turn_25",
                    "segment_ids": ["seg_0"],
                    "code": "out_of_unit_bounds",
                    "word_index": 0,
                },
            }
        ]
    source = _write(tmp_path / "result.json", payload)
    assert (
        main(["transcribe", "export", "--input", str(source), "--format", "md", "--timestamps"])
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    refusal = json.loads(captured.err)
    assert refusal["code"] == "timing_required_for_timestamps"
    assert refusal["segment_id"] == "seg_0"
    if outcome == "abstained":
        assert "already requested but abstained" in refusal["fix"]
        assert "alignment_unavailable" in refusal["fix"]
        assert "rerunning the same request is not an established timing repair" in refusal["fix"]
    elif outcome == "produced":
        assert "already records word_timestamps" in refusal["fix"]
    else:
        assert "transcribe the original source with" in refusal["fix"]


def test_malformed_alignment_ledger_remains_a_normal_cli_refusal(tmp_path, capsys):
    from audio_cli.cli import main

    payload = _payload(
        [{"segment_id": "seg_0", "text": "Like.", "start": 0, "end": 1}],
        outcomes={"word_timestamps": "abstained", "segment_timestamps": "produced"},
    )
    payload["abstentions"] = [
        {
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "end": 1,
            "alignment": {"unit_id": "u", "segment_ids": ["seg_0"], "code": "provider_unavailable"},
        }
    ]
    path = _write(tmp_path / "malformed.json", payload)
    assert main(["transcribe", "export", "--input", str(path), "--format", "md"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == "export_input_invalid"
