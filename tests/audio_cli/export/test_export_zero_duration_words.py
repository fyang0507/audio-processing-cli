"""Never discard lexical words when their subtitle cue candidate collapses."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli.export import export_documents
from audio_cli.export.cues import V1_CUE_POLICY, Cue, CueError, build_cues
from audio_cli.export.lexical import plain_positions
from tests.audio_cli.export.export_test_support import _payload, _timed_segment, _write


@pytest.mark.parametrize(
    ("text", "words", "expected"),
    [
        (
            "Before. Um. After.",
            [("Before", 0, 1), ("Um", 2, 2), ("After", 3, 4)],
            (Cue(0, 1000, "Before."), Cue(2000, 4000, "Um. After.")),
        ),
        (
            "Before um after",
            [("Before", 0, 1), ("um", 2, 2), ("after", 3, 4)],
            (Cue(0, 1000, "Before"), Cue(2000, 4000, "um after")),
        ),
        ("Um. After.", [("Um", 0, 0), ("After", 1, 2)], (Cue(0, 2000, "Um. After."),)),
        ("Before. Um.", [("Before", 0, 1), ("Um", 2, 2)], (Cue(0, 2000, "Before. Um."),)),
        (
            "Kind of. Records.",
            [("Kind", 0, 0), ("of", 0, 0), ("Records", 1, 2)],
            (Cue(0, 2000, "Kind of. Records."),),
        ),
        (
            "Tiny. After.",
            [("Tiny", 0.0001, 0.0004), ("After", 1, 2)],
            (Cue(0, 2000, "Tiny. After."),),
        ),
        (
            "Before. Tiny.",
            [("Before", 0, 1), ("Tiny", 1.9996, 2.0004)],
            (Cue(0, 2000, "Before. Tiny."),),
        ),
        (
            "First. Second.",
            [("First", 1, 1), ("Second", 2, 2)],
            (Cue(1000, 2000, "First. Second."),),
        ),
    ],
)
def test_collapsed_candidate_keeps_every_word_and_prefers_following(text, words, expected):
    segment = _timed_segment(text, words)
    original = deepcopy(segment)
    built = build_cues([segment], duration=5)
    assert built.cues == expected
    assert segment == original
    assert not built.warnings
    assert plain_positions("".join(cue.text for cue in built.cues))[0] == plain_positions(text)[0]


@pytest.mark.parametrize("times", [[(1, 1), (1, 1)], [(1.0001, 1.0002), (1.0003, 1.0004)]])
def test_whole_segment_collapsed_on_grid_refuses(times):
    segment = _timed_segment(
        "Kind of.", [(text, *bounds) for text, bounds in zip(("Kind", "of"), times, strict=True)]
    )
    with pytest.raises(CueError, match="cannot form a positive subtitle interval"):
        build_cues([segment], duration=2)


@pytest.mark.parametrize("speakers", [(None, None), ("S1", "S1"), ("S1", "S2")])
@pytest.mark.parametrize("collapsed_first", [True, False])
def test_collapse_never_borrows_another_segment_or_speaker(speakers, collapsed_first):
    bounds = [(0, 0), (1, 2)] if collapsed_first else [(0, 1), (2, 2)]
    segments = [
        _timed_segment(text + ".", [(text, *time)], segment_id=f"seg_{index}", speaker=speaker)
        for index, (text, time, speaker) in enumerate(
            zip(("One", "Two"), bounds, speakers, strict=True)
        )
    ]
    with pytest.raises(CueError, match="cannot form a positive subtitle interval"):
        build_cues(segments, duration=3)


@pytest.mark.parametrize("times", [[(1, 1), (0, 0.5)], [(0, 1), (0.5, 0.5)]])
def test_regrouping_does_not_hide_raw_word_order_or_overlap(times):
    segment = _timed_segment(
        "One. Two.", [(text, *bounds) for text, bounds in zip(("One", "Two"), times, strict=True)]
    )
    with pytest.raises(CueError, match="overlaps the preceding word"):
        build_cues([segment], duration=2)


def test_required_merging_uses_soft_limits_with_truthful_warnings():
    segment = _timed_segment("Zero. Following.", [("Zero", 0, 0), ("Following", 10, 11)])
    policy = replace(V1_CUE_POLICY, max_chars_per_line_latin=4)
    built = build_cues([segment], duration=12, policy=policy)
    assert built.cues == (Cue(0, 11000, "Zero.\nFollowing."),)
    assert {warning["code"] for warning in built.warnings} == {
        "cue_duration_overlong",
        "cue_line_overlong",
    }
    assert all(warning["blocking"] is False for warning in built.warnings)
    assert "single timed word" not in str(built.warnings)


@pytest.mark.parametrize("output_format", ["srt", "vtt"])
@pytest.mark.parametrize("existing", [False, True])
def test_unrenderable_segment_refuses_atomic_publication(tmp_path, capsys, output_format, existing):
    source = tmp_path / "original.wav"
    source.write_bytes(b"original")
    payload = _payload(
        [_timed_segment("Zero.", [("Zero", 1, 1)])],
        source_path=str(source),
        outcomes={"word_timestamps": "produced"},
    )
    canonical = _write(tmp_path / "canonical.json", payload)
    canonical_before = canonical.read_bytes()
    output = tmp_path / f"subtitle.{output_format}"
    if existing:
        output.write_bytes(b"existing subtitle")
    args = [
        "transcribe",
        "export",
        "--input",
        str(canonical),
        "--format",
        output_format,
        "-o",
        str(output),
    ]
    if existing:
        args.append("--force")
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    refusal = json.loads(captured.err)
    assert refusal["code"] == "export_input_invalid"
    assert "cannot form a positive subtitle interval" in refusal["reason"]
    assert output.read_bytes() == b"existing subtitle" if existing else not output.exists()
    assert source.read_bytes() == b"original"
    assert canonical.read_bytes() == canonical_before


@pytest.mark.parametrize("output_format", ["srt", "vtt"])
def test_recorded_d3_excerpts_retain_previously_omitted_words(tmp_path, output_format):
    fixture = (
        Path(__file__).resolve().parents[2] / "fixtures/qwen_subtitle_zero_duration_words.json"
    )
    segments = json.loads(fixture.read_text())["segments"]
    payload = _payload(
        segments, duration=150, outcomes={"word_timestamps": "produced", "diarization": "produced"}
    )
    path = _write(tmp_path / "recorded-excerpts.json", payload)
    before = path.read_bytes()
    product = export_documents([path], output_format)
    assert [cue.text for cue in product.cues] == [
        "好，",
        "嗯，然后就是说呢。",
        "the verbatim",
        "kind of records",
    ]
    assert [(cue.start_ms, cue.end_ms) for cue in product.cues] == [
        (40934, 41254),
        (42374, 45414),
        (123770, 124330),
        (126810, 128890),
    ]
    assert all(cue.speaker == "S1" for cue in product.cues)
    assert (
        plain_positions("".join(cue.text for cue in product.cues))[0]
        == plain_positions("".join(segment["text"] for segment in segments))[0]
    )
    assert "嗯" in product.content and "kind of records" in product.content
    assert path.read_bytes() == before
