"""Cue construction, punctuation binding, wrapping, and quantization."""

from __future__ import annotations

# ruff: noqa: F403, F405
from export_test_support import *

def test_cues_map_canonical_punctuation_and_use_only_word_bounds() -> None:
    segment = _timed_segment(
        "Hello, WORLD! Again.",
        [("hello", 0.1, 0.2), ("world", 0.25, 0.4), ("again", 0.6, 0.8)],
    )
    segment.update({"start": 0.0, "end": 2.0})
    built = build_cues([segment], duration=2.0)
    assert built.cues == (
        Cue(100, 400, "Hello, WORLD!"),
        Cue(600, 800, "Again."),
    )


@pytest.mark.parametrize("output_format", ["srt", "vtt"])
def test_timed_export_rejects_non_whitespace_control_characters(
    tmp_path: Path, output_format: str,
) -> None:
    payload = _payload(
        [_timed_segment("Hel\0lo.", [("Hel\0lo", 0.1, 0.8)])],
        outcomes={"word_timestamps": "produced"},
    )
    path = _write(tmp_path / "control.json", payload)

    with pytest.raises(InvalidResultError, match="control character"):
        export_documents([path], output_format)


def test_timed_export_rejects_words_that_split_one_casefolded_character(
    tmp_path: Path,
) -> None:
    payload = _payload(
        [_timed_segment("ß", [("s", 0.0, 0.1), ("s", 1.0, 1.1)])],
        outcomes={"word_timestamps": "produced"},
    )
    path = _write(tmp_path / "split-casefold.json", payload)

    with pytest.raises(InvalidResultError, match="case-fold expansion"):
        export_documents([path], "srt")


def test_cues_preserve_opening_wrappers_with_the_following_word() -> None:
    segment = _timed_segment(
        "He said, “Hello.” Then left.",
        [
            ("He", 0.0, 0.1),
            ("said", 0.1, 0.2),
            ("Hello", 0.2, 0.4),
            ("Then", 0.5, 0.6),
            ("left", 0.6, 0.8),
        ],
    )
    assert [cue.text for cue in build_cues([segment], duration=1.0).cues] == [
        "He said, “Hello.”",
        "Then left.",
    ]


def test_cues_assign_straight_opening_quote_to_the_following_timed_word() -> None:
    segment = _timed_segment(
        'A sufficiently long phrase, "Hello."',
        [
            ("A", 0.0, 0.1),
            ("sufficiently", 0.1, 0.3),
            ("long", 0.3, 0.4),
            ("phrase", 0.4, 0.6),
            ("Hello", 1.5, 2.0),
        ],
    )

    assert build_cues([segment], duration=2.0).cues == (
        Cue(0, 600, "A sufficiently long phrase,"),
        Cue(1500, 2000, '"Hello."'),
    )


def test_literal_unknown_account_tilde_input_is_not_expanded(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    literal_directory = Path("~codex-no-such-account")
    literal_directory.mkdir()
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    path = _write(
        literal_directory / "result.json",
        _payload(
            [{"segment_id": "seg_0", "text": "Literal path."}],
            source_path=str(source),
        ),
    )

    assert export_documents([path], "txt").content == "Literal path.\n"

    output = tmp_path / "literal-output.txt"
    export_documents([path], "txt", output=output)
    assert output.read_text(encoding="utf-8") == "Literal path.\n"


def test_cues_break_at_ellipsis_before_a_trailing_wrapper() -> None:
    segment = _timed_segment(
        "Wait…” Then.",
        [("Wait", 0.0, 0.4), ("Then", 0.45, 0.8)],
    )

    assert [cue.text for cue in build_cues([segment], duration=1.0).cues] == [
        "Wait…”",
        "Then.",
    ]


def test_cue_wrapping_keeps_boundaries_with_leading_whitespace() -> None:
    first = "A" * 25
    second = "B" * 25
    segment = _timed_segment(
        f"   {first} {second}",
        [(first, 0.1, 0.4), (second, 0.5, 0.8)],
    )
    built = build_cues([segment], duration=1.0)
    assert built.cues == (Cue(100, 800, f"{first}\n{second}"),)


def test_cues_drop_collapsed_ms_bounds_and_never_trim_real_overlap() -> None:
    collapsed = _timed_segment("A.", [("A", 0.0001, 0.0004)])
    built = build_cues([collapsed], duration=1.0)
    assert built.cues == ()
    assert [warning["code"] for warning in built.warnings] == [
        "cue_dropped_after_quantization"
    ]

    overlapping = [
        _timed_segment("First.", [("First", 0.1, 0.6)], segment_id="seg_0"),
        _timed_segment("Second.", [("Second", 0.5, 0.8)], segment_id="seg_1"),
    ]
    with pytest.raises(CueError, match="refusing to trim or nudge"):
        build_cues(overlapping, duration=1.0)

    submillisecond_overlap = [
        _timed_segment(
            "First.", [("First", 0.0994, 0.1004)], segment_id="seg_0"
        ),
        _timed_segment(
            "Second.", [("Second", 0.1003, 0.1014)], segment_id="seg_1"
        ),
    ]
    with pytest.raises(CueError, match="before millisecond quantization"):
        build_cues(submillisecond_overlap, duration=1.0)

    collapsed_then_surviving = [
        _timed_segment("A.", [("A", 0.0001, 0.0004)], segment_id="seg_0"),
        _timed_segment("B.", [("B", 0.0003, 0.0014)], segment_id="seg_1"),
    ]
    with pytest.raises(CueError, match="before millisecond quantization"):
        build_cues(collapsed_then_surviving, duration=1.0)
