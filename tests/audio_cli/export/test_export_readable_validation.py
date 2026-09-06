"""Adversarial saved-result boundaries for readable word fallback and display clocks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli.export import InvalidResultError, export_documents
from tests.audio_cli.export.export_test_support import _payload, _timed_segment, _write


@pytest.mark.parametrize("output_format", ["txt", "md"])
@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Hello, then thirty more seconds of speech.", ["hello"]),
        ("Hello.", ["hello", "extra"]),
        ("Hello there.", ["there", "hello"]),
        ("Hello.", ["hello", "..."]),
    ],
)
def test_readable_word_fallback_requires_full_lexical_coverage(
    tmp_path: Path, output_format: str, text: str, tokens: list[str]
) -> None:
    path = _write(
        tmp_path / "inconsistent.json",
        _payload(
            [
                _timed_segment(
                    text, [(word, index, index + 0.5) for index, word in enumerate(tokens)]
                )
            ],
            duration=60,
            outcomes={"word_timestamps": "produced"},
        ),
    )
    before = path.read_bytes()
    with pytest.raises(InvalidResultError) as error:
        export_documents([path], output_format, timestamps=True)
    with pytest.raises(InvalidResultError) as subtitle_error:
        export_documents([path], "srt")
    assert error.value.reason == subtitle_error.value.reason
    assert error.value.input_path == path
    assert text in export_documents([path], output_format).content
    assert path.read_bytes() == before


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_readable_fallback_uses_existing_case_whitespace_and_unicode_punctuation_policy(
    tmp_path: Path, output_format: str
) -> None:
    text = "HELLO,\t世界！ Straße…"
    path = _write(
        tmp_path / "matched.json",
        _payload(
            [
                _timed_segment(
                    text, [("hello", 0.1, 0.2), ("世界", 0.3, 0.4), ("STRASSE", 0.5, 0.9)]
                )
            ],
            outcomes={"word_timestamps": "produced"},
        ),
    )
    header = "# Transcript\n\n" if output_format == "md" else ""
    assert export_documents([path], output_format, timestamps=True).content == (
        header + f"[00:00:00.100 --> 00:00:00.900] {text}\n"
    )
    assert export_documents([path], "srt").cue_count > 0


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_native_bounds_do_not_require_word_stream_lexical_coverage(
    tmp_path: Path, output_format: str
) -> None:
    text = "Hello, then thirty more seconds of speech."
    segment = _timed_segment(text, [("hello", 1, 1.5)])
    segment.update(start=0, end=40)
    path = _write(
        tmp_path / "native.json",
        _payload(
            [segment],
            stack="vibevoice",
            duration=60,
            outcomes={"segment_timestamps": "produced", "word_timestamps": "produced"},
        ),
    )
    assert (
        f"[00:00:00.000 --> 00:00:40.000] {text}\n"
        in export_documents([path], output_format, timestamps=True).content
    )
    with pytest.raises(InvalidResultError, match="punctuation invariant"):
        export_documents([path], "srt")


@pytest.mark.parametrize("output_format", ["txt", "md"])
@pytest.mark.parametrize("case", ["partial-words", "huge-words", "huge-native"])
@pytest.mark.parametrize("write_output", [False, True])
def test_bad_readable_timing_is_bare_structured_cli_refusal_before_publication(
    tmp_path: Path, capsys, output_format: str, case: str, write_output: bool
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic canonical source")
    outcomes = {"word_timestamps": "produced"}
    if case == "partial-words":
        segment = _timed_segment("Hello, then thirty more seconds of speech.", [("hello", 1, 1.5)])
        reason = "word text does not map to segment text under the punctuation invariant"
    elif case == "huge-words":
        segment = _timed_segment("Hello.", [("hello", 1e307, 1.1e307)])
        reason = "too large for readable millisecond quantization"
    else:
        segment = {"segment_id": "seg_0", "text": "Hello.", "start": 1e307, "end": 1.1e307}
        outcomes = {"segment_timestamps": "produced"}
        reason = "too large for readable millisecond quantization"
    path = _write(
        tmp_path / "transcript.json",
        _payload([segment], source_path=str(source), duration=1e308, outcomes=outcomes),
    )
    before = path.read_bytes()
    args = ["export", "--input", str(path), "--format", output_format]
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert segment["text"] in captured.out
    output = tmp_path / "preserved.txt"
    output.write_text("existing output")
    requested = [*args, "--timestamps"]
    if write_output:
        requested.extend(["--output", str(output), "--force"])
    assert cli.main(requested) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["code"] == "export_input_invalid"
    assert error["provided"] == str(path)
    assert reason in error["reason"]
    assert output.read_text() == "existing output"
    assert source.read_bytes() == b"synthetic canonical source"
    assert path.read_bytes() == before
