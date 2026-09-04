"""Timed-export evidence, omissions, and subtitle validation."""

from __future__ import annotations

from export_test_support import (
    InvalidResultError,
    Path,
    TimingRequiredError,
    _payload,
    _timed_segment,
    _write,
    export_documents,
    pytest,
)


def test_timed_export_rejects_mixed_event_only_and_untimed_documents(
    tmp_path: Path,
) -> None:
    event = _payload(
        [{
            "segment_id": "seg_0",
            "text": "[Music]",
            "start": 0.0,
            "end": 0.8,
        }],
        outcomes={
            "word_timestamps": "produced",
            "segment_timestamps": "produced",
        },
        run_range=[0.0, 1.0],
    )
    ordinary = _payload(
        [{
            "segment_id": "seg_0",
            "text": "Ordinary untimed speech.",
            "start": 1.0,
            "end": 1.8,
        }],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
        run_range=[1.0, 2.0],
    )
    ordinary["abstentions"] = [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 1.0,
        "end": 1.8,
    }]
    event_path = _write(tmp_path / "event.json", event)
    ordinary_path = _write(tmp_path / "ordinary.json", ordinary)

    with pytest.raises(TimingRequiredError):
        export_documents([event_path, ordinary_path], "srt")


def test_timed_export_rejects_empty_result_with_only_a_produced_outcome(
    tmp_path: Path,
) -> None:
    payload = _payload([], outcomes={"word_timestamps": "produced"})
    path = _write(tmp_path / "empty-produced.json", payload)

    with pytest.raises(TimingRequiredError):
        export_documents([path], "vtt")


def test_export_subtitles_omit_event_segments_and_render_real_speakers(
    tmp_path: Path,
) -> None:
    speech = _timed_segment(
        "Hello there.",
        [("Hello", 0.31, 0.8), ("there", 0.9, 1.2)],
        speaker="Speaker 0",
    )
    event = {
        "segment_id": "seg_1",
        "start": 1.2,
        "end": 1.5,
        "text": "[Environmental Sounds]",
    }
    payload = _payload(
        [speech, event],
        outcomes={
            "word_timestamps": "produced",
            "diarization": "produced",
            "segment_timestamps": "produced",
        },
    )
    path = _write(tmp_path / "timed.json", payload)
    product = export_documents([path], "vtt")

    assert product.cue_count == 1
    assert product.speaker_labels_rendered is True
    assert "<v Speaker 0>Hello there." in product.content
    assert "Environmental" not in product.content
    assert product.cues[0].start_ms == 310
    assert product.cues[0].end_ms == 1200


def test_event_only_continuation_does_not_block_real_cues_from_another_input(
    tmp_path: Path,
) -> None:
    event = _payload(
        [{"segment_id": "seg_0", "text": "[Music]", "start": 0.0, "end": 0.8}],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
        run_range=[0.0, 1.0],
    )
    speech_segment = _timed_segment("Hello.", [("Hello", 1.1, 1.8)])
    speech_segment.update({"start": 1.0, "end": 1.9})
    speech = _payload(
        [speech_segment],
        outcomes={
            "word_timestamps": "produced",
            "segment_timestamps": "produced",
        },
        run_range=[1.0, 2.0],
    )
    inputs = [
        _write(tmp_path / "event.json", event),
        _write(tmp_path / "speech.json", speech),
    ]
    product = export_documents(inputs, "srt")
    assert product.cue_count == 1
    assert "Hello." in product.content
    assert "Music" not in product.content


def test_mixed_timed_and_abstained_untimed_text_omits_untimed_segment(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed speech.", [("Timed", 0.1, 0.3), ("speech", 0.4, 0.8)])
    untimed = {
        "segment_id": "seg_1",
        "text": "This speech lost its word stream.",
        "start": 0.9,
        "end": 1.9,
    }
    payload = _payload(
        [timed, untimed],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
    )
    payload["abstentions"] = [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.9,
        "end": 1.9,
    }]
    path = _write(tmp_path / "mixed.json", payload)
    product = export_documents([path], "srt")
    assert product.cue_count == 1
    assert "Timed speech." in product.content
    assert "lost its word stream" not in product.content


def test_mixed_timed_and_unbounded_abstained_text_refuses_unprovable_omission(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed.", [("Timed", 0.1, 0.8)])
    untimed = {"segment_id": "seg_1", "text": "Unbounded alignment failure."}
    payload = _payload(
        [timed, untimed],
        outcomes={"word_timestamps": "abstained"},
    )
    payload["abstentions"] = [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.9,
        "end": 1.9,
    }]

    with pytest.raises(TimingRequiredError):
        export_documents([_write(tmp_path / "unbounded.json", payload)], "srt")


def test_bounded_wordless_speech_rejects_an_unrelated_alignment_abstention(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed.", [("Timed", 0.1, 0.8)])
    missing = {
        "segment_id": "seg_1", "text": "Missing timing.",
        "start": 1.0, "end": 1.8,
    }
    payload = _payload(
        [timed, missing],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
    )
    payload["abstentions"] = [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.9,
        "end": 1.9,
    }]

    with pytest.raises(InvalidResultError, match="same-bounds"):
        export_documents([_write(tmp_path / "mismatched.json", payload)], "srt")


def test_mixed_timed_and_wordless_speech_requires_abstention_evidence(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed.", [("Timed", 0.1, 0.8)])
    missing = {"segment_id": "seg_1", "text": "Missing timing."}
    payload = _payload(
        [timed, missing],
        outcomes={"word_timestamps": "produced"},
    )
    path = _write(tmp_path / "contradictory-timing.json", payload)

    with pytest.raises(InvalidResultError, match="alignment_unavailable"):
        export_documents([path], "srt")


def test_explicit_empty_words_remain_valid_for_punctuation_only_text(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed.", [("Timed", 0.1, 0.8)])
    punctuation = {"segment_id": "seg_1", "text": "……？！", "words": []}
    payload = _payload(
        [timed, punctuation],
        outcomes={"word_timestamps": "produced"},
    )

    product = export_documents(
        [_write(tmp_path / "punctuation.json", payload)], "srt"
    )
    assert product.cue_count == 1
    assert "Timed." in product.content


def test_subtitles_fail_closed_without_a_real_word_stream(tmp_path: Path) -> None:
    untimed = _payload([{"segment_id": "seg_0", "text": "Transcript only."}])
    path = _write(tmp_path / "untimed.json", untimed)
    with pytest.raises(TimingRequiredError) as raised:
        export_documents([path], "srt")
    assert raised.value.found == ()
    assert raised.value.source_path == Path("source.wav")
    assert raised.value.stack == "qwen-1.7b"

    empty_claim = _payload(
        [{"segment_id": "seg_0", "text": "[Music]", "start": 0.0, "end": 1.0}],
        outcomes={
            "word_timestamps": "produced",
            "segment_timestamps": "produced",
        },
    )
    empty_path = _write(tmp_path / "empty-claim.json", empty_claim)
    empty_product = export_documents([empty_path], "vtt")
    assert empty_product.cues == ()
    assert empty_product.content == "WEBVTT\n\n"

    false_claim = _payload(
        [{"segment_id": "seg_0", "text": "Ordinary untimed speech."}],
        outcomes={"word_timestamps": "produced"},
    )
    with pytest.raises(InvalidResultError, match="alignment_unavailable"):
        export_documents([_write(tmp_path / "false-claim.json", false_claim)], "vtt")


def test_subtitle_quantization_rejects_finite_but_unrepresentable_bounds(
    tmp_path: Path,
) -> None:
    payload = _payload(
        [_timed_segment("Huge.", [("Huge", 1e307, 1e308)])],
        duration=1e308,
        outcomes={"word_timestamps": "produced"},
    )
    path = _write(tmp_path / "huge-timing.json", payload)
    with pytest.raises(InvalidResultError, match="too large"):
        export_documents([path], "srt")


def test_subtitle_validation_attributes_a_bad_later_document_to_that_input(
    tmp_path: Path,
) -> None:
    first = _payload(
        [_timed_segment("Good.", [("Good", 0.1, 0.8)])],
        outcomes={"word_timestamps": "produced"},
        run_range=[0.0, 1.0],
    )
    second = _payload(
        [_timed_segment("Bad.", [("Mismatch", 1.1, 1.8)])],
        outcomes={"word_timestamps": "produced"},
        run_range=[1.0, 2.0],
    )
    first_path = _write(tmp_path / "first.json", first)
    second_path = _write(tmp_path / "second.json", second)

    with pytest.raises(InvalidResultError) as raised:
        export_documents([first_path, second_path], "srt")

    assert raised.value.input_path == second_path
