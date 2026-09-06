"""Alignment diagnostic links must refer to actual untimed result segments."""

from dataclasses import replace

import pytest

from audio_cli.transcribe.result import ResultError, serialize_result
from tests.audio_cli.transcribe.result.test_result import base_result


def alignment_result(alignment=None):
    entry = {"abstention_id": "ab_0", "reason": "alignment_unavailable", "start": 0, "end": 1}
    if alignment is not None:
        entry["alignment"] = alignment
    return base_result(
        abstentions=[entry],
        requested_capabilities=frozenset({"word_timestamps"}),
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": {"word_timestamps": "abstained"},
            "observed": {},
            "plan": {},
        },
    )


def diagnostic():
    return {
        "unit_id": "turn_25",
        "segment_ids": ["seg_0"],
        "code": "out_of_unit_bounds",
        "word_index": 0,
    }


def test_alignment_diagnostics_and_legacy_absence_round_trip():
    assert (
        serialize_result(alignment_result(diagnostic()))["abstentions"][0]["alignment"]
        == diagnostic()
    )
    assert "alignment" not in serialize_result(alignment_result())["abstentions"][0]


@pytest.mark.parametrize(
    "changes",
    [
        {"unit_id": ""},
        {"code": "guessed"},
        {"code": []},
        {"segment_ids": []},
        {"segment_ids": ["missing"]},
        {"segment_ids": ["seg_0", "seg_0"]},
        {"segment_ids": [0]},
        {"word_index": True},
        {"word_index": -1},
        {"word_index": 0.5},
        {"unknown": "field"},
    ],
)
def test_invalid_diagnostic_or_segment_link_is_refused(changes):
    with pytest.raises(ResultError, match="alignment"):
        serialize_result(alignment_result({**diagnostic(), **changes}))


def test_alignment_link_cannot_claim_produced_words_failed():
    result = alignment_result(diagnostic())
    segment = {
        **result.segments[0],
        "words": [{"word_id": "w_0", "text": "Hello", "start": 0, "end": 1}],
    }
    with pytest.raises(ResultError, match="segment with words"):
        serialize_result(replace(result, segments=[segment]))


def test_new_alignment_metadata_requires_an_abstained_word_timing_outcome():
    result = alignment_result(diagnostic())
    with pytest.raises(ResultError, match="abstained word_timestamps"):
        serialize_result(
            replace(
                result,
                requested_capabilities=frozenset(),
                provenance={
                    "stack": "qwen-1.7b",
                    "outcomes": {},
                    "observed": {},
                    "plan": {},
                },
            )
        )


def test_native_segment_alignment_link_requires_the_same_attempted_bounds():
    result = alignment_result(diagnostic())
    provenance = {
        **result.provenance,
        "outcomes": {"word_timestamps": "abstained", "segment_timestamps": "produced"},
    }
    segment = {**result.segments[0], "start": 2, "end": 3}
    with pytest.raises(ResultError, match="native segment bounds"):
        serialize_result(
            replace(
                result,
                segments=[segment],
                provenance=provenance,
                requested_capabilities=frozenset({"word_timestamps", "segment_timestamps"}),
            )
        )


@pytest.mark.parametrize("missing", ["start", "end", "reason", "abstention_id"])
def test_malformed_outer_abstention_is_a_result_error_before_native_link_access(missing):
    result = alignment_result(diagnostic())
    del result.abstentions[0][missing]
    provenance = {
        **result.provenance,
        "outcomes": {"word_timestamps": "abstained", "segment_timestamps": "produced"},
    }
    segment = {**result.segments[0], "start": 0, "end": 1}
    with pytest.raises(ResultError, match="abstentions"):
        serialize_result(
            replace(
                result,
                segments=[segment],
                provenance=provenance,
                requested_capabilities=frozenset({"word_timestamps", "segment_timestamps"}),
            )
        )


def test_alignment_link_rejects_ambiguous_duplicate_segment_ids():
    result = alignment_result(diagnostic())
    with pytest.raises(ResultError, match="unique segment IDs"):
        serialize_result(
            replace(
                result, segments=[result.segments[0], {**result.segments[0], "text": "Other text."}]
            )
        )


@pytest.mark.parametrize(
    "code", ["provider_unavailable", "text_mismatch", "sentence_reconciliation"]
)
def test_unit_or_text_rejection_cannot_claim_a_returned_word_index(code):
    with pytest.raises(ResultError, match="cannot identify a returned word"):
        serialize_result(alignment_result({**diagnostic(), "code": code}))


@pytest.mark.parametrize("entry", [None, 1])
def test_duplicate_segments_and_malformed_abstention_still_raise_result_error(entry):
    result = alignment_result()
    with pytest.raises(ResultError, match="abstentions"):
        serialize_result(
            replace(result, segments=[result.segments[0], result.segments[0]], abstentions=[entry])
        )
