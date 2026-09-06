"""Saved correction evidence must be independently auditable and fail closed."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from audio_cli.transcribe.result import ResultError, serialize_result
from tests.audio_cli.transcribe.result.test_result import base_result


def correction_result():
    return base_result(
        segments=[
            {
                "segment_id": "seg_0",
                "text": "Hello.",
                "words": [{"word_id": "w_0", "text": "Hello", "start": 1, "end": 2}],
            }
        ],
        requested_capabilities=frozenset({"word_timestamps"}),
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": {"word_timestamps": "produced"},
            "plan": {"roles": {"aligner": {"config": {"max_overrun_ms": 20}}}},
            "observed": {
                "alignment_corrections": [
                    {
                        "unit_id": "unit_0",
                        "segment_id": "seg_0",
                        "word_id": "w_0",
                        "word_index": 0,
                        "original_bounds": [0.99, 2.014],
                        "unit_bounds": [1, 2],
                        "applied_bounds": [1, 2],
                        "start_overrun_ms": 10,
                        "end_overrun_ms": 14,
                        "max_overrun_ms": 20,
                    }
                ]
            },
        },
    )


def test_correction_round_trip_retains_original_and_applied_evidence():
    result = correction_result()
    saved = serialize_result(result)
    assert saved["provenance"] == result.provenance
    assert saved["segments"][0]["words"][0] == {
        "word_id": "w_0",
        "text": "Hello",
        "start": 1,
        "end": 2,
    }
    assert saved["abstentions"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"original_bounds": [1, 1.5]},
        {"original_bounds": [2.01, 2.02]},
        {"original_bounds": [2.014, 0.99]},
        {"original_bounds": [0.99, float("inf")]},
        {"unit_bounds": [-1, 2]},
        {"unit_bounds": [1, 11]},
        {"unit_bounds": [1, 1]},
        {"applied_bounds": [1, 1.9]},
        {"applied_bounds": [True, 2]},
        {"end_overrun_ms": 13},
        {"start_overrun_ms": -10},
        {"max_overrun_ms": 30},
        {"word_index": True},
        {"word_index": -1},
        {"word_index": 1},
        {"word_index": 999},
        {"word_id": "missing"},
        {"segment_id": "missing"},
        {"unit_id": ""},
        {"extra": 1},
    ],
)
def test_forged_or_malformed_correction_is_refused(change):
    result = correction_result()
    result.provenance["observed"]["alignment_corrections"][0].update(change)
    with pytest.raises(ResultError):
        serialize_result(result)


@pytest.mark.parametrize("value", [[], None, {}, [None], [0]])
def test_present_empty_or_malformed_correction_ledger_is_not_absence(value):
    result = correction_result()
    result.provenance["observed"]["alignment_corrections"] = value
    with pytest.raises(ResultError, match="alignment_corrections"):
        serialize_result(result)


def test_corrections_require_executed_limit_and_matching_words():
    result = correction_result()
    result.provenance["plan"] = {}
    with pytest.raises(ResultError, match="executed max_overrun_ms"):
        serialize_result(result)
    result = correction_result()
    result.segments[0]["words"][0]["end"] = 1.9
    with pytest.raises(ResultError, match="positive supplied word"):
        serialize_result(result)


def test_correction_cannot_exceed_the_saved_limit():
    result = correction_result()
    result.provenance["plan"]["roles"]["aligner"]["config"]["max_overrun_ms"] = 12
    result.provenance["observed"]["alignment_corrections"][0]["max_overrun_ms"] = 12
    with pytest.raises(ResultError, match="allowed correction"):
        serialize_result(result)


def test_correction_link_rejects_duplicate_word_ids_and_duplicate_records():
    result = correction_result()
    records = result.provenance["observed"]["alignment_corrections"]
    records.append(deepcopy(records[0]))
    with pytest.raises(ResultError, match="duplicates a corrected word"):
        serialize_result(result)
    result = correction_result()
    result.segments[0]["words"].append(deepcopy(result.segments[0]["words"][0]))
    with pytest.raises(ResultError, match="unique segment and word IDs"):
        serialize_result(result)


def test_native_segment_correction_must_bind_its_native_scope():
    result = correction_result()
    result.segments[0].update(start=0.5, end=2)
    result.provenance["outcomes"]["segment_timestamps"] = "produced"
    result = replace(
        result, requested_capabilities=frozenset({"word_timestamps", "segment_timestamps"})
    )
    with pytest.raises(ResultError, match="native segment"):
        serialize_result(result)


def test_rejected_boundary_arithmetic_scope_and_limit_are_validated():
    result = correction_result()
    correction = result.provenance["observed"].pop("alignment_corrections")[0]
    boundary = {
        key: correction[key]
        for key in (
            "original_bounds",
            "unit_bounds",
            "start_overrun_ms",
            "end_overrun_ms",
            "max_overrun_ms",
        )
    }
    boundary["max_overrun_ms"] = 0.501
    result.provenance["plan"]["roles"]["aligner"]["config"]["max_overrun_ms"] = 0.501
    result.provenance["outcomes"]["word_timestamps"] = "abstained"
    result = replace(
        result,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        abstentions=[
            {
                "abstention_id": "ab_0",
                "reason": "alignment_unavailable",
                "start": 1,
                "end": 2,
                "alignment": {
                    "unit_id": "unit_0",
                    "segment_ids": ["seg_0"],
                    "word_index": 0,
                    "code": "out_of_unit_bounds",
                    "boundary": boundary,
                },
            }
        ],
    )
    assert serialize_result(result)["abstentions"][0]["alignment"]["boundary"] == boundary
    boundary["end_overrun_ms"] = 13
    with pytest.raises(ResultError, match="disagrees with recorded bounds"):
        serialize_result(result)


def test_zero_overrun_cannot_carry_a_small_negative_value():
    result = correction_result()
    row = result.provenance["observed"]["alignment_corrections"][0]
    row["original_bounds"][0] = 1
    row["start_overrun_ms"] = -5e-10
    with pytest.raises(ResultError, match="disagrees with recorded bounds"):
        serialize_result(result)


def test_one_unit_cannot_claim_conflicting_correction_scopes():
    result = correction_result()
    result.segments.append(
        {
            "segment_id": "seg_1",
            "text": "Again.",
            "words": [{"word_id": "w_1", "text": "Again", "start": 3, "end": 4}],
        }
    )
    rows = result.provenance["observed"]["alignment_corrections"]
    rows.append(
        {
            **rows[0],
            "word_index": 1,
            "segment_id": "seg_1",
            "word_id": "w_1",
            "original_bounds": [2.99, 4.014],
            "applied_bounds": [3, 4],
            "unit_bounds": [3, 4],
        }
    )
    with pytest.raises(ResultError, match="conflicting bounds"):
        serialize_result(result)


def test_a_rejected_unit_cannot_also_claim_an_accepted_correction():
    result = correction_result()
    result.segments.append({"segment_id": "seg_1", "text": "Other."})
    result.provenance["outcomes"]["word_timestamps"] = "abstained"
    result = replace(
        result,
        abstentions=[
            {
                "abstention_id": "ab_0",
                "reason": "alignment_unavailable",
                "start": 1,
                "end": 2,
                "alignment": {
                    "unit_id": "unit_0",
                    "segment_ids": ["seg_1"],
                    "code": "provider_unavailable",
                },
            }
        ],
    )
    with pytest.raises(ResultError, match="cannot correct a rejected unit"):
        serialize_result(result)


def test_corrected_word_cannot_hide_a_unit_outside_selected_ownership(tmp_path):
    import json

    from audio_cli.export import InvalidResultError, export_documents

    result = correction_result()
    result.segments[0]["words"][0]["end"] = 1.5
    row = result.provenance["observed"]["alignment_corrections"][0]
    row.update(original_bounds=[0.99, 1.5], applied_bounds=[1, 1.5], end_overrun_ms=0)
    valid = serialize_result(result)
    valid["provenance"]["plan"]["execution"] = {
        "range": {"requested": [1, 1.5], "selected_unit_scope": [1, 1.5]}
    }
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(valid))
    output = tmp_path / "must-not-publish.md"
    with pytest.raises(InvalidResultError, match="outside the selected scope"):
        export_documents([path], "md", output, timestamps=True, provenance=True)
    assert not output.exists()
