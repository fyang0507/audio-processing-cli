"""Explicit endpoint clipping retains evidence and cannot repair invalid streams."""

import copy
import math

import pytest

from audio_cli.transcribe.adapters.aligner import normalize_alignment


def test_negative_source_unit_cannot_authorize_a_boundary_correction():
    result = normalize_alignment(
        {
            "segments": [
                {"unit_id": "unit_0", "words": [{"text": "Hello", "start": -1.01, "end": 0.5}]}
            ]
        },
        [{"unit_id": "unit_0", "start": -1, "end": 1}],
        max_overrun_ms=20,
    )
    assert result.rejections == {"unit_0": {"code": "invalid_bounds"}}
    assert result.words == result.corrections == {}


def normalize(words, *, limit=20, start=0, end=1):
    return normalize_alignment(
        {"segments": [{"unit_id": "u", "words": words}]},
        [{"unit_id": "u", "start": start, "end": end}],
        max_overrun_ms=limit,
    )


@pytest.mark.parametrize("limit", [13.375, 20, 80])
def test_recorded_endpoint_clips_with_explicit_evidence_at_exact_required_limit(limit):
    words = [{"text": "Like", "start": 96.57, "end": 98.01}]
    original = copy.deepcopy(words)
    result = normalize(words, limit=limit, start=96.570437, end=97.996625)
    assert result.words == {"u": [{"text": "Like", "start": 96.570437, "end": 97.996625}]}
    assert result.rejections == {}
    assert result.corrections == {
        "u": [
            {
                "word_index": 0,
                "original_bounds": [96.57, 98.01],
                "applied_bounds": [96.570437, 97.996625],
                "unit_bounds": [96.570437, 97.996625],
                "start_overrun_ms": 0.437,
                "end_overrun_ms": 13.375,
                "max_overrun_ms": limit,
            }
        ]
    }
    assert words == original


def test_limit_just_below_observed_overrun_refuses_without_publishing_corrections():
    result = normalize(
        [{"text": "Like", "start": 96.57, "end": 98.01}],
        limit=math.nextafter(13.375, 0),
        start=96.570437,
        end=97.996625,
    )
    assert result.words == result.corrections == {}
    assert result.rejections["u"]["code"] == "out_of_unit_bounds"
    assert result.rejections["u"]["boundary"]["end_overrun_ms"] == 13.375
    assert result.rejections["u"]["boundary"]["max_overrun_ms"] < 13.375


def test_each_endpoint_is_limited_independently_and_both_corrections_are_retained():
    result = normalize([{"text": "Hi", "start": -0.01, "end": 1.02}], limit=20)
    assert result.words == {"u": [{"text": "Hi", "start": 0, "end": 1}]}
    assert result.corrections["u"][0]["start_overrun_ms"] == 10
    assert result.corrections["u"][0]["end_overrun_ms"] == 20


@pytest.mark.parametrize("limit", [0.501, 20])
def test_rounding_allowance_alone_never_creates_a_new_correction_record(limit):
    result = normalize([{"text": "Hi", "start": -0.000501, "end": 1.000501}], limit=limit)
    assert result.words == {"u": [{"text": "Hi", "start": 0, "end": 1}]}
    assert result.corrections == result.rejections == {}


@pytest.mark.parametrize(
    "word",
    [
        {"text": "Hi", "start": 0.8, "end": 0.2},
        {"text": "Hi", "start": -0.02, "end": -0.01},
        {"text": "Hi", "start": 1.01, "end": 1.02},
        {"text": "Hi", "start": -0.01, "end": 0},
        {"text": "Hi", "start": 1, "end": 1.01},
        {"text": "Hi", "start": -0.01, "end": -0.01},
        {"text": "Hi", "start": 0, "end": float("inf")},
    ],
)
def test_larger_limit_does_not_repair_invalid_wholly_outside_or_collapsing_words(word):
    result = normalize([word], limit=1000000)
    assert result.words == result.corrections == {}
    assert result.rejections == {"u": {"code": "invalid_bounds", "word_index": 0}}


@pytest.mark.parametrize("position", [0, 0.5, 1])
def test_existing_inside_zero_duration_words_remain_valid(position):
    word = {"text": "Hi", "start": position, "end": position}
    result = normalize([word])
    assert result.words == {"u": [word]}
    assert result.rejections == result.corrections == {}


def test_raw_overlap_is_rejected_before_clipping_can_hide_it():
    result = normalize(
        [
            {"text": "one", "start": 0, "end": 1.01},
            {"text": "two", "start": 1.005, "end": 1.01},
        ]
    )
    assert result.words == result.corrections == {}
    assert result.rejections == {"u": {"code": "word_order", "word_index": 1}}


@pytest.mark.parametrize(
    "last,code",
    [
        ({"text": "two", "start": 0.5, "end": 1.03}, "out_of_unit_bounds"),
        ({"text": "", "start": 0.5, "end": 1}, "invalid_token"),
        ({"text": "two", "start": 0.5, "end": "1"}, "invalid_bounds"),
    ],
)
def test_later_rejection_discards_previously_eligible_unit_corrections(last, code):
    result = normalize([{"text": "one", "start": -0.01, "end": 0.5}, last])
    assert result.words == result.corrections == {}
    assert result.rejections["u"]["code"] == code
    assert result.rejections["u"]["word_index"] == 1


@pytest.mark.parametrize(
    "limit",
    [True, False, "20", None, -1, 0, 0.5, float("nan"), float("inf"), -float("inf"), 10**1000],
)
def test_adapter_direct_call_rejects_invalid_limit_even_with_no_units(limit):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        normalize_alignment({"segments": []}, [], max_overrun_ms=limit)
