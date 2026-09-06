"""Rejected model estimates remain evidence, never replacement timing."""

import copy
import json
from pathlib import Path

import pytest

from audio_cli.transcribe.adapters.aligner import normalize_alignment
from audio_cli.transcribe.adapters.qwen import sentence_segments


def test_recorded_quantized_endpoint_remains_rejected_with_its_word_index():
    evidence = json.loads(
        (
            Path(__file__).parents[4]
            / "model_tests/benchmark/results/2026-09-06-qwen-alignment-boundary.json"
        ).read_text()
    )["reproduced"]
    request = [evidence["unit"]]
    response = {"segments": [evidence["stage_response"]]}
    original = copy.deepcopy((request, response))
    result = normalize_alignment(response, request)
    assert result.words == {}
    assert result.rejections == {"turn_25": {"code": "out_of_unit_bounds", "word_index": 0}}
    sentences = sentence_segments(request, result.words, rejections=result.rejections)
    assert sentences == [{"unit_id": "turn_25", "text": "Like.", "speaker": "S1"}]
    assert (request, response) == original


@pytest.mark.parametrize(
    "word,code",
    [
        ({"text": "Hi", "start": 0, "end": 1.001}, "out_of_unit_bounds"),
        ({"text": "Hi", "start": 0, "end": 1.04}, "out_of_unit_bounds"),
        ({"text": "Hi", "start": 0, "end": 80}, "out_of_unit_bounds"),
        ({"text": "Hi", "start": 0.8, "end": 0.2}, "invalid_bounds"),
        ({"text": "Hi", "start": 0.0002, "end": -0.0001}, "invalid_bounds"),
        ({"text": "Hi", "start": 0, "end": "1"}, "invalid_bounds"),
        ({"text": "Hi", "start": 0, "end": float("nan")}, "invalid_bounds"),
        ({"text": "Hi", "start": 0}, "invalid_bounds"),
        ({"text": ".", "start": 0, "end": 1}, "invalid_token"),
        ({"start": 0, "end": 1}, "invalid_token"),
    ],
)
def test_no_model_resolution_allowance_or_invalid_word_repair(word, code):
    result = normalize_alignment(
        {"segments": [{"unit_id": "u", "words": [word]}]},
        [{"unit_id": "u", "text": "Hi.", "start": 0, "end": 1}],
    )
    assert result.words == {}
    assert result.rejections == {"u": {"code": code, "word_index": 0}}


def test_serialization_only_endpoint_correction_and_provider_unavailability():
    result = normalize_alignment(
        {
            "segments": [
                {"unit_id": "rounded", "words": [{"text": "Hi", "start": 0, "end": 1}]},
                {
                    "unit_id": "error",
                    "words": None,
                    "error": {"type": "ValueError", "message": "recorded provider error"},
                },
            ]
        },
        [
            {"unit_id": "rounded", "text": "Hi.", "start": 0.0004, "end": 0.9996},
            {"unit_id": "error", "text": "Bye.", "start": 1, "end": 2},
        ],
    )
    assert result.words == {"rounded": [{"text": "Hi", "start": 0.0004, "end": 0.9996}]}
    assert result.rejections == {"error": {"code": "provider_unavailable"}}


def test_sentence_partition_failure_keeps_all_text_and_records_its_unit():
    rejections = {}
    words = {"u": [{"text": "HelloWorld", "start": 0, "end": 1}]}
    sentences = sentence_segments(
        [{"unit_id": "u", "text": "Hello. World!"}], words, rejections=rejections
    )
    assert sentences == [{"unit_id": "u", "text": "Hello."}, {"unit_id": "u", "text": "World!"}]
    assert rejections == {"u": {"code": "sentence_reconciliation"}}
