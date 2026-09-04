from __future__ import annotations

import pytest

from audio_cli.transcribe.adapters import (
    normalize_aligned_words,
    normalize_qwen_segments,
    normalize_vad_regions,
    reconcile_turns,
    sentence_segments,
    split_sentences,
    strip_qwen_scaffold,
)


def test_qwen_no_hint_scaffold_is_stripped_and_not_replaced_with_a_default() -> None:
    # This exact protocol prefix is recorded by run_qwen_verbatim_probe.py:553-558 for the
    # no-hint private API path. Removing the adapter strip makes this assertion fail.
    assert strip_qwen_scaffold("language English<asr_text>Hello.") == "Hello."
    assert strip_qwen_scaffold("language Chinese<asr_text>你好。") == "你好。"
    assert strip_qwen_scaffold("Hello without a scaffold.") == "Hello without a scaffold."

    segments, missing = normalize_qwen_segments(
        {"units": [{
            "unit_id": "u0", "processed": True,
            "text": "language English<asr_text>Hello.",
        }]},
        [{"unit_id": "u0", "start": 0.0, "end": 1.0}],
    )
    assert segments == [{
        "unit_id": "u0", "start": 0.0, "end": 1.0, "text": "Hello.",
    }]
    assert missing == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"units": []},
        {"units": [{"unit_id": "u0", "processed": 0}]},
        {"units": [{"unit_id": "u0"}]},
    ],
)
def test_qwen_requires_an_exact_explicit_processing_ledger(
    payload: dict,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_qwen_segments(
            payload,
            [{"unit_id": "u0", "start": 0.0, "end": 1.0}],
        )


def test_qwen_explicit_false_is_an_unfinished_unit() -> None:
    completed, unfinished = normalize_qwen_segments(
        {"units": [{"unit_id": "u0", "processed": False}]},
        [{"unit_id": "u0", "start": 0.0, "end": 1.0}],
    )
    assert completed == []
    assert unfinished == [{"unit_id": "u0", "start": 0.0, "end": 1.0}]


def test_qwen_text_is_sentence_segmented_and_aligned_words_follow_punctuation() -> None:
    assert split_sentences("Hello world. 你好！Value 1.5 is kept") == [
        "Hello world.", "你好！", "Value 1.5 is kept",
    ]
    assert split_sentences("Wait... okay.") == ["Wait...", "okay."]
    assert split_sentences("真的?!") == ["真的?!"]
    assert split_sentences("...Hello.") == ["...Hello."]
    assert split_sentences("……？！") == ["……？！"]
    assert sentence_segments(
        [{"unit_id": "u0", "text": "Hello, world. Next!", "speaker": "S1"}],
        {"u0": [
            {"text": "Hello", "start": 0.0, "end": 0.2},
            {"text": "world", "start": 0.2, "end": 0.4},
            {"text": "Next", "start": 0.5, "end": 0.7},
        ]},
    ) == [
        {"unit_id": "u0", "text": "Hello, world.", "speaker": "S1", "words": [
            {"text": "Hello", "start": 0.0, "end": 0.2},
            {"text": "world", "start": 0.2, "end": 0.4},
        ]},
        {"unit_id": "u0", "text": "Next!", "speaker": "S1", "words": [
            {"text": "Next", "start": 0.5, "end": 0.7},
        ]},
    ]
    assert sentence_segments(
        [{"unit_id": "u0", "text": "Hello."}],
        {"u0": [{"text": "Goodbye", "start": 0.0, "end": 0.2}]},
    ) == [{"unit_id": "u0", "text": "Hello."}]


def test_diarizer_reconciles_exact_thresholds_and_drops_raw_embeddings() -> None:
    raw = {"segments": [
        {"startTimeSeconds": 0.0, "endTimeSeconds": 0.6, "speakerId": "S1",
         "embedding": [0.0] * 256},
        {"startTimeSeconds": 0.8, "endTimeSeconds": 1.4, "speakerId": "S1",
         "embedding": [1.0] * 256},
        {"startTimeSeconds": 1.2, "endTimeSeconds": 1.8, "speakerId": "S2",
         "embedding": [2.0] * 256},
        {"startTimeSeconds": 1.85, "endTimeSeconds": 2.0, "speakerId": "S3",
         "embedding": [3.0] * 256},
    ]}
    plan = reconcile_turns(raw, duration_seconds=2.0)
    assert plan.units == ({
        "unit_id": "turn_0", "speaker": "S1", "start": 0.0, "end": 1.2,
    },)
    assert plan.overlaps == ({"start": 1.2, "end": 1.4},)
    assert plan.short_turns == ({"start": 1.4, "end": 1.8},)
    assert plan.raw_fragments == ({"start": 1.85, "end": 2.0},)
    assert "embedding" not in repr(plan)


def test_diarizer_accepts_exact_250ms_fragment_and_500ms_turn() -> None:
    fragment_floor = reconcile_turns(
        {"segments": [{"start_s": 0.0, "end_s": 0.25, "speaker": "S1"}]},
        duration_seconds=0.25,
    )
    assert fragment_floor.raw_fragments == ()
    assert fragment_floor.short_turns == ({"start": 0.0, "end": 0.25},)

    turn_floor = reconcile_turns(
        {"segments": [{"start_s": 0.0, "end_s": 0.5, "speaker": "S1"}]},
        duration_seconds=0.5,
    )
    assert len(turn_floor.units) == 1
    assert turn_floor.raw_fragments == ()


def test_same_speaker_merge_accepts_300ms_gap_and_refuses_301ms() -> None:
    at_floor = reconcile_turns(
        {"segments": [
            {"start_s": 0.0, "end_s": 0.3, "speaker": "S1"},
            {"start_s": 0.6, "end_s": 0.9, "speaker": "S1"},
        ]},
        duration_seconds=0.9,
    )
    assert at_floor.units == ({
        "unit_id": "turn_0", "speaker": "S1", "start": 0.0, "end": 0.9,
    },)

    over_floor = reconcile_turns(
        {"segments": [
            {"start_s": 0.0, "end_s": 0.3, "speaker": "S1"},
            {"start_s": 0.601, "end_s": 0.901, "speaker": "S1"},
        ]},
        duration_seconds=0.901,
    )
    assert over_floor.units == ()
    assert over_floor.short_turns == (
        {"start": 0.0, "end": 0.3},
        {"start": 0.601, "end": 0.901},
    )


def test_raw_fragment_inside_gap_blocks_merge_like_the_recorded_runner() -> None:
    # run_turn_attributed_mlx_asr.py:330-355 merges only one pure `gap` span and its
    # recorded note explicitly says never across a filtered-fragment abstention.
    plan = reconcile_turns(
        {"segments": [
            {"start_s": 0.0, "end_s": 0.3, "speaker": "S1"},
            {"start_s": 0.4, "end_s": 0.5, "speaker": "S2"},
            {"start_s": 0.6, "end_s": 0.9, "speaker": "S1"},
        ]},
        duration_seconds=0.9,
    )
    assert plan.units == ()
    assert plan.raw_fragments == ({"start": 0.4, "end": 0.5},)


def test_discarded_fragment_does_not_split_a_kept_speaker_turn() -> None:
    plan = reconcile_turns(
        {"segments": [
            {"start_s": 0.0, "end_s": 2.0, "speaker": "S1"},
            {"start_s": 0.5, "end_s": 0.6, "speaker": "S2"},
        ]},
        duration_seconds=2.0,
    )
    assert plan.units == ({
        "unit_id": "turn_0", "speaker": "S1", "start": 0.0, "end": 2.0,
    },)
    assert plan.short_turns == ()
    assert plan.raw_fragments == ()


@pytest.mark.parametrize(
    "invalid_bound",
    [False, "0.0", float("nan"), float("inf"), 10**400],
)
def test_diarizer_rejects_non_finite_or_coercible_bounds(
    invalid_bound: object,
) -> None:
    with pytest.raises(ValueError, match="segment start"):
        reconcile_turns(
            {"segments": [{
                "start_s": invalid_bound,
                "end_s": 1.0,
                "speaker": "S1",
            }]},
            duration_seconds=1.0,
        )


@pytest.mark.parametrize("invalid_speaker", [None, False, 1.5, "", "  ", [], {}])
def test_diarizer_rejects_synthetic_or_empty_speaker_labels(
    invalid_speaker: object,
) -> None:
    with pytest.raises(ValueError, match="speaker label"):
        reconcile_turns(
            {"segments": [{
                "start_s": 0.0,
                "end_s": 1.0,
                "speaker": invalid_speaker,
            }]},
            duration_seconds=1.0,
        )


@pytest.mark.parametrize(("speaker", "expected"), [("S1", "S1"), (0, "0")])
def test_diarizer_preserves_source_backed_string_and_integer_speaker_labels(
    speaker: object,
    expected: str,
) -> None:
    plan = reconcile_turns(
        {"segments": [{"start_s": 0.0, "end_s": 1.0, "speaker": speaker}]},
        duration_seconds=1.0,
    )
    assert plan.turns == ({
        "turn_id": "turn_0", "speaker": expected, "start": 0.0, "end": 1.0,
    },)


@pytest.mark.parametrize(
    ("duplicate_key", "duplicate_value", "field"),
    [
        ("startTimeSeconds", 0.25, "start"),
        ("endTimeSeconds", 0.75, "end"),
        ("speakerId", "S2", "speaker"),
    ],
)
def test_diarizer_refuses_conflicting_live_and_artifact_aliases(
    duplicate_key: str,
    duplicate_value: object,
    field: str,
) -> None:
    segment = {
        "start_s": 0.0,
        "end_s": 1.0,
        "speaker": "S1",
        duplicate_key: duplicate_value,
    }

    with pytest.raises(ValueError, match=f"exactly one {field} field"):
        reconcile_turns({"segments": [segment]}, duration_seconds=2.0)


def test_diarizer_refuses_ambiguous_top_level_and_nested_segment_arrays() -> None:
    segment = {"start_s": 0.0, "end_s": 1.0, "speaker": "S1"}

    with pytest.raises(ValueError, match="exactly one accepted location"):
        reconcile_turns(
            {"segments": [segment], "output": {"segments": []}},
            duration_seconds=2.0,
        )


def test_diarizer_accepts_the_single_nested_live_segment_location() -> None:
    plan = reconcile_turns(
        {"output": {"segments": [{
            "startTimeSeconds": 0.0,
            "endTimeSeconds": 1.0,
            "speakerId": "S1",
        }]}},
        duration_seconds=2.0,
    )

    assert plan.turns == ({
        "turn_id": "turn_0", "speaker": "S1", "start": 0.0, "end": 1.0,
    },)


def test_aligner_and_vad_normalizers_publish_only_owned_fields() -> None:
    words = normalize_aligned_words(
        {"segments": [{
            "unit_id": "u0", "language": "Chinese",
            "words": [{"text": "你", "start": 1.2, "end": 1.4, "score": 0.8}],
        }]},
        [{"unit_id": "u0"}],
    )
    assert words == {"u0": [{"text": "你", "start": 1.2, "end": 1.4}]}
    rounded = normalize_aligned_words(
        {"segments": [{
            "unit_id": "u1",
            "words": [{"text": "Hi", "start": 2.28, "end": 4.791}],
        }]},
        [{"unit_id": "u1", "start": 2.280125, "end": 4.790625}],
    )
    assert rounded == {"u1": [{
        "text": "Hi", "start": 2.280125, "end": 4.790625,
    }]}
    assert normalize_vad_regions([{
        "start": 0.1, "end": 0.9, "mean_probability": 0.8,
    }]) == [{"start": 0.1, "end": 0.9}]
    assert normalize_aligned_words(
        {"segments": [{
            "unit_id": "u0",
            "words": [{"text": "bad", "start": -1.0, "end": 0.2}],
        }]},
        [{"unit_id": "u0"}],
    ) == {}


@pytest.mark.parametrize(
    "invalid_bound",
    [False, "0.0", float("nan"), float("inf"), 10**400],
)
def test_vad_rejects_non_finite_or_coercible_bounds(
    invalid_bound: object,
) -> None:
    with pytest.raises(ValueError, match="region 0 start"):
        normalize_vad_regions([{"start": invalid_bound, "end": 1.0}])


@pytest.mark.parametrize(
    "regions",
    [
        [{"start": -0.1, "end": 0.5}],
        [{"start": 0.5, "end": 0.5}],
        [{"start": 0.6, "end": 0.5}],
        [{"start": 0.0, "end": 0.8}, {"start": 0.7, "end": 1.0}],
        [{"start": 1.0, "end": 1.5}, {"start": 0.0, "end": 0.5}],
        [{"start": 0.0, "end": 0.0000004}],
    ],
)
def test_vad_rejects_nonpositive_nonchronological_or_overlapping_regions(
    regions: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError):
        normalize_vad_regions(regions)


@pytest.mark.parametrize(
    "words",
    [
        [
            {"text": "Hel", "start": 1.0, "end": 1.4},
            {"text": "lo", "start": 0.2, "end": 0.8},
        ],
        [
            {"text": "Hel", "start": 0.2, "end": 1.2},
            {"text": "lo", "start": 1.0, "end": 1.8},
        ],
        [
            {"text": "", "start": 0.2, "end": 0.3},
            {"text": "Hello", "start": 0.3, "end": 1.8},
        ],
    ],
)
def test_aligner_rejects_invalid_word_sequences(
    words: list[dict],
) -> None:
    assert normalize_aligned_words(
        {"segments": [{"unit_id": "u0", "words": words}]},
        [{"unit_id": "u0", "start": 0.0, "end": 2.0}],
    ) == {}


@pytest.mark.parametrize(
    "first_words",
    [
        None,
        [{"text": "bad", "start": -1.0, "end": 0.2}],
    ],
)
def test_aligner_rejects_duplicates_even_when_the_first_result_is_unusable(
    first_words: object,
) -> None:
    with pytest.raises(ValueError, match="duplicate unit"):
        normalize_aligned_words(
            {"segments": [
                {"unit_id": "u0", "words": first_words},
                {
                    "unit_id": "u0",
                    "words": [{"text": "valid", "start": 0.2, "end": 0.8}],
                },
            ]},
            [{"unit_id": "u0", "start": 0.0, "end": 1.0}],
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"segments": []},
        {"segments": [{"unit_id": "u0"}]},
        {"segments": [{"unit_id": "u0", "words": {}}]},
    ],
)
def test_aligner_requires_an_exact_explicit_per_unit_ledger(
    payload: dict,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_aligned_words(
            payload,
            [{"unit_id": "u0", "start": 0.0, "end": 1.0}],
        )


def test_aligner_null_words_is_explicit_per_unit_abstention() -> None:
    assert normalize_aligned_words(
        {"segments": [{"unit_id": "u0", "words": None}]},
        [{"unit_id": "u0", "start": 0.0, "end": 1.0}],
    ) == {}


@pytest.mark.parametrize(
    ("word_start", "word_end", "unit_start", "unit_end"),
    [
        (False, 1.0, 0.0, 2.0),
        (0.0, True, 0.0, 2.0),
        ("0.0", 1.0, 0.0, 2.0),
        (0.0, "1.0", 0.0, 2.0),
        (0.0, 1.0, False, 2.0),
        (0.0, 1.0, 0.0, "2.0"),
    ],
)
def test_aligner_rejects_coercible_non_numeric_bounds(
    word_start: object,
    word_end: object,
    unit_start: object,
    unit_end: object,
) -> None:
    assert normalize_aligned_words(
        {"segments": [{
            "unit_id": "u0",
            "words": [{"text": "Hello", "start": word_start, "end": word_end}],
        }]},
        [{"unit_id": "u0", "start": unit_start, "end": unit_end}],
    ) == {}
