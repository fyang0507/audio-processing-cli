from __future__ import annotations

import pytest

from audio_cli.transcribe.adapters import normalize_aligned_words


def test_aligner_normalizer_publishes_only_owned_fields() -> None:
    words = normalize_aligned_words(
        {
            "segments": [
                {
                    "unit_id": "u0",
                    "language": "Chinese",
                    "words": [{"text": "你", "start": 1.2, "end": 1.4, "score": 0.8}],
                }
            ]
        },
        [{"unit_id": "u0"}],
    )
    assert words == {"u0": [{"text": "你", "start": 1.2, "end": 1.4}]}
    rounded = normalize_aligned_words(
        {
            "segments": [
                {
                    "unit_id": "u1",
                    "words": [{"text": "Hi", "start": 2.28, "end": 4.791}],
                }
            ]
        },
        [{"unit_id": "u1", "start": 2.280125, "end": 4.790625}],
    )
    assert rounded == {
        "u1": [
            {
                "text": "Hi",
                "start": 2.280125,
                "end": 4.790625,
            }
        ]
    }
    assert (
        normalize_aligned_words(
            {
                "segments": [
                    {
                        "unit_id": "u0",
                        "words": [{"text": "bad", "start": -1.0, "end": 0.2}],
                    }
                ]
            },
            [{"unit_id": "u0"}],
        )
        == {}
    )


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
    assert (
        normalize_aligned_words(
            {"segments": [{"unit_id": "u0", "words": words}]},
            [{"unit_id": "u0", "start": 0.0, "end": 2.0}],
        )
        == {}
    )


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
            {
                "segments": [
                    {"unit_id": "u0", "words": first_words},
                    {
                        "unit_id": "u0",
                        "words": [{"text": "valid", "start": 0.2, "end": 0.8}],
                    },
                ]
            },
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
    assert (
        normalize_aligned_words(
            {"segments": [{"unit_id": "u0", "words": None}]},
            [{"unit_id": "u0", "start": 0.0, "end": 1.0}],
        )
        == {}
    )


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
    assert (
        normalize_aligned_words(
            {
                "segments": [
                    {
                        "unit_id": "u0",
                        "words": [{"text": "Hello", "start": word_start, "end": word_end}],
                    }
                ]
            },
            [{"unit_id": "u0", "start": unit_start, "end": unit_end}],
        )
        == {}
    )
