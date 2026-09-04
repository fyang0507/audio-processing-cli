from __future__ import annotations

import pytest

from audio_cli.transcribe.adapters import (
    normalize_qwen_segments,
    sentence_segments,
    split_sentences,
    strip_qwen_scaffold,
)


def test_qwen_no_hint_scaffold_is_stripped_and_not_replaced_with_a_default() -> None:
    # This exact protocol prefix is recorded by
    # model_tests/benchmark/run_qwen_verbatim_probe.py:319-336 for the no-hint private API path.
    # Removing the adapter strip makes this assertion fail.
    assert strip_qwen_scaffold("language English<asr_text>Hello.") == "Hello."
    assert strip_qwen_scaffold("language Chinese<asr_text>你好。") == "你好。"
    assert strip_qwen_scaffold("Hello without a scaffold.") == "Hello without a scaffold."

    segments, missing = normalize_qwen_segments(
        {
            "units": [
                {
                    "unit_id": "u0",
                    "processed": True,
                    "text": "language English<asr_text>Hello.",
                }
            ]
        },
        [{"unit_id": "u0", "start": 0.0, "end": 1.0}],
    )
    assert segments == [
        {
            "unit_id": "u0",
            "start": 0.0,
            "end": 1.0,
            "text": "Hello.",
        }
    ]
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
        "Hello world.",
        "你好！",
        "Value 1.5 is kept",
    ]
    assert split_sentences("Wait... okay.") == ["Wait...", "okay."]
    assert split_sentences("真的?!") == ["真的?!"]
    assert split_sentences("...Hello.") == ["...Hello."]
    assert split_sentences("……？！") == ["……？！"]
    assert sentence_segments(
        [{"unit_id": "u0", "text": "Hello, world. Next!", "speaker": "S1"}],
        {
            "u0": [
                {"text": "Hello", "start": 0.0, "end": 0.2},
                {"text": "world", "start": 0.2, "end": 0.4},
                {"text": "Next", "start": 0.5, "end": 0.7},
            ]
        },
    ) == [
        {
            "unit_id": "u0",
            "text": "Hello, world.",
            "speaker": "S1",
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.2},
                {"text": "world", "start": 0.2, "end": 0.4},
            ],
        },
        {
            "unit_id": "u0",
            "text": "Next!",
            "speaker": "S1",
            "words": [
                {"text": "Next", "start": 0.5, "end": 0.7},
            ],
        },
    ]
    assert sentence_segments(
        [{"unit_id": "u0", "text": "Hello."}],
        {"u0": [{"text": "Goodbye", "start": 0.0, "end": 0.2}]},
    ) == [{"unit_id": "u0", "text": "Hello."}]
