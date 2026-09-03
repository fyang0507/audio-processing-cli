from __future__ import annotations

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
