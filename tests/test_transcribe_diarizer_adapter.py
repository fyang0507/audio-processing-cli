from __future__ import annotations

import pytest

from audio_cli.transcribe.adapters import reconcile_turns


def test_diarizer_reconciles_exact_thresholds_and_drops_raw_embeddings() -> None:
    raw = {
        "segments": [
            {
                "startTimeSeconds": 0.0,
                "endTimeSeconds": 0.6,
                "speakerId": "S1",
                "embedding": [0.0] * 256,
            },
            {
                "startTimeSeconds": 0.8,
                "endTimeSeconds": 1.4,
                "speakerId": "S1",
                "embedding": [1.0] * 256,
            },
            {
                "startTimeSeconds": 1.2,
                "endTimeSeconds": 1.8,
                "speakerId": "S2",
                "embedding": [2.0] * 256,
            },
            {
                "startTimeSeconds": 1.85,
                "endTimeSeconds": 2.0,
                "speakerId": "S3",
                "embedding": [3.0] * 256,
            },
        ]
    }
    plan = reconcile_turns(raw, duration_seconds=2.0)
    assert plan.units == (
        {
            "unit_id": "turn_0",
            "speaker": "S1",
            "start": 0.0,
            "end": 1.2,
        },
    )
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
        {
            "segments": [
                {"start_s": 0.0, "end_s": 0.3, "speaker": "S1"},
                {"start_s": 0.6, "end_s": 0.9, "speaker": "S1"},
            ]
        },
        duration_seconds=0.9,
    )
    assert at_floor.units == (
        {
            "unit_id": "turn_0",
            "speaker": "S1",
            "start": 0.0,
            "end": 0.9,
        },
    )

    over_floor = reconcile_turns(
        {
            "segments": [
                {"start_s": 0.0, "end_s": 0.3, "speaker": "S1"},
                {"start_s": 0.601, "end_s": 0.901, "speaker": "S1"},
            ]
        },
        duration_seconds=0.901,
    )
    assert over_floor.units == ()
    assert over_floor.short_turns == (
        {"start": 0.0, "end": 0.3},
        {"start": 0.601, "end": 0.901},
    )


def test_raw_fragment_inside_gap_blocks_merge_like_the_recorded_runner() -> None:
    # model_tests/benchmark/turn_attributed_mlx_asr/plan.py:141-168 merges only one pure `gap`
    # span and its recorded note explicitly says never across a filtered-fragment abstention.
    plan = reconcile_turns(
        {
            "segments": [
                {"start_s": 0.0, "end_s": 0.3, "speaker": "S1"},
                {"start_s": 0.4, "end_s": 0.5, "speaker": "S2"},
                {"start_s": 0.6, "end_s": 0.9, "speaker": "S1"},
            ]
        },
        duration_seconds=0.9,
    )
    assert plan.units == ()
    assert plan.raw_fragments == ({"start": 0.4, "end": 0.5},)


def test_discarded_fragment_does_not_split_a_kept_speaker_turn() -> None:
    plan = reconcile_turns(
        {
            "segments": [
                {"start_s": 0.0, "end_s": 2.0, "speaker": "S1"},
                {"start_s": 0.5, "end_s": 0.6, "speaker": "S2"},
            ]
        },
        duration_seconds=2.0,
    )
    assert plan.units == (
        {
            "unit_id": "turn_0",
            "speaker": "S1",
            "start": 0.0,
            "end": 2.0,
        },
    )
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
            {
                "segments": [
                    {
                        "start_s": invalid_bound,
                        "end_s": 1.0,
                        "speaker": "S1",
                    }
                ]
            },
            duration_seconds=1.0,
        )


@pytest.mark.parametrize("invalid_speaker", [None, False, 1.5, "", "  ", [], {}])
def test_diarizer_rejects_synthetic_or_empty_speaker_labels(
    invalid_speaker: object,
) -> None:
    with pytest.raises(ValueError, match="speaker label"):
        reconcile_turns(
            {
                "segments": [
                    {
                        "start_s": 0.0,
                        "end_s": 1.0,
                        "speaker": invalid_speaker,
                    }
                ]
            },
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
    assert plan.turns == (
        {
            "turn_id": "turn_0",
            "speaker": expected,
            "start": 0.0,
            "end": 1.0,
        },
    )


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
        {
            "output": {
                "segments": [
                    {
                        "startTimeSeconds": 0.0,
                        "endTimeSeconds": 1.0,
                        "speakerId": "S1",
                    }
                ]
            }
        },
        duration_seconds=2.0,
    )

    assert plan.turns == (
        {
            "turn_id": "turn_0",
            "speaker": "S1",
            "start": 0.0,
            "end": 1.0,
        },
    )
