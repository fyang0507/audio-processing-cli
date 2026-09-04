"""Runtime provenance and incomplete-coverage invariants for normalized results."""

from __future__ import annotations

from dataclasses import replace

import pytest

from audio_cli.transcribe.result import (
    ABSENT,
    NormalizedResult,
    ResultError,
    serialize_result,
)


def base_result(**changes) -> NormalizedResult:
    found = NormalizedResult(
        source={"path": "sample.wav", "duration_seconds": 10.0, "timebase": "seconds"},
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        abstentions=[],
        provenance={"stack": "qwen-1.7b", "outcomes": {}, "observed": {}, "plan": {}},
        requested_capabilities=frozenset(),
    )
    return replace(found, **changes)


def test_run_result_requires_one_outcome_per_requested_capability() -> None:
    with pytest.raises(ResultError, match="name every requested"):
        serialize_result(base_result(requested_capabilities=frozenset({"verbatim"})))


def test_observed_wall_and_peak_arithmetic_is_enforced() -> None:
    valid = {
        "stage_wall_seconds": {"decode": 0.1, "asr": 1.2},
        "total_wall_seconds": 1.3,
        "peak_rss_bytes_by_stage": {"asr": 100, "aligner": 80},
        "peak_rss_bytes": 100,
    }
    emitted = serialize_result(
        replace(
            base_result(),
            provenance={"stack": "qwen-1.7b", "outcomes": {}, "observed": valid, "plan": {}},
        )
    )
    assert emitted["provenance"]["observed"] == valid

    for mutation, message in (
        ({**valid, "total_wall_seconds": 1.4}, "sum of stage"),
        ({**valid, "peak_rss_bytes": 180}, "maximum"),
    ):
        with pytest.raises(ResultError, match=message):
            serialize_result(
                replace(
                    base_result(),
                    provenance={
                        "stack": "qwen-1.7b",
                        "outcomes": {},
                        "observed": mutation,
                        "plan": {},
                    },
                )
            )


def test_observed_counts_are_reconciled_with_the_serialized_arrays() -> None:
    observed = {"segments": 1, "words": 0, "abstentions": 0}
    emitted = serialize_result(
        replace(
            base_result(),
            provenance={"stack": "qwen-1.7b", "outcomes": {}, "observed": observed, "plan": {}},
        )
    )
    assert emitted["provenance"]["observed"] == observed

    for mutation in (
        {**observed, "segments": 2},
        {**observed, "words": 1},
        {**observed, "turns": 0},
    ):
        with pytest.raises(ResultError, match="result contains"):
            serialize_result(
                replace(
                    base_result(),
                    provenance={
                        "stack": "qwen-1.7b",
                        "outcomes": {},
                        "observed": mutation,
                        "plan": {},
                    },
                )
            )

    with pytest.raises(ResultError, match="unknown keys"):
        serialize_result(
            replace(
                base_result(),
                provenance={
                    "stack": "qwen-1.7b",
                    "outcomes": {},
                    "observed": {"backend_confidence": 0.9},
                    "plan": {},
                },
            )
        )


def test_coverage_cannot_leave_the_source_timeline() -> None:
    coverage = {
        "scope_intervals": [[0.0, 10.0]],
        "covered_through_seconds": 4.0,
        "covered_fraction": 0.4,
        "covered_intervals": [[0.0, 4.0]],
        "missing_intervals": [[4.0, 11.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    with pytest.raises(ResultError, match="ordered interval"):
        serialize_result(base_result(complete=False, coverage=coverage))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"covered_through_seconds": 5.0}, "first missing"),
        ({"covered_fraction": 0.5}, "covered duration"),
        ({"covered_through_seconds": 5.0, "missing_intervals": [[5.0, 10.0]]}, "without gaps"),
        ({"units_completed": 2}, "leave at least one"),
        ({"scope_intervals": [[0.0, 6.0], [5.0, 10.0]]}, "non-overlapping"),
    ],
)
def test_coverage_ledger_must_be_internally_consistent(
    mutation: dict[str, object],
    message: str,
) -> None:
    coverage = {
        "scope_intervals": [[0.0, 10.0]],
        "covered_through_seconds": 4.0,
        "covered_fraction": 0.4,
        "covered_intervals": [[0.0, 4.0]],
        "missing_intervals": [[4.0, 10.0]],
        "units_total": 2,
        "units_completed": 1,
        **mutation,
    }
    with pytest.raises(ResultError, match=message):
        serialize_result(base_result(complete=False, coverage=coverage))


def test_coverage_validator_exercises_multiple_disjoint_scopes() -> None:
    coverage = {
        "scope_intervals": [[0.0, 4.0], [6.0, 10.0]],
        "covered_through_seconds": 2.0,
        "covered_fraction": 0.5,
        "covered_intervals": [[0.0, 2.0], [6.0, 8.0]],
        "missing_intervals": [[2.0, 4.0], [8.0, 10.0]],
        "units_total": 4,
        "units_completed": 2,
    }
    payload = serialize_result(base_result(complete=False, coverage=coverage))
    assert payload["coverage"] == coverage


def test_coverage_tolerance_does_not_scale_with_large_source_timestamps() -> None:
    coverage = {
        "scope_intervals": [[999_999_000.0, 1_000_000_000.0]],
        "covered_through_seconds": 999_999_500.0,
        "covered_fraction": 0.5009,
        "covered_intervals": [[999_998_999.1, 999_999_500.0]],
        "missing_intervals": [[999_999_500.0, 1_000_000_000.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    source = {
        "path": "long.wav",
        "duration_seconds": 1_000_000_000.0,
        "timebase": "seconds",
    }
    with pytest.raises(ResultError, match="without gaps"):
        serialize_result(base_result(source=source, complete=False, coverage=coverage))


def test_coverage_interval_arrays_must_be_chronological() -> None:
    coverage = {
        "scope_intervals": [[0.0, 3.0]],
        "covered_through_seconds": 1.0,
        "covered_fraction": 0.5,
        "covered_intervals": [[2.0, 2.5], [0.0, 1.0]],
        "missing_intervals": [[1.0, 2.0], [2.5, 3.0]],
        "units_total": 4,
        "units_completed": 2,
    }

    with pytest.raises(ResultError, match="chronological"):
        serialize_result(base_result(complete=False, coverage=coverage))


def test_absent_sentinel_is_not_serialized() -> None:
    emitted = serialize_result(base_result())
    assert ABSENT not in emitted.values()
