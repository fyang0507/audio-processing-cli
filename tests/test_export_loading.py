"""Normalized-result loading and per-document scope validation."""

from __future__ import annotations

# ruff: noqa: F403, F405
from export_test_support import *

def test_loader_accepts_exact_result_and_rejects_schema_drift(tmp_path: Path) -> None:
    payload = _payload([{"segment_id": "seg_0", "text": "Hello."}])
    valid = _write(tmp_path / "valid.json", payload)
    loaded = load_result_document(valid)
    assert loaded.payload == payload
    assert loaded.owned_intervals == ((0.0, 2.0),)

    payload["invented"] = True
    invalid = _write(tmp_path / "invalid.json", payload)
    with pytest.raises(InvalidResultError, match="exact current normalized result shape"):
        load_result_document(invalid)

    duplicate_keys = tmp_path / "duplicate-keys.json"
    duplicate_keys.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(InvalidResultError, match="duplicate JSON object key"):
        load_result_document(duplicate_keys)

    surrogate = _payload([{"segment_id": "seg_0", "text": "\ud800"}])
    surrogate_path = tmp_path / "surrogate.json"
    surrogate_path.write_text(json.dumps(surrogate), encoding="utf-8")
    with pytest.raises(InvalidResultError, match="surrogates not allowed"):
        load_result_document(surrogate_path)

    huge = _payload([], duration=0.0)
    huge["source"]["duration_seconds"] = 10**400
    with pytest.raises(InvalidResultError, match="finite number"):
        load_result_document(_write(tmp_path / "huge-number.json", huge))

    huge_range = _payload([], run_range=[0.0, 1.0])
    huge_range["provenance"]["plan"]["execution"]["range"]["requested"][1] = 10**400
    with pytest.raises(InvalidResultError, match="too large to convert"):
        load_result_document(_write(tmp_path / "huge-range.json", huge_range))

    deep = _payload([])
    deep["provenance"]["plan"] = {"deep": "__DEEP_PLAN__"}
    encoded = json.dumps(deep).replace(
        '"__DEEP_PLAN__"', "[" * 1_500 + "0" + "]" * 1_500
    )
    deep_path = tmp_path / "deep-plan.json"
    deep_path.write_text(encoded, encoding="utf-8")
    with pytest.raises(InvalidResultError, match="recursion depth"):
        load_result_document(deep_path)

    merge_deep = _payload([])
    merge_deep["provenance"]["plan"] = {"deep": "__MERGE_DEEP_PLAN__"}
    merge_encoded = json.dumps(merge_deep).replace(
        '"__MERGE_DEEP_PLAN__"', "[" * 500 + "0" + "]" * 500
    )
    merge_deep_path = tmp_path / "merge-deep-plan.json"
    merge_deep_path.write_text(merge_encoded, encoding="utf-8")
    with pytest.raises(InvalidResultError, match="recursion depth"):
        export_documents([merge_deep_path], "txt")


def test_loader_refuses_a_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    fifo = tmp_path / "stream.json"
    os.mkfifo(fifo)

    with pytest.raises(InvalidResultError, match="input must be a regular file"):
        load_result_document(fifo)

    empty = _payload([], duration=0.0)
    empty_path = _write(tmp_path / "empty.json", empty)
    assert export_documents([empty_path], "md").content == "# Transcript\n\n"


@pytest.mark.parametrize(
    ("word_start", "word_end"),
    [(0.998, 2.0), (2.0, 3.002)],
)
def test_loader_rejects_words_outside_explicit_segment_bounds(
    tmp_path: Path,
    word_start: float,
    word_end: float,
) -> None:
    payload = _payload(
        [{
            "segment_id": "seg_0",
            "text": "Hello.",
            "start": 1.0,
            "end": 3.0,
            "words": [{
                "word_id": "w_0",
                "text": "Hello",
                "start": 1.5,
                "end": 2.5,
            }],
        }],
        duration=4.0,
        outcomes={
            "segment_timestamps": "produced",
            "word_timestamps": "produced",
        },
    )
    payload["segments"][0]["words"][0].update({
        "start": word_start,
        "end": word_end,
    })
    transcript = _write(tmp_path / "contradictory-bounds.json", payload)

    with pytest.raises(
        InvalidResultError,
        match="words must fall inside the segment bounds",
    ):
        export_documents([transcript], "srt")


def test_export_preserves_the_recorded_one_millisecond_firered_end_seam(
    tmp_path: Path,
) -> None:
    payload = _payload(
        [{
            "segment_id": "seg_0",
            "text": "Hello.",
            "start": 1.0,
            "end": 3.0,
            "words": [{
                "word_id": "w_0",
                "text": "Hello",
                "start": 2.0,
                "end": 3.001,
            }],
        }],
        duration=4.0,
        outcomes={
            "segment_timestamps": "produced",
            "word_timestamps": "produced",
        },
    )
    transcript = _write(tmp_path / "firered-one-ms-seam.json", payload)

    product = export_documents([transcript], "srt")

    assert "00:00:02,000 --> 00:00:03,001" in product.content


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("requested", [False, True]),
        ("selected_unit_scope", ["0", "1"]),
    ],
)
def test_loader_rejects_non_numeric_opaque_plan_range_bounds(
    tmp_path: Path, field: str, values: list[object],
) -> None:
    payload = _payload([], run_range=[0.0, 1.0])
    payload["provenance"]["plan"]["execution"]["range"][field] = values

    with pytest.raises(InvalidResultError, match="must contain numbers"):
        load_result_document(_write(tmp_path / f"bad-{field}.json", payload))


def test_loader_validates_plan_range_even_when_partial_coverage_owns_merge(
    tmp_path: Path,
) -> None:
    coverage = {
        "covered_through_seconds": 1.0,
        "covered_fraction": 0.5,
        "covered_intervals": [[0.0, 1.0]],
        "missing_intervals": [[1.0, 2.0]],
        "scope_intervals": [[0.0, 2.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    payload = _payload(
        [],
        complete=False,
        coverage=coverage,
        run_range=[0.0, 2.0],
    )
    payload["provenance"]["plan"]["execution"]["range"]["requested"] = [
        -1.0,
        1.0,
    ]

    with pytest.raises(InvalidResultError, match="is not a source interval"):
        load_result_document(_write(tmp_path / "partial-bad-range.json", payload))


def test_loader_requires_partial_coverage_scope_to_match_selected_range(
    tmp_path: Path,
) -> None:
    coverage = {
        "covered_through_seconds": 1.5,
        "covered_fraction": 0.5,
        "covered_intervals": [[1.0, 1.5]],
        "missing_intervals": [[1.5, 2.0]],
        "scope_intervals": [[1.0, 2.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    payload = _payload(
        [],
        complete=False,
        coverage=coverage,
        run_range=[0.0, 1.0],
    )

    with pytest.raises(InvalidResultError, match="selected unit scope"):
        load_result_document(_write(tmp_path / "scope-mismatch.json", payload))


@pytest.mark.parametrize("partial", [False, True])
def test_loader_rejects_one_document_with_segments_outside_its_owned_scope(
    tmp_path: Path,
    partial: bool,
) -> None:
    coverage = {
        "covered_through_seconds": 1.0,
        "covered_fraction": 0.5,
        "covered_intervals": [[0.0, 1.0]],
        "missing_intervals": [[1.0, 2.0]],
        "scope_intervals": [[0.0, 2.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    payload = _payload(
        [_timed_segment("Outside.", [("Outside", 1.1, 1.4)])],
        outcomes={"word_timestamps": "produced"},
        complete=not partial,
        coverage=coverage if partial else ABSENT,
        run_range=None if partial else [0.0, 1.0],
    )

    with pytest.raises(InvalidResultError, match="outside.*owned intervals"):
        load_result_document(_write(tmp_path / f"outside-{partial}.json", payload))


def test_loader_rejects_duplicate_ids_and_nonchronological_words(tmp_path: Path) -> None:
    duplicate = _payload([
        {"segment_id": "seg_0", "text": "One."},
        {"segment_id": "seg_0", "text": "Two."},
    ])
    with pytest.raises(InvalidResultError, match="duplicates"):
        load_result_document(_write(tmp_path / "duplicate.json", duplicate))

    overlapping = _payload(
        [_timed_segment("One two.", [("One", 0.2, 0.6), ("two", 0.5, 0.8)])],
        outcomes={"word_timestamps": "produced"},
    )
    with pytest.raises(InvalidResultError, match="overlaps the preceding word"):
        load_result_document(_write(tmp_path / "overlap.json", overlapping))
