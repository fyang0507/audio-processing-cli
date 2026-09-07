"""Recorded interval equality is independent of IDs, metric values, and audio quality."""

import copy

import pytest

from audio_cli.pipeline import compare_reports
from tests.audio_cli.pipeline.reports.conftest import region


def compare(enhancement, inspection, saved):
    left, right = saved(enhancement), saved(inspection, "right.json")
    original = left.read_bytes(), right.read_bytes()
    result = compare_reports(left, right)
    navigation = compare_reports(left, right, navigation=True)
    scopes = result.get("measurement_scope_comparison")
    assert navigation.get("comparison_overview", {}).get("measurement_scope_comparison") == scopes
    assert compare_reports(right, left).get("measurement_scope_comparison") == scopes
    assert (left.read_bytes(), right.read_bytes()) == original
    return scopes


def test_fixed_and_fresh_scopes_report_two_independent_differences(enhancement, inspection, saved):
    assert compare(enhancement, inspection, saved) == {
        "basis": "recorded_time_intervals",
        "speech_reference_intervals": "different",
        "non_speech_region_intervals": "different",
    }


def test_same_speech_union_ignores_splits_overlaps_order_and_ids(enhancement, inspection, saved):
    inspection["regions"] = [
        region("different_machine_id", 1, 3),
        region("speech_c", 5, 6, "speech"),
        region("speech_b", 4.5, 5.5, "speech"),
        region("speech_a", 4, 4.5, "speech"),
    ]
    # The references measure different media phases and have different levels.
    # Equality of recorded coverage must assert neither equal RMS nor audio.
    assert (
        enhancement["measurements"]["after"]["regional"]["speech_program_rms_dbfs"]
        != (inspection["observations"][0]["measured_rms_dbfs"])
    )
    enhancement["measurements"]["before"]["regional"] = {
        "speech_program_rms_dbfs": -50,
        "machine_regions": [],
    }
    assert compare(enhancement, inspection, saved) == {
        "basis": "recorded_time_intervals",
        "speech_reference_intervals": "same",
        "non_speech_region_intervals": "same",
    }


@pytest.mark.parametrize("kind", ["speech", "non_speech_program"])
def test_each_scope_can_change_without_changing_the_other(enhancement, inspection, saved, kind):
    inspection["regions"] = copy.deepcopy(enhancement["regions"])
    for row in inspection["regions"]:
        if row["kind"] == kind:
            row["start"] += 0.000001
    scopes = compare(enhancement, inspection, saved)
    assert scopes["speech_reference_intervals"] == ("different" if kind == "speech" else "same")
    assert scopes["non_speech_region_intervals"] == (
        "different" if kind == "non_speech_program" else "same"
    )


@pytest.mark.parametrize(
    "machines",
    [
        [region("a", 1, 2), region("b", 2, 3)],
        [region("a", 1, 3), region("b", 1, 3)],
    ],
    ids=["split_equal_coverage", "duplicate_interval"],
)
def test_non_speech_intervals_preserve_individual_boundaries_and_multiplicity(
    enhancement, inspection, saved, machines
):
    inspection["regions"] = [region("speech", 4, 6, "speech"), *machines]
    scopes = compare(enhancement, inspection, saved)
    assert scopes["speech_reference_intervals"] == "same"
    assert scopes["non_speech_region_intervals"] == "different"


def test_non_speech_interval_order_and_report_local_ids_do_not_matter(
    enhancement, inspection, saved
):
    enhancement["regions"] = [region("a", 1, 3), region("b", 7, 8)]
    inspection["regions"] = [region("a", 7, 8), region("b", 1, 3)]
    assert compare(enhancement, inspection, saved) == {
        "basis": "recorded_time_intervals",
        "non_speech_region_intervals": "same",
    }


def test_empty_candidate_sets_do_not_manufacture_a_speech_reference(enhancement, inspection, saved):
    enhancement["regions"] = []
    inspection["regions"] = []
    assert compare(enhancement, inspection, saved) == {
        "basis": "recorded_time_intervals",
        "non_speech_region_intervals": "same",
    }


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("missing", ["regions", "region_basis"])
def test_missing_legacy_scope_evidence_stays_absent(enhancement, inspection, saved, side, missing):
    report = enhancement if side == "left" else inspection
    report.pop(missing)
    assert compare(enhancement, inspection, saved) is None


@pytest.mark.parametrize("key", ["detection", "regional_measurements", "timeline"])
def test_incomplete_scope_basis_does_not_acquire_a_conclusion(enhancement, inspection, saved, key):
    inspection["region_basis"].pop(key)
    assert compare(enhancement, inspection, saved) is None


@pytest.mark.parametrize("key", ["detection", "regional_measurements"])
def test_unknown_scope_basis_stays_uninterpreted(enhancement, inspection, saved, key):
    inspection["region_basis"][key] = "future_scope"
    assert compare(enhancement, inspection, saved) is None


@pytest.mark.parametrize("side", ["left", "right"])
def test_reference_intervals_require_a_recorded_speech_measurement(
    enhancement, inspection, saved, side
):
    if side == "left":
        enhancement["measurements"]["after"]["regional"].pop("speech_program_rms_dbfs")
    else:
        inspection["observations"].pop(0)
    assert compare(enhancement, inspection, saved) == {
        "basis": "recorded_time_intervals",
        "non_speech_region_intervals": "different",
    }


@pytest.mark.parametrize("value", [None, True, "-19"])
def test_unavailable_inspection_reference_is_not_invented(enhancement, inspection, saved, value):
    inspection["observations"][0]["measured_rms_dbfs"] = value
    assert "speech_reference_intervals" not in compare(enhancement, inspection, saved)


def test_unknown_reference_scope_is_not_assumed_to_use_the_speech_manifest(
    enhancement, inspection, saved
):
    inspection["observations"][0]["scope"] = {"time": "all", "frequency": "all"}
    assert "speech_reference_intervals" not in compare(enhancement, inspection, saved)


def test_no_measurement_or_quality_verdict_is_added(enhancement, inspection, saved):
    inspection["regions"] = copy.deepcopy(enhancement["regions"])
    scopes = compare(enhancement, inspection, saved)
    assert set(scopes) == {
        "basis",
        "speech_reference_intervals",
        "non_speech_region_intervals",
    }
