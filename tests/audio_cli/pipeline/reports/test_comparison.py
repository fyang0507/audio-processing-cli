"""Recorded identity bridges, fixed/fresh intervals, and distinct speech references."""

import copy

import pytest

from audio_cli.pipeline import PipelineError, compare_reports
from tests.audio_cli.pipeline.reports.conftest import region, resolve


def test_output_bridge_keeps_fixed_and_fresh_metrics_without_live_media(
    enhancement, inspection, saved
):
    left, right = saved(enhancement, "left.json"), saved(inspection, "right.json")
    originals = left.read_bytes(), right.read_bytes()
    result = compare_reports(left, right)
    assert result["kind"] == "audio_report_comparison"
    assert result["compatibility"]["basis"] == "recorded_output_sha256"
    verification = result["compatibility"]["timeline_verification"]
    assert verification["value"]["content_alignment"]["status"] == "abstained"
    assert verification["value"]["av_sync"]["status"] == "abstained"
    for key, report in (("left", enhancement), ("right", inspection)):
        side = result[key]
        assert side["source"]["value"] == report["source"]
        for item in side["regions"]:
            for measure in item.get("measurements", []):
                assert measure["value"] == resolve(report, measure["report_pointer"])
    left_reference = result["left"]["speech_reference"]
    right_reference = result["right"]["speech_reference"]
    assert left_reference["regions"][0]["value"]["start"] == 4
    assert right_reference["regions"][0]["value"]["start"] == 3.8
    assert left_reference["measurements"][0]["value"] == -20
    assert right_reference["measurements"][0]["value"]["measured_rms_dbfs"] == -19
    assert result["right"]["rule_evaluations"][0]["value"] == inspection["rule_evaluations"][0]
    matched = result["region_comparison"]["overlaps"][1]
    assert matched["left"]["value"]["region_id"] == "machine_001"
    assert matched["right"]["value"]["region_id"] == "machine_002"
    assert matched["same_interval"] is False and matched["ambiguous"] is False
    assert "status" not in result and "regression" not in str(result)
    assert (left.read_bytes(), right.read_bytes()) == originals
    reverse = compare_reports(right, left)
    assert reverse["compatibility"]["timeline_verification"]["side"] == "right"


def test_same_source_inspection_and_enhancement_ignore_historical_paths(
    enhancement, inspection, saved
):
    inspection["source"] = copy.deepcopy(enhancement["source"])
    inspection["source"]["path"] = "/different/moved/path.wav"
    result = compare_reports(saved(enhancement, "e.json"), saved(inspection, "i.json"))
    assert result["compatibility"]["basis"] == "same_source_sha256"
    assert result["left"]["region_basis"]["value"]["detection"] == "original_source"
    assert result["right"]["region_basis"]["value"]["detection"] == "inspected_source"


def test_overlap_reports_split_merge_and_classification_changes_without_ordinal_matching(
    enhancement, inspection, saved
):
    enhancement["regions"] = [region("machine_001", 1, 4), region("machine_002", 8, 9)]
    inspection["regions"] = [
        region("machine_009", 1, 2),
        region("speech_001", 2, 4, "speech"),
        region("machine_001", 4, 5),
    ]
    result = compare_reports(saved(enhancement, "e.json"), saved(inspection, "i.json"))
    compared = result["region_comparison"]
    assert len(compared["overlaps"]) == 2
    assert all(row["ambiguous"] for row in compared["overlaps"])
    assert [row["same_kind"] for row in compared["overlaps"]] == [True, False]
    assert compared["left_without_overlap"][0]["value"]["start"] == 8
    assert compared["right_without_overlap"][0]["value"]["start"] == 4
    assert "lost" not in str(compared) and "regression" not in str(compared)


def test_absent_region_and_metric_blocks_stay_absent(enhancement, inspection, saved):
    for report in (enhancement, inspection):
        report.pop("regions")
        report["measurements"] = {}
    inspection.pop("observations")
    result = compare_reports(saved(enhancement, "e.json"), saved(inspection, "i.json"))
    assert "region_comparison" not in result
    assert "regions" not in result["left"] and "speech_reference" not in result["right"]
    assert result["left"]["measurements"] == {}
    assert result["right"]["measurements"] == {"inspected": {"report_pointer": "/measurements"}}


def test_legacy_actual_metrics_are_not_reconstructed(enhancement, inspection, saved):
    enhancement["measurements"]["after"].pop("program_actual")
    inspection["measurements"].pop("program_actual")
    result = compare_reports(saved(enhancement, "e.json"), saved(inspection, "i.json"))
    assert "program_actual" not in result["left"]["measurements"]["after"]
    assert "program_actual" not in result["right"]["measurements"]["inspected"]


@pytest.mark.parametrize(
    "field,value", [("sha256", "c" * 64), ("sha256", "bad"), ("decoded_audio", None)]
)
def test_unrelated_or_unestablished_identity_refuses(enhancement, inspection, saved, field, value):
    inspection["source"][field] = value
    with pytest.raises(PipelineError):
        compare_reports(saved(enhancement, "e.json"), saved(inspection, "i.json"))


@pytest.mark.parametrize(
    "mutation",
    [
        "origin",
        "duration",
        "preserved",
        "verification",
        "dry_run",
        "timeline",
        "sample_count",
        "tolerance",
    ],
)
def test_conflicting_timeline_or_unverified_output_refuses(
    enhancement, inspection, saved, mutation
):
    if mutation == "origin":
        inspection["source"]["decoded_audio"]["time_origin"] = "container_start"
    elif mutation == "duration":
        inspection["source"]["decoded_audio"]["duration_seconds"] = 11
    elif mutation == "preserved":
        enhancement["timeline_preserved"] = False
    elif mutation == "verification":
        enhancement["timeline_verification"]["status"] = "not_run"
    elif mutation == "dry_run":
        enhancement["dry_run"] = True
    elif mutation == "timeline":
        inspection["region_basis"]["timeline"] = "trimmed"
    elif mutation == "sample_count":
        inspection["source"]["decoded_audio"]["sample_count"] = True
    else:
        enhancement["timeline_verification"]["tolerance_ms"] = -1
    with pytest.raises(PipelineError):
        compare_reports(saved(enhancement, "e.json"), saved(inspection, "i.json"))


def test_same_source_conflicting_sample_durations_refuse(inspection, saved):
    other = copy.deepcopy(inspection)
    other["source"]["decoded_audio"].update(sample_count=528000, duration_seconds=11)
    with pytest.raises(PipelineError, match="conflicting decoded durations"):
        compare_reports(saved(inspection, "a.json"), saved(other, "b.json"))


@pytest.mark.parametrize("seconds,accepted", [(10.05, True), (10.050021, False)])
def test_output_bridge_consumes_recorded_duration_tolerance(
    enhancement, inspection, saved, seconds, accepted
):
    count = round(seconds * 48000)
    for metadata in (enhancement["output"], inspection["source"]):
        metadata["decoded_audio"].update(
            sample_count=count, duration_seconds=round(count / 48000, 6)
        )
    left, right = saved(enhancement, "e.json"), saved(inspection, "i.json")
    if accepted:
        assert compare_reports(left, right)["compatibility"]["basis"] == "recorded_output_sha256"
    else:
        with pytest.raises(PipelineError, match="contradicts its recorded duration verification"):
            compare_reports(left, right)
