"""Navigation retains occurrence provenance while shortening repeated report detail."""

import copy
import json

import pytest

from audio_cli import cli
from audio_cli.pipeline import PipelineError, compare_reports, summarize_report

from .conftest import region, resolve


def occurrences(navigation):
    return [row for group in navigation["limits"]["groups"] for row in group["occurrences"]]


def check_pointers(value, report):
    if isinstance(value, list):
        for row in value:
            check_pointers(row, report)
    elif isinstance(value, dict):
        if "report_pointer" in value:
            original = resolve(report, value["report_pointer"])
            if "value" in value:
                assert value["value"] == original
        for row in value.values():
            check_pointers(row, report)


def test_groups_keep_nested_limits_and_all_scopes(enhancement, saved):
    component = enhancement["stages"][0]["component_evaluations"][0]
    component["measured_at"] = "after_environment_filters"
    # Equal evidence appears twice, but the inherited scope and phase differ.
    enhancement["stages"][0]["operations"] = [
        {
            "scope": {"time": {"start": 1, "end": 3}},
            "speech_preservation": copy.deepcopy(component["speech_preservation"]),
        }
    ]
    component["noise_reference_scopes"] = [
        {"status": "rejected", "reason": "speech_overlap", "start": 1, "end": 2}
    ]
    enhancement["unresolved"] = [
        {
            "status": "bounded_outside_target",
            "measured_at": "encoded_output",
            "region_id": "machine_001",
        }
    ]
    path = saved(enhancement)
    nav = summarize_report(path, navigation=True)
    default = summarize_report(path, include_evidence_limits=True)
    assert nav["limits"]["indexed_occurrences"] == 6
    assert nav["limits"]["distinct_recorded_values"] == 3
    rows = occurrences(nav)
    assert {r["report_pointer"] for r in rows} == {
        r["report_pointer"] for r in default["evidence_limits"]
    }
    for old in default["evidence_limits"]:
        row = next(r for r in rows if r["report_pointer"] == old["report_pointer"])
        assert row.get("context") == old.get("context")
        assert row.get("region_scopes") == old.get("region_scopes")
    repeats = [r for r in rows if r["report_pointer"].endswith("/speech_preservation")]
    assert repeats[0]["finding_index"] == repeats[1]["finding_index"]
    assert {g["phase"] for g in nav["limits"]["groups"]} == {
        "after_environment_filters",
        "unknown",
        "encoded_output",
    }
    check_pointers(nav, enhancement)


def test_equal_previews_do_not_merge_different_full_evidence(enhancement, saved):
    enhancement["unresolved"] = [
        {"status": "abstained", "measurements": {"different": value}} for value in (1, 2)
    ]
    nav = summarize_report(saved(enhancement), navigation=True)
    rows = [r for r in occurrences(nav) if r["report_pointer"].startswith("/unresolved/")]
    assert rows[0]["finding_index"] != rows[1]["finding_index"]


def test_phase_and_collection_counts_do_not_become_delivered_failures(enhancement, saved):
    enhancement["rule_evaluations"] = [{"status": "outside_target"}] * 10
    enhancement["stages"][0].update(
        inside_target_regions=["machine_001"],
        final_region_evaluations=[
            {"status": "inside_target", "region_id": f"r{i}"} for i in range(9)
        ],
    )
    enhancement["unresolved"] = [
        {"status": "outside_target", "measured_at": "predicted_pre_encode"}
    ]
    nav = summarize_report(saved(enhancement), navigation=True)
    counts = {row["report_pointer"]: row for row in nav["outcome_counts"]}
    assert counts["/rule_evaluations"]["phase"] == "before"
    assert counts["/rule_evaluations"]["statuses"] == {"outside_target": 10}
    assert counts["/stages/0/inside_target_regions"]["count"] == 1
    assert counts["/stages/0/final_region_evaluations"]["count"] == 9
    assert counts["/stages/0/final_region_evaluations"]["phase"] == "unknown"
    assert {g["phase"] for g in nav["limits"]["groups"]} == {
        "unknown",
        "before",
        "predicted_pre_encode",
    }
    assert nav["measurements"]["after"]["program_actual"]["value"] == {
        "integrated_loudness_lufs": -16.1
    }
    assert "success" not in nav
    check_pointers(nav, enhancement)


def test_navigation_comparison_counts_every_pair_and_keeps_references(
    enhancement, inspection, saved
):
    enhancement["regions"] = [region("a", 0, 3), region("b", 4, 6, "speech"), region("c", 8, 9)]
    inspection["regions"] = [
        region("x", 0, 1),
        region("y", 1, 3),
        region("z", 5, 7, "speech"),
        region("n", 9, 10),
    ]
    left, right = saved(enhancement), saved(inspection, "right.json")
    detailed = compare_reports(left, right)
    nav = compare_reports(left, right, navigation=True)
    assert nav["comparison_overview"]["counts"] == {
        "overlap_pairs": 3,
        "ambiguous_overlap_pairs": 2,
        "same_interval_pairs": 0,
        "same_kind_pairs": 3,
        "left_without_overlap": 1,
        "right_without_overlap": 1,
        "left_regions_with_multiple_overlaps": 1,
        "right_regions_with_multiple_overlaps": 0,
    }
    for row, full in zip(
        nav["region_comparison"]["overlaps"], detailed["region_comparison"]["overlaps"], strict=True
    ):
        for side in ("left", "right"):
            assert row[f"{side}_report_pointer"] == full[side]["report_pointer"]
        assert row["intersection"] == full["intersection"]
        assert row["ambiguous"] == full["ambiguous"]
    assert nav["left"]["speech_reference"] != nav["right"]["speech_reference"]
    assert nav["left"]["region_basis"]["value"]["detection"] == "original_source"
    assert nav["right"]["region_basis"]["value"]["detection"] == "inspected_source"
    check_pointers(nav["left"], enhancement)
    check_pointers(nav["right"], inspection)


def test_absent_evidence_stays_absent(enhancement, inspection, saved):
    for report in (enhancement, inspection):
        report.pop("regions")
        report.pop("region_basis")
    enhancement["measurements"] = {"predicted": {"program": {"output_i": -12}}}
    nav = compare_reports(saved(enhancement), saved(inspection, "right.json"), navigation=True)
    assert "region_comparison" not in nav
    assert "comparison_overview" not in nav
    assert "regions" not in nav["left"]
    assert "speech_reference" not in nav["left"]
    assert nav["left"]["measurements"] == {
        "predicted": {"report_pointer": "/measurements/predicted"}
    }


def test_observation_links_keep_every_recorded_scope(enhancement, saved):
    enhancement["rule_evaluations"] = [
        {"status": "outside_target", "observation_id": "local_observation"}
    ]
    enhancement["observations"] = [
        {
            "observation_id": "local_observation",
            "scope": {"time": {"start": start, "end": start + 1}},
        }
        for start in (1, 3)
    ]
    nav = summarize_report(saved(enhancement), navigation=True)
    row = next(r for r in occurrences(nav) if r["report_pointer"] == "/rule_evaluations/0")
    assert row["observation_scopes"] == [
        {"value": observation, "report_pointer": f"/observations/{i}"}
        for i, observation in enumerate(enhancement["observations"])
    ]
    check_pointers(nav, enhancement)


def test_cli_navigation_is_opt_in_offline_and_nonmutating(
    enhancement, inspection, saved, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("navigation must never inspect media")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "SileroOnnxVad", forbidden)
    left, right = saved(enhancement), saved(inspection, "right.json")
    before = left.read_bytes(), right.read_bytes()
    default_summary = summarize_report(left)
    default_comparison = compare_reports(left, right)
    for args, kind in (
        (["summary", str(left)], "audio_report_navigation"),
        (["compare", str(left), str(right)], "audio_report_comparison_navigation"),
    ):
        assert cli.main(["report", *args, "--navigation"]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        assert json.loads(captured.out)["kind"] == kind
        if args[0] == "compare":
            assert captured.out.index('"comparison_overview"') < captured.out.index('"left":')
    assert summarize_report(left) == default_summary
    assert compare_reports(left, right) == default_comparison
    assert (left.read_bytes(), right.read_bytes()) == before


@pytest.mark.parametrize("option", ["--metrics", "--evidence-limits"])
def test_navigation_does_not_silently_ignore_detail_flags(enhancement, saved, option, capsys):
    assert cli.main(["report", "summary", str(saved(enhancement)), "--navigation", option]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot be combined" in json.loads(captured.err)["error"]["message"]


def test_navigation_comparison_keeps_identity_refusal(enhancement, inspection, saved):
    inspection["source"]["sha256"] = "c" * 64
    with pytest.raises(PipelineError):
        compare_reports(saved(enhancement), saved(inspection, "right.json"), navigation=True)
