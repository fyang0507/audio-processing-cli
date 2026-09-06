"""Opt-in metric and evidence views must neither relabel nor invent measurements."""

import copy

import pytest

from audio_cli.pipeline import PipelineError, summarize_report
from tests.audio_cli.pipeline.reports.conftest import resolve


def test_default_summary_stays_unchanged_and_options_are_independent(enhancement, saved):
    enhancement["measurements"]["after"].update(
        duration_delta_ms=0.125, duration_basis="decoded_pcm"
    )
    path = saved(enhancement)
    original = path.read_bytes()
    default = summarize_report(path)
    metrics = summarize_report(path, include_metrics=True)
    limits = summarize_report(path, include_evidence_limits=True)
    assert "measurements" not in default and "evidence_limits" not in default
    assert {k: v for k, v in metrics.items() if k not in {"measurements", "regions"}} == default
    assert {k: v for k, v in limits.items() if k != "evidence_limits"} == default
    assert "evidence_limits" not in metrics and "measurements" not in limits
    for scope in ("before", "predicted", "after"):
        measured = metrics["measurements"][scope]["program_actual"]
        assert measured["value"] == resolve(enhancement, measured["report_pointer"])
    assert metrics["measurements"]["after"]["duration_delta_ms"] == {
        "value": 0.125,
        "report_pointer": "/measurements/after/duration_delta_ms",
    }
    assert metrics["measurements"]["after"]["duration_basis"]["value"] == "decoded_pcm"
    assert "output_i" not in str(metrics["measurements"])
    assert path.read_bytes() == original


def test_legacy_missing_metrics_are_not_inferred_from_loudnorm_outputs(enhancement, saved):
    enhancement["measurements"] = {"after": {"program": {"input_i": -17, "output_i": -12}}}
    enhancement.pop("regions")
    enhancement.pop("unresolved")
    result = summarize_report(saved(enhancement), include_metrics=True)
    assert result["measurements"] == {"after": {"report_pointer": "/measurements/after"}}
    assert "regions" not in result and "unresolved" not in result


def test_limits_keep_exact_nested_pointer_and_scope_provenance(enhancement, saved):
    component = enhancement["stages"][0]["component_evaluations"][0]
    component["measured_at"] = "before_initial_program_loudness"
    component["noise_reference_scope_basis"] = "source_timeline_after_speech_and_program_exclusion"
    component["noise_reference_scopes"] = [
        {"start": 0, "end": 0.1, "status": "rejected", "reason": "too_short"}
    ]
    component["future/~evidence"] = {"status": "abstained", "reason": "unknown"}
    enhancement["unresolved"] = [{"region_id": "machine_001", "status": "outside_target"}]
    result = summarize_report(saved(enhancement), include_evidence_limits=True)
    limits = {row["report_pointer"]: row for row in result["evidence_limits"]}
    for pointer, row in limits.items():
        assert row["evidence"] == resolve(enhancement, pointer)
        for value in row.get("context", {}).values():
            assert value["value"] == resolve(enhancement, value["report_pointer"])
    prefix = "/stages/0/component_evaluations/0"
    nested = limits[f"{prefix}/speech_preservation"]
    assert nested["context"]["scope"]["report_pointer"] == f"{prefix}/scope"
    assert nested["context"]["measured_at"]["value"] == "before_initial_program_loudness"
    assert f"{prefix}/future~1~0evidence" in limits
    assert limits["/unresolved/0"]["region_scopes"][0]["value"]["start"] == 1
    assert "measured_at" not in limits["/unresolved/0"]["context"]
    assert "/timeline_verification/content_alignment" in limits
    assert "/timeline_verification/av_sync" in limits


def test_calibration_limit_is_visible_beneath_applied_parent(enhancement, saved):
    enhancement["stages"][0]["guide_calibration"] = {
        "status": "applied",
        "target_attained": False,
        "limiting_reasons": ["maximum_gain"],
        "scope": "model_guide_only",
    }
    limits = summarize_report(saved(enhancement), include_evidence_limits=True)["evidence_limits"]
    assert any(row["report_pointer"] == "/stages/0/guide_calibration" for row in limits)


@pytest.mark.parametrize("value", [None, [], "-16", True])
def test_malformed_actual_values_refuse_only_requested_metric_projection(enhancement, saved, value):
    enhancement["measurements"]["after"]["program_actual"]["integrated_loudness_lufs"] = value
    path = saved(enhancement)
    summarize_report(path)
    with pytest.raises(PipelineError, match="must be a number"):
        summarize_report(path, include_metrics=True)


def test_flattening_arbitrary_extension_status_does_not_crash(enhancement, saved):
    enhancement["stages"][0]["extension"] = {"status": {"vendor": "custom"}}
    before = copy.deepcopy(enhancement)
    summarize_report(saved(enhancement), include_evidence_limits=True)
    assert enhancement == before
