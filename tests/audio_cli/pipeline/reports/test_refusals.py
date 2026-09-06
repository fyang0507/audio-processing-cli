"""Raw JSON and malformed saved evidence refuse before a result can escape."""

import json
import os

import pytest

from audio_cli.pipeline import PipelineError, compare_reports, summarize_report


@pytest.mark.parametrize(
    "raw",
    [
        '{"kind":"audio_inspection","kind":"audio_enhancement_report"}',
        '{"nested":{"value":1,"value":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e999}',
        "[1]",
        "null",
        "{",
    ],
)
def test_every_raw_input_boundary_rejects_invalid_json(tmp_path, raw, enhancement, saved):
    valid = saved(enhancement)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(raw)
    with pytest.raises(PipelineError):
        summarize_report(invalid, include_metrics=True, include_evidence_limits=True)
    for pair in ((invalid, valid), (valid, invalid)):
        with pytest.raises(PipelineError):
            compare_reports(*pair)


@pytest.mark.parametrize(
    "mutation",
    [
        "kind",
        "version",
        "regions",
        "interval",
        "duplicate_region",
        "observations",
        "regional",
        "actual",
        "surrogate",
    ],
)
def test_known_malformed_report_shapes_refuse(enhancement, inspection, saved, mutation):
    if mutation == "kind":
        inspection["kind"] = "transcription_result"
    elif mutation == "version":
        inspection["schema_version"] = "2"
    elif mutation == "regions":
        inspection["regions"] = {}
    elif mutation == "interval":
        inspection["regions"][0]["end"] = -1
    elif mutation == "duplicate_region":
        inspection["regions"].append(inspection["regions"][0])
    elif mutation == "observations":
        inspection["observations"] = [None]
    elif mutation == "regional":
        enhancement["measurements"]["after"]["regional"] = []
    elif mutation == "actual":
        inspection["measurements"]["program_actual"] = None
    else:
        inspection["unknown_extension"] = "bad\ud800"
    with pytest.raises(PipelineError):
        compare_reports(saved(enhancement, "e.json"), saved(inspection, "i.json"))


def test_nonfinite_nested_evidence_is_not_hidden_by_projection(enhancement, saved):
    enhancement["stages"][0]["extension"] = {"future_evidence": float("nan")}
    path = saved(enhancement)
    assert "NaN" in path.read_text()
    with pytest.raises(PipelineError):
        summarize_report(path, include_evidence_limits=True)
    with pytest.raises(PipelineError):
        compare_reports(path, path)


def test_compare_regular_symlink_and_nonregular_input(enhancement, saved, tmp_path):
    path = saved(enhancement)
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    assert compare_reports(link, path)["left"]["report"] == str(path.resolve())
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(PipelineError, match="regular file"):
        compare_reports(fifo, path)


def test_noop_missing_metrics_do_not_become_nulls(enhancement, saved):
    enhancement["measurements"] = {"before": {"program_actual": {}}}
    result = summarize_report(saved(enhancement), include_metrics=True)
    assert result["measurements"]["before"]["program_actual"]["value"] == {}
    assert "null" not in json.dumps(result["measurements"])


@pytest.mark.parametrize("kind", [None, [], {}, 1])
def test_nonstring_report_kind_refuses_with_pipeline_error(enhancement, saved, kind):
    enhancement["kind"] = kind
    path = saved(enhancement)
    with pytest.raises(PipelineError):
        summarize_report(path)
    with pytest.raises(PipelineError):
        compare_reports(path, path)


def test_region_outside_recorded_timeline_refuses(enhancement, inspection, saved):
    inspection["regions"][0]["end"] = 100
    with pytest.raises(PipelineError, match="exceeds the recorded source timeline"):
        compare_reports(saved(enhancement, "e.json"), saved(inspection, "i.json"))
