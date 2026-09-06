"""Offline summaries retain scoped uncertainty and navigate canonical measurements."""

import json
import os
import subprocess
import sys

import pytest

from audio_cli import cli
from audio_cli.pipeline import summarize_report
from audio_cli.pipeline.models import PipelineError


@pytest.fixture
def report():
    return {
        "kind": "audio_enhancement_report",
        "schema_version": "1",
        "source": {"path": "/missing/source.wav", "sha256": "recorded"},
        "profile": {"name": "product-demo", "version": "5", "target_lufs": -16},
        "dry_run": False,
        "rendered": True,
        "timeline_preserved": True,
        "region_basis": {"regional_measurements": "fixed_source_regions"},
        "stages": [
            {
                "name": "environment-denoise",
                "status": "applied",
                "reason": "eligible_environmental_cleanup_resolved",
                "operations": [{"type": "highpass", "large_data": list(range(1000))}],
                "component_evaluations": [
                    {
                        "component": "broadband-denoise",
                        "status": "abstained",
                        "reason": "insufficient_reference",
                        "scope": {"time": {"start": 1.1, "end": 2.2}},
                        "reference_evidence": {"future_component_field": [3, 7]},
                    }
                ],
            },
            {"name": "voice-enhance", "status": "skipped", "reason": "skipped_by_user"},
        ],
        "measurements": {
            "before": {"program_actual": {"integrated_loudness_lufs": -32}},
            "predicted": {"program_actual": {"integrated_loudness_lufs": -16}},
            "after": {"program_actual": {"integrated_loudness_lufs": -16.1}},
        },
        "unresolved": [
            {
                "stage": "source-balance",
                "region_id": "machine_001",
                "status": "bounded_outside_target",
                "reason": "correction_bound_reached",
                "measured_at": "encoded_output",
                "scope": {"time": {"start": 5.2, "end": 7.5}, "frequency": "all"},
                "target_difference_db": {"minimum": -4, "maximum": -2},
                "difference_from_speech_db": -6,
                "local_evidence": {"future_field": ["preserve", {"detail": True}]},
            }
        ],
        "final_peak_validation": {"status": "pass", "measured_true_peak_dbtp": -1.8},
    }


def save(tmp_path, report):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_summary_is_offline_json_preserves_scopes_and_does_not_modify_report(
    tmp_path, report, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("summary must never probe or process audio")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "SileroOnnxVad", forbidden)
    monkeypatch.setattr(cli.EnhancementPipeline, "run", forbidden)
    path = save(tmp_path, report)
    before = path.read_bytes()
    assert cli.main(["report", "summary", path.name]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["report"] == str(path)
    assert path.read_bytes() == before
    assert len(captured.out) < len(before)
    assert "status" not in summary  # No invented overall success.
    assert "measurements" not in summary
    assert "operations" not in summary["stages"][0]
    assert summary["stages"][0]["status"] == "applied"
    assert summary["stages"][1]["status"] == "skipped"
    assert summary["unresolved"][0] == {
        **report["unresolved"][0],
        "report_pointer": "/unresolved/0",
    }
    assert summary["stages"][0]["component_evaluations"][0] == {
        **report["stages"][0]["component_evaluations"][0],
        "report_pointer": "/stages/0/component_evaluations/0",
    }
    for scope, location in summary["measurement_scopes"].items():
        pointer = location["program_actual"]["report_pointer"]
        value = report
        for field in pointer.lstrip("/").split("/"):
            value = value[field]
        assert value == report["measurements"][scope]["program_actual"]


def test_older_report_absence_is_not_an_empty_or_successful_outcome(tmp_path, report):
    for field in ("unresolved", "region_basis", "final_peak_validation", "timeline_preserved"):
        report.pop(field)
    report["measurements"]["after"] = {"program": {"input_i": -16, "output_i": -12}}
    report["stages"][0].pop("component_evaluations")
    summary = summarize_report(save(tmp_path, report))
    for field in ("unresolved", "region_basis", "final_peak_validation", "timeline_preserved"):
        assert field not in summary
    assert summary["measurement_scopes"]["after"] == {"report_pointer": "/measurements/after"}
    assert "component_evaluations" not in summary["stages"][0]
    assert "timeline_verification" not in summary


@pytest.mark.parametrize("checked", [False, True])
def test_timeline_verification_preserves_exact_nested_abstentions(
    tmp_path, report, checked, capsys
):
    # Shape supplied by the timing owner's pipeline/timing.py in commit 156547e.
    verification = {
        "status": "pass" if checked else "not_run",
        "scope": "decoded_audio_duration_only",
        "tolerance_ms": 50,
        "content_alignment": {"status": "abstained", "reason": "not_measured"},
        "av_sync": {"status": "abstained", "reason": "not_measured"},
    }
    report["timeline_verification"] = verification
    if not checked:
        report["rendered"] = False
        report["dry_run"] = True
        report.pop("timeline_preserved")
        report["measurements"].pop("after")
    path = save(tmp_path, report)
    before = path.read_bytes()
    assert cli.main(["report", "summary", str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["timeline_verification"] == verification
    assert ("timeline_preserved" in summary) is checked
    assert path.read_bytes() == before


def test_dry_run_keeps_prediction_scope_without_fabricating_encoded_measurements(tmp_path, report):
    report["rendered"] = False
    report["dry_run"] = True
    report["measurements"].pop("after")
    report["unresolved"][0]["measured_at"] = "predicted_pre_encode"
    report["final_peak_validation"]["status"] = "predicted_pass"
    summary = summarize_report(save(tmp_path, report))
    assert "after" not in summary["measurement_scopes"]
    assert summary["unresolved"][0]["measured_at"] == "predicted_pre_encode"
    assert "measured_at" not in summary["stages"][0]["component_evaluations"][0]
    assert summary["final_peak_validation"]["status"] == "predicted_pass"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "audio_inspection"),
        ("schema_version", "2"),
        ("schema_version", 1),
        ("rendered", "true"),
        ("dry_run", None),
        ("measurements", []),
        ("stages", None),
        ("stages", [{"status": None}]),
        ("unresolved", None),
        ("unresolved", ["abstained"]),
        ("region_basis", "source"),
        ("timeline_preserved", None),
        ("timeline_verification", None),
        ("timeline_verification", []),
        ("timeline_verification", True),
        ("timeline_verification", "pass"),
        ("final_peak_validation", None),
    ],
)
def test_malformed_known_fields_and_unknown_versions_refuse(tmp_path, report, field, value, capsys):
    report[field] = value
    path = save(tmp_path, report)
    assert cli.main(["report", "summary", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["type"] == "PipelineError"


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', "[]", '{"a":NaN}', '{"a":1e999}', "{"])
def test_invalid_json_cannot_silently_select_or_invent_evidence(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw)
    with pytest.raises(PipelineError):
        summarize_report(path)


def test_missing_file_is_a_json_error(tmp_path, capsys):
    assert cli.main(["report", "summary", str(tmp_path / "missing.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["type"] == "PipelineError"


def run_summary_cli(path):
    return subprocess.run(
        [sys.executable, "-m", "audio_cli", "report", "summary", str(path)],
        capture_output=True,
        timeout=10,
        env={**os.environ, "PYTHONIOENCODING": "utf-8:strict"},
    )


@pytest.mark.parametrize("location", ["reason", "nested_value", "nested_key"])
def test_unpaired_surrogates_refuse_before_any_cli_stdout(tmp_path, report, location):
    if location == "reason":
        report["stages"][0]["reason"] = "bad\ud800"
    elif location == "nested_value":
        report["unresolved"][0]["local_evidence"] = {"detail": ["bad\ud800"]}
    else:
        report["timeline_verification"] = {"nested": {"bad\ud800": "value"}}
    completed = run_summary_cli(save(tmp_path, report))
    assert completed.returncode == 2
    assert completed.stdout == b""
    error = json.loads(completed.stderr)["error"]
    assert error["type"] == "PipelineError"
    assert "surrogates not allowed" in error["message"]


@pytest.mark.parametrize("kind", ["fifo", "directory"])
@pytest.mark.parametrize("via_symlink", [False, True])
def test_nonregular_reports_refuse_without_blocking(tmp_path, kind, via_symlink):
    path = tmp_path / "special.json"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    if via_symlink:
        link = tmp_path / "linked.json"
        link.symlink_to(path.name)
        path = link
    # A regression to blocking FIFO open fails via the subprocess timeout, not a
    # hung test suite. No writer is attached to make the FIFO artificially ready.
    completed = run_summary_cli(path)
    assert completed.returncode == 2
    assert completed.stdout == b""
    error = json.loads(completed.stderr)["error"]
    assert error["type"] == "PipelineError"
    assert "input must be a regular file" in error["message"]


def test_regular_report_symlink_and_valid_unicode_remain_supported(tmp_path, report):
    report["stages"][0]["reason"] = "保留原文 🎧"
    path = save(tmp_path, report)
    before = path.read_bytes()
    link = tmp_path / "linked.json"
    link.symlink_to(path.name)
    completed = run_summary_cli(link)
    assert completed.returncode == 0
    assert completed.stderr == b""
    summary = json.loads(completed.stdout)
    assert summary["report"] == str(path.resolve())
    assert summary["stages"][0]["reason"] == "保留原文 🎧"
    assert path.read_bytes() == before
    assert link.is_symlink()


def test_real_report_builder_shape_can_be_projected(tmp_path, monkeypatch):
    from test_pipeline_reports import PROFILE, report_inputs

    from audio_cli.pipeline import reporting

    monkeypatch.setattr(reporting, "ffmpeg_version", lambda: "synthetic")
    report = reporting.build_report(PROFILE, *report_inputs(tmp_path))
    summary = summarize_report(save(tmp_path, report))
    assert summary["profile"] == {"name": PROFILE.name, "version": PROFILE.version}
    for original, row in zip(report["unresolved"], summary["unresolved"], strict=True):
        assert {key: value for key, value in row.items() if key != "report_pointer"} == original
