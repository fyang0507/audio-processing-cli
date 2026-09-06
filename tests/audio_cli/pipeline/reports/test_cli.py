"""Exercise the report options through the public CLI without opening media."""

import json

from audio_cli import cli


def test_cli_details_and_comparison_are_offline(
    enhancement, inspection, saved, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("saved report commands must never inspect media")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "SileroOnnxVad", forbidden)
    left = saved(enhancement)
    right = saved(inspection, "inspection.json")
    before = (left.read_bytes(), right.read_bytes())
    assert cli.main(["report", "summary", str(left), "--metrics", "--evidence-limits"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert "measurements" in summary and "evidence_limits" in summary
    assert cli.main(["report", "compare", str(left), str(right)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    compared = json.loads(captured.out)
    assert compared["kind"] == "audio_report_comparison"
    assert compared["region_comparison"]["basis"] == "positive_time_interval_overlap"
    assert (left.read_bytes(), right.read_bytes()) == before


def test_cli_comparison_refuses_unrelated_sources(enhancement, inspection, saved, capsys):
    inspection["source"]["sha256"] = "c" * 64
    assert (
        cli.main(
            [
                "report",
                "compare",
                str(saved(enhancement)),
                str(saved(inspection, "inspection.json")),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["type"] == "PipelineError"
