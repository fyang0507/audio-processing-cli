import json
from pathlib import Path

from audio_cli.cli import main


def test_list_stages_requires_no_input(capsys) -> None:
    result = main(["enhance", "--profile", "product-demo", "--list-stages"])
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["profile_version"] == "4"
    assert len(payload["stages"]) == 5


def test_cli_rejects_unknown_skip_without_loading_model(tmp_path, capsys) -> None:
    source = tmp_path / "missing.wav"
    result = main(
        [
            "enhance",
            str(source),
            "--profile",
            "product-demo",
            "--skip",
            "unknown",
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().err)
    assert result == 2
    assert "Unknown stage" in payload["error"]["message"]


def test_cli_rejects_invalid_scope_without_creating_artifacts(
    tmp_path, capsys, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "enhanced.wav"
    adjustments = tmp_path / "adjustments.json"
    adjustments.write_text(
        json.dumps(
            {
                "adjustments": [
                    {
                        "type": "gain",
                        "gain_db": 4,
                        "scope": {
                            "time": {"start": 4.0, "end": 6.0},
                            "frequency": "all",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("audio_cli.cli.probe_media", lambda path: {})
    monkeypatch.setattr(
        "audio_cli.cli.media_summary",
        lambda path, probe: {"duration_seconds": 5.0},
    )

    result = main(
        [
            "enhance",
            str(source),
            "--profile",
            "product-demo",
            "--adjustments",
            str(adjustments),
            "--output",
            str(output),
        ]
    )
    payload = json.loads(capsys.readouterr().err)

    assert result == 2
    assert payload["error"] == {
        "type": "AdjustmentError",
        "code": "scope_out_of_range",
        "message": "adjustments[0].scope.time must satisfy 0 <= start < end <= 5.000",
        "field": "adjustments[0].scope.time",
        "provided": {"start": 4.0, "end": 6.0},
        "allowed": {
            "minimum_start_seconds": 0.0,
            "maximum_end_seconds": 5.0,
            "constraint": "start < end",
        },
    }
    assert not output.exists()
    assert not (tmp_path / "enhanced.wav.report.json").exists()


def test_inspect_refuses_an_existing_report_and_replaces_it_with_force(
    tmp_path, capsys, monkeypatch
) -> None:
    """`inspect --report` hard-coded `force=False` and `inspect` had no `--force`.

    So an existing report could never be replaced, while `enhance` — the command that writes both
    a render and a report — has taken `--force` all along. The check also moves to the front: the
    analysis decodes the file and runs speech detection over it, and refusing the destination
    afterwards threw all of that away.
    """
    source = tmp_path / "source.wav"
    source.write_bytes(b"not really a wav, and nothing here reads it")
    report = tmp_path / "inspection.json"
    report.write_text('{"stale": true}', encoding="utf-8")

    # No stubs: the refusal has to come before anything is loaded or decoded.
    assert main(["inspect", str(source), "--report", str(report)]) == 2
    message = json.loads(capsys.readouterr().err)["error"]["message"]
    assert "pass --force" in message
    assert json.loads(report.read_text()) == {"stale": True}

    monkeypatch.setattr("audio_cli.cli.SileroOnnxVad", lambda path: object())
    monkeypatch.setattr(
        "audio_cli.cli.inspect_source",
        lambda source, *, profile, detector: {"kind": "audio_inspection", "fresh": True},
    )
    assert main(["inspect", str(source), "--report", str(report), "--force"]) == 0
    capsys.readouterr()
    assert json.loads(report.read_text())["fresh"] is True


def test_enhance_refuses_existing_targets_and_replaces_them_with_force(
    tmp_path, capsys, monkeypatch
) -> None:
    """`enhance --force` was wired but unasserted, which is how `inspect`'s missing one survived.

    A mutation scan flipped both of its `force=args.force` arguments to `False` and the suite
    stayed green. Both targets are checked, so both are exercised: the render and the report that
    defaults to sit beside it.
    """
    source = tmp_path / "source.wav"
    source.write_bytes(b"nothing here is decoded on the refusal path")
    output = tmp_path / "enhanced.wav"
    report = tmp_path / "enhanced.wav.report.json"
    argv = ["enhance", str(source), "--profile", "product-demo", "-o", str(output)]

    output.write_text("an earlier render")
    assert main(argv) == 2
    assert "pass --force" in json.loads(capsys.readouterr().err)["error"]["message"]
    assert output.read_text() == "an earlier render"

    # The report is a second target with its own check, and it is refused on its own.
    output.unlink()
    report.write_text('{"stale": true}')
    assert main(argv) == 2
    assert "Report already exists" in json.loads(capsys.readouterr().err)["error"]["message"]
    assert json.loads(report.read_text()) == {"stale": True}

    class StubPipeline:
        def __init__(self, profile, **kwargs) -> None:
            pass

        def run(self, source, *, output, dry_run, allow_enhanced_input):
            Path(output).write_text("a new render")
            return {"rendered": True, "fresh": True}

    monkeypatch.setattr("audio_cli.cli.probe_media", lambda path: {})
    monkeypatch.setattr("audio_cli.cli.media_summary",
                        lambda path, probe: {"duration_seconds": 4.0})
    monkeypatch.setattr("audio_cli.cli.SileroOnnxVad", lambda path: object())
    monkeypatch.setattr("audio_cli.cli.EnhancementPipeline", StubPipeline)

    output.write_text("an earlier render")
    assert main([*argv, "--force"]) == 0
    capsys.readouterr()
    assert output.read_text() == "a new render"
    assert json.loads(report.read_text())["fresh"] is True
