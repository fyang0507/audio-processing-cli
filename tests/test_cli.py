import json

from audio_cli.cli import main


def test_list_stages_requires_no_input(capsys) -> None:
    result = main(["enhance", "--profile", "product-demo", "--list-stages"])
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["profile_version"] == "3"
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
