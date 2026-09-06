"""An explicit denoiser selection must reach processing or refuse before work."""

import json
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli.pipeline import DenoiserModel


def test_missing_model_refuses_before_media_and_does_not_fetch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(
        cli, "probe_media", lambda *a: pytest.fail("model refusal must precede audio")
    )
    code = cli.main(
        ["enhance", "input.wav", "--profile", "product-demo", "--dry-run", "--denoiser", "rnnoise"]
    )
    out, err = capsys.readouterr()
    assert code != 0 and out == ""
    error = json.loads(err)["error"]
    assert error["code"] == "package_not_provisioned"
    assert error["fix"] == "audio packages pull rnnoise-voice"
    assert not (tmp_path / "empty-cache").exists()


def test_explicit_model_with_skipped_stage_refuses_before_verification(monkeypatch, capsys):
    monkeypatch.setattr(cli, "verified_artifact", lambda *a: pytest.fail("unused model"))
    code = cli.main(
        [
            "enhance",
            "input.wav",
            "--profile",
            "product-demo",
            "--dry-run",
            "--denoiser",
            "rnnoise",
            "--skip",
            "environment-denoise",
        ]
    )
    out, err = capsys.readouterr()
    assert code == 2 and out == ""
    assert "requires an enabled environment-denoise" in json.loads(err)["error"]["message"]


@pytest.mark.parametrize("selected", [False, True])
def test_only_explicit_model_choice_reaches_pipeline_with_verified_identity(
    selected, monkeypatch, capsys
):
    provenance = {"package": "rnnoise-voice", "sha256": "verified-local-digest", "bytes": 123}
    model_path = Path("/managed/model.rnnn")
    resolutions = []
    pipelines = []

    def verify(package):
        resolutions.append(package)
        return model_path, provenance

    class Pipeline:
        def __init__(self, profile, **kwargs):
            pipelines.append(kwargs)

        def run(self, *args, **kwargs):
            return {"rendered": False}

    monkeypatch.setattr(cli, "verified_artifact", verify)
    monkeypatch.setattr(cli, "probe_media", lambda *a: {})
    monkeypatch.setattr(cli, "media_summary", lambda *a: {"duration_seconds": 2})
    monkeypatch.setattr(cli, "EnhancementPipeline", Pipeline)
    args = ["enhance", "input.wav", "--profile", "product-demo", "--dry-run"]
    if selected:
        args += ["--denoiser", "rnnoise"]
    assert cli.main(args) == 0
    assert resolutions == (["rnnoise-voice"] if selected else [])
    assert pipelines[0]["denoiser_model"] == (
        DenoiserModel(model_path, provenance["sha256"], provenance) if selected else None
    )
    assert json.loads(capsys.readouterr().out) == {"rendered": False}
