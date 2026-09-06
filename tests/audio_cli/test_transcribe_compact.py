"""Compact planning is exactly the default decision payload minus the generated sample."""

from __future__ import annotations

import json

import pytest

from audio_cli import cli
from audio_cli.transcribe import plan as plan_module
from tests.audio_cli.test_transcribe_cli import probe


@pytest.mark.parametrize("stack", ["qwen-1.7b", "qwen-0.6b", "firered", "vibevoice"])
@pytest.mark.parametrize("provisioned", [False, True])
def test_compact_keeps_every_decision_without_generating_sample(
    monkeypatch, capsys, stack, provisioned
):
    entries = {}
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())
    monkeypatch.setattr(cli, "load_registry", lambda: {"packages": entries})
    args = ["transcribe", "plan", "--input", "original.wav", "--stack", stack]
    assert cli.main(args) == 0
    default = json.loads(capsys.readouterr().out)
    if provisioned:
        entries.update({item["package"]: {"state": "ready"} for item in default["packages"]})
        assert cli.main(args) == 0
        default = json.loads(capsys.readouterr().out)
    assert ("next" in default) is not provisioned
    assert "sample_output" in default

    def forbidden(**kwargs):
        raise AssertionError("compact mode generated sample_output")

    monkeypatch.setattr(plan_module, "build_sample_output", forbidden)
    assert cli.main([*args, "--compact"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        key: value for key, value in default.items() if key != "sample_output"
    }
