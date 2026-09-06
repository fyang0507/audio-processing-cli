"""Exercise discovery and remediation through the actual CLI dispatch surface."""

from __future__ import annotations

import json
import shlex

import pytest

from audio_cli import cli
from audio_cli.command import packages_pull_command
from audio_cli.transcribe import refusals
from tests.audio_cli.test_transcribe_cli import probe


@pytest.mark.parametrize("command", ["capabilities", "plan", "run"])
@pytest.mark.parametrize("missing", ["--input", "--stack"])
def test_missing_request_fields_do_not_invent_a_command(command, missing, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete request reached external state")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    supplied = ["--stack", "vibevoice"] if missing == "--input" else ["--input", "actual.wav"]
    assert cli.main(["transcribe", command, *supplied]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    payload = json.loads(output.err)
    assert payload["field"] == missing
    assert "original command" in payload["fix"]
    assert not payload["fix"].startswith("audio ")
    assert "meeting.m4a" not in payload["fix"]


@pytest.mark.parametrize("command", ["capabilities", "plan", "run"])
def test_transcription_help_identifies_required_semantic_fields(command, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["transcribe", command, "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "Required: explicit transcription stack" in help_text
    assert "Required: original media path" in help_text


def test_catalog_next_is_executable_with_literal_unusual_input(monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())
    monkeypatch.setattr(cli, "load_registry", dict)
    source = "-two ' words $(echo wrong).wav"
    assert (
        cli.main(["transcribe", "capabilities", "--stack", "qwen-1.7b", f"--input={source}"]) == 0
    )
    catalog = json.loads(capsys.readouterr().out)
    args = shlex.split(catalog["next"])
    assert args[0] == "audio"
    assert args[args.index("--input") + 1] == f"./{source}"
    assert "<capabilities>" not in catalog["next"]
    assert cli.main(args[1:]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["next"] == "audio packages pull qwen3-asr-1.7b-8bit"


def test_plan_next_downloads_only_the_missing_resolved_package(monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())
    registry = {"packages": {"qwen3-asr-1.7b-8bit": {"state": "ready"}}}
    monkeypatch.setattr(cli, "load_registry", lambda: registry)
    invocation = [
        "transcribe",
        "plan",
        "--input",
        "sample.wav",
        "--stack",
        "qwen-1.7b",
        "--want",
        "word_timestamps",
    ]
    assert cli.main(invocation) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["next"] == "audio packages pull qwen3-forcedaligner"
    assert "next" not in plan["sample_output"]["provenance"]["plan"]
    calls = []

    class RecordingProvisioner:
        def pull(self, selection, *, repair, stack):
            calls.append(([item.id for item in selection], repair, stack))
            return {"pulled": []}

    monkeypatch.setattr(cli, "Provisioner", RecordingProvisioner)
    assert cli.main(shlex.split(plan["next"])[1:]) == 0
    capsys.readouterr()
    assert calls == [(["qwen3-forcedaligner"], False, None)]
    registry["packages"]["qwen3-forcedaligner"] = {"state": "ready"}
    assert cli.main(invocation) == 0
    assert "next" not in json.loads(capsys.readouterr().out)


def test_missing_package_refusal_does_not_expand_to_the_stack():
    missing = [{"package": "qwen3-forcedaligner", "kind": "weights", "bytes": 1276475979}]
    refusal = refusals.packages_not_provisioned(missing, 1276475979, [])
    assert refusal.payload["fix"] == "audio packages pull qwen3-forcedaligner"
    assert refusal.exit_code == 3


def test_package_command_quotes_names_and_preserves_repair():
    assert shlex.split(packages_pull_command(["-option-like", "two words"], repair=True)) == [
        "audio",
        "packages",
        "pull",
        "--repair",
        "--",
        "-option-like",
        "two words",
    ]
    with pytest.raises(ValueError):
        packages_pull_command([])


def test_run_provenance_excludes_discovery_actions_even_for_unprovisioned_plan():
    from pathlib import Path

    from audio_cli.transcribe.catalog import input_metadata
    from audio_cli.transcribe.execution.runtime import _core_plan
    from audio_cli.transcribe.plan import serialize_plan
    from audio_cli.transcribe.planner import build_plan, resolve_request

    request = resolve_request(stack_id="vibevoice", input_path=Path("original.wav"), wants=None)
    plan = build_plan(request, input_metadata(Path("original.wav"), probe()))
    discovery = serialize_plan(plan)
    assert discovery["next"] == "audio packages pull vibevoice-asr-7b"
    assert _core_plan(plan) == discovery["sample_output"]["provenance"]["plan"]


@pytest.mark.parametrize(
    ("stack", "options", "code", "correction"),
    [
        ("qwen-1.7b", ["--want", "word_timing"], "capability_unknown", "'word_timestamps'"),
        (
            "qwen-1.7b",
            ["--want", "vad,word_timestamps", "--vad", "silro-vad"],
            "option_value_unsupported",
            "'vad,word_timestamps'",
        ),
        ("qwen-1.7b", ["--diarizer", "fluidauio"], "option_value_unsupported", "'diarization'"),
        ("qwen-1.7b", ["--language", "EN"], "option_value_unsupported", "'English'"),
        (
            "vibevoice",
            ["--language", "Chinese"],
            "option_unsupported_on_stack",
            "remove --language",
        ),
        (
            "vibevoice",
            ["--want", "diarization", "--diarizer", "fluidaudio"],
            "pin_conflicts_with_native_capability",
            "remove --diarizer",
        ),
        (
            "qwen-1.7b",
            ["--want", "segment_timestamps"],
            "capability_unsatisfiable_on_stack",
            "use --stack",
        ),
        ("qwen-1.7b", ["--range", "bad"], "range_invalid", "keep the intended source interval"),
    ],
)
def test_run_field_corrections_preserve_original_operation_options(
    stack, options, code, correction, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid request reached media or provisioning")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    original = [
        "transcribe",
        "run",
        "--stack",
        stack,
        "--input",
        "original.wav",
        "--range",
        "10:20",
        "--format",
        "txt",
        "-o",
        "review.txt",
        "--force",
    ]
    if stack.startswith("qwen"):
        original.extend(["--language", "Chinese"])
    assert cli.main([*original, *options]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["code"] == code
    assert correction in error["fix"]
    assert error["fix"].endswith("repeat the original command, preserving every other argument")
    assert not error["fix"].startswith("audio ")
    assert "transcribe plan" not in error["fix"]
