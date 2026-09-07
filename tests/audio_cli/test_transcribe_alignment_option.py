"""The public option validates before I/O and reaches planning/execution requests."""

import json
import shlex
from types import SimpleNamespace

import pytest

from audio_cli import cli
from audio_cli.command import Refusal
from tests.audio_cli.test_transcribe_cli import probe


@pytest.mark.parametrize("command", ["plan", "run"])
@pytest.mark.parametrize("value", ["not-a-number", "nan", "inf", "-inf", "0.5"])
def test_invalid_alignment_policy_refuses_before_media_and_registry(
    command, value, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid alignment policy reached I/O")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    assert (
        cli.main(
            [
                "transcribe",
                command,
                "--stack",
                "qwen-1.7b",
                "--input",
                "original.wav",
                "--want",
                "word_timestamps",
                f"--alignment-max-overrun-ms={value}",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["code"] == "alignment_max_overrun_invalid"
    assert payload["provided"] == value
    assert "error" not in payload


@pytest.mark.parametrize("command", ["plan", "run"])
@pytest.mark.parametrize(("stack", "wants"), [("firered", "word_timestamps"), ("qwen-1.7b", None)])
def test_inapplicable_alignment_policy_refuses_before_media(
    command, stack, wants, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        raise AssertionError("inapplicable alignment policy reached I/O")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    args = [
        "transcribe",
        command,
        "--stack",
        stack,
        "--input",
        "original.wav",
        "--alignment-max-overrun-ms",
        "150",
    ]
    if wants:
        args.extend(("--want", wants))
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == "alignment_option_not_applicable"


def test_plan_outputs_effective_alignment_limit(monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())
    monkeypatch.setattr(cli, "load_registry", lambda: {"packages": {}})
    assert (
        cli.main(
            [
                "transcribe",
                "plan",
                "--stack",
                "vibevoice",
                "--input",
                "original.wav",
                "--want",
                "word_timestamps",
                "--alignment-max-overrun-ms",
                "150.25",
                "--compact",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["roles"]["aligner"]["config"]["max_overrun_ms"] == 150.25


def test_run_passes_normalized_alignment_limit_to_orchestration(monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())

    def execute(request, metadata, **kwargs):
        assert request.alignment_max_overrun_ms == 150.25
        assert request.wants == ("word_timestamps",)
        return SimpleNamespace(payload={"test": "request reached execution"})

    monkeypatch.setattr(cli.transcribe_orchestrator, "run", execute)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--stack",
                "qwen-0.6b",
                "--input",
                "original.wav",
                "--want",
                "word_timestamps",
                "--alignment-max-overrun-ms",
                "150.25",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"test": "request reached execution"}


def test_output_collision_fix_retains_alignment_and_invocation_options(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "already exists.json"
    output.write_text("keep")

    def forbidden(*args, **kwargs):
        raise AssertionError("output collision reached media")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--stack",
                "qwen-1.7b",
                "--input",
                "original.wav",
                "--want",
                "word_timestamps",
                "--alignment-max-overrun-ms",
                "150.25",
                "-o",
                str(output),
                "--receipt",
                "--log-dir",
                str(tmp_path / "logs"),
            ]
        )
        == 2
    )
    fix = json.loads(capsys.readouterr().err)["fix"]
    parts = shlex.split(fix)
    parsed = cli._parser().parse_args(parts[1:])
    assert float(parsed.alignment_max_overrun_ms) == 150.25
    assert parts.count("--alignment-max-overrun-ms") == 1
    assert parsed.receipt and parsed.force
    assert parsed.log_dir == tmp_path / "logs"
    assert output.read_text() == "keep"


def test_generic_runnable_run_fix_keeps_explicit_alignment_policy(monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())

    def execute(*args, **kwargs):
        raise Refusal(
            {
                "code": "backend_failed",
                "fix": "audio transcribe run --input original.wav --stack qwen-1.7b --want word_timestamps",
            },
            exit_code=1,
        )

    monkeypatch.setattr(cli.transcribe_orchestrator, "run", execute)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--stack",
                "qwen-1.7b",
                "--input",
                "original.wav",
                "--want",
                "word_timestamps",
                "--alignment-max-overrun-ms",
                "150.25",
            ]
        )
        == 1
    )
    parsed = cli._parser().parse_args(shlex.split(json.loads(capsys.readouterr().err)["fix"])[1:])
    assert float(parsed.alignment_max_overrun_ms) == 150.25
