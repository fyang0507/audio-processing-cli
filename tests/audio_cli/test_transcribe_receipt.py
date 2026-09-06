"""Run the real publication path with controlled stages and opt-in stdout receipts."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli.transcribe.execution import build_receipt, publication
from audio_cli.transcribe.transport import StageFailure
from tests.audio_cli.test_transcribe_cli import probe
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    FakeTransport,
    registry,
    request,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)
from tests.docs.shipped_command_test_support import documented_block


@pytest.mark.parametrize("partial", [False, True])
def test_receipt_preserves_saved_bytes_and_partial_exit(tmp_path, monkeypatch, capsys, partial):
    resolved, _, _ = request(tmp_path)
    output = tmp_path / "result.json"
    saved = tmp_path / "result.partial.json" if partial else output
    entries = registry(tmp_path)
    monkeypatch.setattr(cli, "probe_media", lambda path: probe(361.0))
    run = cli.transcribe_orchestrator.run
    monkeypatch.setattr(
        cli.transcribe_orchestrator,
        "run",
        lambda *args, **kwargs: run(
            *args, **kwargs, registry=entries, transport=FakeTransport(partial=partial)
        ),
    )
    args = [
        "transcribe",
        "run",
        "--input",
        str(resolved.input_path),
        "--stack",
        "qwen-0.6b",
        "-o",
        str(output),
    ]
    expected_exit = 4 if partial else 0
    assert cli.main(args) == expected_exit
    default = capsys.readouterr()
    original = saved.read_bytes()
    payload = json.loads(original)
    if partial:
        assert default.out == ""
        assert json.loads(default.err)["code"] == "run_incomplete"
        assert not output.exists()
    else:
        assert json.loads(default.out) == payload
        assert default.err == ""

    assert cli.main([*args, "--receipt", "--format", "json", "--force"]) == expected_exit
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert saved.read_bytes() == original
    assert resolved.input_path.read_bytes() == b"source"
    if partial:
        refusal = json.loads(captured.err)
        default_refusal = json.loads(default.err)
        assert {k: v for k, v in refusal.items() if k != "fix"} == {
            k: v for k, v in default_refusal.items() if k != "fix"
        }
        retry = cli._parser().parse_args(shlex.split(refusal["fix"])[1:])
        assert retry.receipt is True
    else:
        assert captured.err == default.err
    assert receipt == {
        "output": str(saved),
        "source": payload["source"],
        "stack": "qwen-0.6b",
        "complete": not partial,
        "counts": {"segments": 1 if partial else 3, "abstentions": 0},
        **({"coverage": payload["coverage"]} if partial else {}),
    }
    assert "segments" not in receipt
    if not partial:
        documented = documented_block(
            "For example, a complete floors-only Qwen run",
            document=Path(__file__).resolve().parents[2] / "docs/TRANSCRIBE_CONTRACT.md",
        )
        documented["output"] = str(output)
        documented["source"]["path"] = str(resolved.input_path.resolve())
        assert receipt == documented


@pytest.mark.parametrize("options", [[], ["--format", "txt"], ["-o", "result", "--format", "md"]])
def test_receipt_invalid_combinations_refuse_before_resolution(monkeypatch, capsys, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid receipt reached request, media, packages, or model work")

    monkeypatch.setattr(cli.transcribe_planner, "resolve_request", forbidden)
    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    monkeypatch.setattr(cli.transcribe_orchestrator, "run", forbidden)
    assert cli.main(["transcribe", "run", "--receipt", *options]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == "receipt_options_invalid"
    if not options:
        assert json.loads(captured.err) == documented_block(
            "`--receipt` without `--output` in default JSON format produces:",
            document=Path(__file__).resolve().parents[2] / "docs/TRANSCRIBE_CONTRACT.md",
        )


def test_receipt_preserves_absent_and_empty_collections():
    payload = {
        "source": {"path": "source.wav", "timebase": "seconds"},
        "provenance": {"stack": "qwen-0.6b"},
        "complete": True,
        "segments": [{"text": "No supplied words"}],
        "abstentions": [],
    }
    receipt = build_receipt(payload, "result.json")
    assert receipt["counts"] == {"segments": 1, "abstentions": 0}
    assert "duration_basis" not in receipt["source"]
    assert "range" not in receipt
    payload["turns"] = []
    payload["segments"].append({"words": [{"text": "A"}, {"text": "B"}]})
    assert build_receipt(payload, "result.json")["counts"] == {
        "segments": 2,
        "abstentions": 0,
        "turns": 0,
        "words": 2,
    }


def test_refused_publication_does_not_emit_receipt(tmp_path, monkeypatch, capsys):
    output = tmp_path / "result.json"
    output.write_bytes(b"keep")
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--stack",
                "qwen-0.6b",
                "--input",
                "original.wav",
                "--receipt",
                "-o",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == "output_exists"
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("failure", ["backend", "publication"])
def test_failure_never_reports_an_unpublished_receipt(tmp_path, monkeypatch, capsys, failure):
    resolved, _, _ = request(tmp_path)
    entries = registry(tmp_path)
    output = tmp_path / "result.json"
    transport = FakeTransport()

    def fail(*args, **kwargs):
        if failure == "backend":
            raise StageFailure("asr", "qwen3-asr-0.6b-8bit", "synthetic failure")
        raise OSError("synthetic failure")

    if failure == "backend":
        monkeypatch.setattr(transport, "qwen", fail)
    else:
        monkeypatch.setattr(publication, "_write_result_file", fail)
    monkeypatch.setattr(cli, "probe_media", lambda path: probe(361.0))
    run = cli.transcribe_orchestrator.run
    monkeypatch.setattr(
        cli.transcribe_orchestrator,
        "run",
        lambda *args, **kwargs: run(
            *args,
            **kwargs,
            registry=entries,
            transport=transport,
        ),
    )
    assert cli.main(
        [
            "transcribe",
            "run",
            "--input",
            str(resolved.input_path),
            "--stack",
            "qwen-0.6b",
            "-o",
            str(output),
            "--receipt",
        ]
    ) == (1 if failure == "backend" else 2)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == (
        "backend_failed" if failure == "backend" else "output_path_invalid"
    )
    assert not output.exists()


def test_receipt_carries_recorded_range_without_inventing_coverage():
    scope = {"requested": [0, 12], "selected_unit_scope": [0.98, 11.85]}
    payload = {
        "source": {"path": "original.wav", "duration_seconds": 27.75},
        "provenance": {"stack": "firered", "plan": {"execution": {"range": scope}}},
        "complete": True,
        "segments": [],
        "abstentions": [],
    }
    receipt = build_receipt(payload, "range.json")
    assert receipt["range"] == scope
    assert receipt["complete"] is True
    assert "coverage" not in receipt


def test_existing_output_retry_preserves_invocation_options(tmp_path, monkeypatch, capsys):
    resolved, _, _ = request(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    logs = tmp_path / "logs with $ and spaces"

    def forbidden(*args, **kwargs):
        raise AssertionError("output refusal must precede media and log creation")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--stack",
                "qwen-0.6b",
                "--input",
                str(resolved.input_path),
                "-o",
                str(output),
                "--receipt",
                "--log-dir",
                str(logs),
                "--range",
                "0:12",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    refusal = json.loads(captured.err)
    assert refusal["code"] == "output_exists"
    retry = cli._parser().parse_args(shlex.split(refusal["fix"])[1:])
    assert retry.receipt and retry.force
    assert retry.log_dir == logs
    assert retry.run_range == "0:12"
    assert retry.output == output
    assert output.read_text() == "preserve"
    assert not logs.exists()


def test_partial_resume_preserves_log_destination(tmp_path, monkeypatch, capsys):
    resolved, _, _ = request(tmp_path)
    entries = registry(tmp_path)
    output = tmp_path / "result.json"
    logs = tmp_path / "logs with spaces"
    run = cli.transcribe_orchestrator.run
    monkeypatch.setattr(cli, "probe_media", lambda path: probe(361.0))

    def controlled_run(*args, **kwargs):
        kwargs["transport"] = FakeTransport(partial=True)
        return run(*args, **kwargs, registry=entries)

    monkeypatch.setattr(cli.transcribe_orchestrator, "run", controlled_run)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--stack",
                "qwen-0.6b",
                "--input",
                str(resolved.input_path),
                "-o",
                str(output),
                "--receipt",
                "--log-dir",
                str(logs),
            ]
        )
        == 4
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["complete"] is False
    retry = cli._parser().parse_args(shlex.split(json.loads(captured.err)["fix"])[1:])
    assert retry.receipt is True
    assert retry.log_dir == logs
    assert retry.output == tmp_path / "result.rest.json"
