"""A public run produces reusable JSON; readable delivery consumes the saved result."""

from __future__ import annotations

import json

import pytest

from audio_cli import cli
from audio_cli.transcribe import orchestrator, refusals
from tests.audio_cli.test_transcribe_cli import probe
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    FakeTransport,
    registry,
    request,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)


@pytest.fixture
def controlled_run(tmp_path, monkeypatch):
    resolved, _, _ = request(tmp_path)
    entries = registry(tmp_path)
    calls = []
    run = orchestrator.run

    class CountingTransport(FakeTransport):
        def qwen(self, **kwargs):
            calls.append("asr")
            return super().qwen(**kwargs)

    monkeypatch.setattr(cli, "probe_media", lambda path: probe(361.0))
    monkeypatch.setattr(
        orchestrator,
        "run",
        lambda *args, **kwargs: run(
            *args, **kwargs, registry=entries, transport=CountingTransport()
        ),
    )
    arguments = [
        "transcribe",
        "run",
        "--input",
        str(resolved.input_path),
        "--stack",
        "qwen-0.6b",
    ]
    return resolved.input_path, arguments, calls


@pytest.mark.parametrize("format_arguments", [[], ["--format", "json"]])
def test_public_run_stdout_json_creates_no_automatic_result(
    tmp_path, capsys, controlled_run, format_arguments
):
    source, arguments, calls = controlled_run
    existing = set(tmp_path.rglob("*"))

    assert cli.main([*arguments, *format_arguments]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["complete"] is True
    assert payload["source"]["path"] == str(source)
    assert payload["segments"] == [
        {"segment_id": f"seg_{index}", "text": "Hello."} for index in range(3)
    ]
    assert payload["provenance"]["outcomes"] == {}
    assert calls == ["asr"]
    assert set(tmp_path.rglob("*")) == existing
    assert source.read_bytes() == b"source"


def test_one_saved_run_supplies_readable_exports_and_refuses_missing_timing(
    tmp_path, monkeypatch, capsys, controlled_run
):
    source, arguments, calls = controlled_run
    canonical = tmp_path / "result.json"
    assert cli.main([*arguments, "--output", str(canonical)]) == 0
    captured = capsys.readouterr()
    saved = canonical.read_bytes()
    assert json.loads(captured.out) == json.loads(saved)
    assert captured.err == ""
    assert calls == ["asr"]

    def forbidden(*args, **kwargs):
        raise AssertionError("export reached planning, media probing, provisioning, or inference")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    monkeypatch.setattr(cli.transcribe_planner, "resolve_request", forbidden)
    monkeypatch.setattr(orchestrator, "run", forbidden)
    for output_format, expected in (
        ("txt", "Hello.\nHello.\nHello.\n"),
        ("md", "# Transcript\n\nHello.\n\nHello.\n\nHello.\n"),
    ):
        destination = tmp_path / f"review.{output_format}"
        assert (
            cli.main(
                [
                    "transcribe",
                    "export",
                    "--input",
                    str(canonical),
                    "--format",
                    output_format,
                    "--output",
                    str(destination),
                ]
            )
            == 0
        )
        captured = capsys.readouterr()
        assert captured.err == ""
        assert json.loads(captured.out)["output"] == str(destination)
        assert destination.read_text(encoding="utf-8") == expected
        assert canonical.read_bytes() == saved

    for output_format, options, refusal_code in (
        ("srt", [], "timing_required_for_format"),
        ("md", ["--timestamps"], "timing_required_for_timestamps"),
    ):
        destination = tmp_path / f"timed.{output_format}"
        assert (
            cli.main(
                [
                    "transcribe",
                    "export",
                    "--input",
                    str(canonical),
                    "--format",
                    output_format,
                    "--output",
                    str(destination),
                    *options,
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert json.loads(captured.err)["code"] == refusal_code
        assert not destination.exists()
        assert canonical.read_bytes() == saved
    assert calls == ["asr"]
    assert source.read_bytes() == b"source"


@pytest.mark.parametrize("output_format", ["txt", "md"])
@pytest.mark.parametrize("receipt", [False, True])
def test_retired_run_formats_refuse_before_resolution_or_filesystem_changes(
    tmp_path, monkeypatch, capsys, output_format, receipt
):
    def forbidden(*args, **kwargs):
        raise AssertionError("retired format reached a request, media, package, or model service")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    monkeypatch.setattr(cli, "StageTransport", forbidden)
    monkeypatch.setattr(cli.transcribe_planner, "resolve_request", forbidden)
    monkeypatch.setattr(orchestrator, "run", forbidden)
    existing = set(tmp_path.rglob("*"))
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                str(tmp_path / "missing.wav"),
                "--stack",
                "qwen-0.6b",
                "--format",
                output_format,
                "--output",
                str(tmp_path / f"result.{output_format}"),
                "--log-dir",
                str(tmp_path / "logs"),
                *(["--receipt"] if receipt else []),
            ]
        )
    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "run produces normalized JSON only" in captured.err
    assert f"audio transcribe export --input result.json --format {output_format}" in captured.err
    assert "--output result.json" in captured.err
    assert set(tmp_path.rglob("*")) == existing


@pytest.mark.parametrize("command", ["run", "export"])
def test_public_help_explains_reusable_json_and_offline_formatting(command, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["transcribe", command, "--help"])
    assert raised.value.code == 0
    captured = capsys.readouterr()
    help_text = " ".join(captured.out.split())
    assert captured.err == ""
    assert "normalized JSON" in help_text
    assert "audio transcribe export" in help_text
    assert "without rerunning models" in help_text
    if command == "run":
        assert "stdout" in help_text
        assert "--output" in help_text
        assert "--format" not in help_text
        assert "coverage, capability outcomes and abstentions" in help_text
    else:
        assert "multiple readable or subtitle exports" in help_text
        assert "Missing timing is never invented" in help_text


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_frozen_python_human_output_remains_available(tmp_path, output_format):
    resolved, metadata, _ = request(tmp_path)
    destination = tmp_path / f"legacy.{output_format}"

    product = orchestrator.run(
        resolved,
        metadata,
        output=destination,
        output_format=output_format,
        registry=registry(tmp_path),
        transport=FakeTransport(),
    )

    expected = "Hello.\nHello.\nHello.\n"
    if output_format == "md":
        expected = "# Transcript\n\n" + expected
    assert destination.read_text(encoding="utf-8") == expected
    assert orchestrator.render_human(product.payload, output_format) == expected
    assert product.payload["complete"] is True
    assert resolved.input_path.read_bytes() == b"source"


@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_legacy_python_collision_preserves_output_and_gives_python_recovery(
    tmp_path, output_format
):
    resolved, metadata, _ = request(tmp_path)
    destination = tmp_path / f"legacy.{output_format}"
    destination.write_bytes(b"keep the existing readable output\n")
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            output=destination,
            output_format=output_format,
            registry=registry(tmp_path),
            transport=transport,
        )

    assert raised.value.exit_code == 2
    refusal = raised.value.payload
    assert refusal["code"] == "output_exists"
    assert refusal["existing"] == str(destination)
    assert "original Python call" in refusal["fix"]
    assert "force=True" in refusal["fix"]
    assert "--format" not in refusal["fix"]
    assert not refusal["fix"].startswith("audio ")
    assert destination.read_bytes() == b"keep the existing readable output\n"
    assert resolved.input_path.read_bytes() == b"source"
    assert transport.units == []
