from __future__ import annotations

import json

from audio_cli import cli
from audio_cli.transcribe.orchestrator import firered as firered_execution
from audio_cli.transcribe.orchestrator import qwen as orchestrator_qwen


def probe(duration: float = 12.5) -> dict[str, object]:
    return {
        "primary_audio_stream": {
            "codec_type": "audio",
            "duration": str(duration),
            "sample_rate": "48000",
            "channels": 2,
        },
        "format": {"duration": str(duration), "format_name": "wav"},
    }


def test_run_missing_package_is_bare_exit_three_after_probe_but_before_transport(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())
    monkeypatch.setattr(
        orchestrator_qwen,
        "load_registry",
        lambda: {
            "packages": {},
            "environments": {},
        },
    )

    class ForbiddenTransport:
        def __init__(self, *args, **kwargs):
            raise AssertionError("exit 3 reached model transport")

    monkeypatch.setattr(orchestrator_qwen, "StageTransport", ForbiddenTransport)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                "sample.wav",
                "--stack",
                "qwen-0.6b",
            ]
        )
        == 3
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error == {
        "code": "packages_not_provisioned",
        "missing": [
            {
                "package": "qwen3-asr-0.6b-8bit",
                "kind": "weights",
                "bytes": 1010773761,
            }
        ],
        "total_known_download_bytes": 1010773761,
        "unsized_packages": [],
        "fix": "audio packages pull --stack qwen-0.6b",
    }


def test_run_resolves_stack_refusal_before_range_and_media(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "probe_media",
        lambda path: (_ for _ in ()).throw(AssertionError("request refusal reached probe")),
    )
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                "sample.wav",
                "--range",
                "bad",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["code"] == "stack_required"


def test_run_malformed_range_is_bare_and_precedes_media(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "probe_media",
        lambda path: (_ for _ in ()).throw(AssertionError("range refusal reached probe")),
    )
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                "sample.wav",
                "--stack",
                "qwen-0.6b",
                "--range",
                "not-a-range",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "range_invalid"
    assert error["field"] == "--range"
    assert error["provided"] == "not-a-range"
    assert "error" not in error


def test_run_refuses_existing_output_and_partial_before_media(
    tmp_path, monkeypatch, capsys
) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"source")
    output = tmp_path / "result.json"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "probe_media",
        lambda path: (_ for _ in ()).throw(AssertionError("collision reached probe")),
    )
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                str(source),
                "--stack",
                "qwen-0.6b",
                "-o",
                str(output),
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "output_exists"
    assert error["existing"] == str(output)
    assert error["fix"].endswith(f"-o {output} --force")
    assert "error" not in error
    assert output.read_text(encoding="utf-8") == "keep"

    output.unlink()
    partial = tmp_path / "result.partial.json"
    partial.write_text("keep partial", encoding="utf-8")
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                str(source),
                "--stack",
                "qwen-0.6b",
                "-o",
                str(output),
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "output_exists"
    assert error["existing"] == str(partial)
    assert error["fix"].endswith(f"-o {output} --force")
    assert partial.read_text(encoding="utf-8") == "keep partial"


def test_run_force_still_cannot_target_the_canonical_input(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        cli,
        "probe_media",
        lambda path: (_ for _ in ()).throw(AssertionError("source collision reached probe")),
    )
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                str(source),
                "--stack",
                "qwen-0.6b",
                "-o",
                str(source),
                "--force",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "output_is_canonical_input"
    assert error["resolved_target"] == str(source)
    assert not error["fix"].startswith("audio ")
    assert source.read_bytes() == b"source"


def test_run_reports_a_typed_refusal_for_an_output_symlink_loop(
    tmp_path, monkeypatch, capsys
) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"source")
    output = tmp_path / "loop.json"
    output.symlink_to(output.name)
    monkeypatch.setattr(
        cli,
        "probe_media",
        lambda path: (_ for _ in ()).throw(AssertionError("invalid output reached probe")),
    )

    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                str(source),
                "--stack",
                "qwen-0.6b",
                "-o",
                str(output),
                "--force",
            ]
        )
        == 2
    )

    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "output_path_invalid"
    assert error["target"] == str(output)
    assert output.is_symlink()


def test_firered_run_reaches_preflight_but_not_transport_when_unprovisioned(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())
    monkeypatch.setattr(
        firered_execution,
        "load_registry",
        lambda: {
            "packages": {},
            "environments": {},
        },
    )

    class ForbiddenTransport:
        def __init__(self, *args, **kwargs):
            raise AssertionError("exit 3 reached native model transport")

    monkeypatch.setattr(firered_execution, "StageTransport", ForbiddenTransport)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                "sample.wav",
                "--stack",
                "firered",
            ]
        )
        == 3
    )
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "packages_not_provisioned"
    assert [item["package"] for item in error["missing"]] == ["firered-asr2s"]


def test_firered_run_parses_range_before_media(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "probe_media",
        lambda path: (_ for _ in ()).throw(AssertionError("stack refusal reached probe")),
    )
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--input",
                "sample.wav",
                "--stack",
                "firered",
                "--range",
                "not-a-range",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["code"] == "range_invalid"
