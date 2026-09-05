from __future__ import annotations

import json
from pathlib import Path

from audio_cli import cli
from audio_cli.transcribe.catalog import input_metadata, result_source


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


def test_result_source_resolves_relative_media_only_for_durable_output(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    metadata = input_metadata(Path("source.wav"), probe(2.0))
    assert metadata.path == "source.wav"
    assert result_source(metadata, 1.999) == {
        "path": str(tmp_path / "source.wav"),
        "duration_seconds": 1.999,
        "timebase": "seconds",
    }


def test_capabilities_cli_reads_only_the_metadata_probe(monkeypatch, capsys) -> None:
    calls = []

    def recording_probe(path: Path):
        calls.append(path)
        return probe(361.0)

    monkeypatch.setattr(cli, "probe_media", recording_probe)
    assert (
        cli.main(
            [
                "transcribe",
                "capabilities",
                "--stack",
                "qwen-1.7b",
                "--input",
                "sample.wav",
            ]
        )
        == 0
    )
    emitted = json.loads(capsys.readouterr().out)
    assert calls == [Path("sample.wav")]
    assert emitted["processing"]["unit_count"] == 3
    assert emitted["input"] == {
        "path": "sample.wav",
        "duration_seconds": 361.0,
        "container": "wav",
        "sample_rate_hz": 48000,
        "channels": 2,
    }


def test_capability_next_command_quotes_an_input_with_spaces(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())
    assert (
        cli.main(
            [
                "transcribe",
                "capabilities",
                "--stack",
                "qwen-1.7b",
                "--input",
                "two words.wav",
            ]
        )
        == 0
    )
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["next"].startswith("audio transcribe plan --input 'two words.wav'")


def test_plan_cli_resolves_without_loading_or_provisioning(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.setattr(cli, "probe_media", lambda path: probe())

    class ForbiddenProvisioner:
        def __init__(self, *args, **kwargs):
            raise AssertionError("planning must not construct a provisioner")

    monkeypatch.setattr(cli, "Provisioner", ForbiddenProvisioner)
    assert (
        cli.main(
            [
                "transcribe",
                "plan",
                "--input",
                "sample.wav",
                "--stack",
                "qwen-1.7b",
                "--want",
                "word_timestamps",
                "--language",
                "english",
            ]
        )
        == 0
    )
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["roles"]["asr"]["config"]["language"] == "English"
    assert emitted["roles"]["aligner"]["config"]["language_rule"].endswith(
        "the ASR --language hint is never forwarded"
    )
    assert "outcomes" not in emitted


def test_request_refusals_are_bare_and_precede_probe_and_registry(monkeypatch, capsys) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("request refusal reached external state")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    assert (
        cli.main(
            [
                "transcribe",
                "plan",
                "--input",
                "sample.wav",
                "--stack",
                "qwen-1.7b",
                "--want",
                "word_timing",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "capability_unknown"
    assert "error" not in error


def test_missing_stack_and_input_use_the_documented_refusals(capsys) -> None:
    assert (
        cli.main(
            [
                "transcribe",
                "plan",
                "--input",
                "sample.wav",
                "--want",
                "diarization",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["code"] == "stack_required"

    assert (
        cli.main(
            [
                "transcribe",
                "plan",
                "--stack",
                "qwen-1.7b",
                "--want",
                "diarization",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["code"] == "input_required"


def test_globally_unsupported_capability_never_checks_provisioning(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "load_registry",
        lambda: (_ for _ in ()).throw(
            AssertionError("unsupported request reached provisioning state")
        ),
    )
    monkeypatch.setattr(
        cli,
        "probe_media",
        lambda path: (_ for _ in ()).throw(
            AssertionError("unsupported request reached the media probe")
        ),
    )
    assert (
        cli.main(
            [
                "transcribe",
                "plan",
                "--input",
                "sample.wav",
                "--stack",
                "firered",
                "--want",
                "token_lid",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "code": "capability_unsupported",
        "capability": "token_lid",
        "allowed": [],
        "reason": "no_backend_declares",
        "fix": (
            "no stack or add-on satisfies this; code-switching support does not imply "
            "per-token language labels, and the nearest available output is lid, which "
            "labels a whole speech region"
        ),
    }
