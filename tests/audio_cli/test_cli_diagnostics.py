"""Public command wiring keeps diagnostic output separate from result JSON."""

import json
import sys
from types import SimpleNamespace

from audio_cli import cli


def test_requested_log_directory_refuses_before_media_probe(tmp_path, monkeypatch, capsys):
    source = tmp_path / "original.wav"
    source.write_bytes(b"source is never decoded")
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("preserve")

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid log destination must refuse before media work")

    monkeypatch.setattr(cli, "probe_media", forbidden)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--stack",
                "firered",
                "--input",
                str(source),
                "--log-dir",
                str(blocked),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    refusal = json.loads(captured.err)
    assert refusal["code"] == "log_directory_invalid"
    assert refusal["field"] == "--log-dir"
    assert blocked.read_text() == "preserve"


def test_cli_passes_retained_log_root_to_transport(tmp_path, monkeypatch, capsys):
    source = tmp_path / "original.wav"
    source.write_bytes(b"source is never decoded")
    logs = tmp_path / "logs"
    monkeypatch.setattr(cli, "probe_media", lambda path: {})
    monkeypatch.setattr(cli.transcribe_catalog, "input_metadata", lambda *args: object())

    def run(request, metadata, **kwargs):
        transport = kwargs["transport"]
        assert transport.runner.log_directory.parent == logs
        result = transport.runner.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'raw-out\\x00'); "
                "sys.stderr.buffer.write(b'raw-err\\xff')",
            ],
            stage="diagnostic-test",
        )
        assert result.returncode == 0
        return SimpleNamespace(payload={"test_result": "unchanged"})

    monkeypatch.setattr(cli.transcribe_orchestrator, "run", run)
    assert (
        cli.main(
            [
                "transcribe",
                "run",
                "--stack",
                "firered",
                "--input",
                str(source),
                "--log-dir",
                str(logs),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"test_result": "unchanged"}
    assert str(logs) in captured.err
    assert [path.read_bytes() for path in logs.rglob("stdout.log")] == [b"raw-out\x00"]
    assert [path.read_bytes() for path in logs.rglob("stderr.log")] == [b"raw-err\xff"]


def test_enhance_cli_progress_stays_on_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "probe_media", lambda path: {})
    monkeypatch.setattr(cli, "media_summary", lambda *args: {"duration_seconds": 1})
    monkeypatch.setattr(cli, "SileroOnnxVad", lambda *args: object())

    class Pipeline:
        def __init__(self, profile, **kwargs):
            self.progress = kwargs["progress"]

        def run(self, *args, **kwargs):
            self.progress("dry-run", "started")
            self.progress("dry-run", "finished")
            return {"test_result": "unchanged"}

    monkeypatch.setattr(cli, "EnhancementPipeline", Pipeline)
    assert (
        cli.main(
            [
                "enhance",
                str(tmp_path / "original.wav"),
                "--profile",
                "product-demo",
                "--dry-run",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"test_result": "unchanged"}
    assert "enhance: dry-run started" in captured.err
    assert "enhance: dry-run finished; elapsed " in captured.err
    assert "%" not in captured.err
