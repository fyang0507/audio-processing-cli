"""Caller-selected diagnostic storage with real child output and refusal before launch."""

import io
import subprocess
import sys
from pathlib import Path

import pytest

from audio_cli.media import diagnostics
from audio_cli.transcribe.transport import (
    StageFailure,
    StageTransport,
    SubprocessRunner,
    process_runner,
    service,
)


@pytest.mark.parametrize("code", [0, 1, 4, 137])
def test_durable_exact_bytes_distinct_runs_and_repeated_stages(tmp_path, code):
    root = tmp_path / "retained logs" / "nested"
    directories = []
    runs = []
    for _ in range(2):
        progress = io.StringIO()
        runner = SubprocessRunner(progress, log_root=root)
        runs.append(runner.log_directory)
        for _ in range(2):
            completed = runner.run(
                [
                    sys.executable,
                    "-c",
                    "import os; "
                    "os.write(1,b'raw\\r\\n\\xff\\0'+b'x'*70000); "
                    "os.write(2,b'warning\\r\\n\\xfe\\x1b[31m'); "
                    f"raise SystemExit({code})",
                ],
                stage="asr",
            )
            directory = completed.diagnostics_directory
            directories.append(directory)
            assert directory.parent == runner.log_directory
            assert directory.parent.parent == root
            assert "-asr-" in directory.name
            assert directory.stat().st_mode & 0o777 == 0o700
            assert runner.log_directory.stat().st_mode & 0o777 == 0o700
            assert completed.returncode == code
            assert (directory / "stdout.log").read_bytes() == b"raw\r\n\xff\0" + b"x" * 70000
            assert (directory / "stderr.log").read_bytes() == b"warning\r\n\xfe\x1b[31m"
            assert str(directory / "stderr.log") in progress.getvalue()
    assert len(set(runs)) == 2
    assert len(set(directories)) == 4
    assert all(path.is_dir() for path in directories)


@pytest.mark.parametrize("kind", ["file", "parent-file", "loop", "unwritable"])
@pytest.mark.parametrize("factory", [SubprocessRunner, StageTransport])
def test_bad_log_root_refuses_before_child_execution(tmp_path, monkeypatch, kind, factory):
    root = tmp_path / "logs"
    if kind == "file":
        root.write_bytes(b"canonical data")
    elif kind == "parent-file":
        root.write_bytes(b"canonical data")
        root = root / "child"
    elif kind == "loop":
        root.symlink_to(root)
    else:

        def denied(*args, **kwargs):
            raise PermissionError("requested log root is unwritable")

        monkeypatch.setattr(diagnostics.tempfile, "mkdtemp", denied)

    def no_launch(*args, **kwargs):
        pytest.fail("a child launched before requested log storage was validated")

    monkeypatch.setattr(process_runner.subprocess, "Popen", no_launch)
    with pytest.raises(OSError):
        factory(progress=io.StringIO(), log_root=root)
    if kind in {"file", "parent-file"}:
        assert (tmp_path / "logs").read_bytes() == b"canonical data"


def test_stage_storage_failure_does_not_fall_back_or_launch(tmp_path, monkeypatch):
    runner = SubprocessRunner(io.StringIO(), log_root=tmp_path / "logs")

    def denied(*args, **kwargs):
        raise PermissionError("cannot create stage logs")

    def no_launch(*args, **kwargs):
        pytest.fail("a child launched without diagnostic storage")

    monkeypatch.setattr(diagnostics.tempfile, "mkdtemp", denied)
    monkeypatch.setattr(process_runner.subprocess, "Popen", no_launch)
    completed = runner.run([sys.executable, "-c", "pass"], stage="asr")
    assert completed.returncode == 127
    assert "cannot create stage logs" in completed.stderr
    assert list(runner.log_directory.iterdir()) == []


@pytest.mark.parametrize(
    "code,payload,fails",
    [
        (0, '{"metrics":{},"segments":[]}', False),
        (1, '{"metrics":{},"error":{"message":"actual failure"}}', True),
        (0, '{"metrics":{},"metrics":{}}', True),
    ],
)
def test_transport_logs_survive_working_directory_cleanup(
    tmp_path, monkeypatch, code, payload, fails
):
    script = tmp_path / "provider.py"
    script.write_text(
        "import os,pathlib,sys\n"
        "os.write(1,b'loading\\r\\xff')\n"
        "os.write(2,b'warning\\r\\n\\xfe')\n"
        f"pathlib.Path(sys.argv[-1]).write_text({payload!r})\n"
        f"raise SystemExit({code})\n"
    )
    monkeypatch.setattr(service, "managed_environment_path", lambda _: (tmp_path, None))
    monkeypatch.setattr(service.paths, "env_python", lambda _: Path(sys.executable))
    monkeypatch.setattr(StageTransport, "_stage_script", lambda *args: script)
    transport = StageTransport(progress=io.StringIO(), log_root=tmp_path / "logs")
    work = tmp_path / "work"
    work.mkdir()

    def execute():
        return transport.align(
            model=tmp_path, audio=tmp_path / "audio", segments=[], directory=work
        )

    if fails:
        with pytest.raises(StageFailure):
            execute()
    else:
        assert execute().payload == {"metrics": {}, "segments": []}
    for path in work.iterdir():
        path.unlink()
    work.rmdir()
    (directory,) = transport.runner.log_directory.iterdir()
    assert "-aligner-" in directory.name
    assert (directory / "stdout.log").read_bytes() == b"loading\r\xff"
    assert (directory / "stderr.log").read_bytes() == b"warning\r\n\xfe"


def test_decode_uses_selected_root_and_stage_label(tmp_path):
    import numpy as np
    from scipy.io import wavfile

    source = tmp_path / "source.wav"
    wavfile.write(source, 16_000, np.zeros(1600, dtype=np.int16))
    transport = StageTransport(progress=io.StringIO(), log_root=tmp_path / "logs")
    assert transport.decode(source, tmp_path / "canonical.wav").role == "decode"
    (directory,) = transport.runner.log_directory.iterdir()
    assert "-decode-" in directory.name
    assert (directory / "stderr.log").is_file()


def test_custom_runner_and_log_root_are_not_silently_combined(tmp_path):
    with pytest.raises(ValueError, match="supplied runner"):
        StageTransport(runner=object(), log_root=tmp_path / "logs")


def test_durable_interruption_reaps_child_and_preserves_bytes(tmp_path, monkeypatch):
    runner = SubprocessRunner(io.StringIO(), log_root=tmp_path / "logs")
    real_popen = subprocess.Popen
    real_sleep = process_runner.time.sleep
    children = []
    handles = []
    interrupted = False

    def launch(command, **kwargs):
        handles.extend((kwargs["stdout"], kwargs["stderr"]))
        child = real_popen(command, **kwargs)
        children.append(child)
        return child

    def interrupt(seconds):
        nonlocal interrupted
        (log,) = runner.log_directory.glob("*/stderr.log")
        if not interrupted and log.read_bytes() == b"warning\r\xff":
            interrupted = True
            raise KeyboardInterrupt
        real_sleep(seconds)

    monkeypatch.setattr(process_runner.subprocess, "Popen", launch)
    monkeypatch.setattr(process_runner.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run(
            [sys.executable, "-c", "import os,time; os.write(2,b'warning\\r\\xff'); time.sleep(20)"]
        )
    assert children[0].poll() is not None
    assert all(handle.closed for handle in handles)
    (log,) = runner.log_directory.glob("*/stderr.log")
    assert log.read_bytes() == b"warning\r\xff"
