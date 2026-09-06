"""Exercise real child bytes, live log access, exit failure, and interruption."""

import io
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from audio_cli.media import temporary_directory
from audio_cli.transcribe.transport import StageFailure, StageTransport, process_runner


@pytest.fixture(autouse=True)
def local_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        process_runner,
        "temporary_directory",
        lambda prefix, *, preserve: temporary_directory(str(tmp_path / prefix), preserve=preserve),
    )


@pytest.mark.parametrize("code", [0, 1, 4, 137])
def test_raw_streams_preserve_every_byte_and_host_progress_is_bounded(code):
    raw_stdout = b"loading\r25%\r" + b"x" * 70_000 + b"\xff\x00\n"
    raw_stderr = b"backend warning\r\n" + b"\xfe\x1b[31m" * 4000
    progress = io.StringIO()
    completed = process_runner.SubprocessRunner(progress, heartbeat_seconds=0.03).run(
        [
            sys.executable,
            "-c",
            "import os,time; os.write(1,b'loading\\r25%\\r'+b'x'*70000+b'\\xff\\x00\\n'); "
            "os.write(2,b'backend warning\\r\\n'+b'\\xfe\\x1b[31m'*4000); "
            f"time.sleep(0.12); raise SystemExit({code})",
        ]
    )
    directory = completed.diagnostics_directory
    assert completed.returncode == code
    assert (directory / "stdout.log").read_bytes() == raw_stdout
    assert (directory / "stderr.log").read_bytes() == raw_stderr
    assert directory.stat().st_mode & 0o777 == 0o700
    notices = progress.getvalue()
    assert "child running; elapsed" in notices
    assert f"child exited {code}" in notices
    assert "backend warning" not in notices
    assert "%" not in notices
    assert len(notices) < 2000


def test_paths_are_announced_before_child_launch_and_logs_are_live(tmp_path, monkeypatch):
    progress = io.StringIO()
    real_popen = subprocess.Popen

    def launch(command, **kwargs):
        notices = progress.getvalue()
        for stream in ("stdout", "stderr"):
            assert f"raw backend {stream}: {kwargs[stream].name}" in notices
            assert not kwargs[stream].closed
        return real_popen(command, **kwargs)

    monkeypatch.setattr(process_runner.subprocess, "Popen", launch)
    observed = []
    real_sleep = process_runner.time.sleep

    def inspect_live(seconds):
        log = next(tmp_path.glob("audio-transcribe-*/stderr.log"))
        if log.read_bytes() == b"live\n":
            observed.append(log)
        real_sleep(seconds)

    monkeypatch.setattr(process_runner.time, "sleep", inspect_live)
    completed = process_runner.SubprocessRunner(progress).run(
        [sys.executable, "-c", "import os,time; os.write(2,b'live\\n'); time.sleep(0.1)"]
    )
    assert completed.returncode == 0
    assert observed


def test_launch_failure_retains_diagnostics(tmp_path):
    progress = io.StringIO()
    completed = process_runner.SubprocessRunner(progress).run([str(tmp_path / "missing")])
    assert completed.returncode == 127
    assert (completed.diagnostics_directory / "stdout.log").read_bytes() == b""
    assert "missing" in (completed.diagnostics_directory / "stderr.log").read_text()
    assert "child exited 127" in progress.getvalue()


def test_keyboard_interrupt_reaps_child_and_closes_handles_with_bytes_retained(
    tmp_path, monkeypatch
):
    progress = io.StringIO()
    real_popen = subprocess.Popen
    real_sleep = process_runner.time.sleep
    children = []
    handles = []
    interrupted = False

    def launch(command, **kwargs):
        handles.extend([kwargs["stdout"], kwargs["stderr"]])
        child = real_popen(command, **kwargs)
        children.append(child)
        return child

    def interrupt_after_output(seconds):
        nonlocal interrupted
        log = next(tmp_path.glob("audio-transcribe-*/stderr.log"))
        if not interrupted and log.read_bytes() == b"before interrupt\r\n\xff":
            interrupted = True
            raise KeyboardInterrupt
        real_sleep(seconds)

    monkeypatch.setattr(process_runner.subprocess, "Popen", launch)
    monkeypatch.setattr(process_runner.time, "sleep", interrupt_after_output)
    with pytest.raises(KeyboardInterrupt):
        process_runner.SubprocessRunner(progress).run(
            [
                sys.executable,
                "-c",
                "import os,time; os.write(1,b'raw stdout'); "
                "os.write(2,b'before interrupt\\r\\n\\xff'); time.sleep(20)",
            ]
        )
    assert children[0].poll() is not None
    assert all(handle.closed for handle in handles)
    directory = next(tmp_path.glob("audio-transcribe-*"))
    assert (directory / "stdout.log").read_bytes() == b"raw stdout"
    assert (directory / "stderr.log").read_bytes() == b"before interrupt\r\n\xff"
    assert "interrupted; raw diagnostics retained" in progress.getvalue()


def test_signalled_child_retains_raw_output():
    completed = process_runner.SubprocessRunner(io.StringIO()).run(
        [
            sys.executable,
            "-c",
            "import os,signal; os.write(2,b'killed'); os.kill(os.getpid(),signal.SIGTERM)",
        ]
    )
    assert completed.returncode == -signal.SIGTERM
    assert (completed.diagnostics_directory / "stderr.log").read_bytes() == b"killed"


def test_stage_json_protocol_and_failure_detail_are_unchanged(tmp_path, monkeypatch):
    # A real child substitutes only the provider script/interpreter, not transport.
    from audio_cli.transcribe.transport import service

    script = tmp_path / "stage.py"
    script.write_text(
        "import os,pathlib,sys\n"
        "os.write(1,b'backend stdout\\r\\n')\n"
        "os.write(2,b'backend warning\\r\\n')\n"
        'pathlib.Path(sys.argv[-1]).write_text(\'{"metrics":{},"error":{"message":"actual failure"}}\')\n'
        "raise SystemExit(1)\n"
    )
    monkeypatch.setattr(service, "managed_environment_path", lambda _: (tmp_path, None))
    monkeypatch.setattr(service.paths, "env_python", lambda _: Path(sys.executable))
    monkeypatch.setattr(StageTransport, "_stage_script", lambda *args: script)
    progress = io.StringIO()
    with pytest.raises(StageFailure, match="actual failure") as caught:
        StageTransport(progress=progress).align(
            model=tmp_path / "model", audio=tmp_path / "audio.wav", segments=[], directory=tmp_path
        )
    assert caught.value.outcome.returncode == 1
    assert "backend warning\r\n" not in progress.getvalue()
    assert (
        next(tmp_path.glob("audio-transcribe-*/stderr.log")).read_bytes() == b"backend warning\r\n"
    )


def test_temporary_directory_default_cleanup_and_explicit_retention(tmp_path):
    with temporary_directory(str(tmp_path / "default-")) as cleaned:
        assert cleaned.is_dir()
    assert not cleaned.exists()
    with (
        pytest.raises(RuntimeError),
        temporary_directory(str(tmp_path / "keep-"), preserve=True) as kept,
    ):
        (kept / "evidence").write_bytes(b"raw\xff")
        raise RuntimeError("interrupted")
    assert (kept / "evidence").read_bytes() == b"raw\xff"
