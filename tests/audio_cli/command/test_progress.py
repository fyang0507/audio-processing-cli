"""Real timer liveness and cleanup independent of feature execution."""

import io
import threading

import pytest

from audio_cli.command import ProgressReporter


class HeartbeatStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.heartbeat = threading.Event()

    def write(self, message):
        count = super().write(message)
        if "running; elapsed" in message:
            self.heartbeat.set()
        return count


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_heartbeat_while_caller_is_blocked_and_timer_is_joined(failure, capsys):
    stream = HeartbeatStream()
    reporter = ProgressReporter("enhance", stream=stream, heartbeat_seconds=0.01)

    def work():
        with reporter:
            reporter("denoise", "started")
            assert stream.heartbeat.wait(2), "blocked caller received no heartbeat"
            if failure:
                reporter("denoise", "failed")
                raise failure("interrupted work")
            reporter("denoise", "finished")

    if failure:
        with pytest.raises(failure):
            work()
    else:
        work()
    assert reporter._thread is not None and not reporter._thread.is_alive()
    assert not reporter._active
    reporter.close()  # Explicit cleanup remains safe after context exit.
    notices = stream.getvalue()
    assert "enhance: denoise started\n" in notices
    assert f"denoise {'failed' if failure else 'finished'}; elapsed" in notices
    assert "%" not in notices and "ETA" not in notices
    assert capsys.readouterr().out == ""


def test_default_stream_is_stderr_and_nested_events_resume_outer_heartbeat(capsys):
    with ProgressReporter("enhance") as reporter:
        reporter("dry-run", "started")
        reporter("decode", "started")
        reporter("decode", "finished")
        assert [stage for stage, _ in reporter._active] == ["dry-run"]
        reporter("dry-run", "finished")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "enhance: dry-run finished" in captured.err


def test_broken_presentation_stream_does_not_abort_work():
    stream = io.StringIO()
    with ProgressReporter("enhance", stream=stream) as reporter:
        reporter("decode", "started")
        stream.close()
        reporter("decode", "finished")


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_heartbeat_is_refused(interval):
    with pytest.raises(ValueError, match="positive"):
        ProgressReporter("enhance", heartbeat_seconds=interval)
