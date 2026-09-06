"""Enhancement events preserve real reports, canonical bytes and rendered samples."""

import io
import threading

import numpy as np
import pytest
from scipy.io import wavfile

from audio_cli.command import ProgressReporter
from audio_cli.pipeline import EnhancementPipeline, preparation, publication, stages
from audio_cli.profiles import PROFILES
from audio_cli.vad_contract import SpeechRegion


class FixedVad:
    model_version = "fixed-progress-test-vad"

    def detect(self, *args, **kwargs):
        return [SpeechRegion(1.5, 3.5, 0.9, 1.0)]


@pytest.fixture
def source(tmp_path):
    time = np.arange(48_000 * 4) / 48_000
    samples = np.zeros((len(time), 2), dtype=np.float32)
    speech = (time >= 1.5) & (time < 3.5)
    samples[speech] = (0.02 * np.sin(2 * np.pi * 180 * time[speech]))[:, None]
    path = tmp_path / "source.wav"
    wavfile.write(path, 48_000, samples)
    return path


@pytest.mark.parametrize("dry_run", [True, False])
def test_progress_preserves_reports_signals_and_source(source, tmp_path, dry_run, capsys):
    output = tmp_path / "output.wav"
    canonical = source.read_bytes()
    pipeline = EnhancementPipeline(PROFILES["product-demo"], detector=FixedVad())
    expected = pipeline.run(source, output=output, dry_run=dry_run)
    rendered = output.read_bytes() if not dry_run else None
    events = []
    pipeline.progress = lambda stage, status: events.append((stage, status))
    actual = pipeline.run(source, output=output, dry_run=dry_run)
    assert actual == expected
    assert source.read_bytes() == canonical
    if dry_run:
        assert not output.exists()
    else:
        assert output.read_bytes() == rendered
    outer = "dry-run" if dry_run else "enhancement"
    assert events[0] == (outer, "started")
    assert events[-1] == (outer, "finished")
    work = [
        "preparation",
        "decode",
        "inspection",
        "channel-balance",
        "environment-denoise",
        "frequency-adjustments",
        "voice-enhance",
        "source-balance",
        "fullband-adjustments",
        "normalization",
    ]
    if not dry_run:
        work += [
            "encode",
            "encoded-output-inspection",
            "publication",
            "published-output-inspection",
        ]
    assert [stage for stage, status in events[1:-1] if status == "started"] == work
    assert [stage for stage, status in events[1:-1] if status == "finished"] == work
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_blocked_decode_heartbeats_and_exception_closes_all_scopes(source, monkeypatch, error):
    heartbeat = threading.Event()

    class Stream(io.StringIO):
        def write(self, message):
            result = super().write(message)
            if "decode running; elapsed" in message:
                heartbeat.set()
            return result

    def blocked_decode(*args):
        assert heartbeat.wait(2), "no heartbeat during blocked decode"
        raise error("decode interrupted")

    monkeypatch.setattr(preparation, "decode_audio", blocked_decode)
    stream = Stream()
    reporter = ProgressReporter("enhance", stream=stream, heartbeat_seconds=0.01)
    with pytest.raises(error, match="decode interrupted"), reporter:
        EnhancementPipeline(PROFILES["product-demo"], detector=FixedVad(), progress=reporter).run(
            source, output=None, dry_run=True
        )
    assert not reporter._thread.is_alive()
    assert not reporter._active
    assert "decode failed; elapsed" in stream.getvalue()
    assert "dry-run failed; elapsed" in stream.getvalue()
    assert "decode finished" not in stream.getvalue()


@pytest.mark.parametrize(
    "owner,name,phase",
    [
        (stages, "apply_environment_cleanup", "environment-denoise"),
        (publication, "encode_output", "encode"),
        (publication.os, "replace", "publication"),
    ],
)
def test_processing_failure_closes_phase_without_publishing(
    source, tmp_path, monkeypatch, owner, name, phase
):
    def fail(*args, **kwargs):
        raise RuntimeError("work failed")

    monkeypatch.setattr(owner, name, fail)
    events = []
    output = tmp_path / "output.wav"
    with pytest.raises(RuntimeError, match="work failed"):
        EnhancementPipeline(
            PROFILES["product-demo"],
            detector=FixedVad(),
            progress=lambda stage, status: events.append((stage, status)),
        ).run(source, output=output, dry_run=False)
    assert (phase, "failed") in events
    assert (phase, "finished") not in events
    assert events[-1] == ("enhancement", "failed")
    assert not output.exists()
