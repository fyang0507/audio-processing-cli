from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from audio_cli.pipeline import EnhancementPipeline
from audio_cli.profiles import PROFILES, STAGE_ORDER
from audio_cli.vad import SpeechRegion


class FakeVad:
    model_version = "fake-vad-for-tests"

    def detect(self, samples, sample_rate, **kwargs):
        assert sample_rate == 16_000
        return [SpeechRegion(1.5, 3.5, 0.9, 1.0)]


def test_end_to_end_wav_render_reports_every_stage(tmp_path) -> None:
    sample_rate = 48_000
    duration = 4.0
    time = np.arange(round(sample_rate * duration)) / sample_rate
    audio = np.zeros((time.size, 2), dtype=np.float32)
    machine = (time >= 0.3) & (time < 1.1)
    speech = (time >= 1.5) & (time < 3.5)
    audio[machine] = (0.10 * np.sin(2 * np.pi * 900 * time[machine]))[:, None]
    audio[speech] = (0.006 * np.sin(2 * np.pi * 180 * time[speech]))[:, None]
    source = tmp_path / "source.wav"
    output = tmp_path / "enhanced.wav"
    wavfile.write(source, sample_rate, audio)

    pipeline = EnhancementPipeline(PROFILES["product-demo"], detector=FakeVad())
    report = pipeline.run(source, output=output, dry_run=False)

    assert output.is_file()
    assert report["rendered"] is True
    assert report["timeline_preserved"] is True
    assert report["final_peak_validation"]["status"] == "pass"
    assert [stage["name"] for stage in report["stages"]] == list(STAGE_ORDER)
    assert all(
        stage["status"] in {"applied", "no_op", "skipped", "abstained", "failed"}
        for stage in report["stages"]
    )
    after_lufs = report["measurements"]["after"]["program"]["input_i"]
    assert abs(after_lufs - PROFILES["product-demo"].target_lufs) <= 0.6
