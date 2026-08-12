from __future__ import annotations

import numpy as np
from scipy.io import wavfile

import audio_cli.pipeline as pipeline_module
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
    dry_run_report = pipeline.run(source, output=None, dry_run=True)
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
    assert (
        report["resolved_operations_sha256"]
        == dry_run_report["resolved_operations_sha256"]
    )


def test_pipeline_never_corrects_a_machine_region_that_overlaps_speech(
    tmp_path, monkeypatch
) -> None:
    sample_rate = 48_000
    duration = 4.0
    time = np.arange(round(sample_rate * duration)) / sample_rate
    audio = np.zeros((time.size, 2), dtype=np.float32)
    isolated_machine = (time >= 0.3) & (time < 0.8)
    adjacent_machine = (time >= 1.2) & (time < 1.5)
    speech = (time >= 1.5) & (time < 3.5)
    audio[isolated_machine] = (0.10 * np.sin(2 * np.pi * 900 * time[isolated_machine]))[
        :, None
    ]
    audio[adjacent_machine] = (
        0.10 * np.sin(2 * np.pi * 1100 * time[adjacent_machine])
    )[:, None]
    audio[speech] = (0.006 * np.sin(2 * np.pi * 180 * time[speech]))[:, None]
    source = tmp_path / "mixed-regions.wav"
    output = tmp_path / "mixed-regions-enhanced.wav"
    wavfile.write(source, sample_rate, audio)

    correction_calls: list[set[str]] = []
    apply_corrections = pipeline_module.apply_machine_region_corrections

    def record_corrections(samples, rate, analysis, corrections_db, fade_ms):
        correction_calls.append(set(corrections_db))
        return apply_corrections(samples, rate, analysis, corrections_db, fade_ms)

    monkeypatch.setattr(
        pipeline_module, "apply_machine_region_corrections", record_corrections
    )
    report = EnhancementPipeline(PROFILES["product-demo"], detector=FakeVad()).run(
        source, output=output, dry_run=False
    )

    source_stage = next(
        stage for stage in report["stages"] if stage["name"] == "source-balance"
    )
    abstained = set(source_stage["abstained_regions"])
    operated = {operation["region_id"] for operation in source_stage["operations"]}

    assert output.is_file()
    assert source_stage["status"] == "applied"
    assert abstained
    assert operated
    assert abstained.isdisjoint(operated)
    assert all(abstained.isdisjoint(call) for call in correction_calls)
    final_evaluations = {
        item["region_id"]: item["status"]
        for item in source_stage["final_region_evaluations"]
    }
    assert all(
        final_evaluations[region_id] == "abstained_overlap" for region_id in abstained
    )
