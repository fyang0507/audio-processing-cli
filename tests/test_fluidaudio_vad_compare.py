from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "model_tests"
    / "benchmark"
    / "run_fluidaudio_vad_compare.py"
)
SPEC = importlib.util.spec_from_file_location("fluidaudio_vad_compare", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_regions_from_probabilities_reuses_product_hysteresis_and_padding() -> None:
    regions = RUNNER.regions_from_probabilities(
        [0.1, 0.7, 0.7, 0.1, 0.1, 0.1],
        frame_samples=1600,
        total_samples=9600,
        threshold=0.5,
        exit_threshold=0.35,
        min_speech_ms=100,
        min_silence_ms=200,
        speech_pad_ms=100,
    )
    assert regions == [{"start_s": 0.0, "end_s": 0.4}]


def test_activity_score_reports_asymmetric_region_drift() -> None:
    agreement = RUNNER.activity_score(
        [(0.0, 1.0)],
        [(0.5, 1.5)],
        duration_s=2.0,
        frame_s=0.5,
        labels=("onnx", "fluid"),
    )
    assert agreement == {
        "frame_ms": 500.0,
        "total_frames": 4,
        "both_active_s": 0.5,
        "onnx_only_s": 0.5,
        "fluid_only_s": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
