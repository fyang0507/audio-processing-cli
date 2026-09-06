"""Measure a generated clean/noise fixture through the provisioned RNNoise guide.

Developer benchmark only: there is no user-media input or provisioning shortcut.
Provision first with `audio packages pull rnnoise-voice`, then run with uv's dev extra.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from audio_cli.dsp import apply_guided_denoise, calibrate_guide_input
from audio_cli.media import ffmpeg_version, render_rnnoise
from audio_cli.packages import verified_artifact
from audio_cli.profiles import PROFILES
from audio_cli.vad_contract import SpeechRegion

ROOT = Path(__file__).resolve().parents[2]
RATE = 48000


def ratio_db(before, after):
    return float(
        10 * np.log10(np.mean(before.astype(float) ** 2) / np.mean(after.astype(float) ** 2))
    )


def main():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_dsp_denoise import speech_fixture

    model, provenance = verified_artifact("rnnoise-voice")
    clean, noise = speech_fixture(RATE)
    time = np.arange(len(clean)) / RATE
    cases = []
    for name, scale, background in (
        ("stationary", 1.0, noise),
        ("changing_level", 1.0, noise * (0.2 + 2.8 * time[:, None] / 4)),
        ("quiet_changing_level", 0.01, noise * (0.2 + 2.8 * time[:, None] / 4)),
    ):
        target = (clean * scale).astype(np.float32)
        mixed = ((clean + background) * scale).astype(np.float32)
        calibrated, gain, calibration = calibrate_guide_input(
            mixed,
            RATE,
            [SpeechRegion(1, 3, 0.9, 1)],
            target_rms_dbfs=-24.0,
            peak_limit_dbfs=-3.0,
            maximum_gain_db=40.0,
        )
        assert calibrated is not None
        guide = render_rnnoise(calibrated, RATE, model, provenance["sha256"]) / gain
        output, decision, operation = apply_guided_denoise(
            mixed, guide, RATE, PROFILES["product-demo"]
        )
        speech = slice(round(1.05 * RATE), round(2.95 * RATE))
        pause = slice(round(1.8 * RATE), round(2.0 * RATE))
        voiced = slice(round(1.15 * RATE), round(1.6 * RATE))
        fricative = slice(round(2.83 * RATE), round(2.97 * RATE))
        projection = np.sum(output[voiced].astype(float) * target[voiced]) / np.sum(
            target[voiced].astype(float) ** 2
        )
        cases.append(
            {
                "name": name,
                "input_scale": scale,
                "guide_calibration": calibration,
                "decision": decision,
                "operation": operation,
                "input_samples": len(mixed),
                "output_samples": len(output),
                "pause_noise_reduction_db": round(ratio_db(mixed[pause], output[pause]), 6),
                "speech_clean_reference_error_reduction_db": round(
                    ratio_db(mixed[speech] - target[speech], output[speech] - target[speech]), 6
                ),
                "voiced_projection_gain_db": round(float(20 * np.log10(projection)), 6),
                "fricative_energy_difference_from_clean_db": round(
                    ratio_db(output[fricative], target[fricative]), 6
                ),
            }
        )
    files = [
        "tests/test_dsp_denoise.py",
        "src/audio_cli/profiles.py",
        "src/audio_cli/media/denoise.py",
        "src/audio_cli/dsp/denoise/guide.py",
        "src/audio_cli/dsp/denoise/filtering.py",
        "src/audio_cli/dsp/denoise/processor.py",
        str(Path(__file__).relative_to(ROOT)),
    ]
    print(
        json.dumps(
            {
                "topic": "RNNoise-guided linked mask on generated voiced/unvoiced signals",
                "implementation_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "fixture": "tests/test_dsp_denoise.py:speech_fixture(48000)",
                "fixture_seed": 33,
                "sample_rate_hz": RATE,
                "profile": "product-demo@5",
                "scope": "isolated model-guided DSP component before outer speech blend, leveling, and encoding",
                "model": provenance,
                "ffmpeg": ffmpeg_version(),
                "source_sha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files
                },
                "measurement_windows_seconds": {
                    "pause": [1.8, 2.0],
                    "speech": [1.05, 2.95],
                    "voiced": [1.15, 1.6],
                    "fricative": [2.83, 2.97],
                },
                "cases": cases,
                "limits": [
                    "Synthetic signals are not linguistic speech or recognition/intelligibility evidence.",
                    "No listening judgments or user-recording noise measurements are inferred.",
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
