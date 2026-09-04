"""Post-treatment measurements on speech and machine regions."""

from __future__ import annotations

import numpy as np

from .levels import peak_dbfs, rms_dbfs
from .regions import SignalAnalysis, _speech_samples


def regional_measurements(
    audio: np.ndarray,
    sample_rate: int,
    analysis: SignalAnalysis,
) -> dict[str, object]:
    speech_level = rms_dbfs(
        _speech_samples(audio, analysis.speech_regions, sample_rate)
    )
    machines: list[dict[str, object]] = []
    for region in analysis.machine_regions:
        start = max(0, round(region.start * sample_rate))
        end = min(audio.shape[0], round(region.end * sample_rate))
        measured = rms_dbfs(audio[start:end])
        machines.append(
            {
                "region_id": region.region_id,
                "measured_rms_dbfs": round(measured, 3),
                "difference_from_speech_db": round(measured - speech_level, 3),
            }
        )
    return {
        "speech_program_rms_dbfs": round(speech_level, 3),
        "machine_regions": machines,
        "sample_peak_dbfs": round(peak_dbfs(audio), 3),
    }


__all__ = ["regional_measurements"]
