"""Lightweight value and protocol contracts shared by VAD consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SpeechRegion:
    start: float
    end: float
    mean_probability: float
    peak_probability: float

    def as_dict(self) -> dict[str, float]:
        return {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "mean_probability": round(self.mean_probability, 6),
            "peak_probability": round(self.peak_probability, 6),
        }


class VadDetector(Protocol):
    model_version: str

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        threshold: float,
        exit_threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
        speech_pad_ms: int,
    ) -> list[SpeechRegion]: ...


__all__ = ["SpeechRegion", "VadDetector"]
