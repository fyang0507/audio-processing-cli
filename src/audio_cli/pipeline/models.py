"""State passed between explicit enhancement pipeline phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..dsp import SignalAnalysis
from ..vad_contract import VadDetector


class PipelineError(RuntimeError):
    pass


@dataclass
class PreparedRun:
    source: Path
    output: Path | None
    dry_run: bool
    probe: dict[str, Any]
    source_info: dict[str, object]
    detector: VadDetector
    audio: np.ndarray
    sample_rate: int
    analysis: SignalAnalysis
    before_program: dict[str, float]
    before_regional: dict[str, object]


@dataclass
class StageRun:
    current: np.ndarray
    stages: list[dict[str, object]]
    resolved_adjustments: list[dict[str, object]]
    source_stage_report: dict[str, object]
    abstained_source_region_ids: set[str]


@dataclass
class LoudnessRun:
    final_wav: Path
    pre_program: dict[str, float]
    simulated_program: dict[str, float]
    simulated_regional: dict[str, object]
    program_operation: dict[str, object] | None
    simulated_peak_limit: float
