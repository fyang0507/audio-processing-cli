"""Shared result and error types for transcription stage transport."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StageOutcome:
    role: str
    backend: str
    payload: dict[str, Any]
    wall_seconds: float
    returncode: int = 0
    peak_rss_bytes: int | None = None
    peak_mps_live_bytes: int | None = None
    # FireRed deliberately loads VAD, optional LID, ASR, and punctuation in one process.
    # The process peak therefore belongs to ``firered_process`` while its internal wall
    # measurements remain useful under their real role names. Other stages leave this absent.
    wall_seconds_by_stage: dict[str, float] | None = None


class StageFailure(RuntimeError):
    def __init__(
        self,
        role: str,
        backend: str,
        detail: str,
        *,
        outcome: StageOutcome | None = None,
    ) -> None:
        self.role = role
        self.backend = backend
        self.detail = detail
        self.outcome = outcome
        super().__init__(detail)


class ProcessRunner(Protocol):
    def run(self, command: list[str]) -> subprocess.CompletedProcess[str]: ...
