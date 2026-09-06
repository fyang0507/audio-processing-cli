"""Noise-reference scope selection and public evidence over source-time arrays."""

from __future__ import annotations

import numpy as np

from ..regions import SignalAnalysis
from .reference import reference_consistency


def select_reference_runs(
    audio: np.ndarray,
    sample_rate: int,
    analysis: SignalAnalysis,
    intervals: list[tuple[float, float]],
    window: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, object]]:
    """Expose every unexcluded run, including too-short and inconsistent scopes."""
    frame = window.size
    hop = frame // 4
    starts = np.arange(0, max(0, len(audio) - frame + 1), hop)
    eligible = np.ones(starts.size, dtype=bool)
    guard = round(0.12 * sample_rate)
    excluded = [*intervals, *((r.start, r.end) for r in analysis.machine_regions)]
    for start, end in excluded:
        eligible &= (starts + frame <= round(start * sample_rate) - guard) | (
            starts >= round(end * sample_rate) + guard
        )
    indices = np.flatnonzero(eligible)
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    accepted = []
    scopes: list[dict[str, object]] = []
    for run in runs:
        if not run.size:
            continue
        length = (run.size - 1) * hop + frame
        scope: dict[str, object] = {
            "start": round(float(starts[run[0]]) / sample_rate, 6),
            "end": round(float(starts[run[-1]] + frame) / sample_rate, 6),
            "frame_count": int(run.size),
        }
        if length < 0.25 * sample_rate:
            scope.update(status="rejected", reason="noise_reference_too_short")
        else:
            accepted.append(run)
            reason, diagnostics = reference_consistency(audio, starts, [run], window, sample_rate)
            scope.update(
                status="rejected" if reason else "eligible",
                reason=reason or "reference_run_consistent",
                **diagnostics,
            )
        scopes.append(scope)
    return (
        starts,
        accepted,
        {
            "noise_reference_scopes": scopes,
            "noise_reference_guard_ms": 120,
            "noise_reference_minimum_contiguous_ms": 250,
            "noise_reference_scope_basis": "source_timeline_after_speech_and_program_exclusion",
        },
    )
