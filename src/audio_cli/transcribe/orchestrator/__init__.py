"""Compatibility facade for transcription execution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..catalog import InputMetadata
from ..execution.preflight import preflight
from ..execution.publication import render_human, validate_output_targets
from ..execution.runtime import RunProduct, RunRange, parse_range
from ..planner.request import ResolvedRequest
from ..transport.service import StageTransport
from .firered import _run_firered
from .qwen import _run_qwen
from .vibevoice import _run_vibevoice


def run(
    request: ResolvedRequest,
    metadata: InputMetadata,
    *,
    output: Path | None = None,
    output_format: str = "json",
    run_range: RunRange | None = None,
    registry: Mapping[str, Any] | None = None,
    transport: StageTransport | None = None,
    vad_detector: Any | None = None,
    force: bool = False,
) -> RunProduct:
    """Execute any shipped transcription stack over the canonical source timeline.

    ``output_format`` retains the legacy Python API's human-file behavior. The public
    CLI only passes JSON and uses the separate export command for readable files.
    """
    if request.stack.id == "firered":
        runner = _run_firered
    elif request.stack.id == "vibevoice":
        runner = _run_vibevoice
    elif request.stack.id in {"qwen-0.6b", "qwen-1.7b"}:
        runner = _run_qwen
    else:
        raise ValueError(f"unknown transcription stack {request.stack.id!r}")
    return runner(
        request,
        metadata,
        output=output,
        output_format=output_format,
        run_range=run_range,
        registry=registry,
        transport=transport,
        vad_detector=vad_detector,
        force=force,
    )


__all__ = [
    "RunProduct",
    "RunRange",
    "parse_range",
    "preflight",
    "render_human",
    "run",
    "validate_output_targets",
]
