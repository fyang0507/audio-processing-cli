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
from .qwen import _run_qwen


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
    """Execute any shipped transcription stack over the canonical source timeline."""
    if request.stack.id in {"firered", "vibevoice"}:
        # Kept lazy so the Qwen module remains importable while native stage dependencies are
        # absent; model libraries live only in their fresh environment processes.
        from ..native import run_native

        return run_native(
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
    return _run_qwen(
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
