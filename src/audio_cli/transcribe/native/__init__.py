"""Compatibility facade for native FireRed and VibeVoice orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..catalog import InputMetadata
from ..planner.request import ResolvedRequest
from ..transport.service import StageTransport
from .firered import _run_firered
from .vibevoice import _run_vibevoice


def run_native(
    request: ResolvedRequest,
    metadata: InputMetadata,
    *,
    output: Path | None = None,
    output_format: str = "json",
    run_range: Any = None,
    registry: Mapping[str, Any] | None = None,
    transport: StageTransport | None = None,
    vad_detector: Any | None = None,
    force: bool = False,
) -> Any:
    if request.stack.id == "firered":
        return _run_firered(
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
    if request.stack.id == "vibevoice":
        return _run_vibevoice(
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
    raise ValueError(f"{request.stack.id!r} is not a native-structure stack")


__all__ = ["run_native"]
