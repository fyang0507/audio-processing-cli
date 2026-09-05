"""Deprecated compatibility entry point for native-structure stack orchestration.

The provider workflows live in :mod:`audio_cli.transcribe.orchestrator`. This module
preserves only the public import path shipped before that ownership boundary was made
explicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .catalog import InputMetadata
from .orchestrator import run as _run
from .planner import ResolvedRequest
from .transport import StageTransport


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
    """Forward the two formerly native-owned stacks to the canonical dispatcher."""
    if request.stack.id not in {"firered", "vibevoice"}:
        raise ValueError(f"{request.stack.id!r} is not a native-structure stack")
    return _run(
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


__all__ = ["run_native"]
