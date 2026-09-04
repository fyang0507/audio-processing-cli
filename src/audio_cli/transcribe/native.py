"""Compatibility facade for native FireRed and VibeVoice orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .catalog import InputMetadata
from .planner import ResolvedRequest
from .transport import StageTransport
from ._native_common import (
    _core_helpers,
    _diarizer_outputs,
    _executed_plan,
    _finish,
    _run_diarizer,
    _write_complete,
)
from ._native_firered import _run_firered
from ._native_scope import (
    _EmptySampleRange,
    _bind_supplied_firered_region_ledger,
    _checkout,
    _clip_canonical,
    _duration,
    _expanded_scope,
    _firered_intersects,
    _firered_public_bounds,
    _firered_published_region_ledger,
    _firered_published_vad_prefix,
    _firered_region_ledger,
    _intersects,
    _intersects_any,
    _materialized_role_paths,
    _owned,
    _published_scope,
    _selected_scope,
    _speaker_for_span,
)
from ._native_vibevoice import _native_turns, _prefix_coverage, _run_vibevoice


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
