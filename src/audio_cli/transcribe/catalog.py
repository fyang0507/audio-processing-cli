"""Build the stack capability report from static data and an ffprobe result."""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import stacks


@dataclass(frozen=True)
class InputMetadata:
    path: str
    duration_seconds: float
    container: str
    sample_rate_hz: int
    channels: int

    @property
    def source(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration_seconds": self.duration_seconds,
            "timebase": "seconds",
        }

    @property
    def catalog_input(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration_seconds": self.duration_seconds,
            "container": self.container,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
        }


def input_metadata(path: Path, probe: dict[str, object]) -> InputMetadata:
    """Read only the container facts already returned by :func:`probe_media`."""
    stream = probe.get("primary_audio_stream")
    if not isinstance(stream, dict):
        raise ValueError("media probe has no primary audio stream")
    format_info = probe.get("format", {})
    if not isinstance(format_info, dict):
        raise ValueError("media probe format must be an object")
    raw_duration = stream.get("duration", format_info.get("duration"))
    try:
        duration = round(float(raw_duration), 6)
        sample_rate = int(stream.get("sample_rate", 0))
        channels = int(stream.get("channels", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("media probe carries invalid audio metadata") from exc
    if not math.isfinite(duration) or duration < 0 or sample_rate <= 0 or channels <= 0:
        raise ValueError("media probe carries invalid audio metadata")
    suffix = path.suffix.removeprefix(".").lower()
    fallback = str(format_info.get("format_name", "unknown")).split(",", 1)[0]
    return InputMetadata(
        path=str(path),
        duration_seconds=duration,
        container=suffix or fallback,
        sample_rate_hz=sample_rate,
        channels=channels,
    )


def result_source(
    metadata: InputMetadata,
    duration_seconds: float,
    *,
    source_identity: Path | None = None,
) -> dict[str, Any]:
    """Build the durable source identity for a result document.

    Catalogs and plans preserve the caller's spelling, but a saved result can be
    exported from another working directory.  Its canonical-media guard therefore
    needs an absolute source identity captured while the run still has the caller's
    original working directory.
    """
    source = dict(metadata.source)
    # CLI paths are opened literally.  A real directory named ``~alice`` beneath the
    # caller's working directory must not become Alice's home only when durable
    # provenance is recorded; export relies on this identity to protect the source.
    source["path"] = str(
        source_identity if source_identity is not None else Path(metadata.path).resolve()
    )
    source["duration_seconds"] = duration_seconds
    return source


def _unit_count(definition: stacks.StackDefinition, duration: float) -> int | None:
    rule = definition.processing["unit_count_rule"]
    kind = rule["kind"]
    if kind == "one":
        return 1
    if kind == "unknown":
        return None
    if kind == "fixed_seconds":
        return math.ceil(duration / float(rule["seconds"]))
    raise stacks.StackTableError(
        f"{definition.id}: unknown unit-count rule {kind!r}"
    )


def build_catalog(
    definition: stacks.StackDefinition,
    metadata: InputMetadata,
) -> dict[str, Any]:
    capabilities: dict[str, dict[str, str]] = {}
    for capability in stacks.capability_order():
        cell = definition.capabilities[capability]
        resolution = cell["resolution"]
        if resolution in {"native", "native_stage"}:
            availability = "native"
        elif resolution == "add_on":
            availability = "requires_add_on"
        else:
            availability = "impossible"
        entry = {"availability": availability}
        if availability == "impossible":
            entry["reason"] = cell["reason"]
        entry["note"] = cell["catalog_note"]
        capabilities[capability] = entry

    processing = {
        "unit": definition.processing["unit"],
        "unit_count": _unit_count(definition, metadata.duration_seconds),
        "note": definition.processing["note"],
    }
    return {
        "stack": definition.id,
        "family": definition.family,
        "environment": definition.environment,
        "roles": definition.roles,
        "input": metadata.catalog_input,
        "processing": processing,
        "failure_recovery": dict(definition.failure_recovery),
        "cost": {
            "proved": definition.cost["proved"],
            "projected_seconds": round(
                metadata.duration_seconds * definition.cost["seconds_per_input_second"], 1
            ),
        },
        "capabilities": capabilities,
        "next": (
            f"audio transcribe plan --input {shlex.quote(metadata.path)} "
            f"--stack {definition.id} "
            "--want <capabilities>"
        ),
    }
