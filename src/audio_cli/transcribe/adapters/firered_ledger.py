"""Validate and project FireRed's raw processed-region ledger."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _firered_public_bounds(span: Mapping[str, Any]) -> tuple[float, float]:
    """Project a raw VAD region onto FireRed's published millisecond timeline."""

    start = round(int(float(span["start"]) * 1000) / 1000.0, 6)
    end = round(int(float(span["end"]) * 1000) / 1000.0, 6)
    if end <= start:
        raise ValueError("FireRed VAD region is empty at millisecond precision")
    return start, end


def _firered_intersects(span: Mapping[str, Any], scope: tuple[float, float]) -> bool:
    start, end = _firered_public_bounds(span)
    return end > scope[0] and start < scope[1]


def _firered_region_ledger(
    raw_regions: object,
    *,
    source_duration: float,
    requested_scope: tuple[float, float],
) -> list[dict[str, Any]]:
    """Validate the stage's complete processed/unprocessed VAD unit ledger."""

    if not isinstance(raw_regions, list):
        raise TypeError("FireRed stage regions must be an array")
    regions: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    previous_end = 0.0
    saw_unprocessed = False
    for index, item in enumerate(raw_regions):
        field = f"FireRed stage regions[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{field} must be an object")
        if set(item) != {"region_id", "start", "end", "processed"}:
            raise ValueError(f"{field} has an unexpected shape")

        identifier = item["region_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("FireRed stage region ids must be unique non-empty strings")
        identifiers.add(identifier)

        bounds: list[float] = []
        for name in ("start", "end"):
            value = item[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field}.{name} must be a number")
            try:
                parsed = float(value)
            except OverflowError as exc:
                raise ValueError(f"{field}.{name} must be finite") from exc
            if not math.isfinite(parsed):
                raise ValueError(f"{field}.{name} must be finite")
            bounds.append(parsed)
        start, end = bounds
        if start < 0 or end > source_duration or end <= start:
            raise ValueError(f"{field} must have positive bounds within the source timeline")
        if index and start < previous_end:
            raise ValueError("FireRed stage regions must be chronological and non-overlapping")
        previous_end = end
        if not _firered_intersects({"start": start, "end": end}, requested_scope):
            raise ValueError(f"{field} does not intersect the requested processing range")

        processed = item["processed"]
        if not isinstance(processed, bool):
            raise TypeError(f"{field}.processed must be a boolean")
        if saw_unprocessed and processed:
            raise ValueError("FireRed processed regions must form a prefix")
        saw_unprocessed = saw_unprocessed or not processed
        regions.append(
            {
                "region_id": identifier,
                "start": start,
                "end": end,
                "processed": processed,
            }
        )
    return regions


def _firered_published_region_ledger(
    regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain ledger identity/status while projecting every unit to public bounds."""

    published: list[dict[str, Any]] = []
    for item in regions:
        start, end = _firered_public_bounds(item)
        published.append(
            {
                "region_id": item["region_id"],
                "start": start,
                "end": end,
                "processed": item["processed"],
            }
        )
    return published


def _bind_supplied_firered_region_ledger(
    regions: Sequence[Mapping[str, Any]],
    supplied_vad: Sequence[Mapping[str, Any]],
) -> None:
    """Require the stage ledger to preserve every selected external VAD unit exactly."""

    if len(regions) != len(supplied_vad):
        raise ValueError("FireRed stage regions do not contain every supplied Silero VAD region")
    for index, (actual, expected) in enumerate(zip(regions, supplied_vad, strict=True)):
        if actual["region_id"] != f"vad_{index}":
            raise ValueError("FireRed stage region ids do not preserve supplied Silero VAD order")
        if float(actual["start"]) != float(expected["start"]) or float(actual["end"]) != float(
            expected["end"]
        ):
            raise ValueError(
                f"FireRed stage regions[{index}] does not exactly match the supplied "
                "Silero VAD bounds"
            )


def _firered_published_vad_prefix(
    regions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], ...]:
    """Return the already-projected processed region prefix."""

    return tuple(
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
        }
        for item in regions
        if item["processed"]
    )
