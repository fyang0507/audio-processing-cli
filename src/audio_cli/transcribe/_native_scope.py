"""Canonical range and ownership helpers shared by native transcription stacks."""

from __future__ import annotations

import math
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.environments import packages as package_catalog

from . import refusals


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getframerate() != 16_000
            or handle.getnchannels() != 1
            or handle.getsampwidth() != 2
        ):
            raise ValueError("canonical decode is not mono 16 kHz PCM16")
        return round(handle.getnframes() / float(handle.getframerate()), 6)


class _EmptySampleRange(ValueError):
    """A valid second range that owns no sample on the canonical grid."""


def _clip_canonical(
    source: Path, target: Path, *, start: float, end: float
) -> tuple[Path, tuple[float, float]]:
    """Write an internal clip and return its exact source-timeline sample bounds."""
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        # Select samples by their source timestamps: START <= timestamp < END.
        # Using the same ceiling rule on both sides makes adjacent ranged runs
        # meet at one sample index without reprocessing a pre-range sample.
        # These exact bounds drive adapter validation and timeline rebasing.
        frame_count = reader.getnframes()
        first = min(frame_count, max(0, math.ceil(start * rate)))
        last = min(frame_count, max(0, math.ceil(end * rate)))
        if first >= last:
            raise _EmptySampleRange(
                "range selects no complete sample on the canonical 16 kHz timeline"
            )
        reader.setpos(first)
        frames = reader.readframes(last - first)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return target, (first / rate, last / rate)


def _materialized_role_paths(
    entries: Mapping[str, Mapping[str, Any]], identifier: str
) -> dict[str, Path]:
    materialized = entries[identifier].get("materialized", {})
    values = materialized.get("paths") if isinstance(materialized, Mapping) else None
    source = package_catalog()[identifier].source
    repositories = source.get("repos")
    if not isinstance(values, Mapping) or not isinstance(repositories, list):
        raise refusals.package_integrity_failed(({
            "package": identifier,
            "check": "materialized_role_paths",
            "expected": "one path per declared repository role",
            "actual": values,
        },))
    found: dict[str, Path] = {}
    for repository in repositories:
        role = str(repository["role"])
        location = values.get(str(repository["repo"]))
        if not location:
            raise refusals.package_integrity_failed(({
                "package": identifier,
                "check": f"materialized_role_{role}",
                "expected": repository["repo"],
                "actual": None,
            },))
        found[role] = Path(str(location))
    return found


def _checkout(
    entries: Mapping[str, Mapping[str, Any]], identifier: str
) -> Path:
    value = entries[identifier].get("materialized", {}).get("checkout")
    if not value:
        raise refusals.package_integrity_failed(({
            "package": identifier,
            "check": "installed_checkout",
            "expected": "present",
            "actual": value,
        },))
    return Path(str(value))


def _selected_scope(run_range: Any, duration: float) -> tuple[float, float]:
    return (
        float(run_range.start) if run_range is not None else 0.0,
        float(run_range.end) if run_range is not None else duration,
    )


def _published_scope(
    scope: tuple[float, float], duration: float
) -> tuple[float, float]:
    """Express exact processing bounds at the durable schema's precision."""
    return (round(scope[0], 6), min(duration, round(scope[1], 6)))


def _owned(span: Mapping[str, Any], scope: tuple[float, float]) -> bool:
    return scope[0] <= float(span["start"]) < scope[1]


def _intersects(span: Mapping[str, Any], scope: tuple[float, float]) -> bool:
    """Select a whole native unit when any of it intersects the requested range."""
    return float(span["end"]) > scope[0] and float(span["start"]) < scope[1]


def _firered_public_bounds(span: Mapping[str, Any]) -> tuple[float, float]:
    """Project a raw VAD region onto FireRed's published millisecond timeline."""

    start = round(int(float(span["start"]) * 1000) / 1000.0, 6)
    end = round(int(float(span["end"]) * 1000) / 1000.0, 6)
    if end <= start:
        raise ValueError("FireRed VAD region is empty at millisecond precision")
    return start, end


def _firered_intersects(
    span: Mapping[str, Any], scope: tuple[float, float]
) -> bool:
    start, end = _firered_public_bounds(span)
    return end > scope[0] and start < scope[1]


def _intersects_any(
    span: Mapping[str, Any], others: Sequence[Mapping[str, Any]]
) -> bool:
    start, end = float(span["start"]), float(span["end"])
    return any(
        end > float(other["start"]) and start < float(other["end"])
        for other in others
    )


def _expanded_scope(
    regions: Sequence[Mapping[str, Any]], requested: tuple[float, float]
) -> tuple[float, float]:
    if not regions:
        return requested
    return (
        min(float(item["start"]) for item in regions),
        max(float(item["end"]) for item in regions),
    )


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
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
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
            raise ValueError(
                f"{field} must have positive bounds within the source timeline"
            )
        if index and start < previous_end:
            raise ValueError(
                "FireRed stage regions must be chronological and non-overlapping"
            )
        previous_end = end
        if not _firered_intersects(
            {"start": start, "end": end}, requested_scope
        ):
            raise ValueError(
                f"{field} does not intersect the requested processing range"
            )

        processed = item["processed"]
        if not isinstance(processed, bool):
            raise TypeError(f"{field}.processed must be a boolean")
        if saw_unprocessed and processed:
            raise ValueError("FireRed processed regions must form a prefix")
        saw_unprocessed = saw_unprocessed or not processed
        regions.append({
            "region_id": identifier,
            "start": start,
            "end": end,
            "processed": processed,
        })
    return regions


def _firered_published_region_ledger(
    regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain ledger identity/status while projecting every unit to public bounds."""

    published: list[dict[str, Any]] = []
    for item in regions:
        start, end = _firered_public_bounds(item)
        published.append({
            "region_id": item["region_id"],
            "start": start,
            "end": end,
            "processed": item["processed"],
        })
    return published


def _bind_supplied_firered_region_ledger(
    regions: Sequence[Mapping[str, Any]],
    supplied_vad: Sequence[Mapping[str, Any]],
) -> None:
    """Require the stage ledger to preserve every selected external VAD unit exactly."""

    if len(regions) != len(supplied_vad):
        raise ValueError(
            "FireRed stage regions do not contain every supplied Silero VAD region"
        )
    for index, (actual, expected) in enumerate(zip(regions, supplied_vad, strict=True)):
        if actual["region_id"] != f"vad_{index}":
            raise ValueError(
                "FireRed stage region ids do not preserve supplied Silero VAD order"
            )
        if (
            float(actual["start"]) != float(expected["start"])
            or float(actual["end"]) != float(expected["end"])
        ):
            raise ValueError(
                f"FireRed stage regions[{index}] does not exactly match the supplied "
                "Silero VAD bounds"
            )


def _firered_published_vad_prefix(
    regions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], ...]:
    """Return the already-projected processed region prefix."""

    return tuple({
        "start": float(item["start"]),
        "end": float(item["end"]),
    } for item in regions if item["processed"])


def _speaker_for_span(
    span: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]
) -> str | None:
    """Choose the label owning the most source time, without inventing a bound."""
    start, end = float(span["start"]), float(span["end"])
    scored = []
    for index, turn in enumerate(turns):
        overlap = max(
            0.0,
            min(end, float(turn["end"])) - max(start, float(turn["start"])),
        )
        if overlap > 0:
            scored.append((overlap, -index, str(turn["speaker"])))
    return max(scored)[2] if scored else None
