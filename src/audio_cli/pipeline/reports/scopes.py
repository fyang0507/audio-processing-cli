"""Compare recorded regional time scopes without comparing audio or metric values."""

from __future__ import annotations

from typing import Any


def _intervals(view: dict[str, Any], kind: str) -> list[tuple[float, float]] | None:
    basis = view.get("region_basis", {}).get("value", {})
    detection = {
        "audio_enhancement_report": "original_source",
        "audio_inspection": "inspected_source",
    }
    if (
        "regions" not in view
        or basis.get("timeline") != "source"
        or basis.get("regional_measurements") != "fixed_source_regions"
        or basis.get("detection") != detection.get(view["kind"])
    ):
        return None
    return sorted(
        (row["value"]["start"], row["value"]["end"])
        for row in view["regions"]
        if row["value"]["kind"] == kind
    )


def _speech_union(view: dict[str, Any]) -> list[tuple[float, float]] | None:
    intervals = _intervals(view, "speech")
    if not intervals:
        return None
    measurements = view.get("speech_reference", {}).get("measurements", [])
    if view["kind"] == "audio_inspection":
        measurements = [
            row
            for row in measurements
            if row["value"].get("scope") == {"time": "speech_regions", "frequency": "all"}
            and isinstance(row["value"].get("measured_rms_dbfs"), (int, float))
            and not isinstance(row["value"]["measured_rms_dbfs"], bool)
        ]
    if not measurements:
        return None
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def measurement_scope_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, str]:
    """Return independent equality findings only where recorded evidence permits.

    Speech uses the union of nonempty reference intervals. Regional RMS values
    concern individual intervals, so their multiplicity and boundaries remain
    significant even when two collections cover the same overall time range.
    Equality concerns recorded times, not decoded sample masks or audio values.
    """
    comparisons = {
        "speech_reference_intervals": (_speech_union(left), _speech_union(right)),
        "non_speech_region_intervals": (
            _intervals(left, "non_speech_program"),
            _intervals(right, "non_speech_program"),
        ),
    }
    findings = {
        name: "same" if lvalue == rvalue else "different"
        for name, (lvalue, rvalue) in comparisons.items()
        if lvalue is not None and rvalue is not None
    }
    return {"basis": "recorded_time_intervals", **findings} if findings else {}
