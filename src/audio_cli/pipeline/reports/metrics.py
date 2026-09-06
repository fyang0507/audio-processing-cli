"""Project only recorded measurements, preserving their original measurement scopes."""

from __future__ import annotations

from typing import Any

from .loading import mapping


def pointed(value: Any, pointer: str) -> dict[str, Any]:
    return {"value": value, "report_pointer": pointer}


def number(value: Any, pointer: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{pointer} must be a number")
    return value


def records(value: Any, pointer: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{pointer} must be an array")
    return [mapping(row, f"{pointer}/{i}") for i, row in enumerate(value)]


def region_view(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    identifiers = set()
    for index, region in enumerate(records(report["regions"], "/regions")):
        pointer = f"/regions/{index}"
        start = number(region.get("start"), f"{pointer}/start")
        end = number(region.get("end"), f"{pointer}/end")
        if not 0 <= start < end:
            raise ValueError(f"{pointer} must have 0 <= start < end")
        identifier = region.get("region_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f"{pointer}/region_id must be a unique nonempty string")
        identifiers.add(identifier)
        if not isinstance(region.get("kind"), str):
            raise ValueError(f"{pointer}/kind must be a string")
        rows.append(pointed(region, pointer))
    return rows


def measurement_view(report: dict[str, Any]) -> dict[str, Any]:
    measurements = report["measurements"]
    scopes = (
        [("inspected", measurements, "/measurements")]
        if report["kind"] == "audio_inspection"
        else [
            (
                scope,
                mapping(measurements[scope], f"/measurements/{scope}"),
                f"/measurements/{scope}",
            )
            for scope in ("before", "predicted", "after")
            if scope in measurements
        ]
    )
    result = {}
    for scope, measured, pointer in scopes:
        row: dict[str, Any] = {"report_pointer": pointer}
        if "program_actual" in measured:
            actual = mapping(measured["program_actual"], f"{pointer}/program_actual")
            for name, value in actual.items():
                number(value, f"{pointer}/program_actual/{name}")
            row["program_actual"] = pointed(actual, f"{pointer}/program_actual")
        if "regional" in measured:
            regional = mapping(measured["regional"], f"{pointer}/regional")
            for name in ("speech_program_rms_dbfs", "sample_peak_dbfs"):
                if name in regional:
                    number(regional[name], f"{pointer}/regional/{name}")
            if "machine_regions" in regional:
                for index, region in enumerate(records(regional["machine_regions"], pointer)):
                    for name in ("measured_rms_dbfs", "difference_from_speech_db"):
                        if name in region:
                            number(
                                region[name], f"{pointer}/regional/machine_regions/{index}/{name}"
                            )
            row["regional"] = pointed(regional, f"{pointer}/regional")
        if "duration_delta_ms" in measured:
            number(measured["duration_delta_ms"], f"{pointer}/duration_delta_ms")
            row["duration_delta_ms"] = pointed(
                measured["duration_delta_ms"], f"{pointer}/duration_delta_ms"
            )
        if "duration_basis" in measured:
            if not isinstance(measured["duration_basis"], str):
                raise ValueError(f"{pointer}/duration_basis must be a string")
            row["duration_basis"] = pointed(measured["duration_basis"], f"{pointer}/duration_basis")
        result[scope] = row
    # Inspection records its acoustic measurements in observations, not an
    # enhancement regional block. Keep their speech reference and scope intact.
    if report["kind"] == "audio_inspection" and "observations" in report:
        result["inspected"]["observations"] = [
            pointed(row, f"/observations/{index}")
            for index, row in enumerate(records(report["observations"], "/observations"))
        ]
    return result
