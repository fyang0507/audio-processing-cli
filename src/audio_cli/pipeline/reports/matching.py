"""Report all interval overlaps, including split, merged, and differently classified scopes."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .metrics import pointed, records


def scoped_regions(report: dict[str, Any], regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for region in regions:
        row = dict(region)
        measurements = []
        identifier = region["value"]["region_id"]
        if report["kind"] == "audio_inspection":
            for index, observation in enumerate(report.get("observations", [])):
                if observation.get("region_id") == identifier:
                    measurements.append(
                        {"scope": "inspected", **pointed(observation, f"/observations/{index}")}
                    )
        else:
            for scope in ("before", "predicted", "after"):
                measured = report["measurements"].get(scope, {})
                for index, item in enumerate(
                    measured.get("regional", {}).get("machine_regions", [])
                ):
                    if item.get("region_id") == identifier:
                        measurements.append(
                            {
                                "scope": scope,
                                **pointed(
                                    item, f"/measurements/{scope}/regional/machine_regions/{index}"
                                ),
                            }
                        )
        if measurements:
            row["measurements"] = measurements
        result.append(row)
    return result


def match_regions(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps = []
    for i, lrow in enumerate(left):
        for j, rrow in enumerate(right):
            lvalue, rvalue = lrow["value"], rrow["value"]
            start = max(lvalue["start"], rvalue["start"])
            end = min(lvalue["end"], rvalue["end"])
            if start < end:
                overlaps.append((i, j, start, end))
    left_counts = Counter(i for i, _, _, _ in overlaps)
    right_counts = Counter(j for _, j, _, _ in overlaps)
    return {
        "basis": "positive_time_interval_overlap",
        "overlaps": [
            {
                "left": left[i],
                "right": right[j],
                "intersection": {"start": start, "end": end},
                "same_interval": (left[i]["value"]["start"], left[i]["value"]["end"])
                == (right[j]["value"]["start"], right[j]["value"]["end"]),
                "same_kind": left[i]["value"]["kind"] == right[j]["value"]["kind"],
                "ambiguous": left_counts[i] > 1 or right_counts[j] > 1,
            }
            for i, j, start, end in overlaps
        ],
        "left_without_overlap": [row for i, row in enumerate(left) if i not in left_counts],
        "right_without_overlap": [row for j, row in enumerate(right) if j not in right_counts],
    }


def speech_reference(report: dict[str, Any], regions: list[dict[str, Any]]) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "regions": [row for row in regions if row["value"]["kind"] == "speech"],
    }
    if "region_basis" in report:
        reference["region_basis"] = pointed(report["region_basis"], "/region_basis")
    if report["kind"] == "audio_inspection" and "observations" in report:
        levels = [
            {"scope": "inspected", **pointed(row, f"/observations/{i}")}
            for i, row in enumerate(records(report["observations"], "/observations"))
            if row.get("type") == "speech_program_level"
        ]
        if levels:
            reference["measurements"] = levels
    else:
        levels = []
        for scope in ("before", "predicted", "after"):
            regional = report["measurements"].get(scope, {}).get("regional", {})
            if "speech_program_rms_dbfs" in regional:
                levels.append(
                    {
                        "scope": scope,
                        **pointed(
                            regional["speech_program_rms_dbfs"],
                            f"/measurements/{scope}/regional/speech_program_rms_dbfs",
                        ),
                    }
                )
        if levels:
            reference["measurements"] = levels
    return reference
