"""Offline navigation of recorded enhancement decisions, without remeasurement."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from ..media import file_identity_from_descriptor
from .models import PipelineError

DECISION_FIELDS = ("name", "stage", "component", "region_id", "status", "reason", "measured_at")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite JSON number {value}")
    return number


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _read_report(path: Path) -> dict[str, Any]:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        if file_identity_from_descriptor(descriptor, path) is None:
            raise ValueError("input must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            return _mapping(
                json.load(
                    handle,
                    object_pairs_hook=_object,
                    parse_constant=_constant,
                    parse_float=_float,
                ),
                "report",
            )
    finally:
        os.close(descriptor)


def _decisions(value: Any, pointer: str, *, stages: bool = False) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{pointer} must be an array")
    rows = []
    for index, raw in enumerate(value):
        location = f"{pointer}/{index}"
        item = _mapping(raw, location)
        if not isinstance(item.get("status"), str):
            raise ValueError(f"{location}/status must be a string")
        # Components and unresolved rows are open evidence records: retain every
        # field, including future reference scopes, local evidence, and measurements.
        row = {key: item[key] for key in DECISION_FIELDS if key in item} if stages else dict(item)
        row["report_pointer"] = location
        for key in ("component_evaluations", "final_region_evaluations"):
            if key in item:
                row[key] = _decisions(item[key], f"{location}/{key}")
        rows.append(row)
    return rows


def summarize_report(path: Path) -> dict[str, Any]:
    """Keep explicit outcomes, with pointers to their full scope and measurements.

    Older v1 reports may lack unresolved, region_basis, or program_actual. Do not
    manufacture an empty ledger or infer encoded scope for earlier stage decisions.
    No canonical report field is added or rewritten by this projection.
    """
    try:
        report = _read_report(path)
        if report.get("kind") != "audio_enhancement_report" or report.get("schema_version") != "1":
            raise ValueError("expected audio_enhancement_report schema_version '1'")
        for key in ("rendered", "dry_run"):
            if not isinstance(report.get(key), bool):
                raise ValueError(f"{key} must be a boolean")
        source = _mapping(report.get("source"), "source")
        profile = _mapping(report.get("profile"), "profile")
        measurements = _mapping(report.get("measurements"), "measurements")
        summary: dict[str, Any] = {
            "kind": "audio_enhancement_summary",
            "schema_version": "1",
            "report": str(path.resolve()),
            "source": {key: source[key] for key in ("path", "sha256") if key in source},
            "profile": {key: profile[key] for key in ("name", "version") if key in profile},
            "rendered": report["rendered"],
            "dry_run": report["dry_run"],
            "stages": _decisions(report.get("stages"), "/stages", stages=True),
            "measurement_scopes": {},
        }
        for scope in ("before", "predicted", "after"):
            if scope not in measurements:
                continue
            measured = _mapping(measurements[scope], f"measurements/{scope}")
            summary["measurement_scopes"][scope] = {
                "report_pointer": f"/measurements/{scope}",
                **{
                    field: {"report_pointer": f"/measurements/{scope}/{field}"}
                    for field in ("program_actual", "regional")
                    if field in measured
                },
            }
        for key in ("region_basis", "timeline_verification", "timeline_preserved"):
            if key in report:
                if key in {"region_basis", "timeline_verification"}:
                    _mapping(report[key], key)
                elif not isinstance(report[key], bool):
                    raise ValueError(f"{key} must be a boolean")
                summary[key] = report[key]
        if "unresolved" in report:
            summary["unresolved"] = _decisions(report["unresolved"], "/unresolved")
        if "final_peak_validation" in report:
            peak = _mapping(report["final_peak_validation"], "final_peak_validation")
            summary["final_peak_validation"] = {
                **({"status": peak["status"]} if "status" in peak else {}),
                "report_pointer": "/final_peak_validation",
            }
        # Validate the complete machine output before CLI streaming can emit a
        # prefix and then fail on an escaped lone surrogate in retained evidence.
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        return summary
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise PipelineError(f"Cannot summarize enhancement report {path}: {exc}") from exc
