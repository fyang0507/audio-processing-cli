"""Strict, regular-file loading for offline saved reports."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from ...media import file_identity_from_descriptor


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


def mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def read_report(path: Path) -> dict[str, Any]:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        if file_identity_from_descriptor(descriptor, path) is None:
            raise ValueError("input must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            return mapping(
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


def validate_output(value: object) -> None:
    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def validate_report(report: dict[str, Any], *, inspection: bool = False) -> None:
    kinds = (
        {"audio_enhancement_report", "audio_inspection"}
        if inspection
        else {"audio_enhancement_report"}
    )
    kind = report.get("kind")
    if not isinstance(kind, str) or kind not in kinds or report.get("schema_version") != "1":
        raise ValueError(f"expected {sorted(kinds)} schema_version '1'")
    mapping(report.get("source"), "source")
    mapping(report.get("measurements"), "measurements")
    for key in ("region_basis", "timeline_verification", "profile"):
        if key in report:
            mapping(report[key], key)
    if "timeline_preserved" in report and not isinstance(report["timeline_preserved"], bool):
        raise ValueError("timeline_preserved must be a boolean")
    if report["kind"] == "audio_enhancement_report":
        for key in ("rendered", "dry_run"):
            if not isinstance(report.get(key), bool):
                raise ValueError(f"{key} must be a boolean")
        mapping(report.get("profile"), "profile")
    validate_output(report)
