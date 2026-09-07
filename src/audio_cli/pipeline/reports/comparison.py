"""Public offline comparison of two saved inspection or enhancement reports."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from ..models import PipelineError
from .compatibility import compatibility, duration
from .evidence import evidence_limits
from .loading import read_report, validate_output, validate_report
from .matching import match_regions, scoped_regions, speech_reference
from .metrics import measurement_view, pointed, records, region_view
from .navigation import comparison_navigation
from .scopes import measurement_scope_comparison
from .summary import _decisions


def _view(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "report": str(path.resolve()),
        "kind": report["kind"],
        "source": pointed(report["source"], "/source"),
        "measurements": measurement_view(report),
        "evidence_limits": evidence_limits(report),
    }
    for key in ("profile", "region_basis", "timeline_preserved", "timeline_verification"):
        if key in report:
            result[key] = pointed(report[key], f"/{key}")
    if report["kind"] == "audio_enhancement_report":
        result["stages"] = _decisions(report.get("stages"), "/stages", stages=True)
        result.update(rendered=report["rendered"], dry_run=report["dry_run"])
        if "unresolved" in report:
            result["unresolved"] = _decisions(report["unresolved"], "/unresolved")
    if "rule_evaluations" in report:
        result["rule_evaluations"] = [
            pointed(row, f"/rule_evaluations/{i}")
            for i, row in enumerate(records(report["rule_evaluations"], "/rule_evaluations"))
        ]
    if "regions" in report:
        regions = region_view(report)
        source_duration = duration(report["source"], "/source")
        for region in regions:
            if Fraction(str(region["value"]["end"])) > source_duration + Fraction(1, 1_000_000):
                raise ValueError(f"{region['report_pointer']} exceeds the recorded source timeline")
        result["regions"] = scoped_regions(report, regions)
        result["speech_reference"] = speech_reference(report, regions)
    return result


def compare_reports(left: Path, right: Path, *, navigation: bool = False) -> dict[str, Any]:
    """Compare recorded metrics and scopes; return JSON data or raise PipelineError.

    Only report files are read. Identity links and timeline evidence are recorded
    facts, not fresh media verification or content/alignment/quality judgments.
    """
    try:
        reports = [read_report(path) for path in (left, right)]
        for report in reports:
            validate_report(report, inspection=True)
        compatible = compatibility(*reports)
        lview, rview = [
            _view(path, report) for path, report in zip((left, right), reports, strict=True)
        ]
        result = {
            "kind": "audio_report_comparison",
            "schema_version": "1",
            "compatibility": compatible,
            "left": lview,
            "right": rview,
        }
        if "regions" in lview and "regions" in rview:
            result["region_comparison"] = match_regions(lview["regions"], rview["regions"])
        scopes = measurement_scope_comparison(lview, rview)
        if scopes:
            result["measurement_scope_comparison"] = scopes
        if navigation:
            result = comparison_navigation(result, reports, (left, right))
        validate_output(result)
        return result
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise PipelineError(f"Cannot compare reports {left} and {right}: {exc}") from exc
