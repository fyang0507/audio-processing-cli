"""Flatten recorded evidence limits with exact pointers and inherited scope provenance."""

from __future__ import annotations

from typing import Any

from .metrics import pointed, region_view

LIMIT_STATUSES = {
    "abstained",
    "abstained_overlap",
    "outside_target",
    "bounded_outside_target",
    "failed",
    "rejected",
    "not_run",
    "skipped",
    "unmeasured",
}
CONTEXT_FIELDS = (
    "name",
    "stage",
    "component",
    "region_id",
    "scope",
    "affected_scope",
    "measured_at",
    "speech_reference",
    "reference_region",
    "noise_reference_scope_basis",
    "start",
    "end",
)


def evidence_limits(report: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    regions = region_view(report) if "regions" in report else []

    def walk(value: Any, pointer: str, inherited: dict[str, Any]) -> None:
        if isinstance(value, dict):
            context = {
                **inherited,
                **{
                    key: pointed(value[key], f"{pointer}/{key}")
                    for key in CONTEXT_FIELDS
                    if key in value
                },
            }
            status = value.get("status")
            if (isinstance(status, str) and status in LIMIT_STATUSES) or value.get(
                "target_attained"
            ) is False:
                row = {"report_pointer": pointer, "evidence": value}
                if context:
                    row["context"] = context
                if "region_basis" in report:
                    row["region_basis"] = pointed(report["region_basis"], "/region_basis")
                identifier = context.get("region_id", {}).get("value")
                scopes = [
                    region for region in regions if region["value"]["region_id"] == identifier
                ]
                if scopes:
                    row["region_scopes"] = scopes
                result.append(row)
            for key, child in value.items():
                escaped = key.replace("~", "~0").replace("/", "~1")
                walk(child, f"{pointer}/{escaped}", context)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{pointer}/{index}", inherited)

    for key in (
        "stages",
        "unresolved",
        "rule_evaluations",
        "timeline_verification",
        "final_peak_validation",
    ):
        if key in report:
            walk(report[key], f"/{key}", {})
    return result
