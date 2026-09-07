"""Compact indexes into recorded reports, retaining phase and scope provenance."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .evidence import evidence_limits
from .matching import speech_reference
from .metrics import measurement_view, pointed, records, region_view


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _reference(pointer: str) -> dict[str, str]:
    return {"report_pointer": pointer}


def _observations(report: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    identifier = evidence.get("observation_id")
    if identifier is None:
        return []
    return [
        pointed(row, f"/observations/{i}")
        for i, row in enumerate(records(report.get("observations", []), "/observations"))
        if row.get("observation_id") == identifier
    ]


def _phase(report: dict[str, Any], pointer: str, context: dict[str, Any]) -> str:
    if "measured_at" in context:
        return context["measured_at"]["value"]
    if pointer.startswith("/rule_evaluations/"):
        return "inspected" if report["kind"] == "audio_inspection" else "before"
    # A stage name, rendered flag, or unresolved membership does not establish
    # when an individual measurement happened. Preserve unknown provenance.
    return "unknown"


def _limits(report: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    finding_ids: dict[str, int] = {}
    groups: dict[str, dict[str, Any]] = {}
    limits = evidence_limits(report)
    for row in limits:
        evidence = row["evidence"]
        fingerprint = _key(evidence)
        if fingerprint not in finding_ids:
            finding_ids[fingerprint] = len(findings)
            # Full evidence lives at every occurrence's original report pointer.
            # Group equality uses the entire object, never just this preview.
            preview = {k: v for k, v in evidence.items() if not isinstance(v, (list, dict))}
            detail_fields = [k for k, v in evidence.items() if isinstance(v, (list, dict))]
            findings.append({"preview": preview, "detail_fields": detail_fields})
        context = row.get("context", {})
        phase = _phase(report, row["report_pointer"], context)
        scope = {k: v["value"] for k, v in context.items() if k != "measured_at"}
        group_key = _key([phase, scope])
        group = groups.setdefault(group_key, {"phase": phase, "scope": scope, "occurrences": []})
        occurrence = {
            "finding_index": finding_ids[fingerprint],
            **{k: v for k, v in row.items() if k not in {"evidence", "region_basis"}},
        }
        observations = _observations(report, evidence)
        if observations:
            occurrence["observation_scopes"] = observations
        # The report-wide basis is emitted once by report_navigation. Local
        # context and resolved region scopes remain on every occurrence.
        group["occurrences"].append(occurrence)
    return {
        "indexed_occurrences": len(limits),
        "distinct_recorded_values": len(findings),
        "findings": findings,
        "groups": list(groups.values()),
    }


def _counts(report: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    def add(value: Any, pointer: str, phase: str, *, identifiers: bool = False) -> None:
        if identifiers:
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise ValueError(f"{pointer} must be an array of region identifiers")
            row = {"count": len(value)}
        else:
            values = records(value, pointer)
            if any(not isinstance(v.get("status"), str) for v in values):
                raise ValueError(f"{pointer} entries must have string statuses")
            row = {"count": len(values), "statuses": dict(Counter(v["status"] for v in values))}
        result.append({"report_pointer": pointer, "phase": phase, **row})

    if "rule_evaluations" in report:
        phase = "inspected" if report["kind"] == "audio_inspection" else "before"
        add(report["rule_evaluations"], "/rule_evaluations", phase)
        by_rule: dict[str, Counter[str]] = {}
        for evaluation in report["rule_evaluations"]:
            rule = evaluation.get("rule")
            if isinstance(rule, str):
                by_rule.setdefault(rule, Counter())[evaluation["status"]] += 1
        if by_rule:
            result[-1]["by_rule"] = [
                {"rule": rule, "count": sum(statuses.values()), "statuses": dict(statuses)}
                for rule, statuses in by_rule.items()
            ]
    if "unresolved" in report:
        # Mixed phases are retained per occurrence in the limit index.
        add(report["unresolved"], "/unresolved", "mixed_or_unknown")
    for i, stage in enumerate(records(report.get("stages", []), "/stages")):
        for field in ("inside_target_regions", "abstained_regions", "final_region_evaluations"):
            if field in stage:
                add(
                    stage[field],
                    f"/stages/{i}/{field}",
                    stage.get("measured_at", "unknown"),
                    identifiers=field != "final_region_evaluations",
                )
    return result


def _measurements(report: dict[str, Any]) -> dict[str, Any]:
    result = measurement_view(report)
    for measured in result.values():
        if "regional" in measured:
            regional = measured["regional"]
            # Keep aggregate speech metrics visible; individual intervals are
            # reached through their canonical block, never compared by ID.
            value = regional["value"]
            measured["regional"] = {
                "report_pointer": regional["report_pointer"],
                "aggregates": {
                    k: pointed(
                        v,
                        regional["report_pointer"] + "/" + k.replace("~", "~0").replace("/", "~1"),
                    )
                    for k, v in value.items()
                    if not isinstance(v, (dict, list))
                },
            }
            if "machine_regions" in value:
                measured["regional"]["machine_regions"] = {
                    "count": len(value["machine_regions"]),
                    "report_pointer": regional["report_pointer"] + "/machine_regions",
                }
        if "observations" in measured:
            observations = measured["observations"]
            measured["observations"] = {
                "count": len(observations),
                "report_pointer": "/observations",
            }
    return result


def report_navigation(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Navigate one report; pointers always address the named original report."""
    result = {
        "kind": "audio_report_navigation",
        "schema_version": "1",
        "report": str(path.resolve()),
        "report_kind": report["kind"],
        "source": {
            k: pointed(report["source"][k], f"/source/{k}")
            for k in ("path", "sha256", "decoded_audio")
            if k in report["source"]
        },
        "measurements": _measurements(report),
        "outcome_counts": _counts(report),
        "limits": _limits(report),
    }
    for key in (
        "rendered",
        "dry_run",
        "region_basis",
        "timeline_preserved",
        "timeline_verification",
        "final_peak_validation",
    ):
        if key in report:
            result[key] = pointed(report[key], f"/{key}")
    if "profile" in report:
        result["profile"] = {
            k: pointed(report["profile"][k], f"/profile/{k}")
            for k in ("name", "version")
            if k in report["profile"]
        }
    if "stages" in report:
        result["stages"] = [
            {
                "report_pointer": f"/stages/{i}",
                **{k: stage[k] for k in ("name", "status", "reason") if k in stage},
            }
            for i, stage in enumerate(records(report["stages"], "/stages"))
        ]
    if "regions" in report:
        regions = region_view(report)
        result["regions"] = {
            "report_pointer": "/regions",
            "count": len(regions),
            "kinds": dict(Counter(row["value"]["kind"] for row in regions)),
        }
        reference = speech_reference(report, regions)
        reference["regions"] = [_reference(row["report_pointer"]) for row in reference["regions"]]
        result["speech_reference"] = reference
    return result


def comparison_navigation(
    comparison: dict[str, Any], reports: list[dict[str, Any]], paths: tuple[Path, Path]
) -> dict[str, Any]:
    result = {
        "kind": "audio_report_comparison_navigation",
        "schema_version": "1",
        "compatibility": comparison["compatibility"],
        **{
            side: report_navigation(path, report)
            for side, path, report in zip(("left", "right"), paths, reports, strict=True)
        },
    }
    if "region_comparison" not in comparison:
        return result
    matched = comparison["region_comparison"]
    overlaps = matched["overlaps"]
    for side in ("left", "right"):
        # One interval table per report; overlap rows only link into it.
        result[side]["regions"]["intervals"] = [
            {
                "report_pointer": row["report_pointer"],
                **{k: row["value"][k] for k in ("region_id", "start", "end", "kind")},
            }
            for row in comparison[side]["regions"]
        ]
    result["comparison_overview"] = {
        "basis": matched["basis"],
        "counts": {
            "overlap_pairs": len(overlaps),
            "ambiguous_overlap_pairs": sum(row["ambiguous"] for row in overlaps),
            "same_interval_pairs": sum(row["same_interval"] for row in overlaps),
            "same_kind_pairs": sum(row["same_kind"] for row in overlaps),
            "left_without_overlap": len(matched["left_without_overlap"]),
            "right_without_overlap": len(matched["right_without_overlap"]),
            **{
                f"{side}_regions_with_multiple_overlaps": sum(
                    count > 1
                    for count in Counter(row[side]["report_pointer"] for row in overlaps).values()
                )
                for side in ("left", "right")
            },
        },
    }
    if "measurement_scope_comparison" in comparison:
        result["comparison_overview"]["measurement_scope_comparison"] = comparison[
            "measurement_scope_comparison"
        ]
    result["region_comparison"] = {
        "basis": matched["basis"],
        "overlaps": [
            {
                "left_report_pointer": row["left"]["report_pointer"],
                "right_report_pointer": row["right"]["report_pointer"],
                **{k: row[k] for k in ("intersection", "same_interval", "same_kind", "ambiguous")},
            }
            for row in overlaps
        ],
        **{
            field: [{"report_pointer": row["report_pointer"]} for row in matched[field]]
            for field in ("left_without_overlap", "right_without_overlap")
        },
    }
    return result
