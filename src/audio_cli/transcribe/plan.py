"""The resolved transcription plan and its one serialized envelope."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from audio_cli.command import packages_pull_command

from .sample import build_sample_output


@dataclass(frozen=True)
class Plan:
    stack: str
    source: dict[str, Any]
    requested_capabilities: tuple[str, ...]
    roles: dict[str, dict[str, Any]]
    execution: dict[str, Any]
    capabilities: dict[str, dict[str, Any]]
    packages: tuple[dict[str, Any], ...]
    total_known_download_bytes: int
    unsized_packages: tuple[str, ...]
    warnings: tuple[dict[str, Any], ...]
    sample_abstention_reason: str | None = None


def serialize_plan(plan: Plan, *, compact: bool = False) -> dict[str, Any]:
    """Return JSON-safe plan data and generate its sample through the result serializer."""
    core = {
        "roles": plan.roles,
        "execution": plan.execution,
        "capabilities": plan.capabilities,
        "packages": list(plan.packages),
        "total_known_download_bytes": plan.total_known_download_bytes,
        "unsized_packages": list(plan.unsized_packages),
        "warnings": list(plan.warnings),
    }
    payload = dict(core)
    if not compact:
        payload["sample_output"] = build_sample_output(
            source=plan.source,
            stack=plan.stack,
            requested_capabilities=plan.requested_capabilities,
            plan=core,
            abstention_reason=plan.sample_abstention_reason,
        )
    missing = [item["package"] for item in plan.packages if not item["provisioned"]]
    if missing:
        payload["next"] = packages_pull_command(missing)
    # The stack table is checked at load time, but this final round trip also prevents a future
    # template from smuggling a non-JSON type into stdout.
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
