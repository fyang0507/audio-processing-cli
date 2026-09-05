"""The capability half of the transcription stack table.

Package ownership stays in :mod:`audio_cli.environments`.  This module owns only the facts a
caller uses to choose and resolve capabilities, and validates the seam between the two tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from audio_cli import environments as env

from .result.types import CAPABILITY_NAMES

HERE = Path(__file__).resolve().parent
TABLE = HERE / "stacks.json"

RESOLUTIONS = frozenset(
    {
        "native",
        "native_stage",
        "add_on",
        "unsatisfiable_on_stack",
        "unsupported",
    }
)


class StackTableError(RuntimeError):
    """The checked-in stack table is malformed or disagrees with the package manifest."""


@dataclass(frozen=True)
class StackDefinition:
    id: str
    family: str
    environment: str
    roles: str
    characterization: str
    language_vocabulary: str | None
    base_roles: dict[str, str]
    processing: dict[str, Any]
    failure_recovery: dict[str, Any]
    cost: dict[str, Any]
    execution: dict[str, Any]
    warnings: tuple[dict[str, Any], ...]
    capabilities: dict[str, dict[str, Any]]


@cache
def _raw() -> dict[str, Any]:
    try:
        document = json.loads(TABLE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StackTableError(f"could not read {TABLE.name}: {exc}") from exc
    if document.get("schema_version") != 1:
        raise StackTableError(
            f"unsupported stacks schema_version {document.get('schema_version')!r}"
        )
    return document


def capability_order() -> tuple[str, ...]:
    return tuple(_raw()["capability_order"])


def language_vocabulary(name: str) -> tuple[str, ...]:
    try:
        return tuple(_raw()["language_vocabularies"][name])
    except KeyError as exc:
        raise StackTableError(f"unknown language vocabulary {name!r}") from exc


@cache
def stack_definitions() -> dict[str, StackDefinition]:
    found: dict[str, StackDefinition] = {}
    for identifier, body in _raw()["stacks"].items():
        found[identifier] = StackDefinition(
            id=identifier,
            family=body["family"],
            environment=body["environment"],
            roles=body["roles"],
            characterization=body["characterization"],
            language_vocabulary=body.get("language_vocabulary"),
            base_roles=dict(body["base_roles"]),
            processing=dict(body["processing"]),
            failure_recovery=dict(body["failure_recovery"]),
            cost=dict(body["cost"]),
            execution=dict(body["execution"]),
            warnings=tuple(dict(item) for item in body.get("warnings", ())),
            capabilities={name: dict(cell) for name, cell in body["capabilities"].items()},
        )
    return found


def stack_ids() -> tuple[str, ...]:
    return tuple(stack_definitions())


def get_stack(identifier: str) -> StackDefinition:
    try:
        return stack_definitions()[identifier]
    except KeyError as exc:
        raise StackTableError(f"unknown stack {identifier!r}") from exc


def availability_groups(stack: StackDefinition) -> dict[str, list[str]]:
    groups = {"native": [], "requires_add_on": [], "impossible": []}
    for capability in capability_order():
        resolution = stack.capabilities[capability]["resolution"]
        if resolution in {"native", "native_stage"}:
            groups["native"].append(capability)
        elif resolution == "add_on":
            groups["requires_add_on"].append(capability)
        else:
            groups["impossible"].append(capability)
    return groups


def allowed_stacks(capability: str) -> list[str]:
    if capability not in CAPABILITY_NAMES:
        return []
    return [
        stack.id
        for stack in stack_definitions().values()
        if stack.capabilities[capability]["resolution"] in {"native", "native_stage", "add_on"}
    ]


def recommended_stack(capability: str) -> str | None:
    return _raw().get("recommendations", {}).get(capability)


def _source_checkout_root() -> Path | None:
    candidate = HERE.parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "model_tests").is_dir():
        return candidate
    return None


def _evidence_source_problem(evidence: object) -> str | None:
    if not isinstance(evidence, str):
        return f"evidence_source {evidence!r} is not a safe repository evidence path"
    relative = Path(evidence)
    if relative.is_absolute() or not evidence.startswith("model_tests/") or ".." in relative.parts:
        return f"evidence_source {evidence!r} is not a safe repository evidence path"
    repository = _source_checkout_root()
    if repository is not None and not (repository / relative).is_file():
        return f"evidence_source {evidence!r} is not tracked"
    return None


def validate() -> list[str]:
    """Return every table/manifest disagreement rather than failing at the first one."""
    problems: list[str] = []
    order = capability_order()
    if len(order) != len(set(order)):
        problems.append("capability_order contains duplicates")
    if set(order) != CAPABILITY_NAMES:
        problems.append(
            "capability_order disagrees with the result namespace: "
            f"missing {sorted(CAPABILITY_NAMES - set(order))}, "
            f"extra {sorted(set(order) - CAPABILITY_NAMES)}"
        )

    manifest_packages = env.packages()
    manifest_backends = env.backends()
    manifest_stacks = {stack for package in manifest_packages.values() for stack in package.stacks}
    table_stacks = set(stack_definitions())
    if table_stacks != manifest_stacks:
        problems.append(
            f"stack ids disagree: table-only {sorted(table_stacks - manifest_stacks)}, "
            f"manifest-only {sorted(manifest_stacks - table_stacks)}"
        )

    for stack in stack_definitions().values():
        if set(stack.capabilities) != set(order):
            problems.append(f"{stack.id}: capability cells disagree with capability_order")
        if stack.environment not in env.environments():
            problems.append(f"{stack.id}: unknown environment {stack.environment!r}")
        if stack.language_vocabulary is not None:
            try:
                vocabulary = language_vocabulary(stack.language_vocabulary)
            except StackTableError as exc:
                problems.append(f"{stack.id}: {exc}")
            else:
                folded = [name.casefold() for name in vocabulary]
                if not vocabulary or len(folded) != len(set(folded)):
                    problems.append(f"{stack.id}: language vocabulary must be non-empty and unique")

        for role, backend_id in stack.base_roles.items():
            backend = manifest_backends.get(backend_id)
            if backend is None:
                problems.append(f"{stack.id}: base role {role!r} names {backend_id!r}")
            elif backend.role != role:
                problems.append(
                    f"{stack.id}: backend {backend_id!r} fills {backend.role!r}, not {role!r}"
                )
            elif stack.id not in manifest_packages[backend.package].stacks:
                problems.append(f"{stack.id}: package {backend.package!r} does not list this stack")

        for capability in order:
            cell = stack.capabilities.get(capability, {})
            resolution = cell.get("resolution")
            if resolution not in RESOLUTIONS:
                problems.append(f"{stack.id}/{capability}: invalid resolution {resolution!r}")
                continue
            if not isinstance(cell.get("catalog_note"), str) or len(cell["catalog_note"]) < 20:
                problems.append(f"{stack.id}/{capability}: catalog_note is not substantive")
            evidence = cell.get("evidence_source")
            if evidence_problem := _evidence_source_problem(evidence):
                problems.append(f"{stack.id}/{capability}: {evidence_problem}")
            if resolution == "add_on":
                package_id = cell.get("package")
                if package_id not in manifest_packages:
                    problems.append(
                        f"{stack.id}/{capability}: add-on package {package_id!r} is unknown"
                    )
                elif stack.id not in manifest_packages[package_id].stacks:
                    problems.append(
                        f"{stack.id}/{capability}: package {package_id!r} does not list stack"
                    )
            if resolution in {"unsatisfiable_on_stack", "unsupported"}:
                if not cell.get("reason"):
                    problems.append(f"{stack.id}/{capability}: impossible cell has no reason")
            elif "reason" in cell:
                problems.append(f"{stack.id}/{capability}: satisfiable cell carries a reason")
            if resolution == "native_stage" and not cell.get("stage"):
                problems.append(f"{stack.id}/{capability}: native stage is unnamed")
            if resolution == "unsupported" and not cell.get("refusal_fix"):
                problems.append(f"{stack.id}/{capability}: unsupported cell has no refusal_fix")
            if resolution == "native_stage":
                backend = manifest_backends.get(cell.get("backend"))
                if backend is None or backend.role != capability:
                    problems.append(
                        f"{stack.id}/{capability}: native stage backend is missing or "
                        "fills the wrong role"
                    )
            if resolution == "add_on" and cell.get("package") in manifest_packages:
                bindings = [
                    backend
                    for backend in manifest_backends.values()
                    if backend.package == cell["package"]
                ]
                if len(bindings) != 1:
                    problems.append(
                        f"{stack.id}/{capability}: add-on package must expose one backend"
                    )
    for capability in order:
        cells = [stack.capabilities[capability] for stack in stack_definitions().values()]
        available = allowed_stacks(capability)
        if any(cell["resolution"] == "unsatisfiable_on_stack" for cell in cells) and not available:
            problems.append(
                f"{capability}: unsatisfiable_on_stack requires a non-empty alternative"
            )
        unsupported = [cell for cell in cells if cell["resolution"] == "unsupported"]
        if unsupported and available:
            problems.append(f"{capability}: unsupported cannot have an available stack")
        if (
            unsupported
            and len({(cell.get("reason"), cell.get("refusal_fix")) for cell in unsupported}) != 1
        ):
            problems.append(f"{capability}: unsupported reason and refusal_fix disagree by stack")
    return problems
