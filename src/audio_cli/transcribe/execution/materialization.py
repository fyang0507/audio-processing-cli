"""Validated access to package materialization records used by stage execution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from audio_cli.environments import packages as package_catalog

from ..refusals import request as refusals


def _materialized_value(
    entries: Mapping[str, Mapping[str, Any]],
    identifier: str,
    field: str,
    *,
    check: str | None = None,
) -> Any:
    materialized = entries[identifier].get("materialized", {})
    value = materialized.get(field) if isinstance(materialized, Mapping) else None
    if value is None or value == "":
        raise refusals.package_integrity_failed(
            (
                {
                    "package": identifier,
                    "check": check or f"materialized_{field}",
                    "expected": "present",
                    "actual": value,
                },
            )
        )
    return value


def _materialized_path(entries: Mapping[str, Mapping[str, Any]], identifier: str) -> Path:
    return Path(str(_materialized_value(entries, identifier, "path")))


def _materialized_product_path(entries: Mapping[str, Mapping[str, Any]], identifier: str) -> Path:
    return Path(str(_materialized_value(entries, identifier, "product_path")))


def _materialized_product_sha256(entries: Mapping[str, Mapping[str, Any]], identifier: str) -> str:
    return str(_materialized_value(entries, identifier, "product_sha256"))


def _materialized_role_paths(
    entries: Mapping[str, Mapping[str, Any]], identifier: str
) -> dict[str, Path]:
    materialized = entries[identifier].get("materialized", {})
    values = materialized.get("paths") if isinstance(materialized, Mapping) else None
    source = package_catalog()[identifier].source
    repositories = source.get("repos")
    if not isinstance(values, Mapping) or not isinstance(repositories, list):
        raise refusals.package_integrity_failed(
            (
                {
                    "package": identifier,
                    "check": "materialized_role_paths",
                    "expected": "one path per declared repository role",
                    "actual": values,
                },
            )
        )
    found: dict[str, Path] = {}
    for repository in repositories:
        role = str(repository["role"])
        location = values.get(str(repository["repo"]))
        if not location:
            raise refusals.package_integrity_failed(
                (
                    {
                        "package": identifier,
                        "check": f"materialized_role_{role}",
                        "expected": repository["repo"],
                        "actual": None,
                    },
                )
            )
        found[role] = Path(str(location))
    return found


def _checkout(entries: Mapping[str, Mapping[str, Any]], identifier: str) -> Path:
    return Path(
        str(_materialized_value(entries, identifier, "checkout", check="installed_checkout"))
    )


def _paths_exist(materialized: Mapping[str, Any]) -> bool:
    found = []
    if materialized.get("path"):
        found.append(Path(str(materialized["path"])))
    values = materialized.get("paths")
    if isinstance(values, Mapping):
        found.extend(Path(str(value)) for value in values.values())
    return bool(found) and all(path.exists() for path in found)
