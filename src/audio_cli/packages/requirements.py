"""Environment lock and managed-checkout requirement helpers."""

from __future__ import annotations

import os
import re
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import paths
from ..environments import Environment
from . import catalog


def _module_available(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _module_of(dotted: str) -> str:
    return dotted.rsplit(".", 2)[0]


def _class_of(dotted: str) -> str:
    return dotted.rsplit(".", 2)[1]


def _method_of(dotted: str) -> str:
    return dotted.rsplit(".", 1)[1]


def _locked_versions(environment: Environment) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in environment.lock.read_text().splitlines():
        if line.startswith((" ", "#")) or "==" not in line:
            continue
        name, _, rest = line.partition("==")
        versions[_distribution_name(name)] = rest.split()[0].strip(" \\")
    return versions


def _distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower())


def _direct_file_install_path(specification: str) -> Path | None:
    if not specification.startswith("@ "):
        return None
    try:
        parsed = urllib.parse.urlsplit(specification[2:].strip())
    except ValueError:
        return None
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return Path(urllib.parse.unquote(parsed.path))


def _managed_checkout_requirements(
    document: Mapping[str, Any],
    environment_name: str,
    package_ids: set[str] | None = None,
) -> dict[str, Path]:
    """Required direct installs, keyed by manifest-pinned distribution name."""
    package_catalog = catalog.packages()
    entries = document.get("packages", {})
    if not isinstance(entries, Mapping):
        return {}
    required: dict[str, Path] = {}
    for identifier, entry in entries.items():
        if package_ids is not None and identifier not in package_ids:
            continue
        package = package_catalog.get(identifier)
        if (
            not isinstance(entry, Mapping)
            or entry.get("state") != "ready"
            or package is None
            or package.environment != environment_name
            or package.checkout is None
        ):
            continue
        distribution = _distribution_name(str(package.checkout["distribution"]))
        required[distribution] = Path(
            os.path.abspath(paths.checkout_dir(package.environment, identifier))
        )
    return required


def _environment_drift(
    expected: dict[str, str],
    frozen: dict[str, str],
    required_checkouts: Mapping[str, Path],
) -> dict[str, tuple[str | None, str | None]]:
    comparable = dict(frozen)
    direct_drift = _checkout_install_drift(frozen, required_checkouts)
    for name in required_checkouts:
        comparable.pop(name, None)
    locked_drift = {
        name: (expected.get(name), comparable.get(name))
        for name in expected.keys() | comparable.keys()
        if expected.get(name) != comparable.get(name)
    }
    return {**locked_drift, **direct_drift}


def _checkout_install_drift(
    frozen: Mapping[str, str],
    required_checkouts: Mapping[str, Path],
) -> dict[str, tuple[str | None, str | None]]:
    drift: dict[str, tuple[str | None, str | None]] = {}
    for name, required_path in required_checkouts.items():
        installed = frozen.get(name)
        direct_path = _direct_file_install_path(installed) if installed is not None else None
        if direct_path is None or Path(os.path.abspath(direct_path)) != required_path:
            drift[name] = (f"@ {required_path.as_uri()}", installed)
    return drift
