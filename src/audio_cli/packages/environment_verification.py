"""Verify provisioned runtime environments and repair lock drift."""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..environments import Package, environments
from . import registry
from .checkouts import _checkout_integrity_issues
from .integrity import _now, sha256_file
from .locations import managed_checkout_path, managed_environment_path
from .products import _environment_built_runtime_runs
from .requirements import (
    _environment_drift,
    _locked_versions,
    _managed_checkout_requirements,
)
from .toolchain import Toolchain


def verify_environments(
    toolchain: Toolchain,
    document: dict,
    package_catalog: dict[str, Package],
    *,
    repair: bool,
) -> tuple[dict[str, str], dict[str, bool], set[str], list[dict]]:
    failed: list[dict] = []
    environment_states: dict[str, str] = {}
    built_runtime_probes: dict[str, bool] = {}
    invalid_environment_roots: set[str] = set()
    ready_packages_by_environment: dict[str, list[str]] = {}
    for identifier, package_entry in document["packages"].items():
        package = package_catalog.get(identifier)
        if package is not None and package_entry.get("state") == "ready":
            ready_packages_by_environment.setdefault(package.environment, []).append(
                identifier
            )

    for name, environment in environments().items():
        if not environment.provisioned:
            continue
        entry = document["environments"].get(name)
        if entry is None or entry.get("state") != "ready":
            environment_states[name] = "absent"
            dependents = sorted(ready_packages_by_environment.get(name, ()))
            if dependents:
                invalid_environment_roots.add(name)
                actual_state = entry.get("state") if isinstance(entry, dict) else None
                failed.append({
                    "environment": name,
                    "code": "environment_not_ready",
                    "detail": (
                        f"registry state is {actual_state or 'absent'!r} while ready "
                        f"package(s) depend on it: {', '.join(dependents)}"
                    ),
                    "packages": dependents,
                    "fix": (
                        "audio packages pull --repair " + " ".join(dependents)
                    ),
                })
            continue
        environment_path, environment_path_issue = managed_environment_path(name)
        if environment_path_issue is not None:
            invalid_environment_roots.add(name)
            environment_states[name] = "drifted"
            failed.append({
                "environment": name,
                "code": "environment_drifted",
                "detail": environment_path_issue,
                "examples": {
                    "environment_root": {
                        "locked": str(paths.env_dir(name)),
                        "installed": str(environment_path),
                    }
                },
                "fix": (
                    f"Replace the redirected environment path {paths.env_dir(name)} and "
                    "run audio packages verify --repair"
                ),
            })
            continue
        # Swift is a provisioning dependency until its product exists.  Once the executable
        # is ready, runtime and verify launch it directly, so removing Swift from PATH must
        # not turn a still-runnable environment into `blocked`.
        blocked_by = [tool for tool in environment.requires_tool
                      if toolchain.which(tool) is None]
        if blocked_by:
            built_runtime_probes[name] = _environment_built_runtime_runs(
                name, document, toolchain,
            )
            if not built_runtime_probes[name]:
                environment_states[name] = "blocked"
                continue
        if not environment.has_interpreter:
            # No interpreter means no lock to compare against. Its packages carry the
            # checks that apply — that the product builds and runs — so reporting `ok`
            # here says the directory exists, and nothing more.
            environment_states[name] = "ok"
            continue
        expected = _locked_versions(environment)
        frozen = toolchain.frozen_packages(paths.env_python(name))
        required_checkouts = _managed_checkout_requirements(document, name)
        drift = _environment_drift(expected, frozen, required_checkouts)
        if drift and repair:
            ready_checkouts: list[tuple[Package, Path]] = []
            checkouts_are_safe = True
            for identifier, package_entry in sorted(document["packages"].items()):
                package = package_catalog.get(identifier)
                if (
                    package is None
                    or package.environment != name
                    or package.checkout is None
                    or package_entry.get("state") != "ready"
                ):
                    continue
                materialized = package_entry.get("materialized", {})
                if not isinstance(materialized, dict):
                    checkouts_are_safe = False
                    continue
                issues = _checkout_integrity_issues(
                    package, materialized, toolchain
                )
                checkout, checkout_issue = managed_checkout_path(
                    package, materialized.get("checkout")
                )
                if issues or checkout_issue is not None or checkout is None:
                    checkouts_are_safe = False
                    continue
                ready_checkouts.append((package, checkout))
            if checkouts_are_safe:
                lock_digest = sha256_file(environment.lock)
                document["environments"][name] = {
                    "state": "creating",
                    "path": str(paths.env_dir(name)),
                    "python": environment.python,
                    "lock_sha256": lock_digest,
                    "created_utc": _now(),
                }
                registry.save_registry(document)
                toolchain.create_environment(environment, paths.env_dir(name))
                for _package, checkout in ready_checkouts:
                    toolchain.install_checkout(
                        paths.env_python(name), checkout
                    )
                    toolchain.clean_ignored_checkout(checkout)
                document["environments"][name]["state"] = "ready"
                registry.save_registry(document)
                frozen = toolchain.frozen_packages(paths.env_python(name))
                drift = _environment_drift(
                    expected, frozen, required_checkouts
                )
        if drift:
            environment_states[name] = "drifted"
            failed.append({
                "environment": name, "code": "environment_drifted",
                "detail": f"{len(drift)} package(s) differ from {environment.lock.name}",
                "examples": {n: {"locked": drift[n][0], "installed": drift[n][1]}
                             for n in sorted(drift)[:5]},
                "fix": "audio packages verify --repair",
            })
        else:
            environment_states[name] = "ok"

    return environment_states, built_runtime_probes, invalid_environment_roots, failed
