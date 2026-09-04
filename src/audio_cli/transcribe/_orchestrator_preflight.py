"""Preflight package and runtime integrity before transcription model work."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from audio_cli import paths
from audio_cli.environments import environments as environment_catalog
from audio_cli.environments import packages as package_catalog
from audio_cli.packages import (
    _checkout_install_drift,
    _managed_checkout_requirements,
    built_product_candidates,
    checkout_file_matches,
    checkout_patch_expectation,
    hub_materialization_issues,
    managed_checkout_path,
    managed_environment_path,
    managed_provisioning_root_issue,
    managed_url_artifact_path,
    validated_built_product,
)

from . import refusals
from .plan import Plan

def preflight(
    plan: Plan,
    registry: Mapping[str, Any],
    *,
    built_product_probe: Callable[[Path], bool] | None = None,
    python_runtime_probe: Callable[[Path], bool] | None = None,
    frozen_packages_probe: Callable[[Path], dict[str, str]] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Fail before decode or model load if the selected materialization is not usable."""
    # Resolve through the compatibility facade so existing probe monkeypatches remain effective.
    from . import orchestrator as core

    _CheckoutState = core._CheckoutState
    _built_product_runs = core._built_product_runs
    _checkout_file_digest = core._checkout_file_digest
    _frozen_packages = core._frozen_packages
    _inspect_checkout = core._inspect_checkout
    _missing_record = core._missing_record
    _paths_exist = core._paths_exist
    _python_runtime_runs = core._python_runtime_runs
    hash_file = core.hash_file
    built_product_probe = built_product_probe or _built_product_runs
    python_runtime_probe = python_runtime_probe or _python_runtime_runs
    frozen_packages_probe = frozen_packages_probe or _frozen_packages
    entries = registry.get("packages", {})
    if not isinstance(entries, Mapping):
        entries = {}
    missing = []
    for record in plan.packages:
        if record.get("auto_fetch"):
            continue
        entry = entries.get(record["package"])
        if not isinstance(entry, Mapping) or entry.get("state") != "ready":
            missing.append(_missing_record(record))
    if missing:
        known = sum(int(item["bytes"] or 0) for item in missing)
        unsized = [str(item["package"]) for item in missing if item["bytes"] is None]
        raise refusals.packages_not_provisioned(plan.stack, missing, known, unsized)

    failures = []
    catalog = package_catalog()
    environment_entries = registry.get("environments", {})
    if not isinstance(environment_entries, Mapping):
        environment_entries = {}
    runtime_packages: dict[str, str] = {}
    unusable_environments: set[str] = set()
    for record in plan.packages:
        if record.get("auto_fetch"):
            continue
        package = catalog[str(record["package"])]
        if package.environment != "core":
            runtime_packages.setdefault(package.environment, package.id)
    for environment, package_id in runtime_packages.items():
        entry = environment_entries.get(environment, {})
        actual_state = entry.get("state") if isinstance(entry, Mapping) else None
        if actual_state != "ready":
            unusable_environments.add(environment)
            failures.append({
                "package": package_id,
                "check": f"environment_{environment}_ready",
                "expected": "ready",
                "actual": actual_state or "absent",
            })
            continue
        _environment_path, environment_path_issue = managed_environment_path(environment)
        if environment_path_issue is not None:
            unusable_environments.add(environment)
            failures.append({
                "package": package_id,
                "check": f"environment_{environment}_managed_root",
                "expected": str(paths.env_dir(environment)),
                "actual": environment_path_issue,
            })
            continue
        if environment_catalog()[environment].has_interpreter:
            interpreter = paths.env_python(environment)
            if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                unusable_environments.add(environment)
                failures.append({
                    "package": package_id,
                    "check": f"environment_{environment}_python_executable",
                    "expected": True,
                    "actual": False,
                })
            elif not python_runtime_probe(interpreter):
                unusable_environments.add(environment)
                failures.append({
                    "package": package_id,
                    "check": f"environment_{environment}_python_runs",
                    "expected": True,
                    "actual": False,
                })
            else:
                selected_ids = {
                    str(record["package"])
                    for record in plan.packages
                    if not record.get("auto_fetch")
                }
                requirements = _managed_checkout_requirements(
                    registry, environment, selected_ids
                )
                if requirements:
                    install_drift = _checkout_install_drift(
                        frozen_packages_probe(interpreter), requirements
                    )
                    if install_drift:
                        unusable_environments.add(environment)
                        failures.append({
                            "package": package_id,
                            "check": f"environment_{environment}_checkout_install",
                            "expected": {
                                name: expected
                                for name, (expected, _actual) in install_drift.items()
                            },
                            "actual": {
                                name: actual
                                for name, (_expected, actual) in install_drift.items()
                            },
                        })
    selected: dict[str, Mapping[str, Any]] = {}
    for record in plan.packages:
        identifier = str(record["package"])
        if record.get("auto_fetch"):
            source = catalog[identifier].source
            if source.get("type") == "url":
                override = os.environ.get("AUDIO_PROCESSING_VAD_MODEL")
                location = (
                    Path(override).expanduser()
                    if override else paths.models_dir() / str(source["filename"])
                )
                location_issue = None
                if not override:
                    root_issue = managed_provisioning_root_issue()
                    if root_issue is not None:
                        location_issue = root_issue
                    elif location.parent.is_symlink():
                        location_issue = (
                            f"managed artifact parent is a symlink: {location.parent}"
                        )
                    elif location.exists() or location.is_symlink():
                        _managed, location_issue = managed_url_artifact_path(
                            catalog[identifier], location
                        )
                digest_changed = False
                if location.exists() and location.is_file():
                    try:
                        digest_changed = hash_file(location) != source["sha256"]
                    except OSError as exc:
                        location_issue = (
                            f"could not hash managed artifact {location}: {exc}"
                        )
                unusable = (
                    (bool(override) and not location.is_file())
                    or location_issue is not None
                    or (
                        location.exists()
                        and (
                            not location.is_file()
                            or digest_changed
                        )
                    )
                )
                if unusable:
                    failures.append({
                        "package": identifier,
                        "check": "url_artifact_sha256",
                        "expected": source["sha256"],
                        "actual": location_issue or (
                            "missing, not a file, or digest changed"
                        ),
                    })
            continue
        entry = entries[identifier]
        assert isinstance(entry, Mapping)
        selected[identifier] = entry
        # An unusable environment is already diagnosed above.  Do not use any path
        # beneath it for Git inspection, hashing, or product launch.
        if catalog[identifier].environment in unusable_environments:
            continue
        materialized = entry.get("materialized", {})
        if not isinstance(materialized, Mapping) or not _paths_exist(materialized):
            failures.append({
                "package": identifier,
                "check": "materialized_paths_exist",
                "expected": True,
                "actual": False,
            })
            continue
        source = catalog[identifier].source
        if source.get("type") in {"huggingface", "huggingface_multi"}:
            hub_issues = hub_materialization_issues(
                catalog[identifier], dict(materialized)
            )
            if hub_issues:
                failures.append({
                    "package": identifier,
                    "check": "hub_snapshot_integrity",
                    "expected": (
                        "snapshot directories, manifest-filtered files, and "
                        "recorded byte size"
                    ),
                    "actual": hub_issues,
                })
        # Mutable receipt revision fields are history, not the pin.  Hub packages are
        # bound above by manifest repository/revision -> live cache-indexed snapshot;
        # source builds are bound below by the live checkout HEAD.
        checkout_spec = catalog[identifier].checkout
        if checkout_spec is not None:
            checkout_value = materialized.get("checkout")
            checkout, location_issue = managed_checkout_path(
                catalog[identifier], checkout_value
            )
            if location_issue is not None or checkout is None or not checkout.is_dir():
                failures.append({
                    "package": identifier,
                    "check": "checkout_directory_exists",
                    "expected": "exact managed non-symlink checkout directory",
                    "actual": location_issue or False,
                })
                # Keep receipt validation below, but never inspect or hash the candidate
                # path after its managed identity or directory kind failed.
                checkout = None
            expected_commit = checkout_spec.get("resolved_commit")
            if not isinstance(expected_commit, str) or len(expected_commit) != 40:
                failures.append({
                    "package": identifier,
                    "check": "checkout_resolved_commit",
                    "expected": "full 40-hex Git SHA",
                    "actual": expected_commit,
                })
                expected_commit = str(expected_commit)
            actual_commit = materialized.get("checkout_commit")
            accepted_commits = (checkout_spec.get("commit"), expected_commit)
            if actual_commit not in accepted_commits:
                failures.append({
                    "package": identifier,
                    "check": "checkout_commit",
                    "expected": list(accepted_commits),
                    "actual": actual_commit,
                })
            checkout_state: _CheckoutState | None = None
            if checkout is not None and checkout.is_dir():
                try:
                    checkout_state = _inspect_checkout(checkout)
                except ValueError as exc:
                    failures.append({
                        "package": identifier,
                        "check": "checkout_git_state",
                        "expected": "inspectable pinned source checkout",
                        "actual": str(exc),
                    })
                else:
                    if checkout_state.head != expected_commit:
                        failures.append({
                            "package": identifier,
                            "check": "checkout_head",
                            "expected": expected_commit,
                            "actual": checkout_state.head,
                        })
            try:
                expected_patches, expected_names, expected_digests = (
                    checkout_patch_expectation(catalog[identifier])
                )
            except (OSError, ValueError) as exc:
                failures.append({
                    "package": identifier,
                    "check": "shipped_checkout_patch",
                    "expected": "readable installed patch",
                    "actual": str(exc),
                })
                expected_patches, expected_names, expected_digests = (), (), {}
            applied = materialized.get("patches_applied", [])
            if not isinstance(applied, list) or applied != list(expected_patches):
                failures.append({
                    "package": identifier,
                    "check": "checkout_patch_applied",
                    "expected": list(expected_patches),
                    "actual": applied,
                })
            digests = materialized.get("patched_file_digests", {})
            if not isinstance(digests, Mapping) or dict(digests) != expected_digests:
                failures.append({
                    "package": identifier,
                    "check": "patched_file_digests",
                    "expected": expected_digests,
                    "actual": dict(digests) if isinstance(digests, Mapping) else digests,
                })
            if checkout is not None and checkout.is_dir():
                changed: list[str] = []
                for name, expected_digest in expected_digests.items():
                    if not checkout_file_matches(
                        checkout, name, expected_digest, _checkout_file_digest
                    ):
                        changed.append(name)
                if changed:
                    failures.append({
                        "package": identifier,
                        "check": "checkout_patch_integrity",
                        "expected": "manifest-pinned post-patch hashes",
                        "actual": changed,
                    })
            if checkout_state is not None:
                if set(checkout_state.modified) != set(expected_names):
                    failures.append({
                        "package": identifier,
                        "check": "checkout_tracked_changes",
                        "expected": sorted(expected_names),
                        "actual": list(checkout_state.modified),
                    })
                if checkout_state.untracked:
                    failures.append({
                        "package": identifier,
                        "check": "checkout_untracked_files",
                        "expected": [],
                        "actual": list(checkout_state.untracked),
                    })
        if identifier == "fluidaudio":
            fluid_failures_before = len(failures)
            product = str(source["product"])
            # `built` and `product_runs` are pull history. The current trust boundary is the
            # live checkout plus the exact executable path and digest recorded after the build.
            package = catalog[identifier]
            checkout, location_issue = managed_checkout_path(
                package, materialized.get("path")
            )
            if location_issue is not None or checkout is None:
                failures.append({
                    "package": identifier,
                    "check": "built_checkout_path",
                    "expected": str(paths.checkout_dir(package.environment, package.id)),
                    "actual": location_issue,
                })
                candidates = []
            else:
                try:
                    state = _inspect_checkout(checkout)
                except ValueError as exc:
                    failures.append({
                        "package": identifier,
                        "check": "built_checkout_git_state",
                        "expected": "inspectable pinned source checkout",
                        "actual": str(exc),
                    })
                else:
                    try:
                        expected_patches, expected_names, expected_digests = (
                            checkout_patch_expectation(package)
                        )
                    except (OSError, ValueError) as exc:
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_patch",
                            "expected": "readable installed patch",
                            "actual": str(exc),
                        })
                        expected_patches, expected_names, expected_digests = (), (), {}
                    if (
                        state.head != source["commit"]
                        or set(state.modified) != set(expected_names)
                    ):
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_git_state",
                            "expected": {
                                "head": source["commit"],
                                "modified": sorted(expected_names),
                            },
                            "actual": {
                                "head": state.head,
                                "modified": list(state.modified),
                            },
                        })
                    if materialized.get("patches_applied", []) != list(
                        expected_patches
                    ):
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_patch_applied",
                            "expected": list(expected_patches),
                            "actual": materialized.get("patches_applied", []),
                        })
                    if materialized.get("patched_file_digests", {}) != expected_digests:
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_patch_digests",
                            "expected": expected_digests,
                            "actual": materialized.get("patched_file_digests", {}),
                        })
                    changed = [
                        name for name, digest in expected_digests.items()
                        if not checkout_file_matches(
                            checkout, name, digest, _checkout_file_digest
                        )
                    ]
                    if changed:
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_patch_integrity",
                            "expected": "manifest-pinned post-patch hashes",
                            "actual": changed,
                        })
                candidates = built_product_candidates(checkout, product)
            if len(candidates) != 1:
                failures.append({
                    "package": identifier,
                    "check": "built_product_executable",
                    "expected": 1,
                    "actual": len(candidates),
                })
            elif len(failures) == fluid_failures_before:
                executable, product_issue = validated_built_product(
                    checkout, product, dict(materialized)
                )
                if product_issue is not None or executable is None:
                    failures.append({
                        "package": identifier,
                        "check": "built_product_digest",
                        "expected": "pull-recorded path and sha256",
                        "actual": product_issue,
                    })
                elif not built_product_probe(executable):
                    raise refusals.package_build_unusable(identifier, product)
    if failures:
        raise refusals.package_integrity_failed(failures)
    return selected
