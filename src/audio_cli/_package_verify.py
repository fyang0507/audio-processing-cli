"""Live verification and repair orchestration for provisioned packages."""

from __future__ import annotations

from pathlib import Path

from . import paths
from ._package_compat import facade_dependency
from ._package_core import _now, hub_materialization_issues, sha256_file
from ._package_paths import (
    _checkout_integrity_issues,
    _environment_built_runtime_runs,
    built_product_candidates,
    checkout_file_matches,
    checkout_patch_expectation,
    managed_checkout_path,
    managed_environment_path,
    managed_url_artifact_path,
    validated_built_product,
)
from ._package_registry import load_registry, save_registry
from ._package_requirements import (
    _environment_drift,
    _locked_versions,
    _managed_checkout_requirements,
)
from ._package_teardown import _source_revision_report
from .environments import Package, environments, packages


def _load_registry() -> dict:
    return facade_dependency("load_registry", load_registry)()


def _save_registry(document: dict) -> None:
    facade_dependency("save_registry", save_registry)(document)


def _packages() -> dict[str, Package]:
    return facade_dependency("packages", packages)()


class VerifyMixin:
    def verify(self, *, repair: bool = False) -> dict:
        document = _load_registry()
        catalog = _packages()
        verified: list[dict] = []
        failed: list[dict] = []
        environment_states: dict[str, str] = {}
        built_runtime_probes: dict[str, bool] = {}
        invalid_environment_roots: set[str] = set()
        ready_packages_by_environment: dict[str, list[str]] = {}
        for identifier, package_entry in document["packages"].items():
            package = catalog.get(identifier)
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
                          if self.toolchain.which(tool) is None]
            if blocked_by:
                built_runtime_probes[name] = _environment_built_runtime_runs(
                    name, document, self.toolchain,
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
            frozen = self.toolchain.frozen_packages(paths.env_python(name))
            required_checkouts = _managed_checkout_requirements(document, name)
            drift = _environment_drift(expected, frozen, required_checkouts)
            if drift and repair:
                ready_checkouts: list[tuple[Package, Path]] = []
                checkouts_are_safe = True
                for identifier, package_entry in sorted(document["packages"].items()):
                    package = catalog.get(identifier)
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
                        package, materialized, self.toolchain
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
                    _save_registry(document)
                    self.toolchain.create_environment(environment, paths.env_dir(name))
                    for _package, checkout in ready_checkouts:
                        self.toolchain.install_checkout(
                            paths.env_python(name), checkout
                        )
                        self.toolchain.clean_ignored_checkout(checkout)
                    document["environments"][name]["state"] = "ready"
                    _save_registry(document)
                    frozen = self.toolchain.frozen_packages(paths.env_python(name))
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

        for identifier, entry in sorted(document["packages"].items()):
            if entry.get("state") != "ready":
                failed.append({"package": identifier, "code": "package_not_ready",
                               "detail": f"state is {entry.get('state')!r}",
                               "fix": f"audio packages pull --repair {identifier}"})
                continue
            package = catalog.get(identifier)
            if package is None:
                failed.append({
                    "package": identifier,
                    "code": "package_unknown",
                    "detail": "ready registry entry is not present in the installed manifest",
                    "fix": f"Inspect {paths.registry_path()} and remove the stale entry",
                })
                continue
            # The environment-root verdict owns every path below it.  Once that root is
            # redirected, do not inspect a checkout or launch a product reached through it;
            # the environment failure above is the complete, safe diagnosis.
            if package.environment in invalid_environment_roots:
                continue
            record: dict = {"package": identifier}
            materialized = entry.get("materialized", {})
            if not isinstance(materialized, dict):
                failed.append({
                    "package": identifier,
                    "code": "package_integrity_failed",
                    "detail": "materialized receipt is not an object",
                    "fix": f"audio packages pull --repair {identifier}",
                })
                continue
            # `digest: "ok"` is reserved for the one kind that has something to hash against.
            if package.source["type"] == "url":
                location, location_issue = managed_url_artifact_path(
                    package, materialized.get("path")
                )
                digest_matches = False
                if location_issue is None and location is not None:
                    try:
                        digest_matches = (
                            sha256_file(location) == package.source["sha256"]
                        )
                    except OSError as exc:
                        location_issue = f"could not hash managed artifact {location}: {exc}"
                if digest_matches:
                    record["digest"] = "ok"
                else:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": location_issue or (
                            f"{location} is missing or its digest changed"
                        ),
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
            elif package.source["type"] in {
                "huggingface", "huggingface_multi",
            }:
                issues = hub_materialization_issues(package, materialized)
                if issues:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
                # The receipt is not the pin.  It is mutable local history and may predate a
                # manifest correction; only the manifest revisions can be republished as the
                # revisions this verification actually required above.
                record.update(_source_revision_report(package))
            elif package.source["type"] == "git+build":
                checkout_value = materialized.get("path")
                checkout, location_issue = managed_checkout_path(package, checkout_value)
                product = str(package.source["product"])
                issues: list[str] = []
                if location_issue is not None:
                    issues.append(location_issue)
                elif checkout is None or not checkout.is_dir():
                    issues.append(f"built checkout is not a directory: {checkout}")
                else:
                    try:
                        state = self.toolchain.inspect_checkout(checkout)
                    except ValueError as exc:
                        issues.append(str(exc))
                    else:
                        if state.head != package.source["commit"]:
                            issues.append(
                                f"checkout HEAD is {state.head!r}, expected exact commit "
                                f"{package.source['commit']!r}"
                            )
                        try:
                            expected_patches, expected_names, expected_digests = (
                                checkout_patch_expectation(package)
                            )
                        except (OSError, ValueError) as exc:
                            issues.append(f"shipped checkout patch cannot be read: {exc}")
                            expected_patches, expected_names, expected_digests = (), (), {}
                        if set(state.modified) != set(expected_names):
                            issues.append(
                                f"built checkout tracked changes are "
                                f"{list(state.modified)!r}, expected "
                                f"{sorted(expected_names)!r}"
                            )
                        applied = materialized.get("patches_applied", [])
                        if applied != list(expected_patches):
                            issues.append(
                                f"recorded patches are {applied!r}, expected exactly "
                                f"{list(expected_patches)!r}"
                            )
                        recorded_digests = materialized.get(
                            "patched_file_digests", {}
                        )
                        if recorded_digests != expected_digests:
                            issues.append(
                                "recorded patched-file digests differ from manifest"
                            )
                        changed = [
                            name for name, digest in expected_digests.items()
                            if not checkout_file_matches(
                                checkout, name, digest, self.toolchain.file_digest
                            )
                        ]
                        if changed:
                            issues.append(
                                f"live patched-file hashes changed for {sorted(changed)!r}"
                            )
                    candidates = built_product_candidates(checkout, product)
                    if len(candidates) != 1:
                        issues.append(
                            f"expected one executable built product {product!r}, found "
                            f"{len(candidates)}"
                        )
                    else:
                        _executable, product_issue = validated_built_product(
                            checkout, product, materialized
                        )
                        if product_issue is not None:
                            issues.append(product_issue)
                if issues:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
                if package.environment in built_runtime_probes:
                    product_runs = built_runtime_probes[package.environment]
                else:
                    try:
                        product_runs = self.toolchain.built_product_runs(candidates[0])
                    except OSError:
                        product_runs = False
                if not product_runs:
                    failed.append({
                        "package": identifier, "code": "package_build_unusable",
                        "detail": f"built product {product!r} does not run",
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
                record["product_runs"] = True
                record["product_digest"] = "ok"
                record["patches_applied"] = list(expected_patches)
            else:
                locations = [Path(p) for p in (
                    [materialized["path"]] if materialized.get("path")
                    else list((materialized.get("paths") or {}).values()))]
                gone = [str(location) for location in locations if not location.exists()]
                if gone:
                    # The shared-cache consequence: another root's purge, or a manual cache
                    # clear, can take weights out from under a root that still calls them
                    # ready. Better an exit 3 with a fix than a stack that fails mid-run.
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": f"materialized path(s) no longer exist: {', '.join(gone)}",
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue

            if package.checkout is not None:
                issues = _checkout_integrity_issues(package, materialized, self.toolchain)
                if issues:
                    failed.append({
                        "package": identifier, "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    })
                    continue
                record["patches_applied"] = materialized.get("patches_applied", [])
            verified.append(record)

        report: dict = {"verified": verified, "environments": environment_states,
                        "failed": failed}
        report.update(self._verify_mlx_guard(document))
        return report
