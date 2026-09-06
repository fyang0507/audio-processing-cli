"""Verify live package artifacts against manifest and registry receipts."""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..environments import Package
from .artifacts import verify_artifact_file
from .checkouts import (
    _checkout_integrity_issues,
    checkout_file_matches,
    checkout_patch_expectation,
)
from .integrity import hub_materialization_issues
from .locations import managed_checkout_path
from .models import ProvisioningError
from .products import built_product_candidates, validated_built_product
from .teardown import _source_revision_report
from .toolchain import Toolchain


def verify_packages(
    toolchain: Toolchain,
    document: dict,
    package_catalog: dict[str, Package],
    built_runtime_probes: dict[str, bool],
    invalid_environment_roots: set[str],
) -> tuple[list[dict], list[dict]]:
    verified: list[dict] = []
    failed: list[dict] = []
    for identifier, entry in sorted(document["packages"].items()):
        if entry.get("state") != "ready":
            failed.append(
                {
                    "package": identifier,
                    "code": "package_not_ready",
                    "detail": f"state is {entry.get('state')!r}",
                    "fix": f"audio packages pull --repair {identifier}",
                }
            )
            continue
        package = package_catalog.get(identifier)
        if package is None:
            failed.append(
                {
                    "package": identifier,
                    "code": "package_unknown",
                    "detail": "ready registry entry is not present in the installed manifest",
                    "fix": f"Inspect {paths.registry_path()} and remove the stale entry",
                }
            )
            continue
        # The environment-root verdict owns every path below it.  Once that root is
        # redirected, do not inspect a checkout or launch a product reached through it;
        # the environment failure above is the complete, safe diagnosis.
        if package.environment in invalid_environment_roots:
            continue
        record: dict = {"package": identifier}
        materialized = entry.get("materialized", {})
        if not isinstance(materialized, dict):
            failed.append(
                {
                    "package": identifier,
                    "code": "package_integrity_failed",
                    "detail": "materialized receipt is not an object",
                    "fix": f"audio packages pull --repair {identifier}",
                }
            )
            continue
        # Git blob identity is reported explicitly, separately from URL SHA-256 verdicts.
        if package.source["type"] in {"url", "git-blob"}:
            try:
                _path, verdict, _provenance = verify_artifact_file(
                    package, materialized.get("path")
                )
            except ProvisioningError as exc:
                failed.append(exc.as_dict())
                continue
            record.update(verdict)
        elif package.source["type"] in {
            "huggingface",
            "huggingface_multi",
        }:
            issues = hub_materialization_issues(package, materialized)
            if issues:
                failed.append(
                    {
                        "package": identifier,
                        "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    }
                )
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
                    state = toolchain.inspect_checkout(checkout)
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
                    recorded_digests = materialized.get("patched_file_digests", {})
                    if recorded_digests != expected_digests:
                        issues.append("recorded patched-file digests differ from manifest")
                    changed = [
                        name
                        for name, digest in expected_digests.items()
                        if not checkout_file_matches(checkout, name, digest, toolchain.file_digest)
                    ]
                    if changed:
                        issues.append(f"live patched-file hashes changed for {sorted(changed)!r}")
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
                failed.append(
                    {
                        "package": identifier,
                        "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    }
                )
                continue
            if package.environment in built_runtime_probes:
                product_runs = built_runtime_probes[package.environment]
            else:
                try:
                    product_runs = toolchain.built_product_runs(candidates[0])
                except OSError:
                    product_runs = False
            if not product_runs:
                failed.append(
                    {
                        "package": identifier,
                        "code": "package_build_unusable",
                        "detail": f"built product {product!r} does not run",
                        "fix": f"audio packages pull --repair {identifier}",
                    }
                )
                continue
            record["product_runs"] = True
            record["product_digest"] = "ok"
            record["patches_applied"] = list(expected_patches)
        else:
            locations = [
                Path(p)
                for p in (
                    [materialized["path"]]
                    if materialized.get("path")
                    else list((materialized.get("paths") or {}).values())
                )
            ]
            gone = [str(location) for location in locations if not location.exists()]
            if gone:
                # The shared-cache consequence: another root's purge, or a manual cache
                # clear, can take weights out from under a root that still calls them
                # ready. Better an exit 3 with a fix than a stack that fails mid-run.
                failed.append(
                    {
                        "package": identifier,
                        "code": "package_integrity_failed",
                        "detail": f"materialized path(s) no longer exist: {', '.join(gone)}",
                        "fix": f"audio packages pull --repair {identifier}",
                    }
                )
                continue

        if package.checkout is not None:
            issues = _checkout_integrity_issues(package, materialized, toolchain)
            if issues:
                failed.append(
                    {
                        "package": identifier,
                        "code": "package_integrity_failed",
                        "detail": "; ".join(issues),
                        "fix": f"audio packages pull --repair {identifier}",
                    }
                )
                continue
            record["patches_applied"] = materialized.get("patches_applied", [])
        verified.append(record)

    return verified, failed
