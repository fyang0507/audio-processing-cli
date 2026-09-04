"""Integrity checks for declared executable products used by transcription stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from audio_cli import paths
from audio_cli.environments import Package
from audio_cli.packages import (
    built_product_candidates,
    checkout_file_matches,
    checkout_patch_expectation,
    managed_checkout_path,
    validated_built_product,
)

from ..refusals import request as refusals
from . import runtime


def _validate_declared_product(
    identifier: str,
    package: Package,
    materialized: Mapping[str, Any],
    failures: list[dict[str, Any]],
    built_product_probe: Callable[[Path], bool],
) -> None:
    """Validate any manifest package that declares an executable product."""
    source = package.source
    product = str(source["product"])
    failures_before = len(failures)
    # `built` and `product_runs` are pull history. The current trust boundary is the
    # live checkout plus the exact executable path and digest recorded after the build.
    checkout, location_issue = managed_checkout_path(package, materialized.get("path"))
    if location_issue is not None or checkout is None:
        failures.append(
            {
                "package": identifier,
                "check": "built_checkout_path",
                "expected": str(paths.checkout_dir(package.environment, package.id)),
                "actual": location_issue,
            }
        )
        candidates = []
    else:
        try:
            state = runtime._inspect_checkout(checkout)
        except ValueError as exc:
            failures.append(
                {
                    "package": identifier,
                    "check": "built_checkout_git_state",
                    "expected": "inspectable pinned source checkout",
                    "actual": str(exc),
                }
            )
        else:
            try:
                expected_patches, expected_names, expected_digests = checkout_patch_expectation(
                    package
                )
            except (OSError, ValueError) as exc:
                failures.append(
                    {
                        "package": identifier,
                        "check": "built_checkout_patch",
                        "expected": "readable installed patch",
                        "actual": str(exc),
                    }
                )
                expected_patches, expected_names, expected_digests = (), (), {}
            if state.head != source["commit"] or set(state.modified) != set(expected_names):
                failures.append(
                    {
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
                    }
                )
            if materialized.get("patches_applied", []) != list(expected_patches):
                failures.append(
                    {
                        "package": identifier,
                        "check": "built_checkout_patch_applied",
                        "expected": list(expected_patches),
                        "actual": materialized.get("patches_applied", []),
                    }
                )
            if materialized.get("patched_file_digests", {}) != expected_digests:
                failures.append(
                    {
                        "package": identifier,
                        "check": "built_checkout_patch_digests",
                        "expected": expected_digests,
                        "actual": materialized.get("patched_file_digests", {}),
                    }
                )
            changed = [
                name
                for name, digest in expected_digests.items()
                if not checkout_file_matches(
                    checkout,
                    name,
                    digest,
                    runtime._checkout_file_digest,
                )
            ]
            if changed:
                failures.append(
                    {
                        "package": identifier,
                        "check": "built_checkout_patch_integrity",
                        "expected": "manifest-pinned post-patch hashes",
                        "actual": changed,
                    }
                )
        candidates = built_product_candidates(checkout, product)
    if len(candidates) != 1:
        failures.append(
            {
                "package": identifier,
                "check": "built_product_executable",
                "expected": 1,
                "actual": len(candidates),
            }
        )
    elif len(failures) == failures_before:
        executable, product_issue = validated_built_product(checkout, product, dict(materialized))
        if product_issue is not None or executable is None:
            failures.append(
                {
                    "package": identifier,
                    "check": "built_product_digest",
                    "expected": "pull-recorded path and sha256",
                    "actual": product_issue,
                }
            )
        elif not built_product_probe(executable):
            raise refusals.package_build_unusable(identifier, product)
