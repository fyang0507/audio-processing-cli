"""Built-product discovery, receipt validation, and runtime probes."""

from __future__ import annotations

import os
import re
from pathlib import Path

from . import catalog
from .checkouts import checkout_file_matches, checkout_patch_expectation
from .integrity import sha256_file
from .locations import managed_checkout_path
from .toolchain import Toolchain


def built_product_candidates(checkout: Path, product: str) -> list[Path]:
    """Contained non-symlink executable products from the pinned release build."""
    try:
        checkout_root = checkout.resolve(strict=True)
    except (OSError, RuntimeError):
        return []
    found: set[Path] = set()
    for path in checkout.glob(f".build/**/release/{product}"):
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if (
            path.is_file()
            and not path.is_symlink()
            and resolved.is_relative_to(checkout_root)
            and os.access(path, os.X_OK)
        ):
            found.add(resolved)
    return sorted(found)


def validated_built_product(
    checkout: Path,
    product: str,
    materialized: dict,
) -> tuple[Path | None, str | None]:
    """Bind the launchable product to the exact bytes recorded after `pull` built it."""
    recorded_path = materialized.get("product_path")
    recorded_digest = materialized.get("product_sha256")
    if not isinstance(recorded_path, str) or not recorded_path:
        return None, "built product receipt has no non-empty product_path"
    if (
        not isinstance(recorded_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", recorded_digest) is None
    ):
        return None, "built product receipt has no valid product_sha256"
    candidates = built_product_candidates(checkout, product)
    if len(candidates) != 1:
        return None, (f"expected one executable built product {product!r}, found {len(candidates)}")
    executable = candidates[0]
    try:
        relative = executable.relative_to(checkout.resolve(strict=True)).as_posix()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"built product path cannot be bound to its checkout: {exc}"
    if relative != recorded_path:
        return None, (
            f"built product path is {relative!r}, expected receipt path {recorded_path!r}"
        )
    try:
        actual_digest = sha256_file(executable)
    except OSError as exc:
        return None, f"built product digest could not be read: {exc}"
    if actual_digest != recorded_digest:
        return None, (
            f"built product sha256 is {actual_digest!r}, expected receipt digest "
            f"{recorded_digest!r}"
        )
    return executable, None


def _environment_has_built_runtime(name: str, document: dict) -> bool:
    """Whether an environment can run without the tool that provisioned its product."""
    if name != "swift":
        return False
    entry = document.get("packages", {}).get("fluidaudio", {})
    if entry.get("state") != "ready":
        return False
    materialized = entry.get("materialized", {})
    checkout_value = materialized.get("path")
    if not checkout_value:
        return False
    package = catalog.packages()["fluidaudio"]
    checkout, issue = managed_checkout_path(package, checkout_value)
    if issue is not None or checkout is None:
        return False
    executable, product_issue = validated_built_product(
        checkout, str(package.source["product"]), materialized
    )
    return product_issue is None and executable is not None


def _environment_built_runtime_runs(
    name: str,
    document: dict,
    toolchain: Toolchain,
) -> bool:
    """Live runtime exemption from a missing provisioning tool."""
    if not _environment_has_built_runtime(name, document):
        return False
    entry = document["packages"]["fluidaudio"]
    package = catalog.packages()["fluidaudio"]
    checkout, issue = managed_checkout_path(package, entry["materialized"].get("path"))
    if issue is not None or checkout is None:
        return False
    try:
        state = toolchain.inspect_checkout(checkout)
    except ValueError:
        return False
    try:
        expected_patches, expected_names, expected_digests = checkout_patch_expectation(package)
    except (OSError, ValueError):
        return False
    materialized = entry["materialized"]
    if (
        state.head != package.source["commit"]
        or set(state.modified) != set(expected_names)
        or materialized.get("patches_applied", []) != list(expected_patches)
        or materialized.get("patched_file_digests", {}) != expected_digests
        or any(
            not checkout_file_matches(checkout, name, digest, toolchain.file_digest)
            for name, digest in expected_digests.items()
        )
    ):
        return False
    product = str(package.source["product"])
    executable, product_issue = validated_built_product(checkout, product, materialized)
    if product_issue is not None or executable is None:
        return False
    try:
        return toolchain.built_product_runs(executable)
    except OSError:
        return False


__all__ = ["built_product_candidates", "validated_built_product"]
