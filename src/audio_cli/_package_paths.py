"""Managed-path, checkout-integrity, patch, and built-product validation."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from . import paths
from ._package_compat import facade_dependency
from ._package_core import ProvisioningError, sha256_file
from ._package_runtime import Toolchain
from .environments import HERE as ENVIRONMENTS_DIR
from .environments import Package, packages

def _patch_touched_names(patch: Path) -> tuple[str, ...]:
    """Repository-relative files touched by one shipped unified diff."""
    touched: list[str] = []
    for line in patch.read_text(errors="replace").splitlines():
        if line.startswith("+++ ") and not line.endswith("/dev/null"):
            target = line[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            touched.append(target)
    return tuple(touched)


def _patched_files(patch: Path, checkout: Path) -> list[Path]:
    """Files a unified diff touches, so verify can detect a reverted patch."""
    return [checkout / name for name in _patch_touched_names(patch)]


def checkout_patch_expectation(
    package: Package,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    """Exact receipt, tracked-file set, and post-patch hashes owned by the manifest.

    The registry is mutable history, not an integrity root.  The manifest binds the
    exact post-patch bytes independently of Git diff presentation and receipt values.
    """
    specification = (
        package.source
        if package.source.get("type") == "git+build"
        else package.checkout
    )
    if specification is None:
        return (), (), {}
    expected = dict(specification.get("patched_file_sha256", {}))
    patch_name = specification.get("patch")
    if not patch_name:
        return (), (), expected
    patch = ENVIRONMENTS_DIR / str(patch_name)
    names = _patch_touched_names(patch)
    if set(expected) != set(names):
        raise ValueError(
            f"manifest patched_file_sha256 paths {sorted(expected)!r} do not equal "
            f"shipped patch targets {sorted(names)!r}"
        )
    return (Path(str(patch_name)).name,), names, expected


def materialize_checkout_patch(
    package: Package,
    checkout: Path,
    toolchain: Toolchain,
) -> tuple[list[str], dict[str, str]]:
    """Apply and prove the manifest-owned patch before install or build executes."""
    expected_patches, expected_names, expected_digests = (
        checkout_patch_expectation(package)
    )
    applied: list[str] = []
    digests: dict[str, str] = {}
    if expected_patches:
        specification = (
            package.source
            if package.source.get("type") == "git+build"
            else package.checkout
        )
        assert specification is not None
        patch_name = str(specification["patch"])
        patch = ENVIRONMENTS_DIR / patch_name
        if not patch.is_file():
            raise ProvisioningError(
                "patch_missing", f"{patch} is not in the installed wheel",
                patch=patch_name, package=package.id,
            )
        toolchain.apply_patch(checkout, patch)
        applied.append(Path(patch_name).name)
        for touched in _patched_files(patch, checkout):
            if touched.is_file():
                digests[str(touched.relative_to(checkout))] = (
                    toolchain.file_digest(touched)
                )
    if (
        applied != list(expected_patches)
        or set(digests) != set(expected_names)
        or digests != expected_digests
    ):
        raise ProvisioningError(
            "patch_integrity_failed",
            f"{package.id} did not materialize the manifest-pinned patched file hashes",
            package=package.id,
            expected=expected_digests,
            actual=digests,
            fix=f"audio packages pull --repair {package.id}",
        )
    return applied, digests


def managed_checkout_path(
    package: Package, value: object,
) -> tuple[Path | None, str | None]:
    """Resolve only the manifest-derived checkout, rejecting symlink/parent escapes."""
    expected = paths.checkout_dir(package.environment, package.id)
    candidate = Path(str(value)) if value else None
    if candidate is None:
        return None, "checkout path is absent"
    _environment_path, environment_issue = managed_environment_path(package.environment)
    if environment_issue is not None:
        return candidate, f"package environment is not managed: {environment_issue}"
    if Path(os.path.abspath(candidate)) != Path(os.path.abspath(expected)):
        return candidate, f"checkout path {candidate} is not managed path {expected}"
    if candidate.is_symlink():
        return candidate, f"managed checkout path is a symlink: {candidate}"
    try:
        resolved = candidate.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return candidate, f"managed checkout path cannot be resolved: {exc}"
    if not resolved.is_relative_to(root):
        return candidate, f"managed checkout resolves outside provisioning root: {resolved}"
    return candidate, None


def managed_provisioning_root_issue(*, create: bool = False) -> str | None:
    """Require the configured provisioning-root leaf to be a real directory."""

    root = paths.root()
    try:
        state = os.stat(root, follow_symlinks=False)
    except FileNotFoundError:
        if not create:
            return None
        try:
            root.mkdir(parents=True, exist_ok=True)
            state = os.stat(root, follow_symlinks=False)
        except (OSError, RuntimeError) as exc:
            return f"provisioning root cannot be created safely: {root}: {exc}"
    except (OSError, RuntimeError) as exc:
        return f"provisioning root cannot be inspected safely: {root}: {exc}"
    if stat.S_ISLNK(state.st_mode):
        return f"provisioning root is a symlink: {root}"
    if not stat.S_ISDIR(state.st_mode):
        return f"provisioning root is not a directory: {root}"
    return None


def managed_environment_path(name: str) -> tuple[Path, str | None]:
    """Require the manifest-derived environment root without following an inner symlink."""
    expected = paths.env_dir(name)
    root_issue = managed_provisioning_root_issue()
    if root_issue is not None:
        return expected, root_issue
    if expected.parent.is_symlink():
        return expected, f"managed environment parent is a symlink: {expected.parent}"
    if expected.is_symlink():
        return expected, f"managed environment path is a symlink: {expected}"
    try:
        resolved = expected.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return expected, f"managed environment path cannot be resolved: {exc}"
    if not expected.is_dir() or not resolved.is_relative_to(root):
        return expected, f"managed environment is not a contained directory: {resolved}"
    return expected, None


def managed_environment_creation_target_issue(name: str) -> str | None:
    """Refuse provisioning through a redirected envs parent or environment leaf."""
    target = paths.env_dir(name)
    root_issue = managed_provisioning_root_issue(create=True)
    if root_issue is not None:
        return root_issue
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        parent = target.parent.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return f"managed environment parent cannot be resolved: {exc}"
    if target.parent.is_symlink() or not parent.is_relative_to(root):
        return f"managed environment parent is redirected outside provisioning root: {parent}"
    if target.is_symlink():
        return f"managed environment path is a symlink: {target}"
    if target.exists() and not target.is_dir():
        return f"managed environment path is not a directory: {target}"
    return None


def managed_url_artifact_path(
    package: Package, value: object,
) -> tuple[Path | None, str | None]:
    """Bind a single-file receipt to its manifest-owned path and provisioning root."""
    expected = paths.models_dir() / str(package.source["filename"])
    candidate = Path(str(value)) if value else None
    if candidate is None:
        return None, "artifact path is absent"
    root_issue = managed_provisioning_root_issue()
    if root_issue is not None:
        return candidate, root_issue
    if Path(os.path.abspath(candidate)) != Path(os.path.abspath(expected)):
        return candidate, f"artifact path {candidate} is not managed path {expected}"
    if expected.parent.is_symlink():
        return candidate, f"managed artifact parent is a symlink: {expected.parent}"
    if candidate.is_symlink():
        return candidate, f"managed artifact path is a symlink: {candidate}"
    try:
        parent = expected.parent.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return candidate, f"managed artifact path cannot be resolved: {exc}"
    if (
        not expected.parent.is_dir()
        or not parent.is_relative_to(root)
        or not candidate.is_file()
        or not resolved.is_relative_to(root)
    ):
        return candidate, f"managed artifact is not a contained regular file: {resolved}"
    return candidate, None


def checkout_file_matches(
    checkout: Path, name: str, digest: str, hasher=sha256_file,
) -> bool:
    """Require a real in-checkout regular file before comparing its pinned digest."""
    target = checkout / name
    try:
        checkout_root = checkout.resolve(strict=True)
        resolved = target.resolve(strict=True)
        return (
            target.is_file()
            and not target.is_symlink()
            and resolved.is_relative_to(checkout_root)
            and hasher(target) == digest
        )
    except (OSError, RuntimeError):
        return False


def _checkout_integrity_issues(
    package: Package,
    materialized: dict,
    toolchain: Toolchain,
) -> list[str]:
    """Verify a native backend's live checkout instead of trusting its pull receipt."""
    specification = package.checkout
    if specification is None:
        return []

    issues: list[str] = []
    checkout, location_issue = managed_checkout_path(
        package, materialized.get("checkout")
    )
    if location_issue is not None:
        return [location_issue]
    assert checkout is not None
    if not checkout.is_dir():
        return [f"source checkout is not a directory: {checkout}"]

    expected_commit = specification.get("resolved_commit", specification["commit"])
    accepted_receipts = (specification["commit"], expected_commit)
    recorded_commit = materialized.get("checkout_commit")
    if recorded_commit not in accepted_receipts:
        issues.append(
            f"recorded checkout commit is {recorded_commit!r}, expected one of "
            f"{list(accepted_receipts)!r}"
        )
    try:
        state = toolchain.inspect_checkout(checkout)
    except ValueError as exc:
        return [str(exc)]
    if state.head != expected_commit:
        issues.append(
            f"checkout HEAD is {state.head!r}, expected exact commit {expected_commit!r}"
        )

    try:
        expected_patches, expected_names, expected_digests = (
            checkout_patch_expectation(package)
        )
    except (OSError, ValueError) as exc:
        return [f"shipped checkout patch cannot be read: {exc}"]
    applied = materialized.get("patches_applied", [])
    if not isinstance(applied, list) or applied != list(expected_patches):
        issues.append(
            f"recorded patches are {applied!r}, expected exactly {list(expected_patches)!r}"
        )

    expected_modified = set(expected_names)
    recorded = materialized.get("patched_file_digests", {})
    if not isinstance(recorded, dict) or recorded != expected_digests:
        issues.append(
            f"recorded patched-file digests are {recorded!r}, expected manifest values "
            f"{expected_digests!r}"
        )
    changed: list[str] = []
    for name, digest in expected_digests.items():
        if not checkout_file_matches(checkout, name, digest, toolchain.file_digest):
            changed.append(name)
    if changed:
        issues.append(f"live patched-file hashes changed for {sorted(changed)!r}")

    # Exact Git names reject added changes while manifest-owned hashes bind the permitted files'
    # contents without depending on Git's configurable/version-dependent diff presentation.
    if set(state.modified) != expected_modified:
        issues.append(
            f"tracked checkout changes are {list(state.modified)!r}, expected exactly "
            f"{sorted(expected_modified)!r}"
        )
    if state.untracked:
        issues.append(
            f"checkout has ordinary or ignored untracked files: {list(state.untracked)!r}"
        )
    return issues


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
        return None, (
            f"expected one executable built product {product!r}, found {len(candidates)}"
        )
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
    package = facade_dependency("packages", packages)()["fluidaudio"]
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
    package = facade_dependency("packages", packages)()["fluidaudio"]
    checkout, issue = managed_checkout_path(package, entry["materialized"].get("path"))
    if issue is not None or checkout is None:
        return False
    try:
        state = toolchain.inspect_checkout(checkout)
    except ValueError:
        return False
    try:
        expected_patches, expected_names, expected_digests = (
            checkout_patch_expectation(package)
        )
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
    executable, product_issue = validated_built_product(
        checkout, product, materialized
    )
    if product_issue is not None or executable is None:
        return False
    try:
        return toolchain.built_product_runs(executable)
    except OSError:
        return False
