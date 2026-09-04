"""Source-checkout patching, materialization, and integrity."""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..environments import HERE as ENVIRONMENTS_DIR
from ..environments import Package
from . import integrity
from .locations import managed_checkout_path
from .models import ProvisioningError
from .teardown import _delete_managed
from .toolchain import Toolchain


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
    """Exact patch receipt, tracked-file set, and post-patch hashes from the manifest."""
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


def checkout_file_matches(
    checkout: Path, name: str, digest: str, hasher=integrity.sha256_file,
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


def _require_hub_snapshot(
    package: Package,
    repository: str,
    revision: str,
    snapshot: Path,
    patterns: tuple[str, ...],
) -> int:
    """Bind and measure one download before pull performs any subsequent work."""
    try:
        snapshot_index = integrity._hub_snapshot_index()
    except Exception as exc:  # noqa: BLE001 - no index means no trusted materialization
        issues = [f"Hugging Face cache identity cannot be inspected: {exc}"]
    else:
        issues, byte_count = integrity._inspect_hub_snapshot(
            repository, revision, snapshot, patterns, snapshot_index
        )
    if issues:
        raise ProvisioningError(
            "package_integrity_failed",
            "; ".join(issues),
            package=package.id,
            fix=f"audio packages pull --repair {package.id}",
        )
    assert byte_count is not None
    return byte_count


def _checkout_and_install(toolchain: Toolchain, package: Package) -> dict:
    """Pinned source checkout, patch, and a --no-deps install into the environment."""
    if package.checkout is None:
        return {}
    checkout = paths.checkout_dir(package.environment, package.id)
    # A ready package is skipped before this method. Every invocation therefore represents
    # a fresh materialization or explicit repair and starts from an absent checkout; otherwise
    # an ordinary untracked setup hook could execute during install before later verification.
    _delete_managed(checkout)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    resolved_commit = package.checkout.get(
        "resolved_commit", package.checkout["commit"])
    toolchain.clone(package.checkout["repo"], resolved_commit, checkout)

    applied, digests = materialize_checkout_patch(
        package, checkout, toolchain
    )
    _expected_patches, expected_names, _expected_digests = (
        checkout_patch_expectation(package)
    )
    try:
        state = toolchain.inspect_checkout(checkout)
    except ValueError as exc:
        raise ProvisioningError(
            "checkout_integrity_failed", f"could not inspect fresh checkout: {exc}",
            package=package.id, fix=f"audio packages pull --repair {package.id}",
        ) from exc
    if state.head != resolved_commit or set(state.modified) != set(expected_names) \
            or state.untracked:
        raise ProvisioningError(
            "checkout_integrity_failed",
            f"{package.id} checkout was not exact before install",
            package=package.id,
            expected={
                "head": resolved_commit,
                "modified": sorted(expected_names),
                "untracked": [],
            },
            actual={
                "head": state.head,
                "modified": list(state.modified),
                "untracked": list(state.untracked),
            },
            fix=f"audio packages pull --repair {package.id}",
        )

    toolchain.install_checkout(paths.env_python(package.environment), checkout)
    # Stage interpreters use ``-B`` so they never recreate ignored bytecode.  Cleaning
    # anything the wheel build left in the source tree makes a successful pull start
    # from the same inspectable state that run preflight requires.
    toolchain.clean_ignored_checkout(checkout)
    return {"checkout": str(checkout), "checkout_commit": resolved_commit,
            "patches_applied": applied, "patched_file_digests": digests}
