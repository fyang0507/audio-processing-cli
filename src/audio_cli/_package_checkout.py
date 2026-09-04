"""Hub snapshot and source-checkout materialization helpers."""

from __future__ import annotations

from pathlib import Path

from . import paths
from ._package_compat import facade_dependency
from ._package_core import (
    ProvisioningError,
    _hub_snapshot_index,
    _inspect_hub_snapshot,
)
from ._package_paths import checkout_patch_expectation, materialize_checkout_patch
from ._package_teardown import _delete_managed
from .environments import Package


class CheckoutMixin:
    @staticmethod
    def _require_hub_snapshot(
        package: Package,
        repository: str,
        revision: str,
        snapshot: Path,
        patterns: tuple[str, ...],
    ) -> int:
        """Bind and measure one download before pull performs any subsequent work."""
        try:
            snapshot_index = facade_dependency(
                "_hub_snapshot_index", _hub_snapshot_index
            )()
        except Exception as exc:  # noqa: BLE001 - no index means no trusted materialization
            issues = [f"Hugging Face cache identity cannot be inspected: {exc}"]
        else:
            issues, byte_count = _inspect_hub_snapshot(
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

    def _checkout_and_install(self, package: Package, *, repair: bool = False) -> dict:
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
        self.toolchain.clone(package.checkout["repo"], resolved_commit, checkout)

        applied, digests = materialize_checkout_patch(
            package, checkout, self.toolchain
        )
        _expected_patches, expected_names, _expected_digests = (
            checkout_patch_expectation(package)
        )
        try:
            state = self.toolchain.inspect_checkout(checkout)
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

        self.toolchain.install_checkout(paths.env_python(package.environment), checkout)
        # Stage interpreters use ``-B`` so they never recreate ignored bytecode.  Cleaning
        # anything the wheel build left in the source tree makes a successful pull start
        # from the same inspectable state that run preflight requires.
        self.toolchain.clean_ignored_checkout(checkout)
        return {"checkout": str(checkout), "checkout_commit": resolved_commit,
                "patches_applied": applied, "patched_file_digests": digests}
