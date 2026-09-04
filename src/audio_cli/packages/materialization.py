"""Materialize one package from its manifest-owned source."""

from __future__ import annotations

from .. import paths
from ..environments import ManifestError, Package
from . import checkouts, integrity
from .fetcher import Fetcher
from .integrity import sha256_file
from .locations import managed_url_artifact_path
from .models import ProvisioningError
from .products import built_product_candidates
from .teardown import _delete_managed
from .toolchain import Toolchain


def materialize(
    toolchain: Toolchain,
    fetcher: Fetcher,
    package: Package,
    pre_existing: set[str],
    *,
    repair: bool = False,
) -> dict:
    """Put the package on disk. `digest_verified` appears only where a digest was taken.

    One source kind pins a content hash — `url` — and it is the only one whose materialization
    can claim to have been verified against the manifest. The Hub kinds pin a *revision*; no
    `sha256` exists in the manifest to hash a snapshot against, so they record the revision
    and nothing more. They used to record `digest_verified: True` regardless, which made
    `verify` print `digest: "ok"` for a check no code performs.
    """
    kind = package.source["type"]
    if kind == "url":
        # The filename is manifest data, not derived: the shipped Silero backend resolves
        # this exact name, and a pull that invented one would leave two copies on disk and
        # re-download on first use. tests/test_environments.py ties the two together.
        target = paths.models_dir() / package.source["filename"]
        # `repair` needs no force here, and the digest is the reason: url_file re-hashes what
        # is on disk against the manifest pin and downloads again unless it matches, so a
        # match already is the strongest re-materialization available. Forcing the transfer
        # would spend the bytes to arrive at the same file.
        resolved = fetcher.url_file(package.source["url"], package.source["sha256"], target)
        _managed, location_issue = managed_url_artifact_path(package, resolved)
        if location_issue is not None:
            raise ProvisioningError(
                "package_integrity_failed",
                location_issue,
                fix=f"audio packages pull --repair {package.id}",
            )
        return {
            "path": str(resolved),
            "bytes": integrity._tree_bytes(resolved),
            "digest_verified": True,
        }

    if kind == "huggingface":
        revision = package.source["revision"]
        patterns = package.source.get("allow_patterns")
        if patterns is None:
            snapshot = fetcher.hf_snapshot(package.source["repo"], revision, force=repair)
        else:
            snapshot = fetcher.hf_snapshot(
                package.source["repo"],
                revision,
                force=repair,
                allow_patterns=tuple(patterns),
            )
        snapshot_bytes = checkouts._require_hub_snapshot(
            package,
            package.source["repo"],
            revision,
            snapshot,
            tuple(patterns or ()),
        )
        result = {
            "path": str(snapshot),
            "bytes": snapshot_bytes,
            "revision": revision,
            # Only what this pull fetched is ours to delete later, decided before the
            # download rather than after it — see _pre_existing_revisions.
            "hub_revisions": [] if revision in pre_existing else [revision],
            "hub_revisions_pre_existing": sorted(pre_existing),
        }
        result.update(checkouts._checkout_and_install(toolchain, package))
        return result

    if kind == "huggingface_multi":
        snapshots = {}
        total = 0
        ours: list[str] = []
        for repo in package.source["repos"]:
            patterns = repo.get("allow_patterns")
            snapshot = fetcher.hf_snapshot(
                repo["repo"],
                repo["revision"],
                force=repair,
                allow_patterns=tuple(patterns) if patterns is not None else None,
            )
            snapshot_bytes = checkouts._require_hub_snapshot(
                package,
                repo["repo"],
                repo["revision"],
                snapshot,
                tuple(patterns or ()),
            )
            snapshots[repo["repo"]] = str(snapshot)
            total += snapshot_bytes
            if repo["revision"] not in pre_existing:
                ours.append(repo["revision"])
        result = {
            "paths": snapshots,
            "bytes": total,
            # Plural, because this package spans four repositories. A single `revision`
            # key would have to pick one of them, and the receipt promises the revisions
            # a pull materialized.
            "revisions": [repo["revision"] for repo in package.source["repos"]],
            "hub_revisions": ours,
            "hub_revisions_pre_existing": sorted(pre_existing),
        }
        result.update(checkouts._checkout_and_install(toolchain, package))
        return result

    if kind == "git+build":
        checkout = paths.checkout_dir(package.environment, package.id)
        # This method only runs for a non-ready package or an explicit repair. Reusing a
        # prior tree would let an interrupted pull's untracked build hook run before verify.
        _delete_managed(checkout)
        checkout.parent.mkdir(parents=True, exist_ok=True)
        toolchain.clone(package.source["repo"], package.source["commit"], checkout)
        applied, digests = checkouts.materialize_checkout_patch(package, checkout, toolchain)
        try:
            state = toolchain.inspect_checkout(checkout)
        except ValueError as exc:
            raise ProvisioningError(
                "checkout_integrity_failed",
                f"could not inspect fresh checkout: {exc}",
                package=package.id,
                fix=f"audio packages pull --repair {package.id}",
            ) from exc
        _expected_patches, expected_names, _expected_digests = checkouts.checkout_patch_expectation(
            package
        )
        if (
            state.head != package.source["commit"]
            or set(state.modified) != set(expected_names)
            or state.untracked
        ):
            raise ProvisioningError(
                "checkout_integrity_failed",
                f"{package.id} checkout was not exact before build",
                package=package.id,
                expected={
                    "head": package.source["commit"],
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
        toolchain.swift_build(checkout)
        product = package.source["product"]
        runs = toolchain.swift_product_runs(checkout, product)
        if not runs:
            # A build that produces an executable nobody can launch is not a provisioned
            # package, and this used to be recorded and then ignored: `pull` returned exit 0
            # with an empty `warnings`, `verify` reported `failed: []`, and the environment
            # read `ok`, while the one thing the package exists to do was impossible. The
            # entry stays `pulling`, so `list` and `run` both report it absent.
            raise ProvisioningError(
                "package_build_unusable",
                f"{package.id} built, but its product {product!r} does not run, so nothing "
                f"in the {package.environment} environment can use it",
                package=package.id,
                product=product,
                built=True,
                fix=f"audio packages pull --repair {package.id}",
            )
        candidates = built_product_candidates(checkout, product)
        if len(candidates) != 1:
            raise ProvisioningError(
                "package_build_unusable",
                f"{package.id} built, but expected one contained executable product "
                f"{product!r} and found {len(candidates)}",
                package=package.id,
                product=product,
                built=True,
                fix=f"audio packages pull --repair {package.id}",
            )
        executable = candidates[0]
        product_path = executable.relative_to(checkout.resolve(strict=True)).as_posix()
        return {
            "path": str(checkout),
            "bytes": integrity._tree_bytes(checkout / ".build"),
            "revision": package.source["commit"],
            "built": True,
            "product_runs": runs,
            "product_path": product_path,
            "product_sha256": sha256_file(executable),
            "patches_applied": applied,
            "patched_file_digests": digests,
        }

    raise ManifestError(f"{package.id}: unsupported source type {kind!r}")
