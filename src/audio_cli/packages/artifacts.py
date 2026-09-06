"""Fresh integrity verification and resolution of managed single-file packages."""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..environments import Package, git_blob_source_problems
from ..media import bound_directory, regular_file_digests_at
from . import catalog, registry
from .locations import managed_provisioning_root_issue, managed_url_artifact_path
from .models import ProvisioningError


def require_git_blob_source(package: Package) -> None:
    problems = git_blob_source_problems(
        package.source, package.bytes, auto_fetch=package.auto_fetch
    )
    if problems:
        raise ProvisioningError(
            "package_integrity_failed",
            "; ".join(problems),
            package=package.id,
            fix=f"audio packages pull --repair {package.id}",
        )


def verify_artifact_file(package: Package, value: object) -> tuple[Path, dict, dict]:
    """Verify manifest path and actual bytes; never infer integrity from a registry digest."""
    git_blob = package.source["type"] == "git-blob"
    if git_blob:
        require_git_blob_source(package)
    location, issue = managed_url_artifact_path(package, value)
    if issue is None and location is not None:
        try:
            with bound_directory(location.parent, root=paths.root(), create=False) as descriptor:
                found = regular_file_digests_at(descriptor, location.name, git_blob=git_blob)
                if git_blob:
                    if found is None or (found.git_blob_sha1, found.bytes) != (
                        package.source["git_blob_sha1"],
                        package.bytes,
                    ):
                        issue = f"{location} Git blob identity or byte count differs from manifest"
                    else:
                        record = {
                            "revision": package.source["revision"],
                            "git_blob_sha1": found.git_blob_sha1,
                            "bytes": found.bytes,
                        }
                elif found is None or found.sha256 != package.source["sha256"]:
                    issue = f"{location} is missing or its digest changed"
                else:
                    record = {"digest": "ok"}
        except OSError as exc:
            issue = f"could not verify managed artifact {location}: {exc}"
    if issue is not None or location is None:
        raise ProvisioningError(
            "package_integrity_failed",
            issue or "artifact path is absent",
            package=package.id,
            fix=f"audio packages pull --repair {package.id}",
        )
    return (
        location,
        record,
        {
            "package": package.id,
            "source": dict(package.source),
            "bytes": found.bytes,
            "sha256": found.sha256,
        },
    )


def verified_artifact(package_id: str) -> tuple[Path, dict]:
    """Return a ready single-file package only after fresh managed-path and byte verification.

    This read-only composition API never downloads or repairs. Provenance includes the package,
    manifest source, actual bytes, and a locally computed SHA-256 from the same pass that verified
    source identity. This local SHA-256 is not an upstream declaration. The consuming I/O owner
    must open without following links and match that digest before using its private copy.
    """
    package = catalog.packages().get(package_id)
    if package is None or package.source["type"] not in {"url", "git-blob"}:
        raise ProvisioningError(
            "package_unknown",
            f"{package_id!r} is not a declared single-file artifact",
            exit_code=2,
            fix="audio packages list",
        )
    root_issue = managed_provisioning_root_issue()
    if root_issue is not None:
        raise ProvisioningError(
            "package_integrity_failed",
            root_issue,
            package=package.id,
            fix=f"audio packages pull --repair {package.id}",
        )
    try:
        entry = registry.load_registry()["packages"].get(package_id, {})
    except ProvisioningError as exc:
        exc.payload.update(package=package.id, fix=f"audio packages pull --repair {package.id}")
        raise
    if entry.get("state") != "ready":
        raise ProvisioningError(
            "package_not_provisioned",
            f"{package_id} is not ready; explicitly provision it before use",
            package=package_id,
            fix=f"audio packages pull {package_id}",
        )
    materialized = entry.get("materialized", {})
    location, _record, provenance = verify_artifact_file(package, materialized.get("path"))
    return location, provenance


__all__ = ["require_git_blob_source", "verified_artifact", "verify_artifact_file"]
