"""Ownership accounting and safe deletion helpers for package teardown."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from .. import paths
from ..environments import Package
from ..media import bound_directory
from . import catalog
from .models import ProvisioningError


def _toolchain_missing(package: Package, tool: str) -> ProvisioningError:
    """The exit-3 refusal for a package whose external toolchain is absent."""
    return ProvisioningError(
        "toolchain_missing",
        f"{package.id} needs {tool}, which is not on PATH",
        missing_tool=tool,
        package=package.id,
        requires_tool=[tool],
    )


def _source_revision_report(package: Package) -> dict:
    """What a Hub package's `verify` entry can honestly claim, which is not a digest.

    Nothing here hashes a snapshot. The manifest pins a revision and carries no `sha256` for a Hub
    source, so there is nothing to hash *against*; a `digest` key would name a check no code
    performs. The revision is what is pinned, and the existence check above it is the rest of what
    was verified. So the two claims are told apart by which key is present — `digest` where
    contents were hashed against a manifest pin, `revision`/`revisions` where a revision is pinned
    and the snapshot is present — rather than by a `digest_verified: false` confession.
    """
    revisions = _source_revisions(package)
    if len(revisions) == 1:
        return {"revision": revisions[0]}
    if revisions:
        return {"revisions": revisions}
    return {}


def _source_revisions(package: Package) -> list[str]:
    """Every Hub revision a package pins, whether it names one or four."""
    source = package.source
    if source["type"] == "huggingface":
        return [source["revision"]]
    if source["type"] == "huggingface_multi":
        return [repo["revision"] for repo in source["repos"]]
    return []


def _managed_package_locations(package: Package | None) -> list[Path]:
    """Local deletion targets derived only from the installed manifest."""
    if package is None:
        return []
    found: list[Path] = []
    if package.checkout is not None or package.source["type"] == "git+build":
        found.append(paths.checkout_dir(package.environment, package.id))
    if package.source["type"] in {"url", "git-blob"}:
        found.append(paths.models_dir() / str(package.source["filename"]))
    return found


def _owned_tree_bytes(target: Path) -> int:
    """Bytes below a managed target without following any symlink."""
    try:
        if target.is_symlink():
            return target.lstat().st_size
        if target.is_file():
            return target.stat().st_size
        if not target.is_dir():
            return 0
    except OSError:
        return 0
    total = 0
    for directory, directories, filenames in os.walk(target, followlinks=False):
        base = Path(directory)
        for name in [*directories, *filenames]:
            item = base / name
            try:
                if item.is_symlink():
                    total += item.lstat().st_size
                elif item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    return total


def _owned_tree_bytes_at(parent_descriptor: int, name: str) -> int:
    """Measure one descriptor-relative tree without following any symlink."""

    try:
        found = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return 0
    if stat.S_ISLNK(found.st_mode) or stat.S_ISREG(found.st_mode):
        return found.st_size
    if not stat.S_ISDIR(found.st_mode):
        return 0
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        total = 0
        with os.scandir(directory_descriptor) as entries:
            for entry in entries:
                try:
                    total += _owned_tree_bytes_at(directory_descriptor, entry.name)
                except FileNotFoundError:
                    continue
        return total
    finally:
        os.close(directory_descriptor)


def _delete_at(parent_descriptor: int, name: str) -> None:
    """Delete one descriptor-relative leaf without following it outside its parent."""

    try:
        found = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(found.st_mode):
        if not shutil.rmtree.avoids_symlink_attacks:
            raise OSError("platform recursive deletion is not symlink-attack resistant")
        shutil.rmtree(name, dir_fd=parent_descriptor)
    else:
        os.unlink(name, dir_fd=parent_descriptor)


def _delete_managed(target: Path) -> int:
    """Delete through a no-follow parent descriptor, then confirm descriptor-relative absence."""
    root = Path(os.path.abspath(paths.root()))
    absolute = Path(os.path.abspath(target))
    if absolute == root or not absolute.is_relative_to(root):
        raise ProvisioningError(
            "delete_refused",
            f"managed deletion target is outside the provisioning root: {target}",
            target=str(target),
            fix="Inspect the provisioning registry and managed cache root",
        )
    opened = False
    try:
        with bound_directory(
            absolute.parent,
            root=root,
            create=False,
            # Once opened, the descriptor owns the safe operation. If an attacker renames
            # the parent afterward, deleting from that original directory is still contained;
            # following its replacement pathname would not be.
            verify_on_exit=False,
        ) as parent_descriptor:
            opened = True
            before = _owned_tree_bytes_at(parent_descriptor, absolute.name)
            _delete_at(parent_descriptor, absolute.name)
            try:
                os.stat(
                    absolute.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return before
            raise OSError("managed target still exists after deletion")
    except FileNotFoundError:
        return 0
    except OSError as exc:
        if not opened:
            raise ProvisioningError(
                "delete_refused",
                f"managed deletion target cannot be opened without following links: {target}: "
                f"{exc}",
                target=str(target),
                fix=f"Remove or replace redirected parent paths for {target} and retry",
            ) from exc
        raise ProvisioningError(
            "delete_failed",
            f"could not delete managed target {target}: {exc}",
            target=str(target),
            fix=f"Restore access to {target} and retry",
        ) from exc


def _teardown_revisions(
    package: Package | None,
    materialized: dict,
) -> tuple[list[str], list[str]]:
    """Bound mutable ownership receipts to revisions shipped for one known package."""
    allowed = set(_source_revisions(package)) if package is not None else set()

    def strings(value: object) -> set[str]:
        return (
            {item for item in value if isinstance(item, str)} if isinstance(value, list) else set()
        )

    claimed = strings(materialized.get("hub_revisions"))
    pre_existing = strings(materialized.get("hub_revisions_pre_existing"))
    deletable = (claimed & allowed) - pre_existing
    retained = pre_existing | (claimed - allowed)
    return sorted(deletable), sorted(retained)


def _users_by_environment(document: dict) -> dict[str, set[str]]:
    users: dict[str, set[str]] = {}
    package_catalog = catalog.packages()
    for identifier in document["packages"]:
        package = package_catalog.get(identifier)
        if package is not None:
            users.setdefault(package.environment, set()).add(identifier)
    return users


def _selection_bytes(selection: list[Package], document: dict) -> tuple[int, list[str]]:
    known = 0
    unsized: list[str] = []
    for package in selection:
        materialized = document["packages"].get(package.id, {}).get("materialized", {})
        materialized = materialized if isinstance(materialized, dict) else {}
        if package.source["type"] in {"huggingface", "huggingface_multi"}:
            owned, _retained = _teardown_revisions(package, materialized)
            if package.source["type"] == "huggingface":
                if package.source["revision"] in owned and package.bytes is not None:
                    known += package.bytes
            else:
                known += sum(
                    int(repository.get("bytes") or 0)
                    for repository in package.source["repos"]
                    if repository["revision"] in owned
                )
            continue
        recorded = materialized.get("bytes")
        size = recorded if recorded is not None else package.bytes
        if size is None:
            unsized.append(package.id)
        else:
            known += size
    return known, sorted(unsized)
