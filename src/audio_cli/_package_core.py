"""Shared package lifecycle types, constants, and Hub integrity checks."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ._package_compat import facade_dependency
from .environments import Package

REGISTRY_SCHEMA_VERSION = 1

HUB_CACHE_NOTE = (
    "weights live in the shared Hugging Face cache, not under this root. Only revisions this "
    "root recorded as downloaded and the current manifest still pins for that package are "
    "eligible for deletion; pre-existing and out-of-manifest revisions are retained because "
    "they may belong to another tool, another provisioning root, or an earlier experiment"
)
UNTOUCHED = ["user media", "transcript and subtitle outputs"]


@dataclass(frozen=True)
class CheckoutState:
    """Live Git state for a source checkout used by an executable backend."""

    head: str
    modified: tuple[str, ...]
    untracked: tuple[str, ...]


class ProvisioningError(RuntimeError):
    """A provisioning failure that carries the payload its exit code is documented with."""

    def __init__(self, code: str, message: str, *, exit_code: int = 3, **payload) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.payload = payload

    def as_dict(self) -> dict:
        body = {"code": self.code, "detail": self.message}
        body.update(self.payload)
        return body


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    """Total size under a directory, following symlinks into the Hub's blob store."""
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # a broken symlink contributes nothing rather than failing the pull
            continue
    return total


def _hub_snapshot_index() -> dict[tuple[str, str], Path]:
    """Map canonical Hub cache identity to the snapshot path reported by its cache index."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    return {
        (repository.repo_id, revision.commit_hash): Path(revision.snapshot_path)
        for repository in cache.repos
        for revision in repository.revisions
    }


def _inspect_hub_snapshot(
    repository: str,
    revision: str,
    snapshot: Path | None,
    patterns: tuple[str, ...],
    snapshot_index: Mapping[tuple[str, str], Path],
) -> tuple[list[str], int | None]:
    """Bind one returned Hub path before reading any bytes below it."""
    if snapshot is None:
        return [f"{repository} snapshot is not a non-symlink directory: {snapshot}"], None
    expected = snapshot_index.get((repository, revision))
    if expected is None:
        return [
            f"{repository} revision {revision} is absent from the Hugging Face cache index"
        ], None
    # Refuse a caller-returned path that is not lexically the indexed path before
    # resolving or walking it.  A bad downloader return must not make pull inspect an
    # unrelated tree merely to discover that it was unrelated.
    if Path(os.path.abspath(snapshot)) != Path(os.path.abspath(expected)):
        return [
            f"{repository} snapshot path {snapshot} does not equal cache-indexed "
            f"revision path {expected}"
        ], None
    if (
        expected.is_symlink()
        or expected.parent.is_symlink()
        or expected.parent.parent.is_symlink()
    ):
        return [f"{repository} snapshot is not the exact non-symlink cache-index path"], None
    if not expected.is_dir():
        return [
            f"{repository} snapshot is not a non-symlink directory: {expected}"
        ], None
    try:
        expected_resolved = expected.resolve(strict=True)
        repository_cache_root = expected.parent.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return [f"{repository} indexed snapshot cannot be resolved: {exc}"], None

    unsafe_entries: list[str] = []
    snapshot_files: list[str] = []
    try:
        for entry in expected_resolved.rglob("*"):
            if entry.is_symlink():
                target = entry.resolve(strict=True)
                if not target.is_file() or not target.is_relative_to(
                    repository_cache_root
                ):
                    unsafe_entries.append(str(entry.relative_to(expected_resolved)))
            elif entry.is_file():
                target = entry.resolve(strict=True)
                if not target.is_relative_to(repository_cache_root):
                    unsafe_entries.append(str(entry.relative_to(expected_resolved)))
            if entry.is_file():
                snapshot_files.append(entry.relative_to(expected_resolved).as_posix())
    except (OSError, RuntimeError) as exc:
        return [f"{repository} snapshot tree cannot be resolved safely: {exc}"], None
    if unsafe_entries:
        return [
            f"{repository} snapshot entries resolve outside its repository cache: "
            f"{sorted(unsafe_entries)!r}"
        ], None

    issues = [
        f"{repository} is missing allow_pattern {pattern}"
        for pattern in patterns
        if not any(fnmatch.fnmatch(name, pattern) for name in snapshot_files)
    ]
    if issues:
        return issues, None
    try:
        tree_bytes = facade_dependency("_tree_bytes", _tree_bytes)
        return [], tree_bytes(expected_resolved)
    except OSError as exc:
        return [f"{repository} snapshot bytes cannot be measured safely: {exc}"], None


def hub_materialization_issues(
    package: Package, materialized: dict,
) -> list[str]:
    """Return cheap live integrity failures for revision-pinned Hub snapshots."""
    kind = package.source["type"]
    if kind not in {"huggingface", "huggingface_multi"}:
        return []

    snapshots: list[tuple[str, str, Path | None, tuple[str, ...]]] = []
    if kind == "huggingface":
        value = materialized.get("path")
        snapshots.append((
            package.source["repo"],
            package.source["revision"],
            Path(str(value)) if value else None,
            tuple(package.source.get("allow_patterns", ())),
        ))
    else:
        values = materialized.get("paths")
        values = values if isinstance(values, dict) else {}
        for repository in package.source["repos"]:
            value = values.get(repository["repo"])
            snapshots.append((
                repository["repo"],
                repository["revision"],
                Path(str(value)) if value else None,
                tuple(repository.get("allow_patterns", ())),
            ))

    try:
        snapshot_index = facade_dependency(
            "_hub_snapshot_index", _hub_snapshot_index
        )()
    except Exception as exc:  # noqa: BLE001 - no cache identity means no trusted snapshot
        return [f"Hugging Face cache identity cannot be inspected: {exc}"]
    issues: list[str] = []
    actual_bytes = 0
    all_usable = True
    for repository, revision, snapshot, patterns in snapshots:
        snapshot_issues, byte_count = _inspect_hub_snapshot(
            repository, revision, snapshot, patterns, snapshot_index
        )
        issues.extend(snapshot_issues)
        if byte_count is None:
            all_usable = False
        else:
            actual_bytes += byte_count
    recorded_bytes = materialized.get("bytes")
    if isinstance(recorded_bytes, bool) or not isinstance(recorded_bytes, int):
        issues.append("materialization receipt has no integer bytes measurement")
    elif all_usable and actual_bytes != recorded_bytes:
        issues.append(
            f"snapshot bytes changed: recorded {recorded_bytes}, current {actual_bytes}"
        )
    return issues
