"""Managed provisioning locations and containment checks."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .. import paths
from ..environments import Package


def managed_checkout_path(
    package: Package,
    value: object,
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
    package: Package,
    value: object,
) -> tuple[Path | None, str | None]:
    """Bind a single-file receipt to its manifest-owned path and provisioning root."""
    expected = paths.models_dir() / str(package.source["filename"])
    candidate = Path(str(value)) if value else None
    if candidate is None:
        return None, "artifact path is absent"
    # abspath erases '..' lexically, but the OS traverses preceding symlinks first.
    # Refuse that ambiguity in either input before normalizing or inspecting paths.
    if ".." in candidate.parts or ".." in expected.parts:
        return candidate, "artifact path contains parent traversal"
    root_issue = managed_provisioning_root_issue()
    if root_issue is not None:
        return candidate, root_issue
    if Path(os.path.abspath(candidate)) != Path(os.path.abspath(expected)):
        return candidate, f"artifact path {candidate} is not managed path {expected}"
    if expected.parent.is_symlink():
        return candidate, f"managed artifact parent is a symlink: {expected.parent}"
    if expected.is_symlink():
        return expected, f"managed artifact path is a symlink: {expected}"
    try:
        parent = expected.parent.resolve(strict=True)
        resolved = expected.resolve(strict=True)
        root = paths.root().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return candidate, f"managed artifact path cannot be resolved: {exc}"
    if (
        not expected.parent.is_dir()
        or not parent.is_relative_to(root)
        or not expected.is_file()
        or not resolved.is_relative_to(root)
    ):
        return candidate, f"managed artifact is not a contained regular file: {resolved}"
    # Hashing and the consuming caller must use this same manifest-owned spelling.
    return expected, None


__all__ = [
    "managed_checkout_path",
    "managed_environment_creation_target_issue",
    "managed_environment_path",
    "managed_provisioning_root_issue",
    "managed_url_artifact_path",
]
