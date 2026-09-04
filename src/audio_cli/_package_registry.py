"""Validated, atomic persistence for the package registry."""

from __future__ import annotations

import json
import os
import stat
import uuid
from typing import Any

from . import paths
from ._package_compat import facade_dependency
from ._package_core import REGISTRY_SCHEMA_VERSION, ProvisioningError
from .media import (
    assert_directory_binding,
    bound_directory,
    cleanup_temporary_file,
    file_identity_from_descriptor,
    publish_temporary_file,
)

# Registry
# --------------------------------------------------------------------------------------


def blank_registry() -> dict:
    from . import __version__

    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "tool_version": __version__,
        "root": str(paths.root()),
        "environments": {},
        "packages": {},
    }


def _reject_duplicate_registry_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def load_registry() -> dict:
    target = paths.registry_path()
    try:
        with bound_directory(
            target.parent,
            root=paths.root(),
            create=False,
        ) as parent_descriptor:
            try:
                state = os.stat(
                    target.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return blank_registry()
            if stat.S_ISLNK(state.st_mode):
                raise ProvisioningError(
                    "registry_unreadable",
                    f"{target} is a symlink, not this root's registry",
                    fix=f"Move {target} aside and run audio packages pull again",
                )
            if not stat.S_ISREG(state.st_mode):
                raise ProvisioningError(
                    "registry_unreadable",
                    f"{target} exists but is not a regular file",
                    fix=f"Move {target} aside and run audio packages pull again",
                )
            descriptor = os.open(
                target.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode):
                    raise ProvisioningError(
                        "registry_unreadable",
                        f"{target} exists but is not a regular file",
                        fix=f"Move {target} aside and run audio packages pull again",
                    )
                raw = handle.read()
    except FileNotFoundError:
        # An absent root is the initial, unprovisioned state.  It is different
        # from a present redirected or unreadable root, which fails below.
        return blank_registry()
    except ProvisioningError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ProvisioningError(
            "registry_unreadable", f"could not read {target}: {exc}",
            fix=f"Restore access to {target}, or move it aside and run audio packages pull again",
        ) from exc
    try:
        document = json.loads(
            raw, object_pairs_hook=_reject_duplicate_registry_keys
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ProvisioningError(
            "registry_unreadable", f"{target} is not valid JSON: {exc}",
            fix=f"Move {target} aside and run audio packages pull again",
        ) from exc
    if not isinstance(document, dict):
        raise ProvisioningError(
            "registry_unreadable", f"{target} must contain a JSON object",
            fix=f"Move {target} aside and run audio packages pull again",
        )
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ProvisioningError(
            "registry_unreadable",
            f"{target} has schema_version {document.get('schema_version')!r}, expected "
            f"{REGISTRY_SCHEMA_VERSION}",
        )
    expected_root = str(paths.root())
    if document.get("root") != expected_root:
        raise ProvisioningError(
            "registry_unreadable",
            f"{target} records provisioning root {document.get('root')!r}, expected "
            f"{expected_root!r}",
            fix=f"Restore {target} from this root, or move it aside and pull again",
        )
    for key in ("environments", "packages"):
        value = document.setdefault(key, {})
        if not isinstance(value, dict) or any(
            not isinstance(identifier, str) or not isinstance(entry, dict)
            for identifier, entry in value.items()
        ):
            raise ProvisioningError(
                "registry_unreadable",
                f"{target} field {key!r} must be an object of object entries",
                fix=f"Move {target} aside and run audio packages pull again",
            )
        if key == "packages":
            malformed = sorted(
                identifier
                for identifier, entry in value.items()
                if "materialized" in entry
                and not isinstance(entry["materialized"], dict)
            )
            if malformed:
                raise ProvisioningError(
                    "registry_unreadable",
                    f"{target} package materialized fields must be objects: {malformed}",
                    fix=f"Move {target} aside and run audio packages pull again",
                )
            for identifier, entry in value.items():
                retry_revisions = entry.get("hub_revisions_pre_existing")
                if retry_revisions is not None and (
                    not isinstance(retry_revisions, list)
                    or any(not isinstance(revision, str) for revision in retry_revisions)
                ):
                    raise ProvisioningError(
                        "registry_unreadable",
                        f"{target} package {identifier!r} hub_revisions_pre_existing "
                        "must be an array of strings",
                        fix=f"Move {target} aside and run audio packages pull again",
                    )
                materialized = entry.get("materialized", {})
                byte_count = materialized.get("bytes")
                if byte_count is not None and (
                    isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count < 0
                ):
                    raise ProvisioningError(
                        "registry_unreadable",
                        f"{target} package {identifier!r} materialized bytes must be a "
                        "non-negative integer",
                        fix=f"Move {target} aside and run audio packages pull again",
                    )
                for location_key in ("path", "checkout"):
                    location = materialized.get(location_key)
                    if location is not None and not isinstance(location, str):
                        raise ProvisioningError(
                            "registry_unreadable",
                            f"{target} package {identifier!r} materialized "
                            f"{location_key} must be a string",
                            fix=f"Move {target} aside and run audio packages pull again",
                        )
                locations = materialized.get("paths")
                if locations is not None and (
                    not isinstance(locations, dict)
                    or any(
                        not isinstance(repo, str) or not isinstance(location, str)
                        for repo, location in locations.items()
                    )
                ):
                    raise ProvisioningError(
                        "registry_unreadable",
                        f"{target} package {identifier!r} materialized paths must map "
                        "strings to strings",
                        fix=f"Move {target} aside and run audio packages pull again",
                    )
                for revisions_key in (
                    "hub_revisions", "hub_revisions_pre_existing",
                ):
                    revisions = materialized.get(revisions_key)
                    if revisions is not None and (
                        not isinstance(revisions, list)
                        or any(not isinstance(revision, str) for revision in revisions)
                    ):
                        raise ProvisioningError(
                            "registry_unreadable",
                            f"{target} package {identifier!r} materialized "
                            f"{revisions_key} must be an array of strings",
                            fix=f"Move {target} aside and run audio packages pull again",
                        )
    return document


def save_registry(document: dict) -> None:
    """Atomically publish through one descriptor bound to the managed root."""
    target = paths.registry_path()
    try:
        with bound_directory(
            target.parent,
            root=paths.root(),
            create=True,
        ) as parent_descriptor:
            try:
                existing = os.stat(
                    target.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise ProvisioningError(
                    "registry_unreadable",
                    f"{target} exists but is not a regular file",
                    fix=f"Move {target} aside and run audio packages pull again",
                )

            partial_name = (
                f".audio-registry-{os.getpid()}-{uuid.uuid4().hex}.tmp"
            )
            created = False
            temporary_identity = None
            try:
                descriptor = os.open(
                    partial_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o666,
                    dir_fd=parent_descriptor,
                )
                created = True
                try:
                    temporary_identity = file_identity_from_descriptor(
                        descriptor, target.parent / partial_name
                    )
                finally:
                    if temporary_identity is None:
                        os.close(descriptor)
                if temporary_identity is None:
                    raise OSError(
                        f"registry temporary is not a regular file: "
                        f"{target.parent / partial_name}"
                    )
                with os.fdopen(
                    descriptor, "w", encoding="utf-8", newline="\n"
                ) as handle:
                    handle.write(
                        json.dumps(
                            document,
                            indent=2,
                            sort_keys=True,
                            ensure_ascii=False,
                        ) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())

                facade_dependency(
                    "assert_directory_binding", assert_directory_binding
                )(parent_descriptor, target.parent)
                publish_temporary_file(
                    parent_descriptor,
                    partial_name,
                    target.name,
                    output_path=target,
                    force=True,
                    temporary_identity=temporary_identity,
                )
                created = False
            finally:
                if created and temporary_identity is not None:
                    cleanup_temporary_file(
                        parent_descriptor, partial_name, temporary_identity
                    )
    except ProvisioningError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ProvisioningError(
            "registry_unreadable",
            f"could not write {target} safely: {exc}",
            fix=f"Restore access to {target}, or move it aside and pull again",
        ) from exc
