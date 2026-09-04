"""Descriptor-derived identities for protecting canonical media inputs."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProtectedFileIdentity:
    """A regular file held by device/inode identity across a long operation."""

    path: Path
    device: int
    inode: int


def file_identity_from_descriptor(
    descriptor: int, path: Path
) -> ProtectedFileIdentity | None:
    state = os.fstat(descriptor)
    if not stat.S_ISREG(state.st_mode):
        return None
    return ProtectedFileIdentity(Path(path), state.st_dev, state.st_ino)


def capture_file_identity(path: Path) -> ProtectedFileIdentity | None:
    """Capture an existing regular file without trusting its pathname later."""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return None
    try:
        return file_identity_from_descriptor(descriptor, Path(path))
    finally:
        os.close(descriptor)


def matching_protected_identity(
    directory_descriptor: int,
    name: str,
    identities: Sequence[ProtectedFileIdentity],
) -> ProtectedFileIdentity | None:
    """Return the captured input currently occupying a descriptor-bound name."""
    try:
        state = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(state.st_mode):
        return None
    return next(
        (
            protected
            for protected in identities
            if (state.st_dev, state.st_ino) == (protected.device, protected.inode)
        ),
        None,
    )


def entry_matches_file_identity(
    directory_descriptor: int,
    name: str,
    protected: ProtectedFileIdentity,
) -> bool:
    """Return whether a descriptor-bound name still denotes one captured file."""
    try:
        state = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(state.st_mode)
        and (state.st_dev, state.st_ino) == (protected.device, protected.inode)
    )


__all__ = [
    "ProtectedFileIdentity",
    "capture_file_identity",
    "entry_matches_file_identity",
    "file_identity_from_descriptor",
    "matching_protected_identity",
]
