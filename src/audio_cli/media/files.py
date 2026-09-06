"""Descriptor-bound paths, hashes, and private temporary locations."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


def _resolve_without_version_specific_loops(candidate: Path, *, strict: bool) -> Path:
    try:
        return candidate.resolve(strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(f"Symlink loop from '{candidate}'") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise RuntimeError(f"Symlink loop from '{candidate}'") from exc
        raise


def resolve_path_identity(path: Path) -> Path:
    """Resolve an existing or future path while rejecting loops on every Python 3.11+ runtime."""
    candidate = Path(path)
    try:
        return _resolve_without_version_specific_loops(candidate, strict=True)
    except FileNotFoundError:
        return _resolve_without_version_specific_loops(candidate, strict=False)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise OSError("this platform cannot open managed directories without following links")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def assert_directory_binding(descriptor: int, directory: Path) -> None:
    """Require a pathname to still name the directory held open by ``descriptor``."""

    current = os.stat(directory, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(os.fstat(descriptor), current):
        raise OSError(f"managed directory identity changed during operation: {directory}")


def assert_resolved_directory_binding(descriptor: int, directory: Path) -> None:
    """Require a possibly symlinked caller path to still resolve to an open directory."""

    current = os.stat(directory, follow_symlinks=True)
    if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(os.fstat(descriptor), current):
        raise OSError(f"output directory identity changed during operation: {directory}")


@contextmanager
def bound_directory(
    directory: Path,
    *,
    root: Path,
    create: bool,
    verify_on_exit: bool = True,
) -> Iterator[int]:
    """Open a contained directory chain once, refusing symlinks at every component.

    Callers perform creation, replacement, or deletion relative to the yielded descriptor.
    Renaming a checked parent and replacing its pathname with a symlink therefore cannot retarget
    the operation outside the managed root.
    """

    root_path = Path(os.path.abspath(root))
    directory_path = Path(os.path.abspath(directory))
    if not directory_path.is_relative_to(root_path):
        raise OSError(f"managed directory is outside provisioning root: {directory_path}")
    if create:
        root_path.mkdir(parents=True, exist_ok=True)
    flags = _directory_open_flags()
    descriptors: list[int] = []
    try:
        current = os.open(root_path, flags)
        descriptors.append(current)
        for part in directory_path.relative_to(root_path).parts:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o777, dir_fd=current)
                child = os.open(part, flags, dir_fd=current)
            descriptors.append(child)
            current = child
        try:
            yield current
        except BaseException:
            raise
        else:
            if verify_on_exit:
                assert_directory_binding(current, directory_path)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@dataclass(frozen=True)
class FileDigests:
    """Hashes and actual size measured in one pass over the same regular-file descriptor."""

    sha256: str
    bytes: int
    git_blob_sha1: str | None = None


def sha256_regular_file_at(directory_descriptor: int, name: str) -> str | None:
    """Hash one descriptor-relative regular file without following a leaf symlink."""

    found = regular_file_digests_at(directory_descriptor, name)
    return found.sha256 if found is not None else None


def hash_regular_file_descriptor(descriptor: int, *, git_blob: bool = False) -> FileDigests:
    """Hash an open regular file from its beginning, checking its size and stable identity.

    Git hashes ``blob <actual byte count>\\0`` followed by the bytes. The initial stat supplies
    that prefix, and the read count and final stat must agree with it. This is Git content
    identity over trusted pinned source metadata, not an upstream SHA-256 authentication claim.
    The caller owns the descriptor and its position, which is left at EOF.
    """

    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise OSError("hash input is not a regular file")
    digest = hashlib.sha256()
    blob_digest = hashlib.sha1(usedforsecurity=False) if git_blob else None
    if blob_digest is not None:
        blob_digest.update(f"blob {before.st_size}\0".encode("ascii"))
    os.lseek(descriptor, 0, os.SEEK_SET)
    count = 0
    while chunk := os.read(descriptor, 1024 * 1024):
        count += len(chunk)
        digest.update(chunk)
        if blob_digest is not None:
            blob_digest.update(chunk)
    after = os.fstat(descriptor)
    if (
        count != before.st_size
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise OSError("managed file changed while hashing")
    return FileDigests(digest.hexdigest(), count, blob_digest.hexdigest() if blob_digest else None)


def regular_file_digests_at(
    directory_descriptor: int, name: str, *, git_blob: bool = False
) -> FileDigests | None:
    """Hash a descriptor-relative regular file, refusing opening or hashing replacements."""

    try:
        before = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode):
        return None
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_descriptor,
    )
    try:
        if not os.path.samestat(before, os.fstat(descriptor)):
            raise OSError(f"managed file identity changed while opening: {name}")
        result = hash_regular_file_descriptor(descriptor, git_blob=git_blob)
        current = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if not os.path.samestat(os.fstat(descriptor), current):
            raise OSError(f"managed file identity changed while hashing: {name}")
        return result
    finally:
        os.close(descriptor)


@contextmanager
def temporary_directory(
    prefix: str = "audio-processing-", *, preserve: bool = False
) -> Iterator[Path]:
    """Create a private directory; retained evidence lasts until OS/user cleanup."""
    if preserve:
        yield Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    else:
        with tempfile.TemporaryDirectory(prefix=prefix) as raw:
            yield Path(raw)


@contextmanager
def temporary_output_path(output: Path) -> Iterator[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{output.stem}-",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    temp = Path(raw)
    temp.unlink(missing_ok=True)
    try:
        yield temp
    finally:
        temp.unlink(missing_ok=True)
