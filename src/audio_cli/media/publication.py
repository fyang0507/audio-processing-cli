"""Identity-safe publication of files and structured reports."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import stat
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from . import identity
from .files import assert_resolved_directory_binding, bound_directory


class ProtectedOutputError(OSError):
    """Publication would replace a file captured as canonical input."""

    def __init__(
        self,
        output: Path,
        protected: Path,
        *,
        preserved_at: Path | None = None,
    ) -> None:
        self.output = Path(output)
        self.protected = Path(protected)
        self.preserved_at = Path(preserved_at) if preserved_at is not None else None
        preservation = (
            f"; protected bytes remain at {self.preserved_at}"
            if self.preserved_at is not None
            else ""
        )
        super().__init__(
            f"output {self.output} resolves to protected input {self.protected}"
            f"{preservation}"
        )


def _rename_exchange(
    directory_descriptor: int, left_name: str, right_name: str
) -> None:
    """Atomically exchange two descriptor-relative directory entries."""
    library = ctypes.CDLL(None, use_errno=True)
    left = os.fsencode(left_name)
    right = os.fsencode(right_name)
    if sys.platform == "darwin":
        try:
            function = library.renameatx_np
        except AttributeError as exc:
            raise OSError(errno.ENOTSUP, "atomic rename exchange is unavailable") from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
    elif sys.platform.startswith("linux"):
        try:
            function = library.renameat2
        except AttributeError as exc:
            raise OSError(errno.ENOTSUP, "atomic rename exchange is unavailable") from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
    else:
        raise OSError(errno.ENOTSUP, "atomic rename exchange is unavailable")
    if function(
        directory_descriptor,
        left,
        directory_descriptor,
        right,
        0x00000002,
    ) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), right_name)


def publish_temporary_file(
    directory_descriptor: int,
    temporary_name: str,
    output_name: str,
    *,
    output_path: Path,
    force: bool,
    temporary_identity: identity.ProtectedFileIdentity,
    protected_identities: Sequence[identity.ProtectedFileIdentity] = (),
    replace_non_directory: bool = False,
) -> None:
    """Claim a destination without a check/replace race.

    A force write atomically exchanges the temporary and current destination.  The
    displaced inode then occupies the private temporary name and can be inspected before
    deletion; a protected or refused inode is atomically exchanged back.  Managed-cache
    repair may explicitly replace a non-directory leaf (for example, a symlink) without
    following it.  Directories are always refused.  When no destination exists, a hard
    link remains the atomic create-if-absent primitive.
    """
    if not identity.entry_matches_file_identity(
        directory_descriptor, temporary_name, temporary_identity
    ):
        raise OSError(
            f"writer temporary changed identity before publication: {output_path}"
        )

    if not force:
        protected = identity.matching_protected_identity(
            directory_descriptor, output_name, protected_identities
        )
        if protected is not None:
            raise ProtectedOutputError(output_path, protected.path)
        os.link(
            temporary_name,
            output_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not identity.entry_matches_file_identity(
            directory_descriptor, output_name, temporary_identity
        ):
            raise OSError(
                f"published destination does not contain the writer temporary: "
                f"{output_path}"
            )
        cleanup_temporary_file(
            directory_descriptor, temporary_name, temporary_identity
        )
        return

    for _attempt in range(8):
        try:
            os.stat(
                output_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            try:
                os.link(
                    temporary_name,
                    output_name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                continue
            if not identity.entry_matches_file_identity(
                directory_descriptor, output_name, temporary_identity
            ):
                raise OSError(
                    f"published destination does not contain the writer temporary: "
                    f"{output_path}"
                )
            cleanup_temporary_file(
                directory_descriptor, temporary_name, temporary_identity
            )
            return
        try:
            _rename_exchange(
                directory_descriptor, temporary_name, output_name
            )
        except FileNotFoundError:
            continue

        if not identity.entry_matches_file_identity(
            directory_descriptor, output_name, temporary_identity
        ):
            try:
                _rename_exchange(
                    directory_descriptor, temporary_name, output_name
                )
            except OSError as exc:
                raise OSError(
                    f"could not atomically restore destination after writer "
                    f"temporary identity changed for {output_path}: {exc}"
                ) from exc
            raise OSError(
                f"writer temporary changed identity during publication: {output_path}"
            )

        state = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        protected = identity.matching_protected_identity(
            directory_descriptor, temporary_name, protected_identities
        )
        refused_kind = not stat.S_ISREG(state.st_mode) and (
            not replace_non_directory or stat.S_ISDIR(state.st_mode)
        )
        if refused_kind or protected is not None:
            try:
                _rename_exchange(
                    directory_descriptor, temporary_name, output_name
                )
            except OSError as exc:
                if protected is not None:
                    raise ProtectedOutputError(
                        output_path,
                        protected.path,
                        preserved_at=output_path.parent / temporary_name,
                    ) from exc
                raise OSError(
                    f"could not atomically restore refused destination {output_path}: {exc}"
                ) from exc
            if protected is not None:
                raise ProtectedOutputError(output_path, protected.path)
            raise OSError(
                f"destination changed to a non-regular file during publication: "
                f"{output_path}"
            )
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        return
    raise FileExistsError(
        f"destination changed repeatedly during publication: {output_path}"
    )


def cleanup_temporary_file(
    directory_descriptor: int,
    temporary_name: str,
    temporary_identity: identity.ProtectedFileIdentity,
) -> None:
    """Best-effort removal when the private name still matches at inspection.

    The random sibling is outside the public-path race boundary: a same-credential
    process that discovers and mutates it after this check can already unlink the
    caller's canonical files directly, and POSIX has no portable unlink-if-inode
    primitive.
    """
    if not identity.entry_matches_file_identity(
        directory_descriptor, temporary_name, temporary_identity
    ):
        return
    os.unlink(temporary_name, dir_fd=directory_descriptor)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    force: bool = True,
    protected_paths: Sequence[Path] = (),
    protected_identities: Sequence[identity.ProtectedFileIdentity] = (),
) -> None:
    """Publish text through an exclusive sibling, optionally without clobbering."""
    identities = list(protected_identities)
    for protected_path in protected_paths:
        protected = identity.capture_file_identity(Path(protected_path))
        if protected is not None and all(
            (protected.device, protected.inode) != (known.device, known.inode)
            for known in identities
        ):
            identities.append(protected)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    anchor = Path(resolved_parent.anchor)
    with bound_directory(
        resolved_parent,
        root=anchor,
        create=False,
    ) as parent_descriptor:
        temporary_name = f".audio-write-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        created = False
        temporary_identity = None
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o666,
                dir_fd=parent_descriptor,
            )
            created = True
            try:
                temporary_identity = identity.file_identity_from_descriptor(
                    descriptor, path.parent / temporary_name
                )
            finally:
                if temporary_identity is None:
                    os.close(descriptor)
            if temporary_identity is None:
                raise OSError(
                    f"writer temporary is not a regular file: "
                    f"{path.parent / temporary_name}"
                )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            assert_resolved_directory_binding(parent_descriptor, path.parent)
            publish_temporary_file(
                parent_descriptor,
                temporary_name,
                path.name,
                output_path=path,
                force=force,
                temporary_identity=temporary_identity,
                protected_identities=identities,
            )
            created = False
        finally:
            if created and temporary_identity is not None:
                cleanup_temporary_file(
                    parent_descriptor, temporary_name, temporary_identity
                )


def atomic_write_json(
    path: Path,
    payload: dict[str, object],
    *,
    force: bool = True,
    protected_paths: Sequence[Path] = (),
    protected_identities: Sequence[identity.ProtectedFileIdentity] = (),
) -> None:
    """Written through a sibling and renamed, so a report is never observed half-written.

    Deliberately a plain `open` rather than `mkstemp`. `mkstemp` creates 0600 and `os.replace`
    carries that mode onto the destination, so a report arrived stricter than the render it
    describes -- 0600 beside an 0644 wav, under the same umask, for no reason a reader could act
    on. Everything else this tool publishes honours the umask, including `save_registry`, which
    reached the same conclusion by writing the same way.
    """
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    atomic_write_text(
        path,
        text,
        force=force,
        protected_paths=protected_paths,
        protected_identities=protected_identities,
    )


__all__ = [
    "ProtectedOutputError",
    "atomic_write_json",
    "atomic_write_text",
    "cleanup_temporary_file",
    "publish_temporary_file",
]
