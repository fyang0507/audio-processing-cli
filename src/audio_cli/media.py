from __future__ import annotations

import ctypes
import errno
import json
import os
import stat
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .media_errors import MediaError
from .media_ffmpeg import (
    ENHANCED_MARKER,
    decode_audio,
    encode_output,
    ffmpeg_version,
    is_enhanced_media,
    measure_loudness,
    media_summary,
    probe_media,
    render_loudness_normalized,
    require_runtime,
    write_float_wav,
)
from .media_files import (
    assert_directory_binding,
    assert_resolved_directory_binding,
    bound_directory,
    hash_file,
    sha256_regular_file_at,
    temporary_directory,
    temporary_output_path,
)


@dataclass(frozen=True)
class ProtectedFileIdentity:
    """A regular file held by device/inode identity across a long operation."""

    path: Path
    device: int
    inode: int


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
            identity
            for identity in identities
            if (state.st_dev, state.st_ino) == (identity.device, identity.inode)
        ),
        None,
    )


def entry_matches_file_identity(
    directory_descriptor: int,
    name: str,
    identity: ProtectedFileIdentity,
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
        and (state.st_dev, state.st_ino) == (identity.device, identity.inode)
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
    temporary_identity: ProtectedFileIdentity,
    protected_identities: Sequence[ProtectedFileIdentity] = (),
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
    if not entry_matches_file_identity(
        directory_descriptor, temporary_name, temporary_identity
    ):
        raise OSError(
            f"writer temporary changed identity before publication: {output_path}"
        )

    if not force:
        protected = matching_protected_identity(
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
        if not entry_matches_file_identity(
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
            if not entry_matches_file_identity(
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

        if not entry_matches_file_identity(
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
        protected = matching_protected_identity(
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
    temporary_identity: ProtectedFileIdentity,
) -> None:
    """Best-effort removal when the private name still matches at inspection.

    The random sibling is outside the public-path race boundary: a same-credential
    process that discovers and mutates it after this check can already unlink the
    caller's canonical files directly, and POSIX has no portable unlink-if-inode
    primitive.
    """
    if not entry_matches_file_identity(
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
    protected_identities: Sequence[ProtectedFileIdentity] = (),
) -> None:
    """Publish text through an exclusive sibling, optionally without clobbering."""
    identities = list(protected_identities)
    for protected_path in protected_paths:
        identity = capture_file_identity(Path(protected_path))
        if identity is not None and all(
            (identity.device, identity.inode) != (known.device, known.inode)
            for known in identities
        ):
            identities.append(identity)
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
                temporary_identity = file_identity_from_descriptor(
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
    protected_identities: Sequence[ProtectedFileIdentity] = (),
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
    "ENHANCED_MARKER",
    "MediaError",
    "ProtectedFileIdentity",
    "ProtectedOutputError",
    "assert_directory_binding",
    "assert_resolved_directory_binding",
    "atomic_write_json",
    "atomic_write_text",
    "bound_directory",
    "capture_file_identity",
    "cleanup_temporary_file",
    "decode_audio",
    "encode_output",
    "entry_matches_file_identity",
    "ffmpeg_version",
    "file_identity_from_descriptor",
    "hash_file",
    "is_enhanced_media",
    "matching_protected_identity",
    "measure_loudness",
    "media_summary",
    "probe_media",
    "publish_temporary_file",
    "render_loudness_normalized",
    "require_runtime",
    "sha256_regular_file_at",
    "temporary_directory",
    "temporary_output_path",
    "write_float_wav",
]
