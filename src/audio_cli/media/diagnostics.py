"""Private retained diagnostic directories beneath a caller-selected root."""

import os
import shutil
import stat
import tempfile
from pathlib import Path

from .files import resolve_path_identity


def retained_diagnostics_directory(root: Path, *, prefix: str) -> Path:
    """Create a unique 0700 directory or raise; never fall back to temporary storage.

    The successful creation checks actual filesystem access, not advisory permission
    bits. Existing files are never reused. The caller owns retention and lifecycle.
    """
    if not prefix or "/" in prefix or "\\" in prefix:
        raise ValueError("diagnostic directory prefix must be a single path component")
    try:
        resolved = resolve_path_identity(root)
    except RuntimeError as exc:
        raise OSError(f"invalid diagnostic root: {exc}") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=resolved))


def retain_diagnostic_file(source: Path, target: Path) -> None:
    """Copy exact regular-file bytes to a new private diagnostic artifact."""
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(source_descriptor, "rb") as reader:
        if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
            raise OSError(f"diagnostic source is not a regular file: {source}")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
