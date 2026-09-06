"""Private retained diagnostic directories beneath a caller-selected root."""

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
