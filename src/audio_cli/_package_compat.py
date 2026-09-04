"""Late binding for dependencies historically patched on ``audio_cli.packages``."""

from __future__ import annotations

import sys
from typing import TypeVar


Dependency = TypeVar("Dependency")


def facade_dependency(name: str, fallback: Dependency) -> Dependency:
    """Resolve one overridable dependency without importing the facade recursively."""
    facade = sys.modules.get("audio_cli.packages")
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)
