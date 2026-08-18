"""Where provisioned things live.

One root, resolved one way, so `registry.json`, the environments, and the pinned single-file
artifacts cannot end up in three different places. VOCABULARY.md fixes the tree:

    <root>/
      registry.json
      models/
      envs/<name>/

`AUDIO_PROCESSING_MODEL_CACHE` overrides the root, not the layout inside it. That is a change
from the first Silero backend, which treated the override as the models directory itself and so
produced a different tree depending on whether the variable was set. The default location is
unchanged; only the override case moves, and the one artifact affected is a hash-pinned 2.3 MB
file that re-fetches on demand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def root() -> Path:
    override = os.environ.get("AUDIO_PROCESSING_MODEL_CACHE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "audio-processing-cli"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "audio-processing-cli"
    return Path.home() / ".cache" / "audio-processing-cli"


def models_dir() -> Path:
    return root() / "models"


def envs_dir() -> Path:
    return root() / "envs"


def env_dir(name: str) -> Path:
    return envs_dir() / name


def env_python(name: str) -> Path:
    return env_dir(name) / "bin" / "python"


def registry_path() -> Path:
    return root() / "registry.json"


def checkout_dir(environment: str, package_id: str) -> Path:
    """A package's pinned source checkout, inside the environment that installs it.

    VOCABULARY's tree puts the checkout and its applied patch under the environment directory,
    which also means removing an environment removes its checkouts with it rather than leaving
    them for a separate cleanup step to forget.
    """
    return env_dir(environment) / "src" / package_id
