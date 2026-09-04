"""Host and provisioning diagnostics for ``audio doctor``."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import paths
from ._package_compat import facade_dependency
from ._package_paths import _environment_has_built_runtime
from ._package_registry import load_registry
from ._package_requirements import _module_available
from ._package_runtime import Toolchain
from .environments import environments, packages

def doctor(toolchain: Toolchain | None = None) -> dict:
    """Everything a reader needs before deciding whether a failure is theirs or the tool's."""
    from . import __version__

    toolchain = toolchain or Toolchain()
    document = facade_dependency("load_registry", load_registry)()
    tools = {}
    for tool in ("ffmpeg", "ffprobe", "swift", "uv", "git"):
        location = toolchain.which(tool)
        tools[tool] = {"present": location is not None, "path": location}
    tools["huggingface_hub"] = {"present": _module_available("huggingface_hub"), "path": None}

    root = paths.root()
    usage = shutil.disk_usage(root if root.exists() else Path.home())
    return {
        "tool": {"version": __version__, "path": sys.argv[0],
                 "python": platform.python_version()},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "memory": _memory(),
        "disk": {"total_bytes": usage.total, "free_bytes": usage.free},
        "tools": tools,
        "root": str(root),
        "root_exists": root.exists(),
        "registry": str(paths.registry_path()),
        "environments": {
            name: {
                "state": document["environments"].get(name, {}).get("state", "absent"),
                "python": environment.python,
                "requires_tool": list(environment.requires_tool),
                "blocked_by_missing_tool": [
                    tool for tool in environment.requires_tool
                    if not tools[tool]["present"]
                    and not _environment_has_built_runtime(name, document)
                ],
                "provisional": environment.provisional,
            }
            for name, environment in environments().items() if environment.provisioned
        },
        "packages": {
            identifier: document["packages"].get(identifier, {}).get("state", "absent")
            for identifier in sorted(facade_dependency("packages", packages)())
        },
        "note": (
            "Swift is required to build or repair FluidAudio; a ready built product runs "
            "directly without Swift. Missing provisioning tools are reported rather than fatal."
        ),
    }


def _memory() -> dict:
    """Total and available memory, or nulls where the platform does not report them."""
    total = available = None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10)
            page = 4096
            free = inactive = 0
            for line in result.stdout.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
                elif line.startswith("Pages free:"):
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive = int(line.split(":")[1].strip().rstrip("."))
            available = (free + inactive) * page
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {"total_bytes": total, "available_bytes": available,
            "note": "Host-wide counters; not process-attributable and not summable with "
                    "per-stage peaks."}
