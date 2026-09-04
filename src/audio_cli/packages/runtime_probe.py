"""Inspect one manifest-declared Python target inside a managed environment.

This file is executed directly by the managed environment's interpreter.  It must stay
stdlib-only: the core ``audio_cli`` package is not required to be installed in that environment.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path


def inspect_target(dotted_target: str) -> dict[str, object]:
    """Return the defining module hash and callable parameters for one dotted target."""
    module_name, class_name, method_name = dotted_target.rsplit(".", 2)
    module = importlib.import_module(module_name)
    source_path = Path(inspect.getfile(module))
    owner = getattr(module, class_name)
    method = getattr(owner, method_name)
    return {
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "params": sorted(inspect.signature(method).parameters),
    }


def main(argv: list[str] | None = None) -> int:
    """Write one compact JSON verdict for the parent verifier."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: runtime_probe.py MODULE.CLASS.METHOD", file=sys.stderr)
        return 2
    print(json.dumps(inspect_target(args[0]), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
