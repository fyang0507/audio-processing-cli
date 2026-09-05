"""Enforce the one narrow dynamic-import exception in runtime source."""

from __future__ import annotations

import ast
from pathlib import Path

from top_level_boundary_support import (
    OPAQUE_DYNAMIC_IMPORT_MODULES,
    module_name,
    module_paths,
    python_source_paths,
    unsupported_dynamic_import_lines,
    unsupported_dynamic_import_uses,
)


def test_host_runtime_source_uses_static_imports_except_for_the_child_probe() -> None:
    violations: dict[str, list[tuple[int, str]]] = {}
    for path in python_source_paths():
        if uses := unsupported_dynamic_import_uses(path):
            violations[module_name(path)] = uses

    assert set(violations) == OPAQUE_DYNAMIC_IMPORT_MODULES
    assert [kind for _line, kind in violations["audio_cli.packages.runtime_probe"]] == [
        "import:importlib",
        "attribute:import_module",
    ]


def test_runtime_probe_has_exact_static_imports_and_one_manifest_target_import() -> None:
    path = module_paths()["audio_cli.packages.runtime_probe"]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    static_imports = [
        ast.unparse(node)
        for node in sorted(
            (node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))),
            key=lambda node: (node.lineno, node.col_offset),
        )
    ]
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
    ]
    importlib_references = [
        node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "importlib"
    ]

    assert static_imports == [
        "from __future__ import annotations",
        "import hashlib",
        "import importlib",
        "import inspect",
        "import json",
        "import sys",
        "from pathlib import Path",
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == 1
    assert isinstance(calls[0].args[0], ast.Name)
    assert calls[0].args[0].id == "module_name"
    assert importlib_references == [calls[0].func.value]


def test_dynamic_import_policy_rejects_aliases_and_reflection(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "import importlib\n"
        "(load,) = (importlib.import_module,)\n"
        "other = (__import__ := load)\n"
        "import builtins\n"
        "reflective = vars(builtins)['__import__']\n"
        "from importlib import __import__ as imported_load\n"
        "from importlib.__init__ import import_module as init_load\n"
        "mapping_load = globals()['__builtins__']['__import__']\n"
        "eval(\"__import__('audio_cli.media')\")\n",
        encoding="utf-8",
    )

    assert unsupported_dynamic_import_lines(source) == list(range(1, 10))


def test_dynamic_import_policy_counts_multiple_constructs_on_one_line(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "import importlib\n"
        "module = (__import__('numpy') and importlib.import_module(module_name))\n",
        encoding="utf-8",
    )

    assert [kind for _line, kind in unsupported_dynamic_import_uses(source)] == [
        "import:importlib",
        "attribute:import_module",
        "name:__import__",
    ]


def test_dynamic_import_policy_rejects_constant_reflective_lookups(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "import importlib\n"
        "direct = getattr(importlib, 'import_module')('numpy')\n"
        "mapping = vars(importlib)['import_' + 'module']('numpy')\n",
        encoding="utf-8",
    )

    assert [kind for _line, kind in unsupported_dynamic_import_uses(source)] == [
        "import:importlib",
        "string:import_module",
        "string:import_module",
    ]
