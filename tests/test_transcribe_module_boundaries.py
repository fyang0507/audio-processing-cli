from __future__ import annotations

import ast
from pathlib import Path

TRANSCRIBE = Path(__file__).resolve().parents[1] / "src" / "audio_cli" / "transcribe"
FACADES = {"orchestrator", "native"}


def _local_imports(path: Path, modules: set[str]) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module:
            candidate = node.module.split(".", maxsplit=1)[0]
            if candidate in modules:
                imports.add(candidate)
            continue
        imports.update(alias.name for alias in node.names if alias.name in modules)
    return imports


def _transcribe_graph() -> dict[str, set[str]]:
    paths = {path.stem: path for path in TRANSCRIBE.glob("*.py")}
    modules = set(paths)
    return {
        module: _local_imports(path, modules)
        for module, path in paths.items()
    }


def test_transcribe_implementations_do_not_import_compatibility_facades() -> None:
    graph = _transcribe_graph()
    backedges = {
        module: sorted(imports & FACADES)
        for module, imports in graph.items()
        if module not in {*FACADES, "__init__"} and imports & FACADES
    }
    assert backedges == {}


def test_transcribe_local_import_graph_has_no_cycles() -> None:
    graph = _transcribe_graph()
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            cycle = visiting[visiting.index(module):] + [module]
            raise AssertionError("transcribe import cycle: " + " -> ".join(cycle))
        if module in visited:
            return
        visiting.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)
