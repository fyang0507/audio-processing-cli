from __future__ import annotations

import ast
from pathlib import Path

TRANSCRIBE = Path(__file__).resolve().parents[1] / "src" / "audio_cli" / "transcribe"
FACADES = {"native", "orchestrator", "planner", "refusals", "result", "transport"}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(TRANSCRIBE).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _relative_module(
    source: str,
    *,
    is_package: bool,
    level: int,
    module: str | None,
) -> str:
    package = source.split(".") if is_package else source.split(".")[:-1]
    keep = len(package) - (level - 1)
    base = package[:keep]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _local_imports(
    path: Path,
    module: str,
    modules: set[str],
) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                prefix = "audio_cli.transcribe."
                if alias.name.startswith(prefix):
                    candidate = alias.name.removeprefix(prefix)
                    if candidate in modules:
                        imports.add(candidate)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            candidate = _relative_module(
                module,
                is_package=path.name == "__init__.py",
                level=node.level,
                module=node.module,
            )
        elif node.module == "audio_cli.transcribe":
            candidate = ""
        elif node.module and node.module.startswith("audio_cli.transcribe."):
            candidate = node.module.removeprefix("audio_cli.transcribe.")
        else:
            continue
        children = {
            ".".join(filter(None, (candidate, alias.name))) for alias in node.names
        } & modules
        if children:
            imports.update(children)
        elif candidate in modules:
            imports.add(candidate)
    return imports


def test_absolute_root_imports_resolve_to_facades(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text("from audio_cli.transcribe import planner\n", encoding="utf-8")

    assert _local_imports(source, "native.common", {"native.common", "planner"}) == {"planner"}


def _transcribe_graph() -> dict[str, set[str]]:
    paths = {_module_name(path): path for path in TRANSCRIBE.rglob("*.py") if _module_name(path)}
    modules = set(paths)
    return {module: _local_imports(path, module, modules) for module, path in paths.items()}


def test_transcribe_implementations_do_not_import_compatibility_facades() -> None:
    graph = _transcribe_graph()
    backedges = {
        module: sorted(imports & FACADES)
        for module, imports in graph.items()
        if module not in FACADES and imports & FACADES
    }
    assert backedges == {}


def test_transcribe_local_import_graph_has_no_cycles() -> None:
    graph = _transcribe_graph()
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            cycle = [*visiting[visiting.index(module) :], module]
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
