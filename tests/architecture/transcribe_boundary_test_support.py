"""Source inventory and import-graph mechanics for transcription boundary tests."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from tests.architecture.ast_test_support import constant_builtin_import_fromlist, constant_string

ROOT_FACADE = "<root>"
_INIT_FILES = {"__init__.py", "__init__.pyi"}


def python_source_paths(root: Path) -> list[Path]:
    """Return every runtime or stub source below the transcription package."""
    return sorted((*root.rglob("*.py"), *root.rglob("*.pyi")))


def module_name(path: Path, root: Path) -> str:
    """Map a source path to its module name relative to ``root``."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def source_layout(root: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    """Inventory direct directories, packages, Python-bearing owners, and root modules."""
    sources = python_source_paths(root)
    directories = {
        path.name
        for path in root.iterdir()
        if path.name != "__pycache__" and (path.is_dir() or path.is_symlink())
    }
    importable = {
        path.name for path in root.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    }
    python_bearing = {
        path.relative_to(root).parts[0] for path in sources if len(path.relative_to(root).parts) > 1
    }
    root_modules = {
        path.stem
        for pattern in ("*.py", "*.pyi")
        for path in root.glob(pattern)
        if path.name not in _INIT_FILES
    }
    return directories, importable, python_bearing, root_modules


def _is_package_source(path: Path) -> bool:
    return path.name in _INIT_FILES


def _without_init_suffix(target: str) -> str:
    return "" if target == "__init__" else target.removesuffix(".__init__")


def _dynamic_loader_names(tree: ast.AST) -> tuple[set[str], set[str], set[str], set[str]]:
    importlib_modules = {"importlib"}
    import_functions: set[str] = set()
    builtins_modules = {"builtins"}
    builtin_functions = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_modules.add(alias.asname or alias.name)
                elif alias.name == "builtins":
                    builtins_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in {"builtins", "importlib"}:
            for alias in node.names:
                if node.module == "importlib" and alias.name == "import_module":
                    import_functions.add(alias.asname or alias.name)
                elif node.module == "builtins" and alias.name == "__import__":
                    builtin_functions.add(alias.asname or alias.name)

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target.id]
                value = node.value
            else:
                continue
            if value is None:
                continue
            destination: set[str] | None = None
            if isinstance(value, ast.Name):
                if value.id in importlib_modules:
                    destination = importlib_modules
                elif value.id in import_functions:
                    destination = import_functions
                elif value.id in builtins_modules:
                    destination = builtins_modules
                elif value.id in builtin_functions:
                    destination = builtin_functions
            elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                if value.attr == "import_module" and value.value.id in importlib_modules:
                    destination = import_functions
                elif value.attr == "__import__" and value.value.id in builtins_modules:
                    destination = builtin_functions
            if destination is not None:
                before = len(destination)
                destination.update(targets)
                changed = changed or len(destination) != before
    return importlib_modules, import_functions, builtins_modules, builtin_functions


def _constant_dynamic_import(
    node: ast.AST,
    loader_names: tuple[set[str], set[str], set[str], set[str]],
    *,
    source_package: str,
) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    importlib_modules, import_functions, builtins_modules, builtin_functions = loader_names
    importlib_call = (isinstance(node.func, ast.Name) and node.func.id in import_functions) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_modules
    )
    builtin_call = (isinstance(node.func, ast.Name) and node.func.id in builtin_functions) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "__import__"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in builtins_modules
    )
    if not importlib_call and not builtin_call:
        return None

    argument: ast.AST | None = node.args[0] if node.args else None
    if argument is None:
        argument = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "name"),
            None,
        )
    target = constant_string(argument)
    if target is None:
        raise AssertionError(
            f"dynamic import at line {node.lineno} must have a constant module target"
        )
    if builtin_call and not importlib_call:
        level_node: ast.AST | None = node.args[4] if len(node.args) > 4 else None
        if level_node is None:
            level_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "level"),
                None,
            )
        if level_node is None:
            level = 0
        elif (
            not isinstance(level_node, ast.Constant)
            or isinstance(level_node.value, bool)
            or not isinstance(level_node.value, int)
        ):
            raise AssertionError(
                f"dynamic import at line {node.lineno} must have a constant integer level"
            )
        else:
            level = level_node.value
        if level < 0:
            raise AssertionError(
                f"dynamic import at line {node.lineno} must have a non-negative level"
            )
        if level:
            package = _builtin_relative_package(node, source_package)
            return importlib.util.resolve_name("." * level + target, package)
        return target
    if not target.startswith(".") or not importlib_call:
        return target

    package_node: ast.AST | None = node.args[1] if len(node.args) > 1 else None
    if package_node is None:
        package_node = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "package"),
            None,
        )
    package = (
        source_package
        if isinstance(package_node, ast.Name) and package_node.id == "__package__"
        else constant_string(package_node)
    )
    if package is None:
        raise AssertionError(
            f"relative dynamic import at line {node.lineno} must have a constant package "
            "or use __package__ directly"
        )
    return importlib.util.resolve_name(target, package)


def _builtin_relative_package(node: ast.Call, source_package: str) -> str:
    globals_node: ast.AST | None = node.args[1] if len(node.args) > 1 else None
    if globals_node is None:
        globals_node = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "globals"),
            None,
        )
    if globals_node is None or (
        isinstance(globals_node, ast.Constant) and globals_node.value is None
    ):
        return source_package
    if (
        isinstance(globals_node, ast.Call)
        and isinstance(globals_node.func, ast.Name)
        and globals_node.func.id == "globals"
        and not globals_node.args
        and not globals_node.keywords
    ):
        return source_package
    raise AssertionError(
        f"relative dynamic import at line {node.lineno} must omit globals, pass None, "
        "or call globals() directly"
    )


def _parsed_source(path: Path) -> tuple[ast.AST, tuple[set[str], set[str], set[str], set[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return tree, _dynamic_loader_names(tree)


def local_imports(path: Path, module: str, modules: set[str]) -> set[str]:
    """Return imports whose targets live below ``audio_cli.transcribe``."""
    imports: set[str] = set()
    tree, loader_names = _parsed_source(path)
    source_package = _source_package(path, module)
    for node in ast.walk(tree):
        if dynamic_target := _constant_dynamic_import(
            node, loader_names, source_package=source_package
        ):
            dynamic_targets = [dynamic_target]
            dynamic_targets.extend(
                f"{dynamic_target}.{name}"
                for name in constant_builtin_import_fromlist(node, loader_names)
            )
            for target in dynamic_targets:
                target = _without_init_suffix(target)
                if target == "audio_cli.transcribe":
                    imports.add(ROOT_FACADE)
                elif target.startswith("audio_cli.transcribe."):
                    candidate = target.removeprefix("audio_cli.transcribe.")
                    if candidate in modules:
                        imports.add(candidate)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                prefix = "audio_cli.transcribe."
                if alias.name == "audio_cli.transcribe":
                    imports.add(ROOT_FACADE)
                elif alias.name.startswith(prefix):
                    target = _without_init_suffix(alias.name)
                    candidate = target.removeprefix(prefix)
                    if target == "audio_cli.transcribe":
                        imports.add(ROOT_FACADE)
                    elif candidate in modules:
                        imports.add(candidate)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        explicit_init = node.module == "__init__" or bool(
            node.module and node.module.endswith(".__init__")
        )
        if node.level:
            relative = "." * node.level + (node.module or "")
            target = _without_init_suffix(importlib.util.resolve_name(relative, source_package))
            if target == "audio_cli.transcribe":
                candidate = ""
            elif target.startswith("audio_cli.transcribe."):
                candidate = target.removeprefix("audio_cli.transcribe.")
            else:
                continue
        elif node.module == "audio_cli":
            if any(alias.name == "transcribe" for alias in node.names):
                imports.add(ROOT_FACADE)
            continue
        elif node.module == "audio_cli.transcribe":
            candidate = ""
        elif node.module and node.module.startswith("audio_cli.transcribe."):
            raw_candidate = node.module.removeprefix("audio_cli.transcribe.")
            candidate = _without_init_suffix(raw_candidate)
        else:
            continue
        if explicit_init:
            imports.add(candidate or ROOT_FACADE)
        elif not candidate:
            for alias in node.names:
                imports.add(alias.name if alias.name in modules else ROOT_FACADE)
        else:
            children = {
                ".".join(filter(None, (candidate, alias.name))) for alias in node.names
            } & modules
            if children:
                imports.update(children)
            elif candidate in modules:
                imports.add(candidate)
    return imports


def _source_package(path: Path, module: str) -> str:
    full_module = "audio_cli.transcribe" + (f".{module}" if module else "")
    return full_module if _is_package_source(path) else full_module.rpartition(".")[0]


def external_audio_cli_dependencies(path: Path, module: str) -> set[str]:
    """Return root names imported from outside ``audio_cli.transcribe``."""
    source_package = _source_package(path, module)
    dependencies: set[str] = set()
    tree, loader_names = _parsed_source(path)
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = "." * node.level + (node.module or "")
                base = importlib.util.resolve_name(relative, source_package)
            else:
                base = node.module or ""
            base = _without_init_suffix(base)
            if base == "audio_cli":
                targets.extend(f"audio_cli.{alias.name}" for alias in node.names)
            else:
                targets.append(base)
        elif dynamic_target := _constant_dynamic_import(
            node, loader_names, source_package=source_package
        ):
            targets.append(_without_init_suffix(dynamic_target))
        for target in targets:
            if target == "audio_cli":
                dependencies.add("<root>")
            elif target.startswith("audio_cli.") and not target.startswith("audio_cli.transcribe"):
                dependencies.add(target.split(".", 2)[1])
    return dependencies


def facade_imported_symbols(path: Path, module: str, facade: str) -> set[str]:
    """Return symbols imported from a top-level facade, including unsafe module access."""
    source_package = _source_package(path, module)
    facade_module = f"audio_cli.{facade}"
    symbols: set[str] = set()
    tree, loader_names = _parsed_source(path)
    for node in ast.walk(tree):
        if dynamic_target := _constant_dynamic_import(
            node, loader_names, source_package=source_package
        ):
            dynamic_target = _without_init_suffix(dynamic_target)
            if dynamic_target == facade_module:
                symbols.add("<module>")
            elif dynamic_target.startswith(f"{facade_module}."):
                symbols.add(f"<concrete:{dynamic_target.removeprefix(f'{facade_module}.')}>")
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _without_init_suffix(alias.name)
                if target == facade_module:
                    symbols.add("<module>")
                elif target.startswith(f"{facade_module}."):
                    symbols.add(f"<concrete:{target.removeprefix(f'{facade_module}.')}>")
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            relative = "." * node.level + (node.module or "")
            base = importlib.util.resolve_name(relative, source_package)
        else:
            base = node.module or ""
        base = _without_init_suffix(base)
        if base == "audio_cli":
            if any(alias.name == facade for alias in node.names):
                symbols.add("<module>")
        elif base == facade_module:
            symbols.update(alias.name for alias in node.names)
        elif base.startswith(f"{facade_module}."):
            symbols.add(f"<concrete:{base.removeprefix(f'{facade_module}.')}>")
    return symbols


def imported_roots(path: Path, module: str) -> set[str]:
    """Return top-level names reached by static or supported dynamic imports."""
    source_package = _source_package(path, module)
    roots: set[str] = set()
    tree, loader_names = _parsed_source(path)
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            targets.append(node.module or "")
        elif dynamic_target := _constant_dynamic_import(
            node, loader_names, source_package=source_package
        ):
            targets.append(dynamic_target)
        roots.update(target.split(".", 1)[0] for target in targets if target)
    return roots


def transcribe_graph(root: Path) -> dict[str, set[str]]:
    """Build the complete local import graph and reject colliding source modules."""
    paths: dict[str, Path] = {}
    for path in python_source_paths(root):
        module = module_name(path, root)
        if not module:
            continue
        if previous := paths.get(module):
            raise AssertionError(f"duplicate module sources for {module}: {previous}, {path}")
        paths[module] = path
    modules = set(paths)
    return {module: local_imports(path, module, modules) for module, path in paths.items()}


def assert_acyclic(graph: dict[str, set[str]]) -> None:
    """Raise with the first cycle found in a local import graph."""
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
