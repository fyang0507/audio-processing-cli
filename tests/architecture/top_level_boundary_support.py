"""Static import discovery for the top-level ``audio_cli`` boundary tests."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterator
from pathlib import Path

from tests.architecture.ast_test_support import constant_builtin_import_fromlist, constant_string

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "audio_cli"
DIRECT_PACKAGES = {
    "command",
    "dsp",
    "environments",
    "export",
    "media",
    "packages",
    "pipeline",
    "transcribe",
}
OPAQUE_DYNAMIC_IMPORT_MODULES = frozenset({"audio_cli.packages.runtime_probe"})


def module_name(path: Path) -> str:
    parts = list(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def python_source_paths() -> list[Path]:
    return sorted((*PACKAGE_ROOT.rglob("*.py"), *PACKAGE_ROOT.rglob("*.pyi")))


def module_paths() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in python_source_paths():
        if "__pycache__" in path.parts:
            continue
        module = module_name(path)
        if previous := modules.get(module):
            raise AssertionError(f"duplicate module sources for {module}: {previous}, {path}")
        modules[module] = path
    return modules


def source_package(module: str, path: Path) -> str:
    return module if path.name in {"__init__.py", "__init__.pyi"} else module.rpartition(".")[0]


def unsupported_dynamic_import_uses(path: Path) -> list[tuple[int, str]]:
    """Inventory each dynamic-loading or ordinary reflective construct."""
    uses: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            uses.extend(
                (node.lineno, f"import:{alias.name}")
                for alias in node.names
                if alias.name.removesuffix(".__init__") in {"builtins", "importlib"}
            )
        elif isinstance(node, ast.ImportFrom) and (
            imported_from := (node.module or "").removesuffix(".__init__")
        ) in {"builtins", "importlib"}:
            uses.extend(
                (node.lineno, f"from:{node.module}:{alias.name}")
                for alias in node.names
                if imported_from == "builtins" or alias.name in {"__import__", "import_module"}
            )
        elif isinstance(node, ast.Name) and node.id in {
            "__builtins__",
            "__import__",
            "builtins",
            "eval",
            "exec",
        }:
            uses.append((node.lineno, f"name:{node.id}"))
        elif isinstance(node, ast.Attribute) and node.attr in {"__import__", "import_module"}:
            uses.append((node.lineno, f"attribute:{node.attr}"))
        elif (reflected_name := constant_string(node)) in {
            "__builtins__",
            "__import__",
            "import_module",
        }:
            uses.append((node.lineno, f"string:{reflected_name}"))
    return sorted(uses)


def unsupported_dynamic_import_lines(path: Path) -> list[int]:
    """Return lines containing at least one unsupported dynamic-loading construct."""
    return sorted({line for line, _kind in unsupported_dynamic_import_uses(path)})


def _from_base(node: ast.ImportFrom, source: str) -> str:
    if not node.level:
        return node.module or ""
    relative = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(relative, source)


def without_init_suffix(target: str) -> str:
    return target.removesuffix(".__init__")


def _dynamic_loader_names(tree: ast.AST) -> tuple[set[str], set[str], set[str], set[str]]:
    importlib_modules = {"importlib"}
    import_functions: set[str] = set()
    builtins_modules = {"builtins"}
    builtin_functions = {"__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name.removesuffix(".__init__")
                binding = alias.asname or alias.name.split(".", 1)[0]
                if imported == "importlib":
                    importlib_modules.add(binding)
                elif imported == "builtins":
                    builtins_modules.add(binding)
        elif isinstance(node, ast.ImportFrom) and (
            imported_from := (node.module or "").removesuffix(".__init__")
        ) in {"builtins", "importlib"}:
            for alias in node.names:
                if imported_from == "importlib" and alias.name == "import_module":
                    import_functions.add(alias.asname or alias.name)
                elif alias.name == "__import__":
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
                for aliases in (
                    importlib_modules,
                    import_functions,
                    builtins_modules,
                    builtin_functions,
                ):
                    if value.id in aliases:
                        destination = aliases
                        break
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


def _dynamic_import_target(
    node: ast.AST,
    loader_names: tuple[set[str], set[str], set[str], set[str]],
    *,
    source: str,
    allow_opaque: bool = False,
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
        if allow_opaque:
            return None
        raise AssertionError("dynamic imports must use a statically resolvable module name")

    if importlib_call and target.startswith("."):
        package_node: ast.AST | None = node.args[1] if len(node.args) > 1 else None
        if package_node is None:
            package_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "package"),
                None,
            )
        package = (
            source
            if isinstance(package_node, ast.Name) and package_node.id == "__package__"
            else constant_string(package_node)
        )
        if package is None:
            raise AssertionError("relative dynamic imports must use a static package")
        return importlib.util.resolve_name(target, package)

    if builtin_call:
        level_node: ast.AST | None = node.args[4] if len(node.args) > 4 else None
        if level_node is None:
            level_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "level"),
                None,
            )
        level = ast.literal_eval(level_node) if level_node is not None else 0
        if not isinstance(level, int) or isinstance(level, bool) or level < 0:
            raise AssertionError("dynamic import levels must be static non-negative integers")
        if level:
            globals_node: ast.AST | None = node.args[1] if len(node.args) > 1 else None
            if globals_node is None:
                globals_node = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "globals"),
                    None,
                )
            direct_globals = (
                globals_node is None
                or (isinstance(globals_node, ast.Constant) and globals_node.value is None)
                or (
                    isinstance(globals_node, ast.Call)
                    and isinstance(globals_node.func, ast.Name)
                    and globals_node.func.id == "globals"
                    and not globals_node.args
                    and not globals_node.keywords
                )
            )
            if not direct_globals:
                raise AssertionError(
                    "relative __import__ calls must use the source module globals directly"
                )
            return importlib.util.resolve_name("." * level + target, source)

    return target


def local_import_targets(
    tree: ast.AST,
    *,
    source: str,
    known_modules: set[str],
    allow_opaque_dynamic: bool = False,
) -> Iterator[tuple[int, str]]:
    """Yield the most specific local module named by every import.

    For ``from audio_cli import paths`` the target is ``audio_cli.paths`` when that
    module exists. For ``from audio_cli import __version__`` the synthetic
    ``audio_cli.__version__`` target lets the allowlist govern that package attribute too.
    """
    emitted: set[tuple[int, str]] = set()
    loader_names = _dynamic_loader_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "audio_cli" or alias.name.startswith("audio_cli."):
                    item = (node.lineno, without_init_suffix(alias.name))
                    if item not in emitted:
                        emitted.add(item)
                        yield item
            continue
        if dynamic_target := _dynamic_import_target(
            node,
            loader_names,
            source=source,
            allow_opaque=allow_opaque_dynamic,
        ):
            dynamic_targets = [dynamic_target]
            dynamic_targets.extend(
                candidate
                for name in constant_builtin_import_fromlist(node, loader_names)
                if (candidate := f"{dynamic_target}.{name}") in known_modules
            )
            for target in dynamic_targets:
                if target != "audio_cli" and not target.startswith("audio_cli."):
                    continue
                item = (node.lineno, without_init_suffix(target))
                if item not in emitted:
                    emitted.add(item)
                    yield item
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _from_base(node, source)
        if base != "audio_cli" and not base.startswith("audio_cli."):
            continue
        normalized_base = without_init_suffix(base)
        base_owner = owner(normalized_base)
        public_names = (
            _facade_public_names(base_owner)
            if base_owner is not None and normalized_base == f"audio_cli.{base_owner}"
            else set()
        )
        for alias in node.names:
            candidate = f"{base}.{alias.name}" if alias.name != "*" else base
            if base.endswith(".__init__") or alias.name in public_names:
                target = normalized_base
            elif candidate in known_modules or (base == "audio_cli" and alias.name != "*"):
                target = candidate
            else:
                target = base
            item = (node.lineno, without_init_suffix(target))
            if item not in emitted:
                emitted.add(item)
                yield item


def owner(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) > 1 and parts[1] in DIRECT_PACKAGES:
        return parts[1]
    return None


def _facade_public_names(package_owner: str) -> set[str]:
    facade = PACKAGE_ROOT / package_owner / "__init__.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"), filename=str(facade))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            value = ast.literal_eval(node.value)
            return set(value) if isinstance(value, list) else set()
    return set()


def own_facade_public_imports(
    path: Path,
    package_owner: str,
    *,
    source: str | None = None,
) -> set[str]:
    public = _facade_public_names(package_owner)
    facade = f"audio_cli.{package_owner}"
    if source is None:
        source = source_package(module_name(path), path)
    imported: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _from_base(node, source)
        if without_init_suffix(base) != facade:
            continue
        explicit_init = base.endswith(".__init__")
        imported.update(alias.name for alias in node.names if explicit_init or alias.name in public)
    return imported


def imports() -> Iterator[tuple[str, Path, int, str]]:
    paths = module_paths()
    known_modules = set(paths)
    for module, path in sorted(paths.items()):
        source_owner = owner(module)
        if source_owner is None:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, target in local_import_targets(
            tree,
            source=source_package(module, path),
            known_modules=known_modules,
            allow_opaque_dynamic=module in OPAQUE_DYNAMIC_IMPORT_MODULES,
        ):
            yield source_owner, path, line, target
