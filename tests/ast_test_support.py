"""Cross-minor helpers for source-backed AST parity assertions."""

from __future__ import annotations

import ast


def constant_string(node: ast.AST | None) -> str | None:
    """Fold the intentionally small string-expression subset accepted by scanners."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and all(
        isinstance(value, ast.Constant) and isinstance(value.value, str) for value in node.values
    ):
        return "".join(value.value for value in node.values)  # type: ignore[union-attr]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = constant_string(node.left)
        right = constant_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def constant_builtin_import_fromlist(
    node: ast.AST,
    loader_names: tuple[set[str], set[str], set[str], set[str]],
) -> tuple[str, ...]:
    """Return a built-in import's literal child-module requests or fail closed."""
    if not isinstance(node, ast.Call):
        return ()
    _importlib_modules, _import_functions, builtins_modules, builtin_functions = loader_names
    builtin_call = (isinstance(node.func, ast.Name) and node.func.id in builtin_functions) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "__import__"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in builtins_modules
    )
    if not builtin_call:
        return ()

    fromlist_node: ast.AST | None = node.args[3] if len(node.args) > 3 else None
    if fromlist_node is None:
        fromlist_node = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "fromlist"),
            None,
        )
    if fromlist_node is None or (
        isinstance(fromlist_node, ast.Constant) and fromlist_node.value is None
    ):
        return ()
    try:
        value = ast.literal_eval(fromlist_node)
    except (ValueError, TypeError) as exc:
        raise AssertionError(
            f"dynamic import at line {node.lineno} must have a constant fromlist"
        ) from exc
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(name, str)
        or not name
        or not all(part.isidentifier() for part in name.split("."))
        for name in value
    ):
        raise AssertionError(
            f"dynamic import at line {node.lineno} must have a constant fromlist "
            "of module-name strings"
        )
    return tuple(value)


def stable_ast_source(node: ast.AST) -> str:
    """Canonicalize parsed code without hashing version-specific AST metadata."""
    return ast.unparse(node)
