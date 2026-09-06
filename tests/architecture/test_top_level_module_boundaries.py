"""Enforce the dependency direction between top-level ``audio_cli`` packages."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.architecture.top_level_boundary_support import (
    DIRECT_PACKAGES,
    PACKAGE_ROOT,
    imports,
    local_import_targets,
    module_name,
    module_paths,
    own_facade_public_imports,
    owner,
    python_source_paths,
    source_package,
    without_init_suffix,
)

# Edges point from a package to the direct packages it may depend on. Imports within the
# same package are always allowed and therefore omitted here.
ALLOWED_PACKAGE_DEPENDENCIES = {
    "command": set(),
    "dsp": set(),
    "environments": set(),
    "export": {"command", "media", "transcribe"},
    "media": set(),
    "packages": {"environments", "media"},
    "pipeline": {"dsp", "media"},
    "transcribe": {"command", "environments", "export", "media", "packages"},
}

# These root modules are small shared contracts or infrastructure seams, not feature
# packages. Keep the list exact so a new root-level dependency requires an architectural
# decision rather than silently turning the package root into a service locator.
ALLOWED_ROOT_MODULE_DEPENDENCIES = {
    "command": set(),
    "dsp": {"audio_cli.adjustments", "audio_cli.profiles", "audio_cli.vad_contract"},
    "environments": set(),
    "export": set(),
    "media": set(),
    "packages": {"audio_cli.__version__", "audio_cli.paths"},
    "pipeline": {
        "audio_cli.__version__",
        "audio_cli.adjustments",
        "audio_cli.profiles",
        "audio_cli.vad",
        "audio_cli.vad_contract",
    },
    "transcribe": {
        "audio_cli.paths",
        "audio_cli.vad",
    },
}
COMPOSITION_ROOT_DEPENDENCIES = {
    "audio_cli.__main__": {"audio_cli.cli"},
    "audio_cli.cli": {
        "audio_cli.adjustments",
        "audio_cli.cli_parser",
        "audio_cli.command",
        "audio_cli.environments",
        "audio_cli.export",
        "audio_cli.export.refusals",
        "audio_cli.media",
        "audio_cli.packages",
        "audio_cli.pipeline",
        "audio_cli.profiles",
        "audio_cli.transcribe.catalog",
        "audio_cli.transcribe.execution",
        "audio_cli.transcribe.orchestrator",
        "audio_cli.transcribe.plan",
        "audio_cli.transcribe.planner",
        "audio_cli.transcribe.refusals",
        "audio_cli.transcribe.stacks",
        "audio_cli.transcribe.transport",
        "audio_cli.vad",
    },
    "audio_cli.cli_parser": {"audio_cli.profiles"},
}
SHARED_ROOT_MODULE_DEPENDENCIES = {
    "audio_cli": set(),
    "audio_cli.adjustments": set(),
    "audio_cli.paths": set(),
    "audio_cli.profiles": set(),
    "audio_cli.vad": {"audio_cli.media", "audio_cli.paths", "audio_cli.vad_contract"},
    "audio_cli.vad_contract": set(),
}


def _dependency_violation(
    source: str,
    target: str,
    *,
    source_module: str | None = None,
) -> str | None:
    target = without_init_suffix(target)
    target_owner = owner(target)
    if target_owner == source:
        if target == f"audio_cli.{source}":
            return f"{source} implementation imports its compatibility facade"
        return None
    if target == "audio_cli.cli" or target.startswith("audio_cli.cli."):
        return "lower package imports cli"
    if target_owner is None:
        if target not in ALLOWED_ROOT_MODULE_DEPENDENCIES[source]:
            return f"{source} -> {target}"
        return None
    if target_owner not in ALLOWED_PACKAGE_DEPENDENCIES[source]:
        return f"{source} -> {target}"
    if source == "transcribe" and target_owner == "export":
        if (
            source_module == "audio_cli.transcribe.refusals"
            and target == "audio_cli.export.refusals"
        ):
            return None
        return (
            "transcribe may import audio_cli.export.refusals only from its frozen "
            "refusals compatibility facade"
        )
    if (
        source == "export"
        and target_owner == "transcribe"
        and not (
            target == "audio_cli.transcribe.result"
            or target.startswith("audio_cli.transcribe.result.")
        )
    ):
        return f"export may consume only audio_cli.transcribe.result, not {target}"
    if source != "export" or target_owner != "transcribe":
        facade = f"audio_cli.{target_owner}"
        if target != facade:
            return f"cross-package import bypasses {facade}: {target}"
    return None


def test_every_direct_package_has_a_declared_boundary() -> None:
    directories = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.name != "__pycache__" and (path.is_dir() or path.is_symlink())
    }
    importable = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }
    python_bearing = {
        path.relative_to(PACKAGE_ROOT).parts[0]
        for path in python_source_paths()
        if len(path.relative_to(PACKAGE_ROOT).parts) > 1
    }

    assert directories == DIRECT_PACKAGES
    assert importable == DIRECT_PACKAGES
    assert python_bearing == DIRECT_PACKAGES


def test_source_tree_has_no_symlinked_entries() -> None:
    offenders = sorted(
        str(path.relative_to(PACKAGE_ROOT)) for path in PACKAGE_ROOT.rglob("*") if path.is_symlink()
    )

    assert offenders == []


def test_canonical_wave_io_is_owned_only_by_media_pcm() -> None:
    owners: set[str] = set()
    for path in python_source_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            (
                isinstance(node, ast.Import)
                and any(alias.name.split(".", 1)[0] == "wave" for alias in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and not node.level
                and (node.module or "").split(".", 1)[0] == "wave"
            )
            for node in ast.walk(tree)
        ):
            owners.add(module_name(path))

    assert owners == {"audio_cli.media.pcm"}


def test_every_root_module_has_a_declared_role() -> None:
    actual = {
        module_name(path) for pattern in ("*.py", "*.pyi") for path in PACKAGE_ROOT.glob(pattern)
    }

    assert actual == set(COMPOSITION_ROOT_DEPENDENCIES) | set(SHARED_ROOT_MODULE_DEPENDENCIES)


def test_direct_package_implementations_do_not_import_public_names_from_own_facade() -> None:
    violations: dict[str, list[str]] = {}
    for package_owner in DIRECT_PACKAGES:
        for path in python_source_paths():
            if owner(module_name(path)) != package_owner or path.name in {
                "__init__.py",
                "__init__.pyi",
            }:
                continue
            if imported := sorted(own_facade_public_imports(path, package_owner)):
                violations[str(path.relative_to(PACKAGE_ROOT.parent))] = imported

    assert violations == {}


def test_composition_roots_follow_declared_dependency_direction() -> None:
    paths = module_paths()
    known_modules = set(paths)
    violations: dict[str, dict[str, list[str]]] = {}
    for module, expected in COMPOSITION_ROOT_DEPENDENCIES.items():
        path = paths[module]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        actual = {
            target
            for _, target in local_import_targets(
                tree,
                source=source_package(module, path),
                known_modules=known_modules,
            )
        }
        if actual != expected:
            violations[module] = {
                "missing": sorted(expected - actual),
                "unexpected": sorted(actual - expected),
            }

    assert violations == {}


def test_shared_root_modules_follow_dependency_direction() -> None:
    paths = module_paths()
    known_modules = set(paths)
    violations: dict[str, dict[str, list[str]]] = {}
    for module, expected in SHARED_ROOT_MODULE_DEPENDENCIES.items():
        path = paths[module]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        actual = {
            target
            for _, target in local_import_targets(
                tree,
                source=source_package(module, path),
                known_modules=known_modules,
            )
        }
        if actual != expected:
            violations[module] = {
                "missing": sorted(expected - actual),
                "unexpected": sorted(actual - expected),
            }

    assert violations == {}


def test_import_resolution_handles_relative_modules_and_root_attributes() -> None:
    known_modules = set(module_paths())
    tree = ast.parse(
        "\n".join(
            (
                "from . import registry",
                "from .. import paths",
                "from audio_cli import __version__",
                "from audio_cli.packages import doctor",
                "from audio_cli.transcribe import result",
                "from audio_cli.transcribe.result import NormalizedResult",
                "from .__init__ import Toolchain",
                "import audio_cli.packages.__init__",
                "import importlib as runtime_imports",
                "from importlib import import_module",
                "runtime_imports.import_module('audio_cli.transcribe.' + 'stages.qwen')",
                "import_module('..environments', __package__)",
                "assigned_loader = runtime_imports.import_module",
                "assigned_loader(name='audio_cli.media')",
                "__import__('audio_cli.media')",
                "__import__('audio_cli.media', fromlist=('files',))",
                "__import__('environments', globals(), locals(), (), 2)",
            )
        )
    )

    targets = {
        target
        for _, target in local_import_targets(
            tree,
            source="audio_cli.packages",
            known_modules=known_modules,
        )
    }

    assert targets == {
        "audio_cli.__version__",
        "audio_cli.environments",
        "audio_cli.packages",
        "audio_cli.packages.registry",
        "audio_cli.paths",
        "audio_cli.media",
        "audio_cli.media.files",
        "audio_cli.transcribe.result",
        "audio_cli.transcribe.stages.qwen",
    }


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (
            "import importlib\nimportlib.import_module(module_name)",
            "statically resolvable module name",
        ),
        (
            "__import__('media', {'__package__': 'audio_cli'}, {}, (), 1)",
            "source module globals directly",
        ),
        (
            "children = ('files',)\n__import__('audio_cli.media', fromlist=children)",
            "constant fromlist",
        ),
    ),
)
def test_opaque_dynamic_imports_are_rejected(source: str, message: str) -> None:
    tree = ast.parse(source)

    with pytest.raises(AssertionError, match=message):
        set(
            local_import_targets(
                tree,
                source="audio_cli.packages",
                known_modules=set(module_paths()),
            )
        )


def test_narrow_cross_package_exceptions_are_enforced() -> None:
    assert _dependency_violation("export", "audio_cli.command") is None
    assert _dependency_violation("export", "audio_cli.transcribe.result") is None
    assert _dependency_violation("export", "audio_cli.transcribe.result.types") is None
    assert (
        _dependency_violation(
            "transcribe",
            "audio_cli.export.refusals",
            source_module="audio_cli.transcribe.refusals",
        )
        is None
    )
    assert _dependency_violation("transcribe", "audio_cli.export.refusals") == (
        "transcribe may import audio_cli.export.refusals only from its frozen refusals "
        "compatibility facade"
    )
    assert _dependency_violation(
        "transcribe",
        "audio_cli.export",
        source_module="audio_cli.transcribe.refusals",
    ) == (
        "transcribe may import audio_cli.export.refusals only from its frozen refusals "
        "compatibility facade"
    )
    assert _dependency_violation("export", "audio_cli.transcribe.orchestrator") == (
        "export may consume only audio_cli.transcribe.result, not audio_cli.transcribe.orchestrator"
    )
    assert _dependency_violation("packages", "audio_cli.cli") == "lower package imports cli"
    assert _dependency_violation("transcribe", "audio_cli.pipeline") == (
        "transcribe -> audio_cli.pipeline"
    )
    assert _dependency_violation("transcribe", "audio_cli.packages.requirements") == (
        "cross-package import bypasses audio_cli.packages: audio_cli.packages.requirements"
    )
    assert _dependency_violation("dsp", "audio_cli.vad_contract") is None
    assert _dependency_violation("dsp", "audio_cli.vad") == "dsp -> audio_cli.vad"
    assert _dependency_violation("packages", "audio_cli.packages") == (
        "packages implementation imports its compatibility facade"
    )
    assert _dependency_violation("packages", "audio_cli.packages.__init__") == (
        "packages implementation imports its compatibility facade"
    )


def test_same_owner_facade_symbols_are_not_misclassified_as_modules(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "from audio_cli.packages import doctor\nfrom . import registry\n",
        encoding="utf-8",
    )

    assert own_facade_public_imports(
        source,
        "packages",
        source="audio_cli.packages",
    ) == {"doctor"}


def test_top_level_packages_follow_the_dependency_dag() -> None:
    violations: list[str] = []
    for source, path, line, target in imports():
        if violation := _dependency_violation(
            source,
            target,
            source_module=module_name(path),
        ):
            violations.append(f"{path.relative_to(PACKAGE_ROOT.parent.parent)}:{line}: {violation}")

    assert violations == [], "top-level dependency violations:\n" + "\n".join(violations)


def test_direct_packages_have_exact_shared_root_dependencies() -> None:
    actual = {owner: set() for owner in DIRECT_PACKAGES}
    for source, _, _, target in imports():
        if owner(target) is None:
            actual[source].add(target)

    assert actual == ALLOWED_ROOT_MODULE_DEPENDENCIES
