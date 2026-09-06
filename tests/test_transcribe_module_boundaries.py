from __future__ import annotations

from pathlib import Path

from transcribe_boundary_inventory import EXPECTED_PEER_MODULES
from transcribe_boundary_test_support import (
    ROOT_FACADE,
    assert_acyclic,
    external_audio_cli_dependencies,
    facade_imported_symbols,
    local_imports,
    module_name,
    python_source_paths,
    source_layout,
    transcribe_graph,
)

TRANSCRIBE = Path(__file__).resolve().parents[1] / "src" / "audio_cli" / "transcribe"
FACADES = {
    ROOT_FACADE,
    "adapters",
    "execution",
    "orchestrator",
    "planner",
    "refusals",
    "result",
    "transport",
}
COMPATIBILITY_MODULES = {"native"}
PEER_PACKAGES = {
    "adapters",
    "execution",
    "orchestrator",
    "planner",
    "refusals",
    "result",
    "stages",
    "transport",
}
ALLOWED_PEER_MODULES = {
    "adapters": set(),
    "execution": {
        "adapters.silero",
        "planner.request",
        "refusals.request",
        "result.types",
        "transport.types",
    },
    "orchestrator": {
        "adapters.aligner",
        "adapters.diarizer",
        "adapters.firered",
        "adapters.firered_ledger",
        "adapters.qwen",
        "adapters.vibevoice",
        "execution.materialization",
        "execution.preflight",
        "execution.publication",
        "execution.runtime",
        "execution.vad",
        "planner.build",
        "planner.request",
        "refusals.request",
        "result.serialization",
        "result.types",
        "transport.service",
        "transport.types",
    },
    "planner": {"refusals.request"},
    "refusals": set(),
    "result": set(),
    "stages": set(),
    "transport": set(),
}
ALLOWED_EXTERNAL_DEPENDENCIES = {
    "adapters": set(),
    "execution": {"command", "environments", "media", "packages", "paths", "vad"},
    "orchestrator": {"environments", "media", "packages", "vad"},
    "planner": {"environments"},
    "refusals": {"command"},
    "result": set(),
    "stages": set(),
    "transport": {"media", "packages", "paths"},
}
COMPATIBILITY_EXTERNAL_DEPENDENCIES = {"refusals": {"export"}}
ALLOWED_PACKAGE_FACADE_SYMBOLS = {
    "adapters": set(),
    "execution": {
        "Toolchain",
        "built_product_candidates",
        "checkout_file_matches",
        "checkout_install_drift",
        "checkout_patch_expectation",
        "hub_materialization_issues",
        "managed_checkout_path",
        "managed_checkout_requirements",
        "managed_environment_path",
        "managed_provisioning_root_issue",
        "managed_url_artifact_path",
        "validated_built_product",
    },
    "orchestrator": {"load_registry"},
    "planner": set(),
    "refusals": set(),
    "result": set(),
    "stages": set(),
    "transport": {"managed_environment_path", "validated_built_product"},
}
ROOT_MODULE_DEPENDENCIES = {
    "catalog": {"stacks"},
    "native": {"catalog", "orchestrator", "planner", "transport"},
    "plan": {"sample"},
    "sample": {"result.serialization", "result.types"},
    "stacks": {"result.types"},
}
ROOT_EXTERNAL_DEPENDENCIES = {
    "catalog": {"command"},
    "native": set(),
    "plan": {"command"},
    "sample": set(),
    "stacks": {"environments"},
}
ALLOWED_ROOT_CONTRACT_DEPENDENCIES = {
    "adapters": set(),
    "execution": {"plan"},
    "orchestrator": {"catalog"},
    "planner": {"catalog", "plan", "stacks"},
    "refusals": {"stacks"},
    "result": set(),
    "stages": set(),
    "transport": set(),
}
ALLOWED_STAGE_IMPORTS = {
    "stages": set(),
    "stages._firered_protocol": set(),
    "stages.aligner": set(),
    "stages.firered": {"stages._firered_protocol"},
    "stages.qwen": set(),
    "stages.vibevoice": set(),
}


def _python_source_paths() -> list[Path]:
    return python_source_paths(TRANSCRIBE)


def test_every_transcribe_package_has_a_declared_boundary() -> None:
    directories, importable, python_bearing, root_modules = source_layout(TRANSCRIBE)

    assert directories == PEER_PACKAGES
    assert importable == PEER_PACKAGES
    assert python_bearing == PEER_PACKAGES
    assert root_modules == set(ROOT_MODULE_DEPENDENCIES)


def test_every_transcribe_peer_has_an_exact_module_inventory() -> None:
    modules = set(_transcribe_graph())
    actual = {
        owner: {module for module in modules if module == owner or module.startswith(f"{owner}.")}
        for owner in PEER_PACKAGES
    }

    assert actual == EXPECTED_PEER_MODULES


def _module_name(path: Path) -> str:
    return module_name(path, TRANSCRIBE)


def _transcribe_graph() -> dict[str, set[str]]:
    return transcribe_graph(TRANSCRIBE)


def test_transcribe_implementations_do_not_import_compatibility_facades() -> None:
    graph = _transcribe_graph()
    backedges = {
        module: sorted(imports & FACADES)
        for module, imports in graph.items()
        if module not in FACADES | COMPATIBILITY_MODULES and imports & FACADES
    }
    assert backedges == {}


def test_transcribe_packages_follow_dependency_direction() -> None:
    graph = _transcribe_graph()
    violations: dict[str, list[str]] = {}
    for module, imports in graph.items():
        owner = module.partition(".")[0]
        if owner not in PEER_PACKAGES:
            continue
        cross_package = {
            imported
            for imported in imports
            if imported.partition(".")[0] in PEER_PACKAGES and imported.partition(".")[0] != owner
        }
        unexpected = sorted(cross_package - ALLOWED_PEER_MODULES[owner])
        if unexpected:
            violations[module] = unexpected

    assert violations == {}


def test_transcribe_packages_have_exact_root_contract_dependencies() -> None:
    graph = _transcribe_graph()
    root_contracts = set(ROOT_MODULE_DEPENDENCIES)
    actual = {owner: set() for owner in PEER_PACKAGES}
    for module, imports in graph.items():
        owner = module.partition(".")[0]
        if owner in actual:
            actual[owner].update(imports & root_contracts)

    assert actual == ALLOWED_ROOT_CONTRACT_DEPENDENCIES


def test_undeclared_root_contract_import_is_a_boundary_violation(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "from ...transcribe.stacks import recommended_stack\n"
        "from ...transcribe.stages import qwen\n",
        encoding="utf-8",
    )
    root_contracts = set(ROOT_MODULE_DEPENDENCIES)
    imports = local_imports(
        source,
        "transport.service",
        {"stacks", "stages", "stages.qwen", "transport.service"},
    )

    assert imports & root_contracts - ALLOWED_ROOT_CONTRACT_DEPENDENCIES["transport"] == {"stacks"}
    assert "stages.qwen" in imports


def test_transcribe_packages_follow_external_dependency_direction() -> None:
    violations: dict[str, list[str]] = {}
    for path in _python_source_paths():
        module = _module_name(path)
        if not module:
            continue
        owner = module.partition(".")[0]
        if owner not in PEER_PACKAGES:
            continue
        allowed = ALLOWED_EXTERNAL_DEPENDENCIES[owner] | (
            COMPATIBILITY_EXTERNAL_DEPENDENCIES.get(module, set())
        )
        unexpected = sorted(external_audio_cli_dependencies(path, module) - allowed)
        if unexpected:
            violations[module] = unexpected

    assert violations == {}


def test_export_compatibility_dependency_is_confined_to_refusals_facade() -> None:
    actual: dict[str, set[str]] = {}
    for path in _python_source_paths():
        module = _module_name(path)
        dependencies = external_audio_cli_dependencies(path, module)
        if "export" in dependencies:
            actual[module] = {"export"}

    assert actual == COMPATIBILITY_EXTERNAL_DEPENDENCIES


def test_transcribe_can_only_read_declared_package_facade_symbols() -> None:
    actual = {owner: set() for owner in PEER_PACKAGES}
    for path in _python_source_paths():
        module = _module_name(path)
        if module and (owner := module.partition(".")[0]) in PEER_PACKAGES:
            actual[owner].update(facade_imported_symbols(path, module, "packages"))

    assert actual == ALLOWED_PACKAGE_FACADE_SYMBOLS


def test_transcribe_root_contracts_have_exact_dependency_directions() -> None:
    graph = _transcribe_graph()
    violations: dict[str, dict[str, list[str]]] = {}
    for module, expected_local in ROOT_MODULE_DEPENDENCIES.items():
        path = TRANSCRIBE / f"{module}.py"
        actual_local = graph[module]
        actual_external = external_audio_cli_dependencies(path, module)
        expected_external = ROOT_EXTERNAL_DEPENDENCIES[module]
        if actual_local != expected_local or actual_external != expected_external:
            violations[module] = {
                "missing_local": sorted(expected_local - actual_local),
                "unexpected_local": sorted(actual_local - expected_local),
                "missing_external": sorted(expected_external - actual_external),
                "unexpected_external": sorted(actual_external - expected_external),
            }

    assert violations == {}


def test_transcribe_root_facade_has_only_declared_contract_dependencies() -> None:
    modules = set(_transcribe_graph())
    imports = local_imports(TRANSCRIBE / "__init__.py", "", modules)

    assert imports == {"result", "sample"}


def test_stages_are_isolated_from_host_modules_and_each_other() -> None:
    graph = _transcribe_graph()
    inbound = {
        module: sorted(imported for imported in imports if imported.startswith("stages"))
        for module, imports in graph.items()
        if not module.startswith("stages")
        and any(imported.startswith("stages") for imported in imports)
    }
    internal = {
        module: sorted(imports - ALLOWED_STAGE_IMPORTS[module])
        for module, imports in graph.items()
        if module.startswith("stages") and imports - ALLOWED_STAGE_IMPORTS[module]
    }

    assert inbound == {}
    assert internal == {}


def test_transcribe_local_import_graph_has_no_cycles() -> None:
    assert_acyclic(_transcribe_graph())
