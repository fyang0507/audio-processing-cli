"""Keep decomposed subsystems behind real package boundaries."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "audio_cli"
SUBSYSTEMS = ("dsp", "media", "packages", "pipeline")
FLAT_SUBSYSTEM_PREFIXES = ("_package_", "dsp_", "media_", "pipeline_")
TRANSCRIBE_ROOT = PACKAGE_ROOT / "transcribe"
TRANSCRIBE_FACADES = {
    "orchestrator": [
        "RunProduct",
        "RunRange",
        "parse_range",
        "preflight",
        "render_human",
        "run",
        "validate_output_targets",
    ],
    "planner": ["ResolvedRequest", "build_plan", "parse_wants", "resolve_request"],
    "refusals": [
        "Refusal",
        "backend_failed",
        "capability_unknown",
        "capability_unsatisfiable_on_stack",
        "capability_unsupported",
        "command_path_argument",
        "export_input_invalid",
        "export_inputs_incompatible",
        "input_required",
        "option_unsupported_on_stack",
        "option_value_unsupported",
        "output_exists",
        "output_is_canonical_input",
        "output_path_invalid",
        "output_required_for_force",
        "package_build_unusable",
        "package_integrity_failed",
        "packages_not_provisioned",
        "pin_conflicts_with_native_capability",
        "range_invalid",
        "run_incomplete",
        "stack_required",
        "timing_required_for_format",
    ],
    "result": [
        "ABSENT",
        "ABSTENTION_REASONS",
        "CAPABILITY_NAMES",
        "SCHEMA_VERSION",
        "JsonMapping",
        "NormalizedResult",
        "OptionalArray",
        "OptionalMapping",
        "ResultError",
        "serialize_result",
    ],
    "transport": [
        "ProcessRunner",
        "StageFailure",
        "StageOutcome",
        "StageTransport",
        "SubprocessRunner",
    ],
}
EXPORT_REFUSALS = {
    "export_input_invalid",
    "export_inputs_incompatible",
    "output_exists",
    "output_is_canonical_input",
    "output_path_invalid",
    "output_required_for_force",
    "timing_required_for_format",
}
EXPORT_COMPATIBILITY_REFUSALS = {
    "export_input_invalid",
    "export_inputs_incompatible",
    "output_required_for_force",
    "timing_required_for_format",
}
COMMAND_PRIMITIVES = {
    "Refusal",
    "build_refusal",
    "command_path_argument",
    "export_command",
    "output_exists",
    "output_is_canonical_input",
    "output_path_invalid",
    "transcribe_plan_command",
    "transcribe_run_command",
}
DOCUMENT_FRAGMENT_LAYOUT = {
    "TRANSCRIBE_CONTRACT_": (
        REPO_ROOT,
        REPO_ROOT / "docs" / "transcribe-contract",
    ),
    "TRANSCRIBE_HAPPY_PATH_": (
        REPO_ROOT,
        REPO_ROOT / "docs" / "transcribe-happy-path",
    ),
    "VOCABULARY_": (
        REPO_ROOT,
        REPO_ROOT / "docs" / "vocabulary",
    ),
    "FINDINGS_": (
        REPO_ROOT / "model_tests",
        REPO_ROOT / "model_tests" / "findings",
    ),
}


def test_decomposed_subsystems_use_packages_not_filename_namespaces() -> None:
    for subsystem in SUBSYSTEMS:
        assert (PACKAGE_ROOT / subsystem / "__init__.py").is_file()
        assert not (PACKAGE_ROOT / f"{subsystem}.py").exists()

    offenders = sorted(
        path.name
        for path in PACKAGE_ROOT.glob("*.py")
        if path.name.startswith(FLAT_SUBSYSTEM_PREFIXES)
    )
    assert not offenders, (
        f"module families belong in real subpackages, not filename namespaces: {offenders}"
    )


def test_subsystem_facades_declare_only_public_exports() -> None:
    for subsystem in SUBSYSTEMS:
        module = importlib.import_module(f"audio_cli.{subsystem}")
        exported = getattr(module, "__all__", None)
        assert isinstance(exported, list) and exported
        assert len(exported) == len(set(exported))
        assert all(not name.startswith("_") for name in exported)
        assert all(hasattr(module, name) for name in exported)


def test_transcribe_execution_families_use_packages_not_private_root_modules() -> None:
    for package in ("execution", *TRANSCRIBE_FACADES):
        assert (TRANSCRIBE_ROOT / package / "__init__.py").is_file()
        assert not (TRANSCRIBE_ROOT / f"{package}.py").exists()

    legacy_fragments = {
        "planner_build.py",
        "planner_request.py",
        "process_runner.py",
        "refusal_export.py",
        "refusal_request.py",
        "result_types.py",
        "result_validation.py",
        "transport_types.py",
    }
    assert not legacy_fragments & {path.name for path in TRANSCRIBE_ROOT.glob("*.py")}

    private_root_modules = sorted(
        path.name for path in TRANSCRIBE_ROOT.glob("_*.py") if path.name != "__init__.py"
    )
    assert not private_root_modules, (
        "transcription module families belong in real subpackages, not private "
        f"root modules: {private_root_modules}"
    )
    assert not (TRANSCRIBE_ROOT / "transport" / "decode.py").exists()
    assert not (TRANSCRIBE_ROOT / "adapters" / "decode.py").exists()
    assert (TRANSCRIBE_ROOT / "stages" / "_firered_protocol.py").is_file()


def test_transcribe_has_one_deliberate_private_module() -> None:
    private_modules = sorted(
        str(path.relative_to(TRANSCRIBE_ROOT))
        for pattern in ("_*.py", "_*.pyi")
        for path in TRANSCRIBE_ROOT.rglob(pattern)
        if path.name not in {"__init__.py", "__init__.pyi"}
    )

    assert private_modules == ["stages/_firered_protocol.py"]


def test_provider_orchestration_has_one_package_owner() -> None:
    orchestrator = TRANSCRIBE_ROOT / "orchestrator"
    for implementation in ("qwen.py", "firered.py", "vibevoice.py", "common.py"):
        assert (orchestrator / implementation).is_file()
    assert not (orchestrator / "scope.py").exists()
    assert not (TRANSCRIBE_ROOT / "execution" / "canonical.py").exists()
    assert (PACKAGE_ROOT / "media" / "pcm.py").is_file()
    assert (TRANSCRIBE_ROOT / "execution" / "materialization.py").is_file()
    assert (TRANSCRIBE_ROOT / "adapters" / "firered_ledger.py").is_file()
    compatibility_package = TRANSCRIBE_ROOT / "native"
    assert (TRANSCRIBE_ROOT / "native.py").is_file()
    assert not (compatibility_package / "__init__.py").exists()
    assert not list(compatibility_package.rglob("*.py"))


def test_native_compatibility_module_preserves_only_the_old_public_entry_point(
    monkeypatch,
) -> None:
    native = importlib.import_module("audio_cli.transcribe.native")
    assert native.__all__ == ["run_native"]
    assert list(inspect.signature(native.run_native).parameters) == [
        "request",
        "metadata",
        "output",
        "output_format",
        "run_range",
        "registry",
        "transport",
        "vad_detector",
        "force",
    ]

    forwarded: list[tuple[object, object, dict[str, object]]] = []

    def fake_run(request, metadata, **kwargs):
        forwarded.append((request, metadata, kwargs))
        return "forwarded"

    monkeypatch.setattr(native, "_run", fake_run)
    metadata = object()
    accepted: list[object] = []
    for stack_id in ("firered", "vibevoice"):
        request = SimpleNamespace(stack=SimpleNamespace(id=stack_id))
        accepted.append(request)
        assert native.run_native(request, metadata, force=True) == "forwarded"

    forwarded_arguments = {
        "output": None,
        "output_format": "json",
        "run_range": None,
        "registry": None,
        "transport": None,
        "vad_detector": None,
        "force": True,
    }
    assert forwarded == [(request, metadata, forwarded_arguments) for request in accepted]

    for stack_id in ("qwen-0.6b", "qwen-1.7b", "unknown"):
        request = SimpleNamespace(stack=SimpleNamespace(id=stack_id))
        with pytest.raises(
            ValueError,
            match=rf"^{stack_id!r} is not a native-structure stack$",
        ):
            native.run_native(request, metadata)


def test_transcribe_public_execution_imports_stay_stable() -> None:
    for name, expected in TRANSCRIBE_FACADES.items():
        module = importlib.import_module(f"audio_cli.transcribe.{name}")
        assert module.__all__ == expected
        assert all(hasattr(module, exported) for exported in expected)

    assert importlib.import_module("audio_cli.transcribe.execution").__all__ == []


def test_export_owns_its_command_refusal_builders() -> None:
    export = importlib.import_module("audio_cli.export")
    transcribe_refusals = importlib.import_module("audio_cli.transcribe.refusals")

    assert set(export.__all__) >= EXPORT_REFUSALS
    assert set(transcribe_refusals.__all__) >= EXPORT_COMPATIBILITY_REFUSALS
    for name in EXPORT_COMPATIBILITY_REFUSALS:
        assert getattr(transcribe_refusals, name) is getattr(export, name)
    assert (PACKAGE_ROOT / "export" / "refusals.py").is_file()
    assert not (TRANSCRIBE_ROOT / "refusals" / "export.py").exists()

    tree = ast.parse((TRANSCRIBE_ROOT / "refusals" / "__init__.py").read_text(encoding="utf-8"))
    bridge_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "audio_cli.export.refusals"
    ]
    assert [type(node) for node in tree.body] == [
        ast.Expr,
        ast.ImportFrom,
        ast.ImportFrom,
        ast.Assign,
    ]
    docstring = tree.body[0]
    assert isinstance(docstring, ast.Expr)
    assert isinstance(docstring.value, ast.Constant)
    assert isinstance(docstring.value.value, str)
    assert bridge_imports == [tree.body[1]]
    assert {alias.name for alias in bridge_imports[0].names} == EXPORT_COMPATIBILITY_REFUSALS
    assert all(alias.asname is None for alias in bridge_imports[0].names)
    request_import = tree.body[2]
    assert isinstance(request_import, ast.ImportFrom)
    assert (request_import.level, request_import.module) == (1, "request")
    exports = tree.body[3]
    assert isinstance(exports, ast.Assign)
    assert [target.id for target in exports.targets if isinstance(target, ast.Name)] == ["__all__"]
    assert ast.literal_eval(exports.value) == transcribe_refusals.__all__


def test_shared_command_primitives_have_one_package_owner() -> None:
    command = importlib.import_module("audio_cli.command")
    export = importlib.import_module("audio_cli.export")
    transcribe_refusals = importlib.import_module("audio_cli.transcribe.refusals")

    assert set(command.__all__) == COMMAND_PRIMITIVES
    assert export.output_is_canonical_input is command.output_is_canonical_input
    assert transcribe_refusals.output_is_canonical_input is command.output_is_canonical_input
    assert export.output_path_invalid is command.output_path_invalid
    assert transcribe_refusals.output_path_invalid is command.output_path_invalid
    assert not (PACKAGE_ROOT / "cli_commands.py").exists()
    assert not (PACKAGE_ROOT / "cli_refusals.py").exists()


def test_document_fragment_families_use_directories() -> None:
    for prefix, (old_root, directory) in DOCUMENT_FRAGMENT_LAYOUT.items():
        offenders = sorted(path.name for path in old_root.glob(f"{prefix}*.md"))
        assert not offenders, (
            "document families belong in dedicated directories, not filename "
            f"namespaces: {offenders}"
        )
        fragments = sorted(directory.glob("*.md"))
        assert fragments, f"document fragment directory is empty: {directory}"
        assert all(not path.name.startswith(prefix) for path in fragments)


def test_provisioner_keeps_only_its_supported_workflow_surface() -> None:
    from audio_cli.packages import Provisioner

    public_methods = {
        name
        for name, value in vars(Provisioner).items()
        if not name.startswith("_") and callable(value)
    }
    assert public_methods == {
        "ensure_environment",
        "pull",
        "purge",
        "remove",
        "verify",
    }
    assert not {
        "_checkout_and_install",
        "_collect_environments",
        "_materialize",
        "_pre_existing_revisions",
        "_require_hub_snapshot",
        "_verify_mlx_guard",
    } & set(vars(Provisioner))
