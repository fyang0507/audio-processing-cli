"""Keep decomposed subsystems behind real package boundaries."""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "audio_cli"
SUBSYSTEMS = ("dsp", "media", "packages", "pipeline")
FLAT_SUBSYSTEM_PREFIXES = ("_package_", "dsp_", "media_", "pipeline_")
TRANSCRIBE_ROOT = PACKAGE_ROOT / "transcribe"
TRANSCRIBE_FACADES = {
    "native": ["run_native"],
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
    assert (TRANSCRIBE_ROOT / "stages" / "_firered_protocol.py").is_file()


def test_transcribe_public_execution_imports_stay_stable() -> None:
    for name, expected in TRANSCRIBE_FACADES.items():
        module = importlib.import_module(f"audio_cli.transcribe.{name}")
        assert module.__all__ == expected
        assert all(hasattr(module, exported) for exported in expected)

    assert importlib.import_module("audio_cli.transcribe.execution").__all__ == []


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
