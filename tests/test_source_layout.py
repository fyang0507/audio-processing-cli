"""Keep decomposed subsystems behind real package boundaries."""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "audio_cli"
SUBSYSTEMS = ("dsp", "media", "packages", "pipeline")
FLAT_SUBSYSTEM_PREFIXES = ("_package_", "dsp_", "media_", "pipeline_")
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
        "module families belong in real subpackages, not filename namespaces: "
        f"{offenders}"
    )


def test_subsystem_facades_declare_only_public_exports() -> None:
    for subsystem in SUBSYSTEMS:
        module = importlib.import_module(f"audio_cli.{subsystem}")
        exported = getattr(module, "__all__", None)
        assert isinstance(exported, list) and exported
        assert len(exported) == len(set(exported))
        assert all(not name.startswith("_") for name in exported)
        assert all(hasattr(module, name) for name in exported)


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
