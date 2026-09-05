"""Keep third-party model runtimes inside isolated transcription stages."""

from __future__ import annotations

import sys
from pathlib import Path

from transcribe_boundary_inventory import EXPECTED_STAGE_RUNTIME_IMPORTS
from transcribe_boundary_test_support import imported_roots, module_name, python_source_paths

TRANSCRIBE = Path(__file__).resolve().parents[1] / "src" / "audio_cli" / "transcribe"
HOST_IMPORTS = frozenset(sys.stdlib_module_names) | {"__future__", "audio_cli"}


def test_model_runtime_imports_are_confined_to_isolated_stages() -> None:
    host_violations: dict[str, list[str]] = {}
    stage_third_party: dict[str, set[str]] = {}
    for path in python_source_paths(TRANSCRIBE):
        module = module_name(path, TRANSCRIBE)
        third_party = imported_roots(path, module) - HOST_IMPORTS
        if module == "stages" or module.startswith("stages."):
            stage_third_party[module] = third_party
        elif third_party:
            host_violations[module or "<root>"] = sorted(third_party)

    assert host_violations == {}
    assert stage_third_party == EXPECTED_STAGE_RUNTIME_IMPORTS


def test_import_scanner_detects_lazy_and_dynamic_model_runtime_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "orchestrator.py"
    source.write_text(
        "def load():\n"
        "    import torch\n"
        "    from importlib import import_module as load_module\n"
        "    return load_module('mlx.' + 'core'), torch\n",
        encoding="utf-8",
    )

    assert imported_roots(source, "orchestrator.qwen") >= {"mlx", "torch"}
