"""Unit tests for transcription boundary import resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
from transcribe_boundary_test_support import (
    ROOT_FACADE,
    external_audio_cli_dependencies,
    facade_imported_symbols,
    local_imports,
)


def test_absolute_root_imports_resolve_to_facades(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "import audio_cli.transcribe\n"
        "from audio_cli import transcribe\n"
        "from audio_cli.transcribe import ABSENT\n"
        "from audio_cli.transcribe import planner\n",
        encoding="utf-8",
    )

    assert local_imports(
        source,
        "orchestrator.common",
        {"orchestrator.common", "planner"},
    ) == {ROOT_FACADE, "planner"}


def test_explicit_init_imports_resolve_to_facades(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "from .__init__ import run\n"
        "from .__init__ import qwen\n"
        "from audio_cli.transcribe.orchestrator.__init__ import run as absolute_run\n"
        "from audio_cli.transcribe.orchestrator.__init__ import qwen as absolute_qwen\n"
        "import audio_cli.transcribe.orchestrator.__init__\n"
        "import importlib as runtime_imports\n"
        "runtime_imports.import_module('audio_cli.transcribe.orchestrator.__init__')\n"
        "runtime_imports.import_module(name='audio_cli.transcribe.__init__')\n",
        encoding="utf-8",
    )

    assert local_imports(
        source,
        "orchestrator.common",
        {"orchestrator", "orchestrator.common", "orchestrator.qwen"},
    ) == {ROOT_FACADE, "orchestrator"}


def test_stub_initializers_resolve_relative_imports_as_packages(tmp_path: Path) -> None:
    source = tmp_path / "__init__.pyi"
    source.write_text(
        "from .. import helper\nfrom ....media import canonical_decode_command\n",
        encoding="utf-8",
    )

    assert local_imports(
        source,
        "transport.nested",
        {"helper", "transport.helper", "transport.nested"},
    ) == {"transport.helper"}
    assert external_audio_cli_dependencies(source, "transport.nested") == {"media"}


def test_external_import_resolution_can_detect_forbidden_dependencies(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "from audio_cli import paths\n"
        "from audio_cli.media import atomic_write_text\n"
        "__import__('audio_cli.packages')\n",
        encoding="utf-8",
    )

    assert external_audio_cli_dependencies(source, "adapters.example") == {
        "media",
        "packages",
        "paths",
    }


def test_dynamic_imports_are_governed_by_local_and_facade_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text(
        "import builtins as runtime_builtins\n"
        "import importlib as runtime_imports\n"
        "from importlib import import_module as load_module\n"
        "runtime_imports.import_module(name='audio_cli.transcribe.stages.qwen')\n"
        "load_module('..stages.' + 'qwen', __package__)\n"
        "load_module(name='..plan', package=__package__)\n"
        "assigned_loader = runtime_imports.import_module\n"
        "assigned_loader(name='audio_cli.transcribe.stacks')\n"
        "chained_loader = assigned_loader\n"
        "chained_loader('audio_cli.transcribe.catalog')\n"
        "runtime_builtins.__import__('sample', globals(), locals(), (), 2)\n"
        "runtime_builtins.__import__(\n"
        "    name='refusals.request', globals=globals(), locals=locals(), fromlist=(), level=2\n"
        ")\n"
        "runtime_builtins.__import__(\n"
        "    'stages', globals(), locals(), ('aligner',), 2\n"
        ")\n"
        "runtime_builtins.__import__(name='audio_cli.packages')\n"
        "load_module('audio_cli.packages.pull')\n",
        encoding="utf-8",
    )

    assert local_imports(
        source,
        "orchestrator.common",
        {
            "catalog",
            "orchestrator.common",
            "plan",
            "refusals.request",
            "sample",
            "stacks",
            "stages.aligner",
            "stages.qwen",
        },
    ) == {
        "catalog",
        "plan",
        "refusals.request",
        "sample",
        "stacks",
        "stages.aligner",
        "stages.qwen",
    }
    assert facade_imported_symbols(source, "orchestrator.common", "packages") == {
        "<concrete:pull>",
        "<module>",
    }


@pytest.mark.parametrize(
    "body, error",
    (
        (
            "module_name = 'audio_cli.transcribe.stacks'\nimportlib.import_module(module_name)\n",
            "constant module target",
        ),
        (
            "package_name = __package__\nimportlib.import_module('..plan', package_name)\n",
            "constant package",
        ),
        (
            "dynamic_level = 2\n__import__('stacks', level=dynamic_level)\n",
            "constant integer level",
        ),
        (
            "context = make_context()\n__import__('stacks', globals=context, level=2)\n",
            "must omit globals",
        ),
        (
            "__import__(\n"
            "    'qwen', {'__package__': 'audio_cli.transcribe.stages'}, {}, (), 1\n"
            ")\n",
            "must omit globals",
        ),
        (
            "children = ('pull',)\n__import__('audio_cli.packages', fromlist=children)\n",
            "constant fromlist",
        ),
    ),
)
def test_opaque_dynamic_imports_fail_the_boundary_scan(
    tmp_path: Path, body: str, error: str
) -> None:
    source = tmp_path / "implementation.py"
    source.write_text("import importlib\n" + body, encoding="utf-8")

    with pytest.raises(AssertionError, match=error):
        local_imports(source, "orchestrator.common", {"orchestrator.common", "plan", "stacks"})
