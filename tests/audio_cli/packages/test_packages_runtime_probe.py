"""The managed-environment private-API probe is checked in and stdlib-only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audio_cli.packages.runtime_probe import inspect_target, main


def _fixture_module(tmp_path: Path, module_name: str) -> tuple[str, Path]:
    path = tmp_path / f"{module_name}.py"
    path.write_text(
        "class Example:\n"
        "    def method(self, value, optional=None):\n"
        "        return value, optional\n",
        encoding="utf-8",
    )
    return f"{module_name}.Example.method", path


def test_probe_hashes_the_defining_module_and_reads_the_method_signature(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target, path = _fixture_module(tmp_path, "runtime_probe_inspection_fixture")
    monkeypatch.syspath_prepend(tmp_path)

    assert inspect_target(target) == {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "params": ["optional", "self", "value"],
    }


def test_probe_cli_emits_one_compact_json_verdict(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    target, path = _fixture_module(tmp_path, "runtime_probe_cli_fixture")
    monkeypatch.syspath_prepend(tmp_path)

    assert main([target]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "params": ["optional", "self", "value"],
    }


def test_probe_cli_rejects_any_shape_other_than_one_manifest_target(capsys) -> None:
    assert main([]) == 2
    assert "MODULE.CLASS.METHOD" in capsys.readouterr().err
