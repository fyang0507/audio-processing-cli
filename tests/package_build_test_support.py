"""Offline native-package fixture: real Git, uv and a stdlib PEP 517 build backend."""

from __future__ import annotations

import sys
from dataclasses import replace

import pytest
from package_test_support import FakeFetcher, FakeToolchain

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli.packages import catalog

# No dependencies, downloads or model imports. The backend makes the ordinary setuptools
# residue observed in the FireRed artifact plus ignored bytecode, including on failure.
BACKEND = """from pathlib import Path
import os
import zipfile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    for name in (
        "build/lib/fireredasr2s/__init__.py",
        "fireredasr2s.egg-info/PKG-INFO",
        "__pycache__/build_backend.cpython-311.pyc",
    ):
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("build residue")
    mode = os.environ.get("AUDIO_TEST_BUILD_MODE", "ok")
    if mode == "fail":
        raise RuntimeError("synthetic build failure after residue")
    if mode == "tracked":
        Path("tracked_source.py").write_text("tampered by build")
    name = "fireredasr2s-0.0.1-py3-none-any.whl"
    metadata = "fireredasr2s-0.0.1.dist-info/"
    entries = {
        "fireredasr2s/__init__.py": "VALUE = 1\\n",
        metadata + "METADATA": "Metadata-Version: 2.1\\nName: fireredasr2s\\nVersion: 0.0.1\\n",
        metadata + "WHEEL": "Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
    }
    entries[metadata + "RECORD"] = "".join(path + ",,\\n" for path in [*entries, metadata + "RECORD"])
    with zipfile.ZipFile(Path(wheel_directory) / name, "w") as wheel:
        for path, content in entries.items():
            wheel.writestr(path, content)
    return name
"""


class OfflineBuildToolchain(FakeToolchain):
    """Fake model locks, real disposable source builds and direct-install provenance."""

    def __init__(self):
        super().__init__()
        self.build_residue: list[tuple[str, ...]] = []
        self.interrupt = False
        self.skip_cleanup = False

    def run(self, args, *, cwd=None, timeout=3600):
        if args[0] in {"git", "uv"}:
            self.calls.append(list(args))
            return pkg.Toolchain.run(self, args, cwd=cwd, timeout=timeout)
        return super().run(args, cwd=cwd, timeout=timeout)

    def create_environment(self, environment, target):
        # Do not call the fake's stub writer: a real venv's interpreter is a symlink.
        # Preserve checkouts on repair while modeling sync's removal of direct installs.
        self.created.append(environment.name)
        self._synced.add(environment.name)
        self._direct_installs[environment.name] = {}
        if (target / "bin/python").exists():
            result = self.run(
                ["uv", "pip", "uninstall", "--python", str(target / "bin/python"), "fireredasr2s"]
            )
            assert result.returncode == 0, result.stderr
        result = self.run(
            ["uv", "venv", "--allow-existing", "--python", sys.executable, str(target)]
        )
        assert result.returncode == 0, result.stderr

    def clone(self, repo, commit, target):
        pkg.Toolchain.clone(self, repo, commit, target)

    def inspect_checkout(self, checkout):
        return pkg.Toolchain.inspect_checkout(self, checkout)

    def require_checkout_binding(self, checkout):
        pkg.Toolchain.require_checkout_binding(self, checkout)

    def install_checkout(self, environment_python, checkout):
        try:
            pkg.Toolchain.install_checkout(self, environment_python, checkout)
            self._direct_installs[environment_python.parent.parent.name] = (
                pkg.Toolchain.frozen_packages(self, environment_python)
            )
            if self.interrupt:
                raise KeyboardInterrupt("synthetic interruption after build")
        finally:
            self.build_residue.append(self.inspect_checkout(checkout).untracked)

    def clean_checkout_install_artifacts(self, checkout):
        if not self.skip_cleanup:
            super().clean_checkout_install_artifacts(checkout)


@pytest.fixture
def offline_build(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_OFFLINE", "true")
    monkeypatch.setenv("UV_NO_CACHE", "true")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    source = tmp_path / "source-fixture"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "build_backend"\nbackend-path = ["."]\n'
    )
    (source / ".gitignore").write_text("__pycache__/\n")
    (source / "build_backend.py").write_text(BACKEND)
    (source / "tracked_source.py").write_text("VALUE = 1\n")
    real = pkg.Toolchain()
    for command in (
        ["git", "init", "--quiet"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Audio Tests",
            "-c",
            "user.email=audio@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "disposable build fixture",
        ],
    ):
        result = real.run(command, cwd=source)
        assert result.returncode == 0, result.stderr
    commit = real.inspect_checkout(source).head
    packages = dict(env.packages())
    original = packages["firered-asr2s"]
    package = replace(
        original,
        checkout={
            **original.checkout,
            "repo": str(source),
            "commit": commit,
            "resolved_commit": commit,
        },
    )
    packages[package.id] = package
    monkeypatch.setattr(catalog, "packages", lambda: packages)
    toolchain = OfflineBuildToolchain()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    return provisioner, package


__all__ = ["OfflineBuildToolchain", "offline_build"]
