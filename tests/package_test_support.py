"""The `audio packages` lifecycle, against a fabricated root.

Provisioning the real thing costs 30 GiB and a Swift toolchain, so nothing here downloads or
builds. The two external surfaces are injected, which leaves the parts that actually carry the
rules under test: the registry state machine, the crash-safety guarantee, reference counting,
digest verification, and every payload shape the spec documents publish.

The one thing a fake cannot check is whether the real `uv` and Hub calls work. Those are
exercised by the provisioning probes in `model_tests/benchmark/` and recorded there.

What a fake here *must* be able to represent is the state a repair produces. Both doubles below
say so at the point where it matters, because both were wrong about it once: a double that always
rewrites a snapshot, or that can never stop being drifted, turns a repair test green without the
repair ever running.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from audio_cli import environments as env
from audio_cli import media as media_module
from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.cli import main
from audio_cli.packages import catalog as package_catalog
from audio_cli.packages import (
    environment_verification as package_environment_verification,
)
from audio_cli.packages import fetcher as package_fetcher
from audio_cli.packages import integrity as package_integrity
from audio_cli.packages import registry as package_registry
from audio_cli.packages import requirements as package_requirements
from audio_cli.packages import teardown as package_teardown

VIBE_MODEL_REVISION = "d0c9efdb8d614685062c04425d91e01b6f37d944"
VIBE_TOKENIZER_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    """Every test gets an empty root, so none of them can see a real provisioning."""
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(tmp_path / "root"))

    def snapshot_index() -> dict[tuple[str, str], Path]:
        found: dict[tuple[str, str], Path] = {}
        for package in env.packages().values():
            source = package.source
            repositories = (
                source["repos"]
                if source["type"] == "huggingface_multi"
                else [source]
                if source["type"] == "huggingface"
                else []
            )
            for repository in repositories:
                found[(repository["repo"], repository["revision"])] = (
                    tmp_path
                    / "hub"
                    / f"models--{repository['repo'].replace('/', '--')}"
                    / "snapshots"
                    / repository["revision"]
                )
        return found

    monkeypatch.setattr(package_integrity, "_hub_snapshot_index", snapshot_index)
    return tmp_path / "root"


class FakeToolchain(pkg.Toolchain):
    """Records what would have been run, and pretends it succeeded.

    Drift is *clearable state*, which is the one thing this double has to get right. `uv pip
    sync` is convergent — it makes an environment equal to its lock (ENVIRONMENTS.md) — so an
    environment that has just been recreated cannot still be drifted. This applied `drift`
    unconditionally and never cleared it, which made a repaired environment unrepresentable: the
    only way to observe `ok` was to swap in a second toolchain, and a test doing that never
    executes `verify`'s repair branch at all. Same standard as `FakeFetcher.hf_snapshot` below,
    for the same reason: a double that cannot represent the state a repair produces makes a
    broken repair pass.
    """

    def __init__(
        self,
        *,
        missing: tuple[str, ...] = (),
        drift: dict | None = None,
        private_api_hash: str | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.missing = set(missing)
        self._drift = dict(drift or {})
        self.created: list[str] = []
        self._synced: set[str] = set()
        self._direct_installs: dict[str, dict[str, str]] = {}
        self.private_api_hash = private_api_hash
        self._checkout_commits: dict[Path, str] = {}
        self._checkout_tracked: dict[Path, dict[str, bytes]] = {}

    def which(self, tool: str) -> str | None:
        return None if tool in self.missing else f"/usr/bin/{tool}"

    def run(self, args, *, cwd=None, timeout=3600):
        self.calls.append(list(args))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        # Opt-in, because the default has to keep yielding *no* verdict: that is the path
        # `mlx_audio_private_api_error` covers, and the exemption for it in
        # tests/test_shipped_commands_match_the_document.py has to stay reachable. The only
        # checked-in runtime-probe invocation in the tool is the private-API probe.
        if (
            self.private_api_hash is not None
            and len(args) >= 2
            and Path(str(args[1])).name == "runtime_probe.py"
        ):
            guards = {guard["kind"]: guard for guard in env.environments()["mlx"].guards}
            Result.stdout = json.dumps(
                {
                    "sha256": self.private_api_hash,
                    "params": sorted(guards["signature"]["required_parameters"]),
                }
            )
        return Result()

    def create_environment(self, environment, target: Path) -> None:
        self.created.append(environment.name)
        (target / "bin").mkdir(parents=True, exist_ok=True)
        interpreter = target / "bin" / "python"
        assert not interpreter.is_symlink(), (
            "fake environment must not overwrite a real interpreter"
        )
        interpreter.write_text("#!/bin/sh\n")
        # Re-syncing from the lock is what removes drift, so this is where it goes.
        self._synced.add(environment.name)
        self._direct_installs[environment.name] = {}

    def frozen_packages(self, environment_python: Path) -> dict[str, str]:
        name = environment_python.parent.parent.name
        installed = package_requirements._locked_versions(env.environments()[name])
        if name not in self._synced:
            installed.update(self._drift)
        installed.update(self._direct_installs.get(name, {}))
        return installed

    def clone(self, repo: str, commit: str, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
        touched = target / "vibevoice" / "modular"
        touched.mkdir(parents=True, exist_ok=True)
        (touched / "modeling_vibevoice_asr.py").write_text("original\n")
        fluid_process = target / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
        fluid_process.parent.mkdir(parents=True, exist_ok=True)
        fluid_process.write_text("original fluid process\n", encoding="utf-8")
        package = env.packages()[target.name]
        expected = (
            package.checkout.get("resolved_commit", package.checkout["commit"])
            if package.checkout is not None
            else package.source["commit"]
        )
        self._checkout_commits[target] = expected
        self._checkout_tracked[target] = {
            str(path.relative_to(target)): path.read_bytes()
            for path in target.rglob("*")
            if path.is_file()
        }

    def apply_patch(self, checkout: Path, patch: Path) -> None:
        if checkout.name == "fluidaudio":
            target = checkout / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
            target.write_text("patched fluid process\n", encoding="utf-8")
            return
        target = checkout / "vibevoice" / "modular" / "modeling_vibevoice_asr.py"
        if target.is_file():
            target.write_text("patched\n")

    def install_checkout(self, environment_python: Path, checkout: Path) -> None:
        self.calls.append(["install", str(checkout)])
        environment_name = environment_python.parent.parent.name
        package = env.packages()[checkout.name]
        distribution = package.checkout["distribution"]
        self._direct_installs.setdefault(environment_name, {})[distribution] = (
            f"@ {checkout.resolve(strict=False).as_uri()}"
        )

    def file_digest(self, path: Path) -> str:
        fluid = env.packages()["fluidaudio"].source.get("patched_file_sha256", {})
        for name, digest in fluid.items():
            if (
                path.as_posix().endswith(f"/{name}")
                and path.read_bytes() == b"patched fluid process\n"
            ):
                return digest
        expected = env.packages()["vibevoice-asr-7b"].checkout["patched_file_sha256"]
        for name, digest in expected.items():
            if path.as_posix().endswith(f"/{name}") and path.read_bytes() == b"patched\n":
                return digest
        return pkg.sha256_file(path)

    def inspect_checkout(self, checkout: Path) -> pkg.CheckoutState:
        tracked = self._checkout_tracked[checkout]
        modified = tuple(
            sorted(
                name
                for name, contents in tracked.items()
                if not (checkout / name).is_file() or (checkout / name).read_bytes() != contents
            )
        )
        files = {str(path.relative_to(checkout)) for path in checkout.rglob("*") if path.is_file()}
        return pkg.CheckoutState(
            head=self._checkout_commits[checkout],
            modified=modified,
            untracked=tuple(sorted(files - set(tracked))),
        )

    def require_checkout_binding(self, checkout: Path) -> None:
        # This double's clone inventory represents its Git metadata; real-Git tests
        # override this method and exercise the production binding probe.
        assert checkout in self._checkout_tracked

    def swift_build(self, checkout: Path) -> None:
        product = env.packages()[checkout.name].source["product"]
        artifact = checkout / ".build" / "fake-target" / "release" / product
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"x" * 1024)
        artifact.chmod(0o755)

    def swift_product_runs(self, checkout: Path, product: str) -> bool:
        return True


class FakeFetcher(pkg.Fetcher):
    """Writes a plausible snapshot instead of downloading one."""

    WEIGHTS = b"w" * 2048

    def __init__(
        self, tmp_path: Path, *, corrupt: bool = False, already_cached: tuple[str, ...] = ()
    ) -> None:
        self.hub = tmp_path / "hub"
        self.corrupt = corrupt
        self.already_cached = set(already_cached)
        self.snapshots: list[tuple[str, str]] = []
        self.forced: list[tuple[str, str]] = []
        self.filtered: list[tuple[str, str, tuple[str, ...]]] = []

    def cached_revisions(self) -> set[str]:
        return set(self.already_cached)

    def hf_snapshot(
        self,
        repo: str,
        revision: str,
        *,
        force: bool = False,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        """Faithful about the one behaviour `--repair` turns on.

        `snapshot_download` returns a revision the cache already holds as it stands — corrupt or
        not — and only `force_download` re-fetches its files. A fake that always rewrote them
        would make a broken repair pass.
        """
        self.snapshots.append((repo, revision))
        if force:
            self.forced.append((repo, revision))
        if allow_patterns is not None:
            self.filtered.append((repo, revision, allow_patterns))
        target = self.hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        artifacts = (
            [
                target / (f"{name[:-3]}/model.mil" if name.endswith("/**") else name)
                for name in allow_patterns
            ]
            if allow_patterns is not None
            else [target / "model.safetensors"]
        )
        if all(artifact.is_file() for artifact in artifacts) and not force:
            return target
        target.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(self.WEIGHTS)
        return target

    def delete_hub_revisions(self, revisions: list[str]) -> tuple[list[str], int]:
        """Deletes the fake snapshot directories this fetcher created, and reports their size."""
        deleted, freed = [], 0
        for revision in revisions:
            for candidate in self.hub.glob(f"models--*/snapshots/{revision}"):
                freed += package_integrity._tree_bytes(candidate)
                for item in sorted(candidate.rglob("*"), reverse=True):
                    item.unlink() if item.is_file() else item.rmdir()
                candidate.rmdir()
                deleted.append(revision)
        return deleted, freed

    def url_file(self, url: str, sha256: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"corrupt" if self.corrupt else b"onnx-bytes")
        if not self.corrupt:
            # Stand in for a verified download without needing the real 2.3 MB artifact.
            pkg.sha256_file(target)
        return target


def _snapshot_index_for(hub: Path) -> dict[tuple[str, str], Path]:
    """The cache-index view corresponding to a FakeFetcher's configured Hub root."""
    found: dict[tuple[str, str], Path] = {}
    for package in env.packages().values():
        source = package.source
        repositories = (
            source["repos"]
            if source["type"] == "huggingface_multi"
            else [source]
            if source["type"] == "huggingface"
            else []
        )
        for repository in repositories:
            found[(repository["repo"], repository["revision"])] = (
                hub
                / f"models--{repository['repo'].replace('/', '--')}"
                / "snapshots"
                / repository["revision"]
            )
    return found


@pytest.fixture
def provisioner(tmp_path):
    return pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))


QWEN_REPO = "mlx-community/Qwen3-ASR-1.7B-8bit"
QWEN_REVISION = "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
ALIGNER_REVISION = "0e1a68e91d815300c7c9754b2a7639378b23db15"
FIRERED_REVISIONS = (
    "2304afed56eacfee6256dee5937ed22ffa0b64ec",
    "1bb4d285c8456429385d9c0810300df4297bc11b",
    "e448fd967f44182a1c323cc30f5d89f2400c28da",
    "7990aaccc6b7aec1e527743bd30201f2c4a03b8c",
)


@pytest.fixture
def the_fake_download_satisfies_the_pin(monkeypatch):
    """Make the fake `url` download match its manifest pin, so a *passing* digest exists.

    The real artifact is 2.3 MB and hash-pinned, and the fake writes ten bytes, so under a fake
    fetcher the only reachable outcome for `silero-vad` is a failed digest. The claim under test
    here is the passing one — `digest: "ok"` has to be earned by a package that has a hash, and
    withheld from every package that does not.
    """
    catalog = dict(env.packages())
    silero = catalog["silero-vad"]
    catalog["silero-vad"] = replace(
        silero,
        source={**silero.source, "sha256": hashlib.sha256(b"onnx-bytes").hexdigest()},
    )
    monkeypatch.setattr(package_catalog, "packages", lambda: catalog)
    return catalog


__all__ = [
    "ALIGNER_REVISION",
    "FIRERED_REVISIONS",
    "QWEN_REPO",
    "QWEN_REVISION",
    "VIBE_MODEL_REVISION",
    "VIBE_TOKENIZER_REVISION",
    "FakeFetcher",
    "FakeToolchain",
    "Path",
    "_snapshot_index_for",
    "env",
    "hashlib",
    "io",
    "isolated_root",
    "json",
    "main",
    "media_module",
    "os",
    "package_catalog",
    "package_environment_verification",
    "package_fetcher",
    "package_integrity",
    "package_registry",
    "package_requirements",
    "package_teardown",
    "paths",
    "pkg",
    "provisioner",
    "pytest",
    "replace",
    "shutil",
    "subprocess",
    "sys",
    "the_fake_download_satisfies_the_pin",
    "types",
]
