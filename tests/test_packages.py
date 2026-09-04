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
                source["repos"] if source["type"] == "huggingface_multi"
                else [source] if source["type"] == "huggingface"
                else []
            )
            for repository in repositories:
                found[(repository["repo"], repository["revision"])] = (
                    tmp_path / "hub"
                    / f"models--{repository['repo'].replace('/', '--')}"
                    / "snapshots" / repository["revision"]
                )
        return found

    monkeypatch.setattr(pkg, "_hub_snapshot_index", snapshot_index)
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

    def __init__(self, *, missing: tuple[str, ...] = (), drift: dict | None = None,
                 private_api_hash: str | None = None) -> None:
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
        # `-c` invocation in the tool is the private-API probe.
        if self.private_api_hash is not None and list(args)[1:2] == ["-c"]:
            guards = {guard["kind"]: guard for guard in env.environments()["mlx"].guards}
            Result.stdout = json.dumps({
                "sha256": self.private_api_hash,
                "params": sorted(guards["signature"]["required_parameters"]),
            })
        return Result()

    def create_environment(self, environment, target: Path) -> None:
        self.created.append(environment.name)
        (target / "bin").mkdir(parents=True, exist_ok=True)
        (target / "bin" / "python").write_text("#!/bin/sh\n")
        # Re-syncing from the lock is what removes drift, so this is where it goes.
        self._synced.add(environment.name)
        self._direct_installs[environment.name] = {}

    def frozen_packages(self, environment_python: Path) -> dict[str, str]:
        name = environment_python.parent.parent.name
        installed = pkg._locked_versions(env.environments()[name])
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
        fluid_process = (
            target / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
        )
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
            target = (
                checkout / "Sources" / "FluidAudioCLI" / "Commands"
                / "ProcessCommand.swift"
            )
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
            if path.as_posix().endswith(f"/{name}") \
                    and path.read_bytes() == b"patched fluid process\n":
                return digest
        expected = env.packages()["vibevoice-asr-7b"].checkout["patched_file_sha256"]
        for name, digest in expected.items():
            if path.as_posix().endswith(f"/{name}") and path.read_bytes() == b"patched\n":
                return digest
        return pkg.sha256_file(path)

    def inspect_checkout(self, checkout: Path) -> pkg.CheckoutState:
        tracked = self._checkout_tracked[checkout]
        modified = tuple(sorted(
            name
            for name, contents in tracked.items()
            if not (checkout / name).is_file() or (checkout / name).read_bytes() != contents
        ))
        files = {
            str(path.relative_to(checkout))
            for path in checkout.rglob("*")
            if path.is_file()
        }
        return pkg.CheckoutState(
            head=self._checkout_commits[checkout],
            modified=modified,
            untracked=tuple(sorted(files - set(tracked))),
        )

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

    def __init__(self, tmp_path: Path, *, corrupt: bool = False,
                 already_cached: tuple[str, ...] = ()) -> None:
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
        target = (
            self.hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        )
        artifacts = (
            [
                target / (
                    f"{name[:-3]}/model.mil" if name.endswith("/**") else name
                )
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
                freed += pkg._tree_bytes(candidate)
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
            source["repos"] if source["type"] == "huggingface_multi"
            else [source] if source["type"] == "huggingface"
            else []
        )
        for repository in repositories:
            found[(repository["repo"], repository["revision"])] = (
                hub / f"models--{repository['repo'].replace('/', '--')}"
                / "snapshots" / repository["revision"]
            )
    return found


@pytest.fixture
def provisioner(tmp_path):
    return pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))


def test_an_empty_root_reports_everything_absent() -> None:
    report = pkg.list_report()
    assert report["packages"] == []
    assert set(report["environments"].values()) == {"absent"}
    assert report["total_known_bytes"] == 0


def test_list_uses_manifest_identity_and_license_facts_for_known_packages() -> None:
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    document = pkg.blank_registry()
    document["packages"][package.id] = {
        "state": "ready",
        "environment": "tampered-environment",
        "license_declared": "tampered-license",
        "license_reviewed": not package.license_reviewed,
        "materialized": {"bytes": 7},
    }
    pkg.save_registry(document)

    listed = pkg.list_report()["packages"]

    assert listed == [{
        "package": package.id,
        "environment": package.environment,
        "state": "ready",
        "bytes": 7,
        "license_declared": package.license_declared,
        "license_reviewed": package.license_reviewed,
        "used_by_stacks": list(package.stacks),
    }]


def test_path_report_locates_things_before_anything_is_provisioned() -> None:
    """VOCABULARY: a session with no provisioning history must still find everything."""
    report = pkg.path_report()
    assert report["root"] == str(paths.root())
    assert report["registry"].endswith("registry.json")
    assert report["environments"]["mlx"]["state"] == "absent"
    assert report["environments"]["mlx"]["python"].endswith("envs/mlx/bin/python")


def test_pull_creates_the_environment_and_marks_the_package_ready(provisioner) -> None:
    selection = pkg.select(["qwen3-asr-1.7b-8bit"])
    receipt = provisioner.pull(selection)

    assert receipt["environments_created"] == ["mlx"]
    assert receipt["pulled"][0]["package"] == "qwen3-asr-1.7b-8bit"
    assert receipt["pulled"][0]["revision"] == "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"

    document = pkg.load_registry()
    assert document["packages"]["qwen3-asr-1.7b-8bit"]["state"] == "ready"
    assert document["environments"]["mlx"]["state"] == "ready"
    assert document["environments"]["mlx"]["lock_sha256"] == env.lock_digest("mlx")
    assert pkg.missing_packages(selection) == []


def test_pull_warns_that_a_declared_license_is_not_a_reviewed_one(
    provisioner, tmp_path,
) -> None:
    receipt = provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    warning = receipt["warnings"][0]
    assert warning["code"] == "license_unreviewed"
    assert warning["blocking"] is False
    assert warning["packages"] == ["qwen3-asr-1.7b-8bit"]

    # And says nothing where the terms were actually read.
    other = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))
    assert other.pull(pkg.select(["speaker-diarization-coreml"]))["warnings"] == []


def test_a_crashed_pull_does_not_read_as_provisioned(tmp_path) -> None:
    """The failure mode this state machine exists for: exit 3 must still fire afterwards."""

    class ExplodingFetcher(FakeFetcher):
        def hf_snapshot(self, repo: str, revision: str, *, force: bool = False) -> Path:
            raise pkg.ProvisioningError("download_failed", "network went away")

    provisioner = pkg.Provisioner(toolchain=FakeToolchain(),
                                 fetcher=ExplodingFetcher(tmp_path))
    selection = pkg.select(["qwen3-asr-1.7b-8bit"])
    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull(selection)

    document = pkg.load_registry()
    assert document["packages"]["qwen3-asr-1.7b-8bit"]["state"] == "pulling"
    assert not pkg.is_ready(document, "qwen3-asr-1.7b-8bit")
    assert [p.id for p in pkg.missing_packages(selection)] == ["qwen3-asr-1.7b-8bit"]

    # And it is still nameable, which is what purge needs.
    assert "qwen3-asr-1.7b-8bit" in pkg.Provisioner().purge(dry_run=True)["would_remove"][
        "packages"]


def test_a_crashed_pull_is_recoverable_by_pulling_again(tmp_path) -> None:
    calls = {"n": 0}

    class FlakyFetcher(FakeFetcher):
        def hf_snapshot(self, repo: str, revision: str, *, force: bool = False) -> Path:
            calls["n"] += 1
            if calls["n"] == 1:
                raise pkg.ProvisioningError("download_failed", "first attempt died")
            return super().hf_snapshot(repo, revision, force=force)

    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FlakyFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")


def test_the_registry_is_written_atomically(provisioner, isolated_root) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    leftovers = list(isolated_root.glob(".audio-registry-*.tmp"))
    assert leftovers == [], f"temporary registry files survived: {leftovers}"
    assert json.loads(paths.registry_path().read_text())["schema_version"] == 1


def test_registry_temporary_never_follows_a_precreated_symlink(
    tmp_path, monkeypatch,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.json"
    victim.write_text("owned elsewhere\n", encoding="utf-8")
    monkeypatch.setattr(
        "audio_cli.packages.uuid.uuid4", lambda: types.SimpleNamespace(hex="fixed")
    )
    temporary = target.with_name(f".audio-registry-{os.getpid()}-fixed.tmp")
    temporary.symlink_to(victim)

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.save_registry(pkg.blank_registry())

    assert caught.value.code == "registry_unreadable"
    assert victim.read_text(encoding="utf-8") == "owned elsewhere\n"
    assert temporary.is_symlink()
    assert not target.exists()


def test_registry_writer_refuses_a_known_substituted_temporary(
    tmp_path, monkeypatch,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.json"
    victim.write_text("owned elsewhere\n", encoding="utf-8")
    monkeypatch.setattr(
        "audio_cli.packages.uuid.uuid4", lambda: types.SimpleNamespace(hex="fixed")
    )
    temporary = target.with_name(f".audio-registry-{os.getpid()}-fixed.tmp")
    real_assert = pkg.assert_directory_binding

    def substitute_temporary(descriptor: int, directory: Path) -> None:
        real_assert(descriptor, directory)
        temporary.unlink()
        temporary.symlink_to(victim)

    monkeypatch.setattr(pkg, "assert_directory_binding", substitute_temporary)

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.save_registry(pkg.blank_registry())

    assert caught.value.code == "registry_unreadable"
    assert "temporary changed identity" in caught.value.message
    assert victim.read_text(encoding="utf-8") == "owned elsewhere\n"
    assert temporary.is_symlink()
    assert not target.exists()


def test_registry_writer_rolls_back_a_temporary_substitution_at_publication(
    tmp_path, monkeypatch,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    original = pkg.blank_registry()
    original["tool_version"] = "original"
    target.write_text(json.dumps(original), encoding="utf-8")
    before = target.read_bytes()
    victim = tmp_path / "victim.json"
    victim.write_text("owned elsewhere\n", encoding="utf-8")
    monkeypatch.setattr(
        "audio_cli.packages.uuid.uuid4", lambda: types.SimpleNamespace(hex="fixed")
    )
    temporary = target.with_name(f".audio-registry-{os.getpid()}-fixed.tmp")
    real_exchange = media_module._rename_exchange
    substituted = False

    def substitute_at_exchange(
        directory_descriptor: int, left_name: str, right_name: str,
    ) -> None:
        nonlocal substituted
        if not substituted:
            substituted = True
            os.unlink(left_name, dir_fd=directory_descriptor)
            os.symlink(str(victim), left_name, dir_fd=directory_descriptor)
        real_exchange(directory_descriptor, left_name, right_name)

    monkeypatch.setattr(media_module, "_rename_exchange", substitute_at_exchange)

    with pytest.raises(pkg.ProvisioningError) as caught:
        replacement = pkg.blank_registry()
        replacement["tool_version"] = "replacement"
        pkg.save_registry(replacement)

    assert caught.value.code == "registry_unreadable"
    assert substituted is True
    assert target.read_bytes() == before
    assert victim.read_text(encoding="utf-8") == "owned elsewhere\n"
    assert temporary.is_symlink()


def test_registry_writer_cannot_follow_a_parent_swapped_after_open(
    tmp_path, monkeypatch,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True)
    displaced_root = tmp_path / "root-before-swap"
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / target.name
    victim.write_text("owned elsewhere\n", encoding="utf-8")

    def swap_parent():
        target.parent.rename(displaced_root)
        target.parent.symlink_to(outside, target_is_directory=True)
        return types.SimpleNamespace(hex="fixed")

    monkeypatch.setattr("audio_cli.packages.uuid.uuid4", swap_parent)
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.save_registry(pkg.blank_registry())

    assert caught.value.code == "registry_unreadable"
    assert "directory identity changed" in caught.value.message
    assert victim.read_text(encoding="utf-8") == "owned elsewhere\n"
    assert not list(outside.glob(".audio-registry-*.tmp"))
    assert not list(displaced_root.glob(".audio-registry-*.tmp"))


@pytest.mark.parametrize("destination_kind", ["directory", "symlink"])
def test_registry_writer_refuses_a_nonregular_destination(
    tmp_path, destination_kind: str,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True)
    victim = tmp_path / "outside-registry.json"
    victim.write_text("owned elsewhere\n", encoding="utf-8")
    if destination_kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(victim)

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.save_registry(pkg.blank_registry())

    assert caught.value.code == "registry_unreadable"
    assert "not a regular file" in caught.value.message
    assert victim.read_text(encoding="utf-8") == "owned elsewhere\n"
    if destination_kind == "directory":
        assert target.is_dir()
    else:
        assert target.is_symlink()


def test_remove_takes_the_environment_only_with_its_last_package(provisioner) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit", "qwen3-forcedaligner",
                                 "firered-asr2s"]))

    first = provisioner.remove(["qwen3-asr-1.7b-8bit"])
    assert first["environments_removed"] == []
    assert "mlx" in first["environments_kept"]
    assert "qwen3-forcedaligner still needs mlx" in first["environments_kept_reason"]

    second = provisioner.remove(["qwen3-forcedaligner"])
    assert second["environments_removed"] == ["mlx"]
    assert second["environments_kept"] == ["torch-firered"]
    assert not paths.env_dir("mlx").exists()


def test_remove_reports_hub_revisions_and_never_claims_to_own_the_cache(provisioner) -> None:
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    report = provisioner.remove(["vibevoice-asr-7b"])
    assert report["hub_revisions_deleted"] == [
        VIBE_MODEL_REVISION,
        VIBE_TOKENIZER_REVISION,
    ]
    assert "shared Hugging Face cache" in report["hub_cache_note"]


def test_remove_deletes_the_checkout_and_the_revision_it_materialized(provisioner) -> None:
    """The recorded revision is deleted; the shared cache around it is not touched."""
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    checkout = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
    snapshots = [
        Path(value)
        for value in pkg.load_registry()["packages"]["vibevoice-asr-7b"]
        ["materialized"]["paths"].values()
    ]
    snapshot = snapshots[0]
    sibling = snapshot.parent / "another-revision"
    sibling.mkdir(parents=True)
    (sibling / "weights").write_bytes(b"someone else's")
    assert checkout.is_dir() and all(item.is_dir() for item in snapshots)

    report = provisioner.remove(["vibevoice-asr-7b"])
    assert not checkout.exists()
    assert not any(item.exists() for item in snapshots), (
        "the revisions this tool materialized should be reclaimed"
    )
    assert sibling.is_dir(), "a revision this tool never recorded is not ours to delete"
    assert report["reclaimed_bytes"] > 0
    assert report["hub_revisions_not_found"] == []


def test_remove_rejects_a_package_that_was_never_provisioned(provisioner) -> None:
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.remove(["firered-asr2s"])
    assert caught.value.code == "package_not_provisioned"
    assert caught.value.exit_code == 2


def test_purge_dry_run_reports_and_removes_nothing(provisioner) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit", "fluidaudio"]))
    report = provisioner.purge(dry_run=True)
    assert report["would_remove"]["environments"] == ["mlx", "swift"]
    assert report["unsized_packages"] == []  # fluidaudio's size is measured at pull time
    assert report["untouched"] == pkg.UNTOUCHED
    assert paths.env_dir("mlx").exists()
    assert pkg.load_registry()["packages"]

    done = provisioner.purge(dry_run=False)
    # The dry run projects registry figures; the real one reports what the filesystem and the
    # Hub cache actually gave back, so they are related but not equal by construction.
    assert done["reclaimed_bytes"] > 0
    assert pkg.load_registry()["packages"] == {}
    assert not paths.env_dir("mlx").exists()
    assert not paths.env_dir("swift").exists()


def test_purge_finds_everything_from_the_registry_alone(provisioner) -> None:
    """VOCABULARY: purge reads registry.json, not shell history."""
    provisioner.pull(pkg.select(["firered-asr2s"]))
    fresh = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(Path("/tmp")))
    assert fresh.purge(dry_run=True)["would_remove"]["packages"] == ["firered-asr2s"]


def test_verify_passes_on_a_freshly_pulled_root(provisioner) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    report = provisioner.verify()
    assert report["failed"] == []
    assert report["environments"]["mlx"] == "ok"
    assert [entry["package"] for entry in report["verified"]] == ["qwen3-asr-1.7b-8bit"]


def test_verify_reports_a_reverted_patch(provisioner) -> None:
    """The check must be able to fail, or the invariant it states is decoration."""
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    assert provisioner.verify()["failed"] == []

    patched = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b") / "vibevoice" / "modular" / \
        "modeling_vibevoice_asr.py"
    patched.write_text("original\n")

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert failure[0]["fix"].startswith("audio packages pull --repair")


def test_verify_reports_a_drifted_environment_and_repairs_it(tmp_path) -> None:
    """One provisioner, drifted before and `ok` after, with evidence the repair path ran.

    This used to assert the second half against a *different* toolchain, which reports `ok` with
    or without the flag — so `repair=True` was decorative and `verify`'s `if drift and repair`
    branch never executed. Note `"mlx"` here is the pip package whose lock pins 0.32.0, not the
    environment that happens to share its name.
    """
    pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)).pull(
        pkg.select(["qwen3-asr-1.7b-8bit"]))

    toolchain = FakeToolchain(drift={"mlx": "0.31.0"})
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))

    report = provisioner.verify()
    assert report["environments"]["mlx"] == "drifted"
    assert report["failed"][0]["code"] == "environment_drifted"
    assert report["failed"][0]["examples"]["mlx"] == {"locked": "0.32.0", "installed": "0.31.0"}
    assert report["failed"][0]["fix"] == "audio packages verify --repair"
    # Reported, not repaired: the flag is what re-syncs, so without it the state persists and a
    # second look says the same thing rather than the report having consumed it.
    assert toolchain.created == [], "verify without --repair re-synced an environment"
    assert provisioner.verify()["environments"]["mlx"] == "drifted"

    repaired = provisioner.verify(repair=True)
    assert repaired["environments"]["mlx"] == "ok"
    assert repaired["failed"] == []
    assert toolchain.created == ["mlx"], (
        "the environment was never re-synced, so nothing repaired the drift"
    )
    # And it stayed repaired, which is what distinguishes a re-sync from a suppressed report.
    assert provisioner.verify()["environments"]["mlx"] == "ok"


def test_verify_treats_an_unlocked_extra_distribution_as_repairable_drift(
    tmp_path,
) -> None:
    pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)).pull(
        pkg.select(["qwen3-asr-1.7b-8bit"])
    )
    direct_reference = "@ file:///external/startup-hook"
    toolchain = FakeToolchain(drift={"startup-hook": direct_reference})
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))

    report = provisioner.verify()

    assert report["environments"]["mlx"] == "drifted"
    assert report["failed"][0]["examples"]["startup-hook"] == {
        "locked": None,
        "installed": direct_reference,
    }
    repaired = provisioner.verify(repair=True)
    assert repaired["environments"]["mlx"] == "ok"
    assert repaired["failed"] == []
    assert toolchain.created == ["mlx"]
    assert provisioner.verify()["environments"]["mlx"] == "ok"


def test_freeze_parser_keeps_direct_editable_and_unknown_installed_lines(
    monkeypatch,
) -> None:
    toolchain = pkg.Toolchain()
    stdout = "\n".join((
        "locked_package==1.2.3",
        "rogue @ file:///tmp/rogue",
        "-e file:///tmp/editable#egg=editable_hook",
        "-e http://[invalid",
        "future-freeze-syntax",
    ))
    monkeypatch.setattr(
        toolchain,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout=stdout, stderr=""
        ),
    )

    assert toolchain.frozen_packages(Path("/unused/python")) == {
        "locked-package": "1.2.3",
        "rogue": "@ file:///tmp/rogue",
        "editable-hook": "-e file:///tmp/editable#egg=editable_hook",
        "<editable:http://[invalid>": "-e http://[invalid",
        "<unparsed:future-freeze-syntax>": "future-freeze-syntax",
    }


def test_environment_drift_allows_only_manifest_owned_direct_checkout_paths(
    tmp_path,
) -> None:
    managed = tmp_path / "managed"
    external = tmp_path / "external"
    drift = pkg._environment_drift(
        {},
        {
            "native-backend": f"@ {managed.as_uri()}",
            "startup-hook": f"@ {external.as_uri()}",
        },
        {"native-backend": managed},
    )

    assert drift == {
        "startup-hook": (None, f"@ {external.as_uri()}"),
    }


@pytest.mark.parametrize(
    "frozen",
    [
        {},
        {"wrong-name": "@ file:///managed/checkout"},
        {"native-backend": "@ file:///external/checkout"},
    ],
)
def test_environment_drift_requires_the_exact_named_checkout_install(
    frozen: dict[str, str],
) -> None:
    required = Path("/managed/checkout")
    drift = pkg._environment_drift(
        {}, frozen, {"native-backend": required}
    )
    assert drift["native-backend"] == (
        f"@ {required.as_uri()}", frozen.get("native-backend")
    )


@pytest.mark.parametrize("package_id", ["vibevoice-asr-7b", "firered-asr2s"])
def test_verify_repair_reinstalls_ready_checkout_after_lock_sync(
    tmp_path: Path,
    package_id: str,
) -> None:
    toolchain = FakeToolchain()
    provisioner = pkg.Provisioner(
        toolchain=toolchain, fetcher=FakeFetcher(tmp_path)
    )
    package = env.packages()[package_id]
    provisioner.pull(pkg.select([package_id]))
    environment_name = package.environment
    distribution = package.checkout["distribution"]
    assert distribution in toolchain._direct_installs[environment_name]

    # Model a lock change plus uv sync: the direct install exists before repair,
    # create_environment removes it, and install_checkout must restore it.
    locked_name = next(iter(pkg._locked_versions(env.environments()[environment_name])))
    toolchain._synced.discard(environment_name)
    toolchain._drift[locked_name] = "0.invalid"
    document = pkg.load_registry()
    document["environments"][environment_name]["lock_sha256"] = "stale"
    pkg.save_registry(document)
    toolchain.calls.clear()
    toolchain.created.clear()

    repaired = provisioner.verify(repair=True)

    assert repaired["environments"][environment_name] == "ok"
    assert repaired["failed"] == []
    assert toolchain.created == [environment_name]
    assert [call[0] for call in toolchain.calls if call[0] == "install"] == [
        "install"
    ]
    assert distribution in toolchain._direct_installs[environment_name]
    registry = pkg.load_registry()
    assert registry["environments"][environment_name]["state"] == "ready"
    assert registry["environments"][environment_name]["lock_sha256"] == (
        pkg.sha256_file(env.environments()[environment_name].lock)
    )
    assert provisioner.verify()["environments"][environment_name] == "ok"

    created_before = list(toolchain.created)
    provisioner.pull(pkg.select([package_id]))
    assert toolchain.created == created_before


def test_verify_repair_never_installs_a_tampered_ready_checkout(
    tmp_path: Path,
) -> None:
    toolchain = FakeToolchain()
    provisioner = pkg.Provisioner(
        toolchain=toolchain, fetcher=FakeFetcher(tmp_path)
    )
    package_id = "vibevoice-asr-7b"
    package = env.packages()[package_id]
    provisioner.pull(pkg.select([package_id]))
    patched = (
        paths.checkout_dir(package.environment, package.id)
        / "vibevoice/modular/modeling_vibevoice_asr.py"
    )
    patched.write_text("tampered\n", encoding="utf-8")
    locked_name = next(iter(pkg._locked_versions(env.environments()[package.environment])))
    toolchain._synced.discard(package.environment)
    toolchain._drift[locked_name] = "0.invalid"
    toolchain.calls.clear()
    toolchain.created.clear()

    report = provisioner.verify(repair=True)

    assert report["environments"][package.environment] == "drifted"
    assert any(
        item.get("package") == package_id
        and item["code"] == "package_integrity_failed"
        for item in report["failed"]
    )
    assert toolchain.created == []
    assert not [call for call in toolchain.calls if call[0] == "install"]


def test_verify_refuses_a_symlinked_environment_root_even_when_freeze_matches(
    tmp_path,
) -> None:
    toolchain = FakeToolchain(private_api_hash="would-execute-external-python")
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    managed = paths.env_dir("mlx")
    external = tmp_path / "external-mlx"
    managed.rename(external)
    managed.symlink_to(external, target_is_directory=True)
    marker = external / "owned-elsewhere"
    marker.write_text("keep\n", encoding="utf-8")
    toolchain.calls.clear()

    report = provisioner.verify()

    assert report["environments"]["mlx"] == "drifted"
    assert report["failed"][0]["code"] == "environment_drifted"
    assert "environment path is a symlink" in report["failed"][0]["detail"]
    assert report["mlx_audio_private_api_matches_expected"] is False
    assert not any("-c" in call for call in toolchain.calls)
    repaired = provisioner.verify(repair=True)
    assert repaired["environments"]["mlx"] == "drifted"
    assert marker.read_text(encoding="utf-8") == "keep\n"

    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["qwen3-forcedaligner"]))
    assert caught.value.code == "environment_drifted"
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_pull_refuses_a_symlinked_provisioning_root_before_skip_or_fetch(
    tmp_path,
) -> None:
    package_id = "qwen3-asr-1.7b-8bit"
    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)
    )
    provisioner.pull(pkg.select([package_id]))
    root = paths.root()
    external = tmp_path / "external-provisioning-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)
    marker = external / "owned-elsewhere"
    marker.write_text("keep\n", encoding="utf-8")

    class NoFetch(FakeFetcher):
        def cached_revisions(self) -> set[str]:
            raise AssertionError("symlinked provisioning root reached cache inspection")

    toolchain = FakeToolchain()
    guarded = pkg.Provisioner(toolchain=toolchain, fetcher=NoFetch(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        # This ready package formerly took the skip fast path and reported success.
        guarded.pull(pkg.select([package_id]))

    assert caught.value.code == "environment_drifted"
    assert caught.value.message == f"provisioning root is a symlink: {root}"
    assert toolchain.calls == []
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_pull_refuses_a_symlinked_root_before_loading_its_registry(
    tmp_path, monkeypatch,
) -> None:
    root = paths.root()
    external = tmp_path / "external-provisioning-root"
    external.mkdir()
    root.symlink_to(external, target_is_directory=True)

    def forbidden_registry_read():
        raise AssertionError("symlinked provisioning root reached registry loading")

    monkeypatch.setattr(pkg, "load_registry", forbidden_registry_read)
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.Provisioner(
            toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)
        ).pull(pkg.select(["qwen3-asr-0.6b-8bit"]))

    assert caught.value.code == "environment_drifted"
    assert caught.value.message == f"provisioning root is a symlink: {root}"


def test_verify_never_probes_below_a_symlinked_provisioning_root(tmp_path) -> None:
    package_id = "vibevoice-asr-7b"
    toolchain = FakeToolchain(private_api_hash="would-execute-external-python")
    provisioner = pkg.Provisioner(
        toolchain=toolchain, fetcher=FakeFetcher(tmp_path)
    )
    provisioner.pull(pkg.select([package_id]))
    root = paths.root()
    external = tmp_path / "external-provisioning-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)
    marker = external / "owned-elsewhere"
    marker.write_text("keep\n", encoding="utf-8")
    toolchain.calls.clear()
    toolchain.created.clear()

    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.verify(repair=True)

    assert caught.value.code == "registry_unreadable"
    assert str(root) in caught.value.message
    assert toolchain.calls == []
    assert toolchain.created == []
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_every_managed_path_rejects_a_symlinked_provisioning_root(tmp_path) -> None:
    root = paths.root()
    environment = paths.env_dir("torch-vibevoice")
    checkout = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
    artifact = (
        paths.models_dir()
        / env.packages()["silero-vad"].source["filename"]
    )
    checkout.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")
    external = tmp_path / "external-provisioning-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)
    expected_issue = f"provisioning root is a symlink: {root}"

    assert pkg.managed_environment_path("torch-vibevoice") == (
        environment, expected_issue,
    )
    assert pkg.managed_checkout_path(
        env.packages()["vibevoice-asr-7b"], checkout
    ) == (checkout, f"package environment is not managed: {expected_issue}")
    assert pkg.managed_url_artifact_path(
        env.packages()["silero-vad"], artifact
    ) == (artifact, expected_issue)


def test_verify_refuses_an_in_root_symlinked_environments_parent(tmp_path) -> None:
    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)
    )
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    envs = paths.envs_dir()
    alternate = paths.root() / "alternate-envs"
    envs.rename(alternate)
    envs.symlink_to(alternate, target_is_directory=True)

    report = provisioner.verify()

    assert report["environments"]["mlx"] == "drifted"
    assert "environment parent is a symlink" in report["failed"][0]["detail"]
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["qwen3-forcedaligner"]))
    assert caught.value.code == "environment_drifted"


def test_verify_does_not_inspect_or_launch_below_a_redirected_environment_parent(
    tmp_path,
) -> None:
    class ProbeTrap(FakeToolchain):
        armed = False

        def inspect_checkout(self, checkout: Path) -> pkg.CheckoutState:
            if self.armed:
                raise AssertionError(f"verify inspected redirected checkout {checkout}")
            return super().inspect_checkout(checkout)

        def built_product_runs(self, executable: Path) -> bool:
            if self.armed:
                raise AssertionError(f"verify launched redirected product {executable}")
            return True

    toolchain = ProbeTrap()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["vibevoice-asr-7b", "fluidaudio"]))
    envs = paths.envs_dir()
    alternate = paths.root() / "alternate-envs"
    envs.rename(alternate)
    envs.symlink_to(alternate, target_is_directory=True)
    toolchain.armed = True

    report = provisioner.verify()

    assert report["verified"] == []
    assert {item["environment"] for item in report["failed"]} == {
        "torch-vibevoice", "swift",
    }
    assert all(item["code"] == "environment_drifted" for item in report["failed"])


def test_verify_fails_closed_when_ready_packages_have_no_ready_environment(
    tmp_path,
) -> None:
    class ProbeTrap(FakeToolchain):
        armed = False

        def inspect_checkout(self, checkout: Path) -> pkg.CheckoutState:
            if self.armed:
                raise AssertionError(f"verify inspected checkout without environment {checkout}")
            return super().inspect_checkout(checkout)

        def built_product_runs(self, executable: Path) -> bool:
            if self.armed:
                raise AssertionError(f"verify launched product without environment {executable}")
            return True

    toolchain = ProbeTrap()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["vibevoice-asr-7b", "fluidaudio"]))
    document = pkg.load_registry()
    document["environments"].pop("torch-vibevoice")
    document["environments"]["swift"]["state"] = "creating"
    pkg.save_registry(document)
    toolchain.armed = True

    report = provisioner.verify()

    assert report["verified"] == []
    assert report["environments"]["torch-vibevoice"] == "absent"
    assert report["environments"]["swift"] == "absent"
    failures = {item["environment"]: item for item in report["failed"]}
    assert set(failures) == {"torch-vibevoice", "swift"}
    assert failures["torch-vibevoice"]["code"] == "environment_not_ready"
    assert failures["swift"]["code"] == "environment_not_ready"
    assert failures["torch-vibevoice"]["packages"] == ["vibevoice-asr-7b"]
    assert failures["swift"]["packages"] == ["fluidaudio"]


def test_verify_reports_a_corrupted_single_file_artifact(tmp_path) -> None:
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(),
                                 fetcher=FakeFetcher(tmp_path, corrupt=True))
    provisioner.pull(pkg.select(["silero-vad"]))
    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]


def test_verify_reports_the_private_api_guard_as_unchecked_without_the_environment(
    provisioner,
) -> None:
    """No mlx environment means no verdict, not a passing one."""
    report = provisioner.verify()
    assert report["mlx_audio_private_api_matches_expected"] is None
    assert report["mlx_audio_private_api_expected_source_hash"] == (
        "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250"
    )


def test_selecting_by_stack_covers_every_package_that_stack_can_use() -> None:
    chosen = {package.id for package in pkg.select(stack="qwen-1.7b")}
    assert chosen == {"silero-vad", "qwen3-asr-1.7b-8bit", "qwen3-forcedaligner",
                      "fluidaudio", "speaker-diarization-coreml"}


def test_named_selection_stably_deduplicates_repeated_ids(provisioner) -> None:
    selection = pkg.select([
        "qwen3-asr-1.7b-8bit",
        "qwen3-forcedaligner",
        "qwen3-asr-1.7b-8bit",
    ])
    assert [package.id for package in selection] == [
        "qwen3-asr-1.7b-8bit", "qwen3-forcedaligner",
    ]

    receipt = provisioner.pull(pkg.select([
        "qwen3-asr-1.7b-8bit", "qwen3-asr-1.7b-8bit",
    ]))
    assert [item["package"] for item in receipt["pulled"]] == ["qwen3-asr-1.7b-8bit"]
    assert receipt["skipped"] == []


def test_repair_deduplicates_before_forcing_a_download(tmp_path) -> None:
    fetcher = FakeFetcher(tmp_path)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))

    receipt = provisioner.pull(pkg.select([
        "qwen3-asr-1.7b-8bit", "qwen3-asr-1.7b-8bit",
    ]), repair=True)

    assert fetcher.forced == [(QWEN_REPO, QWEN_REVISION)]
    assert [item["package"] for item in receipt["pulled"]] == ["qwen3-asr-1.7b-8bit"]


def test_unknown_names_fail_with_the_menu_rather_than_a_guess() -> None:
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.select(["whisper-large"])
    assert caught.value.code == "package_unknown"
    assert "qwen3-asr-1.7b-8bit" in caught.value.payload["allowed"]

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.select(stack="whisper")
    assert caught.value.payload["allowed"] == ["firered", "qwen-0.6b", "qwen-1.7b", "vibevoice"]


def test_a_missing_required_tool_blocks_only_the_package_that_needs_it(tmp_path) -> None:
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(missing=("swift",)),
                                 fetcher=FakeFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["fluidaudio"]))
    assert caught.value.code == "toolchain_missing"
    assert caught.value.payload["requires_tool"] == ["swift"]

    # The mlx package is unaffected, which is the point of reporting rather than failing.
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")


def test_doctor_separates_pull_toolchains_from_runtime_requirements() -> None:
    report = pkg.doctor(toolchain=FakeToolchain(missing=("swift",)))
    assert report["tools"]["swift"]["present"] is False
    assert report["environments"]["swift"]["requires_tool"] == ["swift"]
    assert report["environments"]["swift"]["blocked_by_missing_tool"] == ["swift"]
    assert report["environments"]["mlx"]["blocked_by_missing_tool"] == []
    assert report["environments"]["torch-vibevoice"]["provisional"] is True
    assert report["packages"]["firered-asr2s"] == "absent"
    assert env.packages()["fluidaudio"].requires_tool == ("swift",)


def test_registry_with_a_future_schema_version_is_refused(isolated_root) -> None:
    """A newer tool's registry must not be silently reinterpreted by an older one."""
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"


@pytest.mark.parametrize("foreign_kind", ["copy", "symlink"])
@pytest.mark.parametrize("teardown", ["remove", "purge"])
def test_foreign_registry_never_authorizes_hub_deletion(
    tmp_path, foreign_kind: str, teardown: str,
) -> None:
    """Hub ownership belongs to one provisioning root, not to a movable receipt."""
    foreign_root = tmp_path / "foreign-root"
    foreign_registry = foreign_root / "registry.json"
    foreign_root.mkdir()
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    document = pkg.blank_registry()
    document["root"] = str(foreign_root)
    document["packages"][package.id] = {
        "state": "ready",
        "environment": package.environment,
        "materialized": {
            "hub_revisions": [package.source["revision"]],
            "bytes": 0,
        },
    }
    foreign_registry.write_text(json.dumps(document), encoding="utf-8")
    target = paths.registry_path()
    target.parent.mkdir(parents=True)
    if foreign_kind == "copy":
        shutil.copyfile(foreign_registry, target)
    else:
        target.symlink_to(foreign_registry)

    class DeletionTripwire(FakeFetcher):
        def delete_hub_revisions(self, revisions: list[str]) -> tuple[list[str], int]:
            raise AssertionError(f"foreign registry authorized deletion of {revisions}")

    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(), fetcher=DeletionTripwire(tmp_path)
    )
    with pytest.raises(pkg.ProvisioningError) as caught:
        if teardown == "remove":
            provisioner.remove([package.id])
        else:
            provisioner.purge(dry_run=False)

    assert caught.value.code == "registry_unreadable"


@pytest.mark.parametrize("document", [[], {"schema_version": 1, "packages": []}, {
    "schema_version": 1, "packages": {"broken": []},
}])
def test_registry_container_shapes_are_validated(document) -> None:
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"


def test_registry_duplicate_keys_are_refused_before_last_wins_semantics(
    capsys,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_root = json.dumps(str(paths.root()))
    target.write_text(
        "{"
        '"schema_version":1,'
        '"tool_version":"test",'
        '"root":"attacker-controlled",'
        f'"root":{expected_root},'
        '"environments":{},'
        '"packages":{}'
        "}",
        encoding="utf-8",
    )

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"
    assert "duplicate JSON object key 'root'" in caught.value.message

    assert main(["packages", "list"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"


@pytest.mark.parametrize("command", ["list", "path", "verify"])
def test_cli_readers_refuse_a_nonfile_registry(command, capsys) -> None:
    paths.registry_path().mkdir(parents=True)

    assert main(["packages", command]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"
    assert "not a regular file" in error["detail"]


@pytest.mark.parametrize(
    "revisions_key", ["hub_revisions", "hub_revisions_pre_existing"]
)
def test_registry_rejects_nonstring_revision_ownership_receipts(
    revisions_key, capsys,
) -> None:
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    document = pkg.blank_registry()
    document["packages"][package.id] = {
        "state": "pulling",
        "materialized": {revisions_key: [package.source["revision"], {}]},
    }
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"

    assert main(["packages", "pull", "--repair", package.id]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"


def test_repair_refuses_a_malformed_top_level_retry_ownership_ledger(capsys) -> None:
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    document = pkg.blank_registry()
    document["packages"][package.id] = {
        "state": "pulling",
        "hub_revisions_pre_existing": [package.source["revision"], {}],
    }
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")

    assert main(["packages", "pull", "--repair", package.id]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"
    assert "hub_revisions_pre_existing" in error["detail"]


def test_registry_read_error_is_a_machine_readable_cli_failure(
    monkeypatch, capsys,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")
    original = pkg.os.open

    def unreadable(path, flags, mode=0o777, *, dir_fd=None):
        if path == target.name and dir_fd is not None:
            raise PermissionError("denied")
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(pkg.os, "open", unreadable)
    assert main(["packages", "verify"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"
    assert "Restore access" in error["fix"]


def test_registry_loader_never_follows_a_leaf_substituted_before_open(
    tmp_path, monkeypatch,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(pkg.blank_registry()), encoding="utf-8")
    victim = tmp_path / "external-registry.json"
    victim.write_text("external bytes must not be read\n", encoding="utf-8")
    original = pkg.os.open
    substituted = False

    def substitute(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal substituted
        if path == target.name and dir_fd is not None and not substituted:
            substituted = True
            target.unlink()
            target.symlink_to(victim)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(pkg.os, "open", substitute)

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()

    assert caught.value.code == "registry_unreadable"
    assert substituted is True
    assert target.is_symlink()
    assert victim.read_text(encoding="utf-8") == "external bytes must not be read\n"


@pytest.mark.parametrize("command", ["list", "path", "verify"])
def test_cli_readers_refuse_nonobject_materialized_registry_entry(
    command, capsys,
) -> None:
    document = pkg.blank_registry()
    document["packages"]["silero-vad"] = {
        "state": "ready", "materialized": "not-an-object",
    }
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")

    assert main(["packages", command]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"


@pytest.mark.parametrize("command", ["list", "path", "verify"])
def test_cli_readers_refuse_nonnumeric_materialized_bytes(command, capsys) -> None:
    document = pkg.blank_registry()
    document["packages"]["silero-vad"] = {
        "state": "ready", "materialized": {"bytes": "not-a-number"},
    }
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")

    assert main(["packages", command]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"


def test_cli_exit_codes(capsys, provisioner) -> None:
    assert main(["packages", "list"]) == 0
    assert json.loads(capsys.readouterr().out)["packages"] == []

    assert main(["packages", "path"]) == 0
    capsys.readouterr()

    # verify with nothing provisioned has nothing to fail on
    assert main(["packages", "verify"]) == 0
    capsys.readouterr()

    assert main(["packages", "remove", "firered-asr2s"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "package_not_provisioned"

    assert main(["packages", "pull", "--want", "diarization"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "stack_required"

    assert main(["doctor"]) == 0
    assert "environments" in json.loads(capsys.readouterr().out)


def test_verify_exits_three_when_a_check_fails(capsys, provisioner) -> None:
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    patched = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b") / "vibevoice" / "modular" / \
        "modeling_vibevoice_asr.py"
    patched.write_text("original\n")
    assert main(["packages", "verify"]) == 3
    codes = {item["code"] for item in json.loads(capsys.readouterr().out)["failed"]}
    assert "package_integrity_failed" in codes


def test_the_patch_ships_inside_the_package() -> None:
    """It is applied at pull time on a user's machine, so it cannot live in model_tests/."""
    patch = env.HERE / "patches" / "vibevoice-logits-to-keep.patch"
    assert patch.is_file()
    assert "modeling_vibevoice_asr.py" in patch.read_text()


def test_fluidaudio_patch_applies_to_exact_pinned_source_and_forces_offline_models(
    tmp_path: Path,
) -> None:
    """Exercise the real patch; the provisioning fake cannot prove Swift semantics."""
    repository = Path(__file__).resolve().parents[1]
    fixture = repository / "tests/fixtures/fluidaudio/ProcessCommand.swift"
    assert pkg.sha256_file(fixture) == (
        "2a90c1f8848b21a89a18361458ac3c58fc785fb110d93e706cc2af9b0f01dc41"
    ), "fixture drifted from FluidAudio 19600a485baa4998812e4654b70d2bab8f2c9949"
    target = (
        tmp_path / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
    )
    target.parent.mkdir(parents=True)
    shutil.copyfile(fixture, target)
    patch = env.HERE / "patches/fluidaudio-pinned-model-dir.patch"

    check = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True)

    assert pkg.sha256_file(target) == (
        "cba176cb6a612f347a8fcc72be2af9ce766246a781338a27e5e668c862f9e683"
    )
    patched = target.read_text(encoding="utf-8")
    model_argument = patched.index("args.modelDirectory")
    offline_gate = patched.index("ModelHub.offlineMode = true")
    model_load = patched.index("OfflineDiarizerModels.load(from: modelDir)")
    assert model_argument < offline_gate < model_load
    assert "OfflineDiarizerModels.defaultModelsDirectory()" not in patched


# --------------------------------------------------------------------------------------
# The shared Hugging Face cache
# --------------------------------------------------------------------------------------

ALIGNER_REVISION = "0e1a68e91d815300c7c9754b2a7639378b23db15"
FIRERED_REVISIONS = (
    "2304afed56eacfee6256dee5937ed22ffa0b64ec", "1bb4d285c8456429385d9c0810300df4297bc11b",
    "e448fd967f44182a1c323cc30f5d89f2400c28da", "7990aaccc6b7aec1e527743bd30201f2c4a03b8c",
)


def test_a_pre_existing_revision_is_never_deleted_by_teardown(tmp_path) -> None:
    """Weights live in a shared cache, so "materialized here" cannot mean "ours to delete".

    The failure this prevents: pull a package whose revision another tool already cached, then
    purge, and the other tool's weights are gone. It was real — a scratch root recorded a
    three-week-old revision as its own and offered it to `purge`.
    """
    fetcher = FakeFetcher(tmp_path, already_cached=(ALIGNER_REVISION,))
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    receipt = provisioner.pull(pkg.select(["qwen3-forcedaligner"]))

    assert receipt["pulled"][0]["hub_revisions_pre_existing"] == [ALIGNER_REVISION]
    assert receipt["pulled_known_bytes"] == 0
    materialized = pkg.load_registry()["packages"]["qwen3-forcedaligner"]["materialized"]
    assert materialized["hub_revisions"] == []
    assert materialized["hub_revisions_pre_existing"] == [ALIGNER_REVISION]

    dry = provisioner.purge(dry_run=True)
    assert dry["would_remove"]["hub_revisions"] == []
    assert dry["would_keep"]["hub_revisions"] == [ALIGNER_REVISION]
    assert dry["reclaimable_known_bytes"] == 0

    done = provisioner.purge(dry_run=False)
    assert done["hub_revisions_deleted"] == []
    assert done["hub_revisions_retained"] == [ALIGNER_REVISION]
    snapshot = (
        tmp_path / "hub" / "models--mlx-community--Qwen3-ForcedAligner-0.6B-8bit"
        / "snapshots" / ALIGNER_REVISION
    )
    assert snapshot.is_dir(), "purge deleted a revision it did not download"


def test_remove_keeps_pre_existing_revisions_and_deletes_its_own(tmp_path) -> None:
    """A multi-repo package where some revisions were cached and some were not."""
    fetcher = FakeFetcher(tmp_path, already_cached=FIRERED_REVISIONS[:2])
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    provisioner.pull(pkg.select(["firered-asr2s"]))

    report = provisioner.remove(["firered-asr2s"])
    assert set(report["hub_revisions_deleted"]) == set(FIRERED_REVISIONS[2:])
    assert set(report["hub_revisions_retained"]) == set(FIRERED_REVISIONS[:2])
    assert "not this root's to delete" in report["hub_revisions_retained_reason"]
    for revision in FIRERED_REVISIONS[:2]:
        assert list((tmp_path / "hub").glob(
            f"models--*/snapshots/{revision}"
        )), f"{revision} was deleted"


def test_a_multi_repo_receipt_names_every_revision_it_materialized(provisioner) -> None:
    """The receipt promises the revisions pulled, and this package spans four repositories."""
    receipt = provisioner.pull(pkg.select(["firered-asr2s"]))["pulled"][0]
    assert set(receipt["revisions"]) == set(FIRERED_REVISIONS)
    assert "revision" not in receipt, "a four-repo package cannot have one revision"


def test_vibevoice_materializes_only_the_pinned_tokenizer_files(provisioner) -> None:
    package = env.packages()["vibevoice-asr-7b"]
    repositories = package.source["repos"]
    tokenizer = next(item for item in repositories if item["role"] == "tokenizer")
    patterns = tuple(tokenizer["allow_patterns"])

    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    materialized = pkg.load_registry()["packages"]["vibevoice-asr-7b"]["materialized"]

    assert set(materialized["paths"]) == {
        "microsoft/VibeVoice-ASR",
        "Qwen/Qwen2.5-7B",
    }
    assert materialized["revisions"] == [
        VIBE_MODEL_REVISION,
        VIBE_TOKENIZER_REVISION,
    ]
    assert provisioner.fetcher.filtered == [(
        "Qwen/Qwen2.5-7B",
        VIBE_TOKENIZER_REVISION,
        patterns,
    )]
    tokenizer_path = Path(materialized["paths"]["Qwen/Qwen2.5-7B"])
    assert {path.name for path in tokenizer_path.iterdir()} == set(patterns)


def test_pull_refuses_an_unindexed_hub_return_before_traversing_it(
    tmp_path, monkeypatch,
) -> None:
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    external = tmp_path / "outside-cache" / package.source["revision"]
    external.mkdir(parents=True)

    class UnindexedFetcher(FakeFetcher):
        def hf_snapshot(self, repo, revision, *, force=False, allow_patterns=None):
            self.snapshots.append((repo, revision))
            return external

    original_tree_bytes = pkg._tree_bytes

    def refuse_external_traversal(path: Path) -> int:
        if Path(path) == external:
            raise AssertionError("pull traversed an unindexed downloader return")
        return original_tree_bytes(path)

    monkeypatch.setattr(pkg, "_tree_bytes", refuse_external_traversal)
    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(), fetcher=UnindexedFetcher(tmp_path)
    )

    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull([package])

    assert caught.value.code == "package_integrity_failed"
    assert "does not equal cache-indexed revision path" in caught.value.message
    entry = pkg.load_registry()["packages"][package.id]
    assert entry["state"] == "pulling"
    assert "materialized" not in entry


def test_multi_hub_pull_refuses_a_missing_allowlist_before_later_work(
    tmp_path,
) -> None:
    original = env.packages()["vibevoice-asr-7b"]
    tokenizer = next(
        repository
        for repository in original.source["repos"]
        if repository["role"] == "tokenizer"
    )
    asr = next(
        repository
        for repository in original.source["repos"]
        if repository["role"] == "asr"
    )
    package = replace(
        original,
        source={**original.source, "repos": [tokenizer, asr]},
    )

    class MissingTokenizerFile(FakeFetcher):
        def hf_snapshot(self, repo, revision, *, force=False, allow_patterns=None):
            snapshot = super().hf_snapshot(
                repo,
                revision,
                force=force,
                allow_patterns=allow_patterns,
            )
            if allow_patterns is not None:
                (snapshot / "tokenizer.json").unlink()
            return snapshot

    toolchain = FakeToolchain()
    fetcher = MissingTokenizerFile(tmp_path)
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=fetcher)

    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull([package])

    assert caught.value.code == "package_integrity_failed"
    assert "missing allow_pattern tokenizer.json" in caught.value.message
    assert fetcher.snapshots == [(tokenizer["repo"], tokenizer["revision"])]
    assert not any(call and call[0] == "install" for call in toolchain.calls)
    entry = pkg.load_registry()["packages"][package.id]
    assert entry["state"] == "pulling"
    assert "materialized" not in entry


def test_speaker_model_globs_require_nested_files_not_only_directories(
    provisioner,
) -> None:
    package = env.packages()["speaker-diarization-coreml"]
    patterns = tuple(package.source["allow_patterns"])

    provisioner.pull(pkg.select([package.id]))
    materialized = pkg.load_registry()["packages"][package.id]["materialized"]
    snapshot = Path(materialized["path"])

    assert provisioner.fetcher.filtered == [(
        package.source["repo"], package.source["revision"], patterns,
    )]
    assert pkg.hub_materialization_issues(package, materialized) == []
    nested = snapshot / "Segmentation.mlmodelc" / "model.mil"
    replaced_bytes = nested.stat().st_size
    nested.unlink()
    (snapshot / "same-size-filler.bin").write_bytes(b"x" * replaced_bytes)

    issues = pkg.hub_materialization_issues(package, materialized)
    assert "missing allow_pattern Segmentation.mlmodelc/**" in " ".join(issues)


def test_real_hub_fetcher_forwards_the_tokenizer_allowlist(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo: str, **kwargs):
        calls.update({"repo": repo, **kwargs})
        target = tmp_path / "snapshot"
        target.mkdir()
        return str(target)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    patterns = ("tokenizer.json", "tokenizer_config.json")
    found = pkg.Fetcher().hf_snapshot(
        "Qwen/Qwen2.5-7B",
        VIBE_TOKENIZER_REVISION,
        allow_patterns=patterns,
    )

    assert found == tmp_path / "snapshot"
    assert calls == {
        "repo": "Qwen/Qwen2.5-7B",
        "revision": VIBE_TOKENIZER_REVISION,
        "force_download": False,
        "allow_patterns": list(patterns),
    }


def test_verify_flags_weights_another_root_deleted(provisioner, tmp_path) -> None:
    """The residual shared-cache risk must fail loudly rather than at run time."""
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert provisioner.verify()["failed"] == []

    snapshot = Path(
        pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]["path"])
    shutil.rmtree(snapshot)

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert failure[0]["fix"] == "audio packages pull --repair qwen3-asr-1.7b-8bit"


def test_verify_flags_hub_snapshot_byte_drift(provisioner) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    materialized = pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"][
        "materialized"
    ]
    snapshot = Path(materialized["path"])
    next(path for path in snapshot.rglob("*") if path.is_file()).unlink()

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "snapshot bytes changed" in failure[0]["detail"]


@pytest.mark.parametrize("mutation", ["external_file", "snapshot_symlink"])
def test_verify_rejects_hub_snapshot_redirection(
    provisioner, tmp_path, mutation: str,
) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    materialized = pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"][
        "materialized"
    ]
    snapshot = Path(materialized["path"])
    if mutation == "external_file":
        model = next(path for path in snapshot.rglob("*") if path.is_file())
        external = tmp_path / "external-model.safetensors"
        external.write_bytes(model.read_bytes())
        model.unlink()
        model.symlink_to(external)
    else:
        external = tmp_path / "external-snapshot"
        snapshot.rename(external)
        snapshot.symlink_to(external, target_is_directory=True)

    failure = provisioner.verify()["failed"]

    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "non-symlink" in failure[0]["detail"] or "outside" in failure[0]["detail"]


def test_verify_flags_a_missing_allowlisted_tokenizer_file(provisioner) -> None:
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    materialized = pkg.load_registry()["packages"]["vibevoice-asr-7b"][
        "materialized"
    ]
    tokenizer = Path(materialized["paths"]["Qwen/Qwen2.5-7B"])
    (tokenizer / "tokenizer.json").unlink()

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "missing allow_pattern tokenizer.json" in failure[0]["detail"]


def test_verify_requires_allowlisted_tokenizer_matches_to_be_files(provisioner) -> None:
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    materialized = pkg.load_registry()["packages"]["vibevoice-asr-7b"][
        "materialized"
    ]
    tokenizer = Path(materialized["paths"]["Qwen/Qwen2.5-7B"])
    required = tokenizer / "tokenizer.json"
    replaced_bytes = required.stat().st_size
    required.unlink()
    required.mkdir()
    # Preserve the receipt's total tree size so only the file-kind check can catch this.
    (tokenizer / "same-size-filler.bin").write_bytes(b"x" * replaced_bytes)

    failure = provisioner.verify()["failed"]

    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "missing allow_pattern tokenizer.json" in failure[0]["detail"]


def test_verify_rejects_hub_path_outside_the_cache_index(
    provisioner, tmp_path,
) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    document = pkg.load_registry()
    materialized = document["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]
    snapshot = Path(materialized["path"])
    moved = tmp_path / "attacker-controlled" / snapshot.name
    shutil.copytree(snapshot, moved)
    materialized["path"] = str(moved)
    pkg.save_registry(document)

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "does not equal cache-indexed revision path" in failure[0]["detail"]


def test_pull_receipt_bytes_are_named_for_this_pull_not_the_total(provisioner) -> None:
    """The figure legitimately goes down between pulls, so it must not read as cumulative."""
    first = provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    second = provisioner.pull(pkg.select(["qwen3-asr-0.6b-8bit"]))
    # The old name invited reading a per-pull figure as a running total, and it is not one:
    # each receipt covers only its own packages, while `list` accumulates.
    assert "reclaimable_known_bytes" not in first
    assert first["pulled_known_bytes"] == env.packages()[
        "qwen3-asr-1.7b-8bit"
    ].bytes
    assert second["pulled_known_bytes"] == env.packages()[
        "qwen3-asr-0.6b-8bit"
    ].bytes
    document = pkg.load_registry()
    assert pkg.list_report()["total_known_bytes"] == sum(
        entry["materialized"]["bytes"]
        for entry in document["packages"].values()
    )


def test_path_says_where_weights_actually_live(provisioner) -> None:
    """A reader who assumes the root holds the weights concludes a 17 GiB pull did nothing."""
    report = pkg.path_report()
    assert "Hugging Face cache" in report["weights"]["location"]
    assert report["models"]["exists"] is False
    assert "silero-vad" in report["models"]["holds"]


def test_path_reports_single_and_multi_repo_materializations_without_null_aliases(
    provisioner,
) -> None:
    provisioner.pull(pkg.select([
        "qwen3-asr-1.7b-8bit", "firered-asr2s", "vibevoice-asr-7b",
    ]))
    document = pkg.load_registry()
    report = pkg.path_report()["packages"]

    single = report["qwen3-asr-1.7b-8bit"]
    single_materialized = document["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]
    assert single["location"] == single_materialized["path"]
    assert "locations" not in single
    assert "checkout" not in single

    for identifier in ("firered-asr2s", "vibevoice-asr-7b"):
        entry = report[identifier]
        materialized = document["packages"][identifier]["materialized"]
        assert entry["locations"] == dict(sorted(materialized["paths"].items()))
        assert entry["checkout"] == materialized["checkout"]
        assert "location" not in entry


def test_a_retry_does_not_disown_its_own_partial_download(tmp_path) -> None:
    """An interrupted pull leaves a partly-published snapshot; the retry must still own it.

    `snapshot_download` publishes files as they land, so a cache scan on the second attempt
    reports the revision as present. Re-deciding there would classify this root's own 16 GiB
    as somebody else's and teardown would refuse to reclaim it.
    """
    revision = "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"

    class InterruptedThenCachedFetcher(FakeFetcher):
        """Fails the first attempt, and afterwards reports the revision as cached."""

        def hf_snapshot(self, repo: str, revision_: str, *, force: bool = False) -> Path:
            path = super().hf_snapshot(repo, revision_, force=force)
            if not self.already_cached:
                self.already_cached = {revision_}   # the partial snapshot is now visible
                raise pkg.ProvisioningError("download_failed", "interrupted mid-download")
            return path

    fetcher = InterruptedThenCachedFetcher(tmp_path)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    selection = pkg.select(["qwen3-asr-1.7b-8bit"])

    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull(selection)
    assert pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"][
        "hub_revisions_pre_existing"] == []

    provisioner.pull(selection)
    materialized = pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]
    assert materialized["hub_revisions"] == [revision], (
        "the retry disowned the download the first attempt started"
    )
    assert materialized["hub_revisions_pre_existing"] == []
    assert provisioner.purge(dry_run=True)["would_remove"]["hub_revisions"] == [revision]


# --------------------------------------------------------------------------------------
# What `pull` and `verify` used to claim: a digest nobody took, a repair nobody wired, and
# work nobody needed
# --------------------------------------------------------------------------------------

QWEN_REPO = "mlx-community/Qwen3-ASR-1.7B-8bit"
QWEN_REVISION = "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"


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
    monkeypatch.setattr(pkg, "packages", lambda: catalog)
    return catalog


def test_verify_earns_the_word_digest_instead_of_borrowing_it(
    provisioner, the_fake_download_satisfies_the_pin
) -> None:
    """`digest: "ok"` used to mean "the path exists" for every package but one.

    `_materialize` set `digest_verified: True` on both Hub kinds, where the manifest pins a
    revision and carries no `sha256` to hash a snapshot against. So `verify` reported a digest
    check it never ran, and its `"unverified"` branch was unreachable for everything this code
    can pull. What a Hub package can honestly report is the revision.
    """
    provisioner.pull(pkg.select(["silero-vad", "qwen3-asr-1.7b-8bit", "firered-asr2s"]))

    for identifier in ("qwen3-asr-1.7b-8bit", "firered-asr2s"):
        materialized = pkg.load_registry()["packages"][identifier]["materialized"]
        assert "digest_verified" not in materialized, (
            f"{identifier} recorded a digest claim; nothing hashed it"
        )

    report = provisioner.verify()
    assert report["failed"] == []
    entries = {entry["package"]: entry for entry in report["verified"]}

    # Hashed against a manifest pin, which one package has.
    assert entries["silero-vad"] == {"package": "silero-vad", "digest": "ok"}
    # Revision pinned, contents not hashed — and told apart by which key is present.
    assert entries["qwen3-asr-1.7b-8bit"] == {
        "package": "qwen3-asr-1.7b-8bit", "revision": QWEN_REVISION}
    assert set(entries["firered-asr2s"]["revisions"]) == set(FIRERED_REVISIONS)
    assert "digest" not in entries["firered-asr2s"]

    # And the earned claim is still a measurement: break the bytes and it goes away.
    Path(pkg.load_registry()["packages"]["silero-vad"]["materialized"]["path"]).write_bytes(
        b"tampered")
    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]


def test_verify_rejects_a_hash_matching_url_artifact_outside_its_managed_path(
    provisioner, the_fake_download_satisfies_the_pin, tmp_path,
) -> None:
    provisioner.pull(pkg.select(["silero-vad"]))
    external = tmp_path / "external.onnx"
    external.write_bytes(b"onnx-bytes")
    document = pkg.load_registry()
    document["packages"]["silero-vad"]["materialized"]["path"] = str(external)
    pkg.save_registry(document)

    failure = provisioner.verify()["failed"]

    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "not managed path" in failure[0]["detail"]
    assert pkg.path_report()["packages"]["silero-vad"]["location"] == str(
        paths.models_dir() / env.packages()["silero-vad"].source["filename"]
    )


def test_verify_rejects_a_symlinked_models_parent_with_matching_url_bytes(
    provisioner, the_fake_download_satisfies_the_pin,
) -> None:
    provisioner.pull(pkg.select(["silero-vad"]))
    models = paths.models_dir()
    alternate = paths.root() / "alternate-models"
    models.rename(alternate)
    models.symlink_to(alternate, target_is_directory=True)

    failure = provisioner.verify()["failed"]

    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert f"managed artifact parent is a symlink: {models}" in failure[0]["detail"]


@pytest.mark.parametrize(
    ("identifier", "receipt_key", "expected"),
    [
        ("qwen3-asr-1.7b-8bit", "revision", [QWEN_REVISION]),
        ("firered-asr2s", "revisions", list(FIRERED_REVISIONS)),
    ],
)
def test_verify_reports_manifest_revisions_not_tampered_receipt_history(
    provisioner, identifier: str, receipt_key: str, expected: list[str],
) -> None:
    provisioner.pull(pkg.select([identifier]))
    document = pkg.load_registry()
    materialized = document["packages"][identifier]["materialized"]
    materialized[receipt_key] = "tampered" if receipt_key == "revision" else ["tampered"]
    pkg.save_registry(document)

    report = provisioner.verify()
    assert report["failed"] == []
    verified = next(item for item in report["verified"] if item["package"] == identifier)
    if receipt_key == "revision":
        assert verified["revision"] == expected[0]
    else:
        assert verified["revisions"] == expected


def test_verify_fails_a_ready_registry_package_missing_from_the_manifest() -> None:
    document = pkg.blank_registry()
    document["packages"]["retired-or-tampered"] = {
        "state": "ready", "environment": "core", "materialized": {},
    }
    pkg.save_registry(document)

    report = pkg.Provisioner(toolchain=FakeToolchain()).verify()
    assert report["verified"] == []
    assert report["failed"] == [{
        "package": "retired-or-tampered",
        "code": "package_unknown",
        "detail": "ready registry entry is not present in the installed manifest",
        "fix": f"Inspect {paths.registry_path()} and remove the stale entry",
    }]


def test_a_stale_digest_claim_in_the_registry_is_not_republished(provisioner) -> None:
    """A root provisioned before this fix carries the fabrication in `registry.json`.

    `verify` reads the registry, so forwarding `digest_verified` would let the claim survive the
    upgrade that removed it. The revision is what gets reported either way.
    """
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    document = pkg.load_registry()
    document["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]["digest_verified"] = True
    pkg.save_registry(document)

    entry = next(item for item in provisioner.verify()["verified"]
                 if item["package"] == "qwen3-asr-1.7b-8bit")
    assert entry == {"package": "qwen3-asr-1.7b-8bit", "revision": QWEN_REVISION}


def test_pull_skips_what_the_registry_already_calls_ready(provisioner) -> None:
    """Measured before this fix: a second pull of a ready package re-did all of the work."""
    first = provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert first["skipped"] == []
    assert provisioner.fetcher.snapshots == [(QWEN_REPO, QWEN_REVISION)]

    second = provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert second["pulled"] == []
    assert second["skipped"] == ["qwen3-asr-1.7b-8bit"]
    assert second["environments_created"] == []
    # Nothing was added, so nothing is claimed to have been.
    assert second["pulled_known_bytes"] == 0
    assert "--repair" in second["skipped_reason"]
    assert provisioner.fetcher.snapshots == [(QWEN_REPO, QWEN_REVISION)], (
        "the second pull re-materialized a package that was already ready"
    )


def test_a_ready_package_is_never_reopened_as_pulling(provisioner, monkeypatch) -> None:
    """The cost of a pointless re-pull is not the time. It is this.

    `pull` writes `state: "pulling"` before any bytes move, which is what makes a crashed pull
    read as absent. Re-running it over a healthy package therefore downgrades that package for as
    long as the work takes, and an interrupt leaves it downgraded.
    """
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))

    states: list[str | None] = []
    real_save = pkg.save_registry

    def spy(document: dict) -> None:
        states.append(document["packages"].get("qwen3-asr-1.7b-8bit", {}).get("state"))
        real_save(document)

    monkeypatch.setattr(pkg, "save_registry", spy)
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert states == [], (
        f"a no-op pull rewrote the registry, states {states}: a ready package must not be "
        "reopened as `pulling`"
    )


def test_repair_re_downloads_a_hub_snapshot_a_re_pull_would_keep(tmp_path) -> None:
    """`--repair` was declared, documented, named in four `fix` strings, and never read.

    Wiring it to re-run `_materialize` is not enough by itself: `snapshot_download` returns a
    revision the cache already holds as it stands, so without `force_download` a repair of a
    corrupt snapshot reports success having moved no bytes.
    """
    fetcher = FakeFetcher(tmp_path)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    weights = Path(
        pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]["path"]
    ) / "model.safetensors"
    weights.write_bytes(b"bit rot")

    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert weights.read_bytes() == b"bit rot", "a plain re-pull is not a repair"
    assert fetcher.forced == []

    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]), repair=True)
    assert fetcher.forced == [(QWEN_REPO, QWEN_REVISION)]
    assert weights.read_bytes() == FakeFetcher.WEIGHTS


def test_repair_replaces_a_checkout_rather_than_patching_what_is_there(provisioner) -> None:
    """A checkout integrity failure names `pull --repair`, so it has to actually repair."""
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    checkout = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
    patched = checkout / "vibevoice" / "modular" / "modeling_vibevoice_asr.py"
    patched.write_text("original\n")
    stray = checkout / "left-behind-by-a-half-finished-pull.txt"
    stray.write_text("x")
    assert [item["code"] for item in provisioner.verify()["failed"]] == [
        "package_integrity_failed"
    ]

    provisioner.pull(pkg.select(["vibevoice-asr-7b"]), repair=True)
    assert provisioner.verify()["failed"] == []
    assert not stray.exists(), "the checkout was patched in place rather than replaced"


@pytest.mark.parametrize("identifier", ["firered-asr2s", "vibevoice-asr-7b"])
@pytest.mark.parametrize(
    "mutation", ["missing", "head", "tracked", "untracked", "ignored"],
)
def test_verify_rejects_every_live_native_checkout_drift(
    provisioner, identifier: str, mutation: str,
) -> None:
    provisioner.pull(pkg.select([identifier]))
    document = pkg.load_registry()
    materialized = document["packages"][identifier]["materialized"]
    snapshots = [Path(path) for path in materialized["paths"].values()]
    checkout = Path(materialized["checkout"])

    if mutation == "missing":
        shutil.rmtree(checkout)
    elif mutation == "head":
        provisioner.toolchain._checkout_commits[checkout] = "0" * 40
    elif mutation == "tracked":
        (checkout / "pyproject.toml").write_text("tampered\n", encoding="utf-8")
    elif mutation == "untracked":
        (checkout / "rogue.py").write_text("rogue\n", encoding="utf-8")
    else:
        ignored = checkout / "__pycache__" / "rogue.cpython-312.pyc"
        ignored.parent.mkdir()
        ignored.write_bytes(b"importable")

    report = provisioner.verify()
    failure = [item for item in report["failed"] if item.get("package") == identifier]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert not [item for item in report["verified"] if item["package"] == identifier]
    if identifier == "firered-asr2s" and mutation == "missing":
        assert all(snapshot.is_dir() for snapshot in snapshots), (
            "the checkout regression accidentally removed the intact Hub evidence"
        )


def test_verify_accepts_a_legacy_short_checkout_receipt_only_when_live_head_is_exact(
    provisioner,
) -> None:
    provisioner.pull(pkg.select(["firered-asr2s"]))
    document = pkg.load_registry()
    materialized = document["packages"]["firered-asr2s"]["materialized"]
    materialized["checkout_commit"] = env.packages()["firered-asr2s"].checkout["commit"]
    pkg.save_registry(document)

    assert provisioner.verify()["failed"] == []
    checkout = Path(materialized["checkout"])
    provisioner.toolchain._checkout_commits[checkout] = materialized["checkout_commit"]
    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "expected exact commit" in failure[0]["detail"]


def test_verify_rejects_a_wrong_checkout_receipt_with_an_exact_live_head(
    provisioner,
) -> None:
    provisioner.pull(pkg.select(["firered-asr2s"]))
    document = pkg.load_registry()
    materialized = document["packages"]["firered-asr2s"]["materialized"]
    materialized["checkout_commit"] = "wrong"
    pkg.save_registry(document)

    report = provisioner.verify()
    failure = [
        item for item in report["failed"] if item.get("package") == "firered-asr2s"
    ]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "recorded checkout commit" in failure[0]["detail"]
    assert not [
        item for item in report["verified"] if item["package"] == "firered-asr2s"
    ]


def test_verify_hashes_live_patch_against_manifest_even_with_a_valid_receipt(
    provisioner,
) -> None:
    identifier = "vibevoice-asr-7b"
    package = env.packages()[identifier]
    provisioner.pull(pkg.select([identifier]))
    document = pkg.load_registry()
    materialized = document["packages"][identifier]["materialized"]
    checkout = Path(materialized["checkout"])
    _patches, names, _digests = pkg.checkout_patch_expectation(package)
    assert len(names) == 1
    patched = checkout / names[0]
    patched.write_text("attacker-controlled\n", encoding="utf-8")

    report = provisioner.verify()
    failure = [item for item in report["failed"] if item.get("package") == identifier]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert failure[0]["detail"] == (
        f"live patched-file hashes changed for {sorted(names)!r}"
    )
    assert not [item for item in report["verified"] if item["package"] == identifier]


def test_real_checkout_probe_includes_ordinary_and_ignored_untracked_files(tmp_path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    toolchain = pkg.Toolchain()
    assert toolchain.run(["git", "init", "--quiet"], cwd=checkout).returncode == 0
    (checkout / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    (checkout / "tracked.py").write_text("original\n", encoding="utf-8")
    assert toolchain.run(["git", "add", "."], cwd=checkout).returncode == 0
    assert toolchain.run([
        "git", "-c", "user.name=Audio Tests", "-c", "user.email=audio@example.invalid",
        "commit", "--quiet", "-m", "fixture",
    ], cwd=checkout).returncode == 0
    assert toolchain.run([
        "git", "config", "core.abbrev", "12",
    ], cwd=checkout).returncode == 0
    assert toolchain.run([
        "git", "config", "diff.noprefix", "true",
    ], cwd=checkout).returncode == 0
    assert toolchain.run([
        "git", "config", "diff.interHunkContext", "100",
    ], cwd=checkout).returncode == 0
    assert toolchain.run([
        "git", "config", "diff.suppressBlankEmpty", "true",
    ], cwd=checkout).returncode == 0

    clean = toolchain.inspect_checkout(checkout)
    assert len(clean.head) == 40
    assert clean.modified == ()
    assert clean.untracked == ()

    (checkout / "tracked.py").write_text("changed\n", encoding="utf-8")
    (checkout / "rogue.py").write_text("rogue\n", encoding="utf-8")
    (checkout / "ignored.pyc").write_bytes(b"importable")
    changed = toolchain.inspect_checkout(checkout)
    assert changed.modified == ("tracked.py",)
    assert changed.untracked == ("ignored.pyc", "rogue.py")


def test_repair_discards_the_swift_checkout_before_rebuilding(provisioner) -> None:
    """A rebuild in place trusts the tree whose state is what `--repair` was called about."""
    provisioner.pull(pkg.select(["fluidaudio"]))
    checkout = paths.checkout_dir("swift", "fluidaudio")
    stray = checkout / "half-applied.txt"
    stray.write_text("x")

    provisioner.pull(pkg.select(["fluidaudio"]), repair=True)
    assert not stray.exists()
    product = env.packages()["fluidaudio"].source["product"]
    assert list(checkout.glob(f".build/**/release/{product}"))


@pytest.mark.parametrize("mutation", ["checkout", "product"])
def test_verify_rejects_a_missing_fluidaudio_checkout_or_product(
    provisioner, mutation: str,
) -> None:
    provisioner.pull(pkg.select(["fluidaudio"]))
    materialized = pkg.load_registry()["packages"]["fluidaudio"]["materialized"]
    checkout = Path(materialized["path"])
    product = env.packages()["fluidaudio"].source["product"]
    executable = next(checkout.glob(f".build/**/release/{product}"))
    if mutation == "checkout":
        shutil.rmtree(checkout)
    else:
        executable.unlink()

    report = provisioner.verify()
    failure = [item for item in report["failed"] if item.get("package") == "fluidaudio"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert not [item for item in report["verified"] if item["package"] == "fluidaudio"]


def test_verify_never_launches_fluidaudio_from_an_external_receipt_path(
    provisioner, tmp_path,
) -> None:
    provisioner.pull(pkg.select(["fluidaudio"]))
    document = pkg.load_registry()
    external = tmp_path / "external-fluid"
    product = external / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    product.chmod(0o755)
    document["packages"]["fluidaudio"]["materialized"]["path"] = str(external)
    pkg.save_registry(document)

    report = provisioner.verify()
    failure = [item for item in report["failed"] if item.get("package") == "fluidaudio"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "not managed path" in failure[0]["detail"]


def test_verify_rejects_a_fluidaudio_product_symlink_escape(
    provisioner, tmp_path,
) -> None:
    provisioner.pull(pkg.select(["fluidaudio"]))
    package = env.packages()["fluidaudio"]
    checkout = paths.checkout_dir(package.environment, package.id)
    product = next(checkout.glob(".build/**/release/fluidaudiocli"))
    external = tmp_path / "external-fluid-product"
    external.write_bytes(product.read_bytes())
    external.chmod(0o755)
    product.unlink()
    product.symlink_to(external)

    report = provisioner.verify()
    failure = [item for item in report["failed"] if item.get("package") == "fluidaudio"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "expected one executable built product" in failure[0]["detail"]


def test_a_url_package_needs_no_forced_download_because_it_is_hash_pinned() -> None:
    """The one place `--repair` does nothing, and the reason it does not have to.

    `url_file` re-hashes what is on disk against the manifest pin and downloads again unless it
    matches, so a match is already the strongest re-materialization available. This runs the real
    fetcher, not the fake, because the claim is about that code path: a matching file returns
    without touching the network at all.
    """
    import urllib.request

    target = paths.models_dir() / "already-correct.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"onnx-bytes")

    def explode(*args, **kwargs):  # noqa: ANN002, ANN003 - a tripwire, never called
        raise AssertionError("url_file went to the network for a file that matches its pin")

    original = urllib.request.urlopen
    urllib.request.urlopen = explode
    try:
        resolved = pkg.Fetcher().url_file(
            "https://example.invalid/x.onnx", hashlib.sha256(b"onnx-bytes").hexdigest(), target)
    finally:
        urllib.request.urlopen = original
    assert resolved == target

    # And it is a check, not a shortcut: a file that does not match is re-fetched.
    target.write_bytes(b"corrupt")
    with pytest.raises(AssertionError, match="went to the network"):
        urllib.request.urlopen = explode
        try:
            pkg.Fetcher().url_file(
                "https://example.invalid/x.onnx",
                hashlib.sha256(b"onnx-bytes").hexdigest(), target)
        finally:
            urllib.request.urlopen = original


def test_url_download_temporary_never_follows_a_precreated_symlink(
    tmp_path, monkeypatch,
) -> None:
    target = paths.models_dir() / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"owned elsewhere")
    monkeypatch.setattr(
        "audio_cli.packages.uuid.uuid4", lambda: types.SimpleNamespace(hex="fixed")
    )
    temporary = target.with_name(f".audio-download-{os.getpid()}-fixed.part")
    temporary.symlink_to(victim)

    with pytest.raises(pkg.ProvisioningError) as raised:
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(b"model").hexdigest(),
            target,
        )

    assert raised.value.code == "download_failed"
    assert victim.read_bytes() == b"owned elsewhere"
    assert temporary.is_symlink()
    assert not target.exists()


def test_hash_matching_url_target_symlink_is_replaced_not_accepted(
    tmp_path, monkeypatch,
) -> None:
    payload = b"model bytes"
    target = paths.models_dir() / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim.onnx"
    victim.write_bytes(payload)
    target.symlink_to(victim)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        "audio_cli.packages.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    resolved = pkg.Fetcher().url_file(
        "https://example.invalid/model.onnx",
        hashlib.sha256(payload).hexdigest(),
        target,
    )

    assert resolved == target
    assert target.is_file() and not target.is_symlink()
    assert target.read_bytes() == payload
    assert victim.read_bytes() == payload


def test_url_download_rolls_back_a_private_temporary_substitution_at_publication(
    tmp_path, monkeypatch,
) -> None:
    payload = b"new model bytes"
    target = paths.models_dir() / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"prior managed cache")
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"owned elsewhere")
    monkeypatch.setattr(
        pkg.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed")
    )

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        pkg.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )
    real_exchange = media_module._rename_exchange
    substituted = False

    def substitute_at_exchange(directory_descriptor, left_name, right_name):
        nonlocal substituted
        if not substituted:
            substituted = True
            os.rename(
                left_name,
                "held-legitimate.part",
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            os.symlink(
                victim,
                left_name,
                dir_fd=directory_descriptor,
            )
        real_exchange(directory_descriptor, left_name, right_name)

    monkeypatch.setattr(media_module, "_rename_exchange", substitute_at_exchange)

    with pytest.raises(pkg.ProvisioningError) as raised:
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(payload).hexdigest(),
            target,
        )

    temporary = target.with_name(
        f".audio-download-{os.getpid()}-fixed.part"
    )
    assert raised.value.code == "download_failed"
    assert substituted is True
    assert target.read_bytes() == b"prior managed cache"
    assert victim.read_bytes() == b"owned elsewhere"
    assert temporary.is_symlink()
    assert (target.parent / "held-legitimate.part").read_bytes() == payload


def test_url_download_never_replaces_a_directory_leaf(
    monkeypatch,
) -> None:
    payload = b"model bytes"
    target = paths.models_dir() / "model.onnx"
    target.mkdir(parents=True)
    marker = target / "owned.txt"
    marker.write_bytes(b"preserve me")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        pkg.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    with pytest.raises(pkg.ProvisioningError) as raised:
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(payload).hexdigest(),
            target,
        )

    assert raised.value.code == "download_failed"
    assert target.is_dir()
    assert marker.read_bytes() == b"preserve me"


def test_url_download_replaces_an_unreadable_cache_entry(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"model bytes"
    target = paths.models_dir() / "model.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"unreadable")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        pkg,
        "sha256_regular_file_at",
        lambda *_args: (_ for _ in ()).throw(PermissionError("unreadable")),
    )
    monkeypatch.setattr(
        "audio_cli.packages.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    assert pkg.Fetcher().url_file(
        "https://example.invalid/model.onnx",
        hashlib.sha256(payload).hexdigest(),
        target,
    ) == target
    assert target.read_bytes() == payload


def test_url_download_cannot_follow_a_parent_swapped_after_open(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"model bytes"
    models = paths.models_dir()
    models.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "model.onnx"
    external.write_bytes(b"owned elsewhere")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def swap_parent():
        models.rename(paths.root() / "models-old")
        models.symlink_to(outside, target_is_directory=True)
        return types.SimpleNamespace(hex="fixed")

    monkeypatch.setattr("audio_cli.packages.uuid.uuid4", swap_parent)
    monkeypatch.setattr(
        "audio_cli.packages.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(payload),
    )

    with pytest.raises(pkg.ProvisioningError) as raised:
        pkg.Fetcher().url_file(
            "https://example.invalid/model.onnx",
            hashlib.sha256(payload).hexdigest(),
            models / "model.onnx",
        )

    assert raised.value.code == "download_failed"
    assert external.read_bytes() == b"owned elsewhere"


def test_a_stack_pull_provisions_around_a_missing_toolchain(tmp_path) -> None:
    """Measured before this fix: `pull --stack qwen-1.7b` with no `swift` left `ready` empty.

    `select` sorts by package id, `fluidaudio` sorts first, and `pull` raised on it — so a machine
    without a Swift toolchain got none of the ASR weights it could have had. §0 of
    TRANSCRIBE_HAPPY_PATH.md promises the opposite.
    """
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(missing=("swift",)),
                                  fetcher=FakeFetcher(tmp_path))
    receipt = provisioner.pull(pkg.select(stack="qwen-1.7b"), stack="qwen-1.7b")

    pulled = [entry["package"] for entry in receipt["pulled"]]
    assert "qwen3-asr-1.7b-8bit" in pulled and "qwen3-forcedaligner" in pulled
    assert "fluidaudio" not in pulled

    blocked = next(item for item in receipt["warnings"]
                   if item["code"] == "toolchain_missing")
    assert blocked["blocking"] is True
    assert blocked["packages"] == ["fluidaudio"]
    assert blocked["requires_tool"] == ["swift"]
    assert "qwen-1.7b" in blocked["detail"]

    document = pkg.load_registry()
    assert pkg.is_ready(document, "qwen3-asr-1.7b-8bit")
    assert "fluidaudio" not in document["packages"], "a blocked package left a registry entry"
    # A blocked package provisioned nothing, so it claims no license and no bytes.
    licenses = next(item for item in receipt["warnings"]
                    if item["code"] == "license_unreviewed")
    assert "fluidaudio" not in licenses["packages"]
    assert receipt["pulled_known_bytes"] > 0


def test_naming_a_toolchain_blocked_package_is_still_exit_three(tmp_path) -> None:
    """The asymmetry: a stack is a superset guess, a named package is an instruction.

    Named *beside a package that did provision*, so this cannot pass by way of the
    nothing-was-provisionable rule below — the refusal has to come from the naming.
    """
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(missing=("swift",)),
                                  fetcher=FakeFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit", "fluidaudio"]))
    assert caught.value.code == "toolchain_missing"
    assert caught.value.exit_code == 3
    assert caught.value.payload["package"] == "fluidaudio"
    assert caught.value.payload["requires_tool"] == ["swift"]
    # What it did provision before refusing stays provisioned, as after any interrupted pull.
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")

    # Order does not soften it either: blocked first is the case that used to abort a stack.
    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull(pkg.select(["fluidaudio", "qwen3-asr-0.6b-8bit"]))
    assert "qwen3-asr-0.6b-8bit" not in pkg.load_registry()["packages"]

    # And a stack in which nothing at all was provisionable is exit 3 too, because there is no
    # partial success to report. No shipped stack is one package wide, so the selection is
    # constructed rather than selected.
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["fluidaudio"]), stack="qwen-1.7b")
    assert caught.value.exit_code == 3


def test_teardown_never_deletes_a_sibling_of_the_models_directory(
    tmp_path, monkeypatch,
) -> None:
    """`str(models_dir) in path` is a substring test where a prefix test was meant.

    Point the Hub cache at `<root>/models_hub` — one plausible `HF_HOME` — and every snapshot
    under it tests positive, so `_locations` hands the shared cache to `_delete`. The revision
    below was in the cache before this root wanted it, which is exactly the case teardown
    promises to retain.
    """
    class HubBesideModels(FakeFetcher):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.hub = Path(str(paths.models_dir()) + "_hub")

    fetcher = HubBesideModels(tmp_path, already_cached=(QWEN_REVISION,))
    monkeypatch.setattr(pkg, "_hub_snapshot_index", lambda: _snapshot_index_for(fetcher.hub))
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit", "silero-vad"]))

    snapshot = Path(
        pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]["path"])
    artifact = Path(pkg.load_registry()["packages"]["silero-vad"]["materialized"]["path"])
    assert str(paths.models_dir()) in str(snapshot), "the fixture no longer sets the trap"

    report = provisioner.remove(["qwen3-asr-1.7b-8bit", "silero-vad"])
    assert report["hub_revisions_retained"] == [QWEN_REVISION]
    assert snapshot.is_dir(), "teardown deleted a directory inside the shared Hugging Face cache"
    # The true positive still holds, or the fix would be "delete nothing".
    assert not artifact.exists(), "the artifact this root wrote under models/ was not reclaimed"


def test_want_is_refused_rather_than_accepted_and_ignored(capsys, monkeypatch) -> None:
    """`--want` reached no code that could honour it. TRANSCRIBE_HAPPY_PATH.md §4.6 on why."""
    assert main(["packages", "pull", "--stack", "qwen-1.7b", "--want", "diarization"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "want_not_implemented"
    assert error["field"] == "--want"
    assert error["provided"] == "diarization"
    assert error["fix"] == "audio packages pull --stack qwen-1.7b"
    assert pkg.load_registry()["packages"] == {}, "the refused command provisioned something"

    # Without --stack it is still the older refusal, whose fix no longer suggests --want either.
    assert main(["packages", "pull", "--want", "diarization"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "stack_required"
    assert "--want" not in error["fix"]

    def provisioner_must_not_start():
        raise AssertionError("a refused --want reached provisioning")

    monkeypatch.setattr("audio_cli.cli.Provisioner", provisioner_must_not_start)
    assert main(["packages", "pull", "--stack", "qwen-1.7b", "--want", ""]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "want_not_implemented"
    assert error["provided"] == ""


def test_a_stack_beside_named_packages_is_a_conflict_not_a_precedence(capsys) -> None:
    """`select` returned early on package ids and dropped `--stack` silently."""
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.select(["silero-vad"], stack="qwen-1.7b")
    assert caught.value.code == "stack_conflicts_with_named_packages"
    assert caught.value.exit_code == 2
    assert caught.value.payload["stack"] == "qwen-1.7b"
    assert caught.value.payload["packages"] == ["silero-vad"]
    assert caught.value.payload["fix"] == "audio packages pull silero-vad"

    assert main(["packages", "pull", "--stack", "qwen-1.7b", "silero-vad"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == \
        "stack_conflicts_with_named_packages"


def test_the_cli_hands_repair_and_the_stack_through_to_pull(monkeypatch, capsys) -> None:
    """The wiring the flags were missing: `--repair` was parsed and read by nothing.

    `--stack` has to reach `pull` as well, because it is what decides whether a toolchain-blocked
    package is a warning or an exit 3.
    """
    seen: dict[str, object] = {}

    class Recorder:
        def pull(self, selection, *, repair: bool = False, stack: str | None = None) -> dict:
            seen.update(packages=[package.id for package in selection], repair=repair,
                        stack=stack)
            return {"pulled": [], "skipped": [], "warnings": []}

    monkeypatch.setattr("audio_cli.cli.Provisioner", Recorder)

    assert main(["packages", "pull", "--repair", "silero-vad"]) == 0
    capsys.readouterr()
    assert seen == {"packages": ["silero-vad"], "repair": True, "stack": None}

    assert main(["packages", "pull", "--stack", "firered"]) == 0
    capsys.readouterr()
    assert seen["stack"] == "firered"
    assert seen["repair"] is False
    assert seen["packages"] == [
        "firered-asr2s", "fluidaudio", "silero-vad", "speaker-diarization-coreml",
    ]


def test_verify_does_not_require_the_provisioning_tool_to_run_a_built_product(
    tmp_path, the_fake_download_satisfies_the_pin
) -> None:
    """Swift builds the binary; verification and transcription execute that binary directly."""
    toolchain = FakeToolchain()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["fluidaudio"]))
    toolchain.missing.add("swift")

    report = provisioner.verify()
    assert report["environments"]["swift"] == "ok"
    assert report["failed"] == []
    assert next(
        item for item in report["verified"] if item["package"] == "fluidaudio"
    )["product_runs"] is True
    assert any(call[0].endswith("fluidaudiocli") for call in toolchain.calls), (
        "verify trusted the receipt instead of launching the built executable"
    )
    assert pkg.doctor(toolchain=toolchain)["environments"]["swift"][
        "blocked_by_missing_tool"
    ] == []


def test_a_weight_only_swift_environment_stays_blocked_without_a_build_tool(tmp_path) -> None:
    """Without the built runtime, a missing provisioning tool still blocks repair."""
    absent_root = pkg.Provisioner(toolchain=FakeToolchain(missing=("swift",)),
                                  fetcher=FakeFetcher(tmp_path))
    assert absent_root.verify()["environments"]["swift"] == "absent"

    absent_root.pull(pkg.select(["speaker-diarization-coreml"]))
    assert absent_root.verify()["environments"]["swift"] == "blocked"


def test_vocabulary_names_every_environment_state_verify_can_emit() -> None:
    """VOCABULARY.md is the naming contract, so a new enum member cannot land unregistered.

    Derived from the source rather than listed by hand: a member added to `verify` without a
    line in the contract fails the first assertion, and one added to both without the backticked
    spelling fails the second.
    """
    import re

    emitted = set(re.findall(r'environment_states\[name\] = "(\w+)"',
                             Path(pkg.__file__).read_text()))
    assert emitted == {"absent", "ok", "drifted", "blocked"}, (
        f"verify emits environment states {sorted(emitted)}; register the new one in "
        "VOCABULARY.md and add it here"
    )
    vocabulary = (Path(__file__).resolve().parents[1] / "VOCABULARY.md").read_text()
    for state in sorted(emitted):
        assert f"`{state}`" in vocabulary, f"VOCABULARY.md does not name the {state!r} state"


def test_remove_validates_every_name_before_deleting_anything(provisioner) -> None:
    """One fat-fingered name used to cost a good package's bytes.

    The assertions that matter are on the filesystem. The registry is the half that looked
    correct — it rolled back with the raise, which is precisely how this hid: `is_ready` alone
    reports `True` both before and after the fix, and the bytes are gone in only one of them.
    """
    provisioner.pull(pkg.select(["silero-vad", "vibevoice-asr-7b"]))
    document = pkg.load_registry()
    artifact = Path(document["packages"]["silero-vad"]["materialized"]["path"])
    snapshots = [
        Path(value)
        for value in document["packages"]["vibevoice-asr-7b"]
        ["materialized"]["paths"].values()
    ]
    checkout = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
    assert artifact.is_file() and all(item.is_dir() for item in snapshots) \
        and checkout.is_dir()

    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.remove(["silero-vad", "vibevoice-asr-7b", "not-a-package"])
    assert caught.value.code == "package_not_provisioned"
    assert caught.value.exit_code == 2

    # All three kinds of location a package can hold, none of them touched.
    assert artifact.is_file(), "a pinned artifact was deleted before the list was validated"
    assert checkout.is_dir(), "a source checkout was deleted before the list was validated"
    assert all(item.is_dir() for item in snapshots), (
        "a Hub revision was deleted before the list was validated"
    )
    assert pkg.is_ready(pkg.load_registry(), "silero-vad")
    assert pkg.is_ready(pkg.load_registry(), "vibevoice-asr-7b")

    # And the refusal is not a disabled teardown: the same names without the typo still work.
    provisioner.remove(["silero-vad", "vibevoice-asr-7b"])
    assert not artifact.exists() and not checkout.exists() \
        and not any(item.exists() for item in snapshots)
    assert pkg.load_registry()["packages"] == {}


def test_remove_names_only_what_it_removed(provisioner) -> None:
    """`removed` is accumulated inside the loop, so it must not be able to name a survivor."""
    provisioner.pull(pkg.select(["silero-vad", "qwen3-asr-1.7b-8bit"]))

    with pytest.raises(pkg.ProvisioningError):
        provisioner.remove(["qwen3-asr-1.7b-8bit", "not-a-package"])
    assert sorted(pkg.load_registry()["packages"]) == ["qwen3-asr-1.7b-8bit", "silero-vad"]

    # A name given twice is removed once and reported once, rather than raising on the second
    # pass over an entry the first pass already dropped.
    report = provisioner.remove(["silero-vad", "silero-vad"])
    assert report["removed"] == ["silero-vad"]
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")


def test_remove_drops_unknown_registry_entry_without_following_its_paths(
    tmp_path,
) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    keep = victim / "keep.txt"
    keep.write_text("owned elsewhere", encoding="utf-8")
    foreign_revision = "f" * 40
    document = pkg.blank_registry()
    document["packages"]["retired-or-tampered"] = {
        "state": "ready",
        "environment": "../../victim",
        "materialized": {
            "checkout": str(victim),
            "path": str(paths.models_dir() / ".." / "victim"),
            "hub_revisions": [foreign_revision],
        },
    }
    document["environments"]["../../victim"] = {"state": "ready"}
    pkg.save_registry(document)

    report = pkg.Provisioner(
        toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)
    ).remove(["retired-or-tampered"])

    assert report["removed"] == ["retired-or-tampered"]
    assert report["hub_revisions_deleted"] == []
    assert report["hub_revisions_retained"] == [foreign_revision]
    assert report["hub_revisions_retained_reason"] == (
        "not owned by this root under the current package manifest, so they are not "
        "this root's to delete"
    )
    assert keep.read_text(encoding="utf-8") == "owned elsewhere"
    assert pkg.load_registry()["packages"] == {}


def test_remove_refuses_a_parent_symlink_escape_and_preserves_ownership(
    provisioner, tmp_path,
) -> None:
    provisioner.pull(pkg.select(["silero-vad"]))
    filename = env.packages()["silero-vad"].source["filename"]
    shutil.rmtree(paths.models_dir())
    external = tmp_path / "external-models"
    external.mkdir()
    victim = external / filename
    victim.write_bytes(b"owned elsewhere")
    paths.models_dir().symlink_to(external, target_is_directory=True)

    with pytest.raises(pkg.ProvisioningError) as raised:
        provisioner.remove(["silero-vad"])

    assert raised.value.code == "delete_refused"
    assert victim.read_bytes() == b"owned elsewhere"
    assert pkg.is_ready(pkg.load_registry(), "silero-vad")


def test_failed_deletion_reports_no_reclaim_and_keeps_registry_owner(
    provisioner, monkeypatch,
) -> None:
    provisioner.pull(pkg.select(["silero-vad"]))
    artifact = Path(
        pkg.load_registry()["packages"]["silero-vad"]["materialized"]["path"]
    )
    monkeypatch.setattr(pkg, "_delete_at", lambda _parent, _name: None)

    with pytest.raises(pkg.ProvisioningError) as raised:
        provisioner.remove(["silero-vad"])

    assert raised.value.code == "delete_failed"
    assert artifact.is_file()
    assert pkg.is_ready(pkg.load_registry(), "silero-vad")


def test_managed_delete_cannot_follow_a_parent_swapped_after_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    managed_parent = paths.root() / "envs"
    managed_target = managed_parent / "victim"
    managed_target.mkdir(parents=True)
    (managed_target / "managed.bin").write_bytes(b"managed")
    outside = tmp_path / "outside"
    outside_target = outside / "victim"
    outside_target.mkdir(parents=True)
    sentinel = outside_target / "KEEP"
    sentinel.write_bytes(b"owned elsewhere")
    original_measure = pkg._owned_tree_bytes_at
    swapped = False

    def swap_after_parent_open(parent_descriptor: int, name: str) -> int:
        nonlocal swapped
        if not swapped:
            swapped = True
            managed_parent.rename(paths.root() / "envs-old")
            managed_parent.symlink_to(outside, target_is_directory=True)
        return original_measure(parent_descriptor, name)

    monkeypatch.setattr(pkg, "_owned_tree_bytes_at", swap_after_parent_open)

    assert pkg._delete_managed(managed_target) == len(b"managed")
    assert sentinel.read_bytes() == b"owned elsewhere"
    assert not (paths.root() / "envs-old" / "victim").exists()


def test_environment_refcount_comes_from_manifest_not_mutable_entry(provisioner) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit", "qwen3-forcedaligner"]))
    document = pkg.load_registry()
    document["packages"]["qwen3-asr-1.7b-8bit"]["environment"] = "elsewhere"
    pkg.save_registry(document)

    report = provisioner.remove(["qwen3-forcedaligner"])

    assert "mlx" in report["environments_kept"]
    assert paths.env_dir("mlx").is_dir()
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")


def test_nonready_pull_replaces_dirty_checkout_before_install(tmp_path) -> None:
    checkout = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
    checkout.mkdir(parents=True)
    stale = checkout / "setup.py"
    stale.write_text("raise RuntimeError('executed stale hook')\n", encoding="utf-8")

    class InstallTripwire(FakeToolchain):
        def install_checkout(self, environment_python: Path, checkout_: Path) -> None:
            assert not (checkout_ / "setup.py").exists()
            super().install_checkout(environment_python, checkout_)

    provisioner = pkg.Provisioner(
        toolchain=InstallTripwire(), fetcher=FakeFetcher(tmp_path)
    )
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))

    assert provisioner.verify()["failed"] == []


def test_verify_rejects_patched_file_symlink_even_when_target_bytes_match(
    provisioner, tmp_path,
) -> None:
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    package = env.packages()["vibevoice-asr-7b"]
    _patches, names, _digests = pkg.checkout_patch_expectation(package)
    checkout = paths.checkout_dir(package.environment, package.id)
    patched = checkout / names[0]
    external = tmp_path / "matching.py"
    external.write_bytes(patched.read_bytes())
    patched.unlink()
    patched.symlink_to(external)

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "live patched-file hashes changed" in failure[0]["detail"]


def test_a_teardown_that_dies_leaves_no_package_reading_as_ready(
    tmp_path, monkeypatch,
) -> None:
    """The narrower half of the same defect, and the reason each entry is saved as it goes.

    A shared Hub cache can fail or vanish mid-teardown. With one write at the end, that failure
    rolled back the registry after every local file had already been deleted — every package
    named still `ready`, nothing behind any of them. This is the only way to reach that window
    now that unknown names are refused up front, so it is worth an injected failure.
    """
    class HostileCache(FakeFetcher):
        def delete_hub_revisions(self, revisions: list[str]) -> tuple[list[str], int]:
            raise OSError("the shared cache went away mid-teardown")

    for teardown in ("remove", "purge"):
        fetcher = HostileCache(tmp_path / teardown)
        monkeypatch.setattr(
            pkg, "_hub_snapshot_index", lambda fetcher=fetcher: _snapshot_index_for(fetcher.hub)
        )
        provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
        provisioner.pull(pkg.select(["silero-vad", "qwen3-asr-1.7b-8bit"]))
        artifact = Path(pkg.load_registry()["packages"]["silero-vad"]["materialized"]["path"])

        with pytest.raises(OSError):
            if teardown == "remove":
                provisioner.remove(["silero-vad", "qwen3-asr-1.7b-8bit"])
            else:
                provisioner.purge(dry_run=False)

        assert not artifact.exists(), f"{teardown} did not delete the local artifact"
        assert pkg.load_registry()["packages"] == {}, (
            f"{teardown} left a package reading as ready with its bytes already deleted"
        )


def test_verify_compares_the_private_api_hash_when_the_environment_answers(tmp_path) -> None:
    """The guard's *passing* verdict was unreachable: every test only ever observed `None`.

    A fake interpreter that never answers exercises the no-verdict path and nothing else, so a
    regression that stopped comparing — or compared against the wrong value — would have looked
    exactly like a clean run. §1.3 publishes `matches_expected: true`, so something has to be able
    to produce it. Found while scanning for the vacuous-flag shape; it is the same family.
    """
    expected = {guard["kind"]: guard
                for guard in env.environments()["mlx"].guards}["source_hash"]["sha256"]

    answering = pkg.Provisioner(toolchain=FakeToolchain(private_api_hash=expected),
                                fetcher=FakeFetcher(tmp_path))
    answering.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    report = answering.verify()
    assert report["mlx_audio_private_api_source_hash"] == expected
    assert report["mlx_audio_private_api_matches_expected"] is True
    assert report["mlx_audio_private_api_signature_ok"] is True
    assert "mlx_audio_private_api_error" not in report

    # And it is a comparison rather than an echo: a source file that moved fails it, with both
    # values published so a reader can see why.
    moved = pkg.Provisioner(toolchain=FakeToolchain(private_api_hash="0" * 64),
                            fetcher=FakeFetcher(tmp_path))
    report = moved.verify()
    assert report["mlx_audio_private_api_source_hash"] == "0" * 64
    assert report["mlx_audio_private_api_matches_expected"] is False
    assert report["mlx_audio_private_api_expected_source_hash"] == expected
    # Stated, not fixed: a moved private API is reported and does not reach `failed`, so `verify`
    # still exits 0. Whether it should is a contract decision, recorded in HANDOFF.md.
    assert report["failed"] == []


def test_repair_forces_every_repository_of_a_multi_repo_package(tmp_path) -> None:
    """A repair that forced only the first of four repositories would leave the rot in place.

    The single-repo test cannot see this: `huggingface` and `huggingface_multi` pass `force`
    through separate code, and a mutation scan showed nothing asserted the second one.
    """
    fetcher = FakeFetcher(tmp_path)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    provisioner.pull(pkg.select(["firered-asr2s"]))
    snapshots = pkg.load_registry()["packages"]["firered-asr2s"]["materialized"]["paths"]
    weights = [Path(location) / "model.safetensors" for location in snapshots.values()]
    assert len(weights) == 4
    for artifact in weights:
        artifact.write_bytes(b"bit rot")

    provisioner.pull(pkg.select(["firered-asr2s"]), repair=True)
    assert {revision for _, revision in fetcher.forced} == set(FIRERED_REVISIONS)
    assert all(artifact.read_bytes() == FakeFetcher.WEIGHTS for artifact in weights), (
        "a repair left at least one of the four repositories unfetched"
    )


def test_the_cli_hands_repair_and_dry_run_to_verify_and_purge(monkeypatch, capsys) -> None:
    """The rest of the flag wiring, for the same reason `--repair` on `pull` needed it.

    `--dry-run` is the one where a dropped argument is destructive: a caller asking what a purge
    would reclaim would have it reclaimed.
    """
    seen: dict[str, object] = {}

    class Recorder:
        def verify(self, *, repair: bool = False) -> dict:
            seen["verify_repair"] = repair
            return {"verified": [], "environments": {}, "failed": []}

        def purge(self, *, dry_run: bool) -> dict:
            seen["purge_dry_run"] = dry_run
            return {"removed": {}}

    monkeypatch.setattr("audio_cli.cli.Provisioner", Recorder)
    for argv, key, expected in (
        (["packages", "verify"], "verify_repair", False),
        (["packages", "verify", "--repair"], "verify_repair", True),
        (["packages", "purge"], "purge_dry_run", False),
        (["packages", "purge", "--dry-run"], "purge_dry_run", True),
    ):
        assert main(argv) == 0
        capsys.readouterr()
        assert seen[key] is expected, f"{' '.join(argv)} reached the provisioner as {seen[key]!r}"


def test_every_built_package_pins_the_product_it_has_to_launch() -> None:
    """The defect this catches: the executable's name living in a runner instead of the manifest.

    `swift_product_runs` ran `swift run ... fluidaudio` while Package.swift at the pinned commit
    declares `.executable(name: "fluidaudiocli")`, so the check answered "no executable product
    named 'fluidaudio'" on a perfectly good build and could never return True. Pinning the name
    beside the commit makes it reviewable when the commit moves.
    """
    built = {identifier: package for identifier, package in env.packages().items()
             if package.source["type"] == "git+build"}
    assert built, "no git+build package, so this invariant has nothing to hold"
    for identifier, package in built.items():
        assert package.source.get("product"), (
            f"{identifier} is built from source but names no product to launch"
        )


def test_the_pinned_product_name_is_the_one_launched(tmp_path) -> None:
    """A pinned name that the runner ignores would be decoration."""
    launched: list[str] = []

    class Recording(FakeToolchain):
        def swift_product_runs(self, checkout: Path, product: str) -> bool:
            launched.append(product)
            return True

    provisioner = pkg.Provisioner(toolchain=Recording(), fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["fluidaudio"]))
    assert launched == [env.packages()["fluidaudio"].source["product"]]


def test_a_build_whose_product_cannot_run_is_not_a_provisioned_package(tmp_path) -> None:
    """The defect this catches: `product_runs: false` recorded and then ignored.

    `pull` returned exit 0 with an empty `warnings`, `verify` reported `failed: []`, and the
    environment read `ok`, while the one thing the package exists for was impossible.
    """
    class Broken(FakeToolchain):
        def swift_product_runs(self, checkout: Path, product: str) -> bool:
            return False

    provisioner = pkg.Provisioner(toolchain=Broken(), fetcher=FakeFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["fluidaudio"]))
    assert caught.value.code == "package_build_unusable"
    assert caught.value.exit_code == 3
    assert caught.value.payload["product"] == "fluidaudiocli"
    assert caught.value.payload["fix"] == "audio packages pull --repair fluidaudio"
    # Fails closed: the entry never reaches `ready`, so `list` and `run` both call it absent.
    assert not pkg.is_ready(pkg.load_registry(), "fluidaudio")


def test_verify_fails_when_the_live_product_does_not_run_despite_a_true_receipt(tmp_path) -> None:
    """The registry's product_runs bit is history; the current executable is the check."""
    class StopsRunning(FakeToolchain):
        def built_product_runs(self, executable: Path) -> bool:
            return False

    provisioner = pkg.Provisioner(toolchain=StopsRunning(), fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["fluidaudio"]))

    document = pkg.load_registry()
    assert document["packages"]["fluidaudio"]["materialized"]["product_runs"] is True
    pkg.save_registry(document)

    report = provisioner.verify()
    failure = [entry for entry in report["failed"] if entry["package"] == "fluidaudio"]
    assert failure, "verify passed a package whose product does not run"
    assert failure[0]["code"] == "package_build_unusable"
    assert failure[0]["fix"] == "audio packages pull --repair fluidaudio"
    assert not [v for v in report["verified"] if v["package"] == "fluidaudio"], (
        "the same package was both verified and failed"
    )


def test_verify_accepts_a_live_product_despite_a_stale_false_receipt(provisioner) -> None:
    provisioner.pull(pkg.select(["fluidaudio"]))
    document = pkg.load_registry()
    document["packages"]["fluidaudio"]["materialized"]["product_runs"] = False
    pkg.save_registry(document)

    report = provisioner.verify()
    assert report["failed"] == []
    verified = next(item for item in report["verified"] if item["package"] == "fluidaudio")
    assert verified["product_runs"] is True


def test_nonlaunching_product_does_not_exempt_a_swiftless_environment(tmp_path) -> None:
    class StopsRunning(FakeToolchain):
        def built_product_runs(self, executable: Path) -> bool:
            return False

    toolchain = StopsRunning()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["fluidaudio"]))
    toolchain.missing.add("swift")

    report = provisioner.verify()
    assert report["environments"]["swift"] == "blocked"
    failure = [item for item in report["failed"] if item.get("package") == "fluidaudio"]
    assert [item["code"] for item in failure] == ["package_build_unusable"]


def test_the_real_toolchain_launches_the_product_it_was_given(tmp_path) -> None:
    """The assertion that would have caught the original defect, and the one above does not.

    A double that replaces `swift_product_runs` only proves `_materialize` passes the pinned name;
    the bug was inside the method, which ignored its surroundings and ran a literal. So this
    exercises the real body and overrides `run` alone.
    """
    commands: list[list[str]] = []

    class RecordingRun(pkg.Toolchain):
        def run(self, args, *, cwd=None, timeout=3600):
            commands.append(list(args))

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

    assert RecordingRun().swift_product_runs(tmp_path, "fluidaudiocli") is True
    assert commands == [["swift", "run", "-c", "release", "fluidaudiocli", "--help"]], (
        f"the product argument did not reach the command: {commands}"
    )


def test_the_real_verify_probe_executes_the_built_product_without_swift(tmp_path) -> None:
    commands: list[list[str]] = []

    class RecordingRun(pkg.Toolchain):
        def run(self, args, *, cwd=None, timeout=3600):
            commands.append(list(args))

            class Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return Result()

    executable = tmp_path / "fluidaudiocli"
    assert RecordingRun().built_product_runs(executable) is True
    assert commands == [[str(executable), "--help"]]
