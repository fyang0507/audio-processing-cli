"""Independent review reproductions: Git redirection, drift recovery and pin ownership."""

from __future__ import annotations

import subprocess

import pytest

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.packages import requirements
from audio_cli.packages.checkouts import install_verified_checkout
from tests.audio_cli.packages.package_build_test_support import offline_build as offline_build
from tests.audio_cli.packages.package_test_support import FakeFetcher, FakeToolchain
from tests.audio_cli.packages.package_test_support import isolated_root as isolated_root


def prepared_checkout(offline_build):
    provisioner, package = offline_build
    checkout = paths.checkout_dir(package.environment, package.id)
    checkout.parent.mkdir(parents=True)
    provisioner.toolchain.clone(package.checkout["repo"], package.checkout["commit"], checkout)
    receipt = {
        "checkout": str(checkout),
        "checkout_commit": package.checkout["commit"],
        "patches_applied": [],
        "patched_file_digests": {},
    }
    return package, checkout, receipt


@pytest.mark.parametrize("redirect", ["core.worktree", "gitfile", "git-symlink", "commondir"])
def test_cleanup_rejects_git_metadata_redirection_before_deleting_external_files(
    offline_build, tmp_path, redirect
):
    package, checkout, receipt = prepared_checkout(offline_build)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_text("unrelated data")

    class RedirectDuringInstall(pkg.Toolchain):
        cleanup_ran = False

        def run(self, args, **kwargs):
            self.cleanup_ran |= "clean" in args
            return super().run(args, **kwargs)

        def install_checkout(self, environment_python, source):
            if redirect == "core.worktree":
                subprocess.run(
                    ["git", "config", "core.worktree", str(external)],
                    cwd=source,
                    check=True,
                    capture_output=True,
                )
            elif redirect == "commondir":
                subprocess.run(["git", "init", "--quiet"], cwd=external, check=True)
                (source / ".git/commondir").write_text(str(external / ".git"))
            else:
                (source / ".git").rename(external / "metadata")
                if redirect == "gitfile":
                    (source / ".git").write_text(f"gitdir: {external / 'metadata'}\n")
                else:
                    (source / ".git").symlink_to(external / "metadata", target_is_directory=True)

    toolchain = RedirectDuringInstall()
    with pytest.raises(pkg.ProvisioningError) as caught:
        install_verified_checkout(package, receipt, toolchain)
    assert caught.value.code == "checkout_cleanup_failed"
    assert not toolchain.cleanup_ran
    assert sentinel.read_text() == "unrelated data"
    assert (checkout / "tracked_source.py").read_text() == "VALUE = 1\n"


def test_cleanup_command_cannot_be_retargeted_after_the_git_binding_probe(offline_build, tmp_path):
    package, checkout, receipt = prepared_checkout(offline_build)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_text("unrelated data")

    class RedirectAtClean(pkg.Toolchain):
        cleanup_ran = False

        def install_checkout(self, environment_python, source):
            (source / "build-residue.py").write_text("generated")

        def run(self, args, **kwargs):
            if "clean" in args:
                self.cleanup_ran = True
                subprocess.run(
                    ["git", "config", "core.worktree", str(external)],
                    cwd=checkout,
                    check=True,
                    capture_output=True,
                )
            return super().run(args, **kwargs)

    toolchain = RedirectAtClean()
    with pytest.raises(pkg.ProvisioningError) as caught:
        install_verified_checkout(package, receipt, toolchain)
    assert caught.value.code == "package_integrity_failed"
    assert toolchain.cleanup_ran
    assert sentinel.read_text() == "unrelated data"
    assert not (checkout / "build-residue.py").exists()


@pytest.mark.parametrize("dirty_checkout", [False, True])
def test_repaired_pull_reconciles_dependency_drift_before_forced_snapshots(
    tmp_path, dirty_checkout
):
    toolchain = FakeToolchain()
    fetcher = FakeFetcher(tmp_path)
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=fetcher)
    package = env.packages()["firered-asr2s"]
    provisioner.pull([package])
    locked = next(iter(requirements._locked_versions(env.environments()[package.environment])))
    toolchain._synced.discard(package.environment)
    toolchain._drift[locked] = "0.invalid"
    if dirty_checkout:
        (paths.checkout_dir(package.environment, package.id) / "setup.py").write_text("untrusted")
    original_snapshot = fetcher.hf_snapshot

    def snapshot_after_sync(*args, **kwargs):
        assert toolchain.created == [package.environment, package.environment]
        assert (
            toolchain.frozen_packages(paths.env_python(package.environment))[locked] != "0.invalid"
        )
        return original_snapshot(*args, **kwargs)

    fetcher.hf_snapshot = snapshot_after_sync
    report = provisioner.pull([package], repair=True)
    assert [receipt["package"] for receipt in report["pulled"]] == [package.id]
    assert len(fetcher.forced) == 4
    assert pkg.load_registry()["packages"][package.id]["state"] == "ready"
    assert provisioner.verify()["failed"] == []


def test_environment_drift_refusal_precedes_new_download_and_its_fix_recovers(tmp_path):
    toolchain = FakeToolchain()
    fetcher = FakeFetcher(tmp_path)
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=fetcher)
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    toolchain._synced.discard("mlx")
    toolchain._drift["unexpected-library"] = "0.invalid"
    selection = pkg.select(["qwen3-forcedaligner"])
    before = list(fetcher.snapshots)
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(selection)
    assert caught.value.payload["fix"] == (
        "run audio packages verify --repair, then audio packages pull --repair qwen3-forcedaligner"
    )
    assert fetcher.snapshots == before
    assert "qwen3-forcedaligner" not in pkg.load_registry()["packages"]
    assert provisioner.verify(repair=True)["failed"] == []
    assert provisioner.pull(selection, repair=True)["pulled"][0]["package"] == "qwen3-forcedaligner"
    assert provisioner.verify()["failed"] == []


def test_missing_direct_install_repair_does_not_claim_an_environment_was_recreated(tmp_path):
    toolchain = FakeToolchain()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    selection = pkg.select(["firered-asr2s"])
    provisioner.pull(selection)
    toolchain._direct_installs["torch-firered"] = {}
    report = provisioner.pull(selection, repair=True)
    assert report["environments_created"] == []
    assert toolchain.created == ["torch-firered"]
    assert provisioner.verify()["failed"] == []


@pytest.mark.parametrize("previous_type", ["huggingface", "huggingface_multi"])
@pytest.mark.parametrize("changed_identity", ["revision", "repository"])
def test_new_manifest_pin_uses_pre_download_cache_ownership_and_survives_teardown(
    tmp_path, previous_type, changed_identity
):
    package = env.packages()["qwen3-forcedaligner"]
    revision = package.source["revision"]
    old = {
        **package.source,
        **({"revision": "a" * 40} if changed_identity == "revision" else {"repo": "previous/repo"}),
    }
    previous_source = (
        old if previous_type == "huggingface" else {"type": "huggingface_multi", "repos": [old]}
    )
    fetcher = FakeFetcher(tmp_path, already_cached=[revision])
    snapshot = fetcher.hf_snapshot(package.source["repo"], revision)
    document = pkg.load_registry()
    document["packages"][package.id] = {
        "state": "ready",
        "source": previous_source,
        "hub_revisions_pre_existing": [],
    }
    pkg.save_registry(document)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    report = provisioner.pull([package], repair=True)
    materialized = pkg.load_registry()["packages"][package.id]["materialized"]
    assert materialized["hub_revisions"] == []
    assert materialized["hub_revisions_pre_existing"] == [revision]
    assert report["pulled_known_bytes"] == 0
    assert "listed 1 of 1" in report["pulled"][0]["pre_existing_note"]
    assert provisioner.purge(dry_run=True)["would_remove"]["hub_revisions"] == []
    assert provisioner.remove([package.id])["hub_revisions_deleted"] == []
    assert snapshot.is_dir()


def test_manifest_transition_retains_unchanged_owned_revision_but_protects_new_shared_pins(
    tmp_path,
):
    package = env.packages()["firered-asr2s"]
    repositories = package.source["repos"]
    revisions = [repository["revision"] for repository in repositories]
    old_repositories = [
        repository if index == 0 else {**repository, "revision": str(index) * 40}
        for index, repository in enumerate(repositories)
    ]
    fetcher = FakeFetcher(tmp_path, already_cached=revisions)
    snapshots = [
        fetcher.hf_snapshot(repository["repo"], repository["revision"])
        for repository in repositories
    ]
    document = pkg.load_registry()
    document["packages"][package.id] = {
        "state": "pulling",
        "source": {**package.source, "repos": old_repositories},
        "hub_revisions_pre_existing": [],
    }
    pkg.save_registry(document)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    report = provisioner.pull([package], repair=True)
    materialized = pkg.load_registry()["packages"][package.id]["materialized"]
    assert materialized["hub_revisions"] == [revisions[0]]
    assert materialized["hub_revisions_pre_existing"] == sorted(revisions[1:])
    assert report["pulled_known_bytes"] == repositories[0]["bytes"]
    removed = provisioner.remove([package.id])
    assert removed["hub_revisions_deleted"] == [revisions[0]]
    assert not snapshots[0].exists()
    assert all(snapshot.is_dir() for snapshot in snapshots[1:])
