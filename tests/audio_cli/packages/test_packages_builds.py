"""Real offline build residue, failure handling, and reachable post-repair verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.packages.checkouts import install_verified_checkout
from tests.audio_cli.packages.package_build_test_support import offline_build as offline_build
from tests.audio_cli.packages.package_test_support import FakeFetcher, FakeToolchain
from tests.audio_cli.packages.package_test_support import isolated_root as isolated_root


def assert_build_ran_cleanly(provisioner, package):
    toolchain = provisioner.toolchain
    assert toolchain.build_residue
    for residue in toolchain.build_residue:
        assert "build/lib/fireredasr2s/__init__.py" in residue
        assert "fireredasr2s.egg-info/PKG-INFO" in residue
        assert any(path.startswith("__pycache__/") for path in residue)
    checkout = paths.checkout_dir(package.environment, package.id)
    assert toolchain.inspect_checkout(checkout).untracked == ()
    assert provisioner.verify()["failed"] == []
    assert pkg.load_registry()["packages"][package.id]["state"] == "ready"
    installed = toolchain._direct_installs[package.environment][package.checkout["distribution"]]
    assert installed == f"@ {checkout.as_uri()}"


def test_firered_real_build_then_repair_removes_ordinary_and_ignored_residue(offline_build):
    provisioner, package = offline_build
    receipt = provisioner.pull([package])
    assert receipt["pulled"][0]["package"] == package.id
    assert_build_ran_cleanly(provisioner, package)
    checkout = paths.checkout_dir(package.environment, package.id)
    # Recreate the recorded legacy state. Verification must still refuse it.
    residue = checkout / "build/lib/fireredasr2s/__init__.py"
    residue.parent.mkdir(parents=True)
    residue.write_text("legacy build residue")
    assert provisioner.verify()["failed"][0]["code"] == "package_integrity_failed"
    provisioner.pull([package], repair=True)
    assert len(provisioner.toolchain.build_residue) == 2
    assert_build_ran_cleanly(provisioner, package)


def test_environment_repair_with_a_real_build_restores_provenance_and_clean_checkout(offline_build):
    provisioner, package = offline_build
    provisioner.pull([package])
    toolchain = provisioner.toolchain
    toolchain._direct_installs[package.environment] = {}
    assert provisioner.verify()["environments"][package.environment] == "drifted"
    report = provisioner.verify(repair=True)
    assert report["failed"] == []
    assert toolchain.created == [package.environment, package.environment]
    assert len(toolchain.build_residue) == 2
    assert_build_ran_cleanly(provisioner, package)


@pytest.mark.parametrize("failure", ["fail", "interrupt"])
def test_failed_and_interrupted_builds_stay_nonready_and_retry_converges(
    offline_build, monkeypatch, failure
):
    provisioner, package = offline_build
    toolchain = provisioner.toolchain
    monkeypatch.setenv("AUDIO_TEST_BUILD_MODE", failure)
    toolchain.interrupt = failure == "interrupt"
    with pytest.raises(KeyboardInterrupt if toolchain.interrupt else pkg.ProvisioningError):
        provisioner.pull([package])
    entry = pkg.load_registry()["packages"][package.id]
    assert entry["state"] == "pulling"
    assert "materialized" not in entry
    checkout = paths.checkout_dir(package.environment, package.id)
    assert toolchain.inspect_checkout(checkout).untracked == ()
    assert any(
        failure["code"] == "package_not_ready" and failure.get("package") == package.id
        for failure in provisioner.verify()["failed"]
    )
    toolchain.interrupt = False
    monkeypatch.setenv("AUDIO_TEST_BUILD_MODE", "ok")
    provisioner.pull([package], repair=True)
    assert_build_ran_cleanly(provisioner, package)


@pytest.mark.parametrize("failure", ["residue", "tracked"])
def test_successful_installer_cannot_publish_a_dirty_checkout_as_ready(
    offline_build, monkeypatch, failure
):
    provisioner, package = offline_build
    provisioner.toolchain.skip_cleanup = failure == "residue"
    monkeypatch.setenv("AUDIO_TEST_BUILD_MODE", failure)
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull([package])
    assert caught.value.code == "package_integrity_failed"
    assert caught.value.payload["fix"] == f"audio packages pull --repair {package.id}"
    assert pkg.load_registry()["packages"][package.id]["state"] == "pulling"
    assert (
        "untracked" in caught.value.message
        if failure == "residue"
        else "tracked" in caught.value.message
    )


def test_existing_untracked_hook_is_neither_executed_nor_cleaned(offline_build):
    provisioner, package = offline_build
    provisioner.pull([package])
    materialized = pkg.load_registry()["packages"][package.id]["materialized"]
    checkout = Path(materialized["checkout"])
    hook = checkout / "setup.py"
    hook.write_text("untrusted hook")
    before = len(provisioner.toolchain.build_residue)
    with pytest.raises(pkg.ProvisioningError) as caught:
        install_verified_checkout(package, materialized, provisioner.toolchain)
    assert caught.value.code == "checkout_integrity_failed"
    assert len(provisioner.toolchain.build_residue) == before
    assert hook.read_text() == "untrusted hook"


def test_incomplete_environment_repair_does_not_restore_ready(tmp_path):
    class BrokenReinstall(FakeToolchain):
        broken = False

        def install_checkout(self, environment_python, checkout):
            if not self.broken:
                super().install_checkout(environment_python, checkout)

    toolchain = BrokenReinstall()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["firered-asr2s"]))
    toolchain.broken = True
    toolchain._direct_installs["torch-firered"] = {}
    report = provisioner.verify(repair=True)
    assert report["environments"]["torch-firered"] == "drifted"
    assert report["failed"][0]["code"] == "environment_drifted"
    assert pkg.load_registry()["environments"]["torch-firered"]["state"] == "creating"
    assert provisioner.verify()["environments"]["torch-firered"] == "absent"


def test_pull_does_not_promote_a_missing_checkout_distribution(tmp_path):
    class MissingInstall(FakeToolchain):
        def install_checkout(self, environment_python, checkout):
            pass

    provisioner = pkg.Provisioner(toolchain=MissingInstall(), fetcher=FakeFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["firered-asr2s"]))
    assert caught.value.code == "environment_drifted"
    assert "fireredasr2s" in caught.value.payload["examples"]
    assert pkg.load_registry()["packages"]["firered-asr2s"]["state"] == "pulling"


def test_unsuccessful_sync_does_not_announce_environment_ready(tmp_path):
    class BrokenSync(FakeToolchain):
        def frozen_packages(self, environment_python):
            return {}

    provisioner = pkg.Provisioner(toolchain=BrokenSync(), fetcher=FakeFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["firered-asr2s"]))
    assert caught.value.code == "environment_drifted"
    assert pkg.load_registry()["environments"]["torch-firered"]["state"] == "creating"
    assert "firered-asr2s" not in pkg.load_registry()["packages"]


def test_cleanup_failure_has_a_reachable_repair_and_cannot_promote_ready(tmp_path):
    class FailedCleanup(FakeToolchain):
        broken = True

        def clean_checkout_install_artifacts(self, checkout):
            if self.broken:
                raise pkg.ProvisioningError("checkout_cleanup_failed", "synthetic cleanup failure")
            super().clean_checkout_install_artifacts(checkout)

    toolchain = FailedCleanup()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["firered-asr2s"]))
    assert caught.value.code == "checkout_cleanup_failed"
    assert caught.value.payload == {
        "package": "firered-asr2s",
        "fix": "audio packages pull --repair firered-asr2s",
    }
    assert pkg.load_registry()["packages"]["firered-asr2s"]["state"] == "pulling"
    toolchain.broken = False
    provisioner.pull(pkg.select(["firered-asr2s"]), repair=True)
    assert provisioner.verify()["failed"] == []


def test_cleanup_refuses_redirected_checkout_without_touching_external_files(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep"
    sentinel.write_text("external")

    class RedirectingInstall(FakeToolchain):
        def install_checkout(self, environment_python, checkout):
            checkout.rename(checkout.with_name("relocated"))
            checkout.symlink_to(outside, target_is_directory=True)

        def clean_checkout_install_artifacts(self, checkout):
            raise AssertionError("cleanup followed a redirected checkout")

    provisioner = pkg.Provisioner(toolchain=RedirectingInstall(), fetcher=FakeFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["firered-asr2s"]))
    assert caught.value.code == "checkout_cleanup_failed"
    assert sentinel.read_text() == "external"
    assert pkg.load_registry()["packages"]["firered-asr2s"]["state"] == "pulling"
