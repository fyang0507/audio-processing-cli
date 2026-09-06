"""Cleanup must retain its directory when a pathname is replaced at command launch."""

from __future__ import annotations

import os

import pytest

from audio_cli import packages as pkg
from audio_cli.packages.checkouts import install_verified_checkout
from tests.audio_cli.packages.package_build_test_support import offline_build as offline_build
from tests.audio_cli.packages.package_test_support import isolated_root as isolated_root
from tests.audio_cli.packages.test_packages_review_regressions import prepared_checkout


@pytest.mark.parametrize("swap_parent", [False, True])
@pytest.mark.parametrize("failed_install", [False, True])
def test_cleanup_launch_cannot_follow_a_replaced_checkout_path(
    offline_build, tmp_path, swap_parent, failed_install
):
    package, checkout, receipt = prepared_checkout(offline_build)
    external_anchor = tmp_path / "external"
    if swap_parent:
        external_anchor.mkdir()
    external_checkout = external_anchor / checkout.name if swap_parent else external_anchor
    pkg.Toolchain().clone(package.checkout["repo"], package.checkout["commit"], external_checkout)
    sentinel = external_checkout / "ordinary-keep.txt"
    sentinel.write_text("owned outside the provisioning root")
    original_anchor = checkout.parent if swap_parent else checkout
    moved_anchor = tmp_path / "moved-original"
    moved_checkout = moved_anchor / checkout.name if swap_parent else moved_anchor

    class SwapImmediatelyBeforeLaunch(pkg.Toolchain):
        cleanup_calls = 0
        cleanup_result = None

        def install_checkout(self, environment_python, source):
            (source / "build-residue.py").write_text("generated ordinary residue")
            if failed_install:
                raise pkg.ProvisioningError(
                    "checkout_install_failed", "synthetic installer failure"
                )

        def run(self, args, *, cwd=None, timeout=3600):
            if "clean" in args:
                self.cleanup_calls += 1
                assert isinstance(cwd, int), "destructive command lost its held directory"
                assert os.path.samestat(os.fstat(cwd), checkout.stat())
                original_anchor.rename(moved_anchor)
                original_anchor.symlink_to(external_anchor, target_is_directory=True)
                self.cleanup_result = super().run(args, cwd=cwd, timeout=timeout)
                return self.cleanup_result
            return super().run(args, cwd=cwd, timeout=timeout)

    toolchain = SwapImmediatelyBeforeLaunch()
    with pytest.raises(pkg.ProvisioningError) as caught:
        install_verified_checkout(package, receipt, toolchain)
    assert caught.value.code == "checkout_cleanup_failed"
    assert toolchain.cleanup_calls == 1, "legacy Toolchain.run injection was bypassed"
    assert toolchain.cleanup_result.returncode == 0
    assert sentinel.is_file(), "cleanup deleted the ordinary external sentinel"
    assert sentinel.read_text() == "owned outside the provisioning root"
    assert "Removing build-residue.py" in toolchain.cleanup_result.stdout
    assert not (moved_checkout / "build-residue.py").exists()
    assert (moved_checkout / "tracked_source.py").read_text() == "VALUE = 1\n"
