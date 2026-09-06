"""FluidAudio repair and verification at managed-checkout boundaries."""

from __future__ import annotations

from tests.audio_cli.packages.package_test_support import Path, env, paths, pkg, pytest, shutil
from tests.audio_cli.packages.package_test_support import isolated_root as isolated_root
from tests.audio_cli.packages.package_test_support import provisioner as provisioner


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
    provisioner,
    mutation: str,
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
    provisioner,
    tmp_path,
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
    provisioner,
    tmp_path,
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
