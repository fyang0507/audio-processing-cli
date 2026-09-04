"""Package verification claims and native-checkout repair behavior."""

from __future__ import annotations

# ruff: noqa: F403, F405
from package_test_support import *

# --------------------------------------------------------------------------------------
# What `pull` and `verify` used to claim: a digest nobody took, a repair nobody wired, and
# work nobody needed
# --------------------------------------------------------------------------------------
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
