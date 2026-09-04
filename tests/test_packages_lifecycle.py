"""Core package lifecycle and registry-publication behavior."""

from __future__ import annotations

from package_test_support import (
    VIBE_MODEL_REVISION,
    VIBE_TOKENIZER_REVISION,
    FakeFetcher,
    FakeToolchain,
    Path,
    env,
    json,
    os,
    package_registry,
    paths,
    pkg,
    pytest,
    types,
)
from package_test_support import (
    isolated_root as isolated_root,
)
from package_test_support import (
    provisioner as provisioner,
)

from audio_cli.media import publication as media_publication


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
        package_registry.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed")
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
        package_registry.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed")
    )
    temporary = target.with_name(f".audio-registry-{os.getpid()}-fixed.tmp")
    real_assert = package_registry.assert_directory_binding

    def substitute_temporary(descriptor: int, directory: Path) -> None:
        real_assert(descriptor, directory)
        temporary.unlink()
        temporary.symlink_to(victim)

    monkeypatch.setattr(
        package_registry, "assert_directory_binding", substitute_temporary
    )

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
        package_registry.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed")
    )
    temporary = target.with_name(f".audio-registry-{os.getpid()}-fixed.tmp")
    real_exchange = media_publication._rename_exchange
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

    monkeypatch.setattr(media_publication, "_rename_exchange", substitute_at_exchange)

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

    monkeypatch.setattr(package_registry.uuid, "uuid4", swap_parent)
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
