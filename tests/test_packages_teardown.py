"""Teardown safety, CLI wiring, and environment reference behavior."""

from __future__ import annotations

from package_test_support import (
    QWEN_REVISION,
    FakeFetcher,
    FakeToolchain,
    Path,
    _snapshot_index_for,
    env,
    json,
    main,
    package_environment_verification,
    package_integrity,
    package_teardown,
    paths,
    pkg,
    pytest,
    shutil,
)
from package_test_support import (
    isolated_root as isolated_root,
)
from package_test_support import (
    provisioner as provisioner,
)
from package_test_support import (
    the_fake_download_satisfies_the_pin as the_fake_download_satisfies_the_pin,
)
from spec_document_loader import read_spec_document


def test_teardown_never_deletes_a_sibling_of_the_models_directory(
    tmp_path,
    monkeypatch,
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
    monkeypatch.setattr(
        package_integrity, "_hub_snapshot_index", lambda: _snapshot_index_for(fetcher.hub)
    )
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit", "silero-vad"]))

    snapshot = Path(pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]["path"])
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
    assert (
        json.loads(capsys.readouterr().err)["error"]["code"]
        == "stack_conflicts_with_named_packages"
    )


def test_the_cli_hands_repair_and_the_stack_through_to_pull(monkeypatch, capsys) -> None:
    """The wiring the flags were missing: `--repair` was parsed and read by nothing.

    `--stack` has to reach `pull` as well, because it is what decides whether a toolchain-blocked
    package is a warning or an exit 3.
    """
    seen: dict[str, object] = {}

    class Recorder:
        def pull(self, selection, *, repair: bool = False, stack: str | None = None) -> dict:
            seen.update(packages=[package.id for package in selection], repair=repair, stack=stack)
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
        "firered-asr2s",
        "fluidaudio",
        "silero-vad",
        "speaker-diarization-coreml",
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
    assert (
        next(item for item in report["verified"] if item["package"] == "fluidaudio")["product_runs"]
        is True
    )
    assert any(call[0].endswith("fluidaudiocli") for call in toolchain.calls), (
        "verify trusted the receipt instead of launching the built executable"
    )
    assert pkg.doctor(toolchain=toolchain)["environments"]["swift"]["blocked_by_missing_tool"] == []


def test_a_weight_only_swift_environment_stays_blocked_without_a_build_tool(tmp_path) -> None:
    """Without the built runtime, a missing provisioning tool still blocks repair."""
    absent_root = pkg.Provisioner(
        toolchain=FakeToolchain(missing=("swift",)), fetcher=FakeFetcher(tmp_path)
    )
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

    emitted = set(
        re.findall(
            r'environment_states\[name\] = "(\w+)"',
            Path(package_environment_verification.__file__).read_text(),
        )
    )
    assert emitted == {"absent", "ok", "drifted", "blocked"}, (
        f"verify emits environment states {sorted(emitted)}; register the new one in "
        "VOCABULARY.md and add it here"
    )
    vocabulary = read_spec_document(Path(__file__).resolve().parents[1] / "VOCABULARY.md")
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
        for value in document["packages"]["vibevoice-asr-7b"]["materialized"]["paths"].values()
    ]
    checkout = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
    assert artifact.is_file() and all(item.is_dir() for item in snapshots) and checkout.is_dir()

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
    assert (
        not artifact.exists()
        and not checkout.exists()
        and not any(item.exists() for item in snapshots)
    )
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

    report = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)).remove(
        ["retired-or-tampered"]
    )

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
    provisioner,
    tmp_path,
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
    provisioner,
    monkeypatch,
) -> None:
    provisioner.pull(pkg.select(["silero-vad"]))
    artifact = Path(pkg.load_registry()["packages"]["silero-vad"]["materialized"]["path"])
    monkeypatch.setattr(package_teardown, "_delete_at", lambda _parent, _name: None)

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
    original_measure = package_teardown._owned_tree_bytes_at
    swapped = False

    def swap_after_parent_open(parent_descriptor: int, name: str) -> int:
        nonlocal swapped
        if not swapped:
            swapped = True
            managed_parent.rename(paths.root() / "envs-old")
            managed_parent.symlink_to(outside, target_is_directory=True)
        return original_measure(parent_descriptor, name)

    monkeypatch.setattr(package_teardown, "_owned_tree_bytes_at", swap_after_parent_open)

    assert package_teardown._delete_managed(managed_target) == len(b"managed")
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

    provisioner = pkg.Provisioner(toolchain=InstallTripwire(), fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))

    assert provisioner.verify()["failed"] == []


def test_verify_rejects_patched_file_symlink_even_when_target_bytes_match(
    provisioner,
    tmp_path,
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
