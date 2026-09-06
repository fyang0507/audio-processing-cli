"""Ownership-scoped cache notes and size accounting across pulls, skips, and repairs."""

from __future__ import annotations

import pytest
from package_test_support import FakeFetcher, FakeToolchain
from package_test_support import isolated_root as isolated_root

from audio_cli import environments as env
from audio_cli import packages as pkg


@pytest.mark.parametrize("identifier", ["firered-asr2s", "vibevoice-asr-7b"])
@pytest.mark.parametrize("cached_count", [0, 1, "all"])
def test_partial_and_full_preexisting_revisions_have_scoped_notes_and_accounting(
    tmp_path, identifier, cached_count
):
    package = env.packages()[identifier]
    repositories = package.source["repos"]
    revisions = [repository["revision"] for repository in repositories]
    cached = revisions if cached_count == "all" else revisions[:cached_count]
    fetcher = FakeFetcher(tmp_path, already_cached=cached)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    report = provisioner.pull([package])
    receipt = report["pulled"][0]
    expected_owned_bytes = sum(
        repository["bytes"] for repository in repositories if repository["revision"] not in cached
    )
    assert report["pulled_known_bytes"] == expected_owned_bytes
    assert "manifest-declared sizes" in report["pulled_known_bytes_note"]
    assert "Not measured network bytes or added disk usage" in report["pulled_known_bytes_note"]
    materialized = pkg.load_registry()["packages"][identifier]["materialized"]
    assert receipt["bytes"] == materialized["bytes"]
    # Synthetic measured snapshots deliberately differ from manifest-declared weight sizes.
    assert receipt["bytes"] != report["pulled_known_bytes"]
    assert set(receipt["revisions"]) == set(revisions)
    assert "digest_verified" not in receipt
    assert "digest" not in receipt
    if cached:
        assert receipt["hub_revisions_pre_existing"] == sorted(cached)
        assert (
            f"listed {len(cached)} of {len(revisions)} pinned revisions"
            in receipt["pre_existing_note"]
        )
        assert (
            f"({'all' if len(cached) == len(revisions) else 'some'} revisions)"
            in receipt["pre_existing_note"]
        )
        assert "missing files may still be fetched" in receipt["pre_existing_note"]
        assert "not downloaded" not in receipt["pre_existing_note"]
    else:
        assert "pre_existing_note" not in receipt
        assert "hub_revisions_pre_existing" not in receipt
    assert set(materialized["hub_revisions"]) == set(revisions) - set(cached)

    previous = pkg.load_registry()
    assert provisioner.pull([package])["pulled_known_bytes"] == 0
    assert pkg.load_registry() == previous

    # A current cache scan cannot reassign this root's historical ownership on repair.
    fetcher.already_cached = set(revisions)
    repaired = provisioner.pull([package], repair=True)
    assert repaired["pulled_known_bytes"] == expected_owned_bytes
    assert repaired["pulled"] == report["pulled"]
    assert {revision for _repo, revision in fetcher.forced} == set(revisions)
    assert "including repairs" in repaired["pulled_known_bytes_note"]
    assert provisioner.verify()["failed"] == []
    removed = provisioner.remove([identifier])
    assert set(removed["hub_revisions_deleted"]) == set(revisions) - set(cached)
    assert set(removed["hub_revisions_retained"]) == set(cached)


def test_preexisting_note_is_bounded_by_current_manifest_on_retry(tmp_path):
    package = env.packages()["qwen3-forcedaligner"]
    revision = package.source["revision"]
    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path, already_cached=[revision])
    )
    provisioner.pull([package])
    document = pkg.load_registry()
    document["packages"][package.id]["hub_revisions_pre_existing"].append("out-of-manifest")
    pkg.save_registry(document)
    report = provisioner.pull([package], repair=True)
    receipt = report["pulled"][0]
    assert receipt["hub_revisions_pre_existing"] == [revision]
    assert "listed 1 of 1" in receipt["pre_existing_note"]
    assert report["pulled_known_bytes"] == 0


def test_fake_environment_never_overwrites_an_interpreter_symlink(tmp_path):
    target = tmp_path / "fake-env"
    (target / "bin").mkdir(parents=True)
    external = tmp_path / "external-interpreter"
    external.write_bytes(b"keep executable")
    (target / "bin/python").symlink_to(external)
    with pytest.raises(AssertionError, match="must not overwrite a real interpreter"):
        FakeToolchain().create_environment(env.environments()["torch-firered"], target)
    assert external.read_bytes() == b"keep executable"
