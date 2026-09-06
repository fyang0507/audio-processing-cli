"""Shared Hugging Face cache ownership and integrity behavior."""

from __future__ import annotations

from tests.audio_cli.packages.package_test_support import (
    ALIGNER_REVISION,
    FIRERED_REVISIONS,
    VIBE_MODEL_REVISION,
    VIBE_TOKENIZER_REVISION,
    FakeFetcher,
    FakeToolchain,
    Path,
    env,
    package_integrity,
    pkg,
    pytest,
    replace,
    shutil,
    sys,
    types,
)
from tests.audio_cli.packages.package_test_support import (
    isolated_root as isolated_root,
)
from tests.audio_cli.packages.package_test_support import (
    provisioner as provisioner,
)


# --------------------------------------------------------------------------------------
# The shared Hugging Face cache
# --------------------------------------------------------------------------------------
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
        tmp_path
        / "hub"
        / "models--mlx-community--Qwen3-ForcedAligner-0.6B-8bit"
        / "snapshots"
        / ALIGNER_REVISION
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
        assert list((tmp_path / "hub").glob(f"models--*/snapshots/{revision}")), (
            f"{revision} was deleted"
        )


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
    assert provisioner.fetcher.filtered == [
        (
            "Qwen/Qwen2.5-7B",
            VIBE_TOKENIZER_REVISION,
            patterns,
        )
    ]
    tokenizer_path = Path(materialized["paths"]["Qwen/Qwen2.5-7B"])
    assert {path.name for path in tokenizer_path.iterdir()} == set(patterns)


def test_pull_refuses_an_unindexed_hub_return_before_traversing_it(
    tmp_path,
    monkeypatch,
) -> None:
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    external = tmp_path / "outside-cache" / package.source["revision"]
    external.mkdir(parents=True)

    class UnindexedFetcher(FakeFetcher):
        def hf_snapshot(self, repo, revision, *, force=False, allow_patterns=None):
            self.snapshots.append((repo, revision))
            return external

    original_tree_bytes = package_integrity._tree_bytes

    def refuse_external_traversal(path: Path) -> int:
        if Path(path) == external:
            raise AssertionError("pull traversed an unindexed downloader return")
        return original_tree_bytes(path)

    monkeypatch.setattr(package_integrity, "_tree_bytes", refuse_external_traversal)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=UnindexedFetcher(tmp_path))

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
        repository for repository in original.source["repos"] if repository["role"] == "tokenizer"
    )
    asr = next(repository for repository in original.source["repos"] if repository["role"] == "asr")
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

    assert provisioner.fetcher.filtered == [
        (
            package.source["repo"],
            package.source["revision"],
            patterns,
        )
    ]
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

    snapshot = Path(pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]["path"])
    shutil.rmtree(snapshot)

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert failure[0]["fix"] == "audio packages pull --repair qwen3-asr-1.7b-8bit"


def test_verify_flags_hub_snapshot_byte_drift(provisioner) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    materialized = pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]
    snapshot = Path(materialized["path"])
    next(path for path in snapshot.rglob("*") if path.is_file()).unlink()

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "snapshot bytes changed" in failure[0]["detail"]


@pytest.mark.parametrize("mutation", ["external_file", "snapshot_symlink"])
def test_verify_rejects_hub_snapshot_redirection(
    provisioner,
    tmp_path,
    mutation: str,
) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    materialized = pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]
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
    materialized = pkg.load_registry()["packages"]["vibevoice-asr-7b"]["materialized"]
    tokenizer = Path(materialized["paths"]["Qwen/Qwen2.5-7B"])
    (tokenizer / "tokenizer.json").unlink()

    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
    assert "missing allow_pattern tokenizer.json" in failure[0]["detail"]


def test_verify_requires_allowlisted_tokenizer_matches_to_be_files(provisioner) -> None:
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    materialized = pkg.load_registry()["packages"]["vibevoice-asr-7b"]["materialized"]
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
    provisioner,
    tmp_path,
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
    assert first["pulled_known_bytes"] == env.packages()["qwen3-asr-1.7b-8bit"].bytes
    assert second["pulled_known_bytes"] == env.packages()["qwen3-asr-0.6b-8bit"].bytes
    document = pkg.load_registry()
    assert pkg.list_report()["total_known_bytes"] == sum(
        entry["materialized"]["bytes"] for entry in document["packages"].values()
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
    provisioner.pull(
        pkg.select(
            [
                "qwen3-asr-1.7b-8bit",
                "firered-asr2s",
                "vibevoice-asr-7b",
            ]
        )
    )
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
                self.already_cached = {revision_}  # the partial snapshot is now visible
                raise pkg.ProvisioningError("download_failed", "interrupted mid-download")
            return path

    fetcher = InterruptedThenCachedFetcher(tmp_path)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    selection = pkg.select(["qwen3-asr-1.7b-8bit"])

    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull(selection)
    assert (
        pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["hub_revisions_pre_existing"] == []
    )

    provisioner.pull(selection)
    materialized = pkg.load_registry()["packages"]["qwen3-asr-1.7b-8bit"]["materialized"]
    assert materialized["hub_revisions"] == [revision], (
        "the retry disowned the download the first attempt started"
    )
    assert materialized["hub_revisions_pre_existing"] == []
    assert provisioner.purge(dry_run=True)["would_remove"]["hub_revisions"] == [revision]
