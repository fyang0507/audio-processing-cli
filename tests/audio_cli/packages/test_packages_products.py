"""Crash-safe teardown, multi-repository repair, and built-runtime behavior."""

from __future__ import annotations

from tests.audio_cli.packages.package_test_support import (
    FIRERED_REVISIONS,
    FakeFetcher,
    FakeToolchain,
    Path,
    _snapshot_index_for,
    env,
    main,
    package_integrity,
    pkg,
    pytest,
)
from tests.audio_cli.packages.package_test_support import (
    isolated_root as isolated_root,
)
from tests.audio_cli.packages.package_test_support import (
    provisioner as provisioner,
)


def test_a_teardown_that_dies_leaves_no_package_reading_as_ready(
    tmp_path,
    monkeypatch,
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
            package_integrity,
            "_hub_snapshot_index",
            lambda fetcher=fetcher: _snapshot_index_for(fetcher.hub),
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
    guards = {guard["kind"]: guard for guard in env.environments()["mlx"].guards}
    expected = guards["source_hash"]["sha256"]

    toolchain = FakeToolchain(private_api_hash=expected)
    answering = pkg.Provisioner(
        toolchain=toolchain,
        fetcher=FakeFetcher(tmp_path),
    )
    answering.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    report = answering.verify()
    assert report["mlx_audio_private_api_source_hash"] == expected
    assert report["mlx_audio_private_api_matches_expected"] is True
    assert report["mlx_audio_private_api_signature_ok"] is True
    assert "mlx_audio_private_api_error" not in report
    probe_calls = [
        call for call in toolchain.calls if Path(str(call[1])).name == "runtime_probe.py"
    ]
    assert [call[2:] for call in probe_calls] == [[guards["signature"]["target"]]]

    # And it is a comparison rather than an echo: a source file that moved fails it, with both
    # values published so a reader can see why.
    moved = pkg.Provisioner(
        toolchain=FakeToolchain(private_api_hash="0" * 64), fetcher=FakeFetcher(tmp_path)
    )
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
    built = {
        identifier: package
        for identifier, package in env.packages().items()
        if package.source["type"] == "git+build"
    }
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
