"""The `audio packages` lifecycle, against a fabricated root.

Provisioning the real thing costs 30 GiB and a Swift toolchain, so nothing here downloads or
builds. The two external surfaces are injected, which leaves the parts that actually carry the
rules under test: the registry state machine, the crash-safety guarantee, reference counting,
digest verification, and every payload shape the spec documents publish.

The one thing a fake cannot check is whether the real `uv` and Hub calls work. Those are
exercised by the provisioning probes in `model_tests/benchmark/` and recorded there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.cli import main


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    """Every test gets an empty root, so none of them can see a real provisioning."""
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(tmp_path / "root"))
    return tmp_path / "root"


class FakeToolchain(pkg.Toolchain):
    """Records what would have been run, and pretends it succeeded."""

    def __init__(self, *, missing: tuple[str, ...] = (), drift: dict | None = None) -> None:
        self.calls: list[list[str]] = []
        self.missing = set(missing)
        self._drift = drift or {}
        self.created: list[str] = []

    def which(self, tool: str) -> str | None:
        return None if tool in self.missing else f"/usr/bin/{tool}"

    def run(self, args, *, cwd=None, timeout=3600):
        self.calls.append(list(args))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    def create_environment(self, environment, target: Path) -> None:
        self.created.append(environment.name)
        (target / "bin").mkdir(parents=True, exist_ok=True)
        (target / "bin" / "python").write_text("#!/bin/sh\n")

    def frozen_packages(self, environment_python: Path) -> dict[str, str]:
        name = environment_python.parent.parent.name
        installed = pkg._locked_versions(env.environments()[name])
        installed.update(self._drift)
        return installed

    def clone(self, repo: str, commit: str, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
        touched = target / "vibevoice" / "modular"
        touched.mkdir(parents=True, exist_ok=True)
        (touched / "modeling_vibevoice_asr.py").write_text("original\n")

    def apply_patch(self, checkout: Path, patch: Path) -> None:
        target = checkout / "vibevoice" / "modular" / "modeling_vibevoice_asr.py"
        if target.is_file():
            target.write_text("patched\n")

    def install_checkout(self, environment_python: Path, checkout: Path) -> None:
        self.calls.append(["install", str(checkout)])

    def swift_build(self, checkout: Path) -> None:
        (checkout / ".build").mkdir(parents=True, exist_ok=True)
        (checkout / ".build" / "product").write_bytes(b"x" * 1024)

    def swift_product_runs(self, checkout: Path) -> bool:
        return True


class FakeFetcher(pkg.Fetcher):
    """Writes a plausible snapshot instead of downloading one."""

    def __init__(self, tmp_path: Path, *, corrupt: bool = False) -> None:
        self.hub = tmp_path / "hub"
        self.corrupt = corrupt
        self.snapshots: list[tuple[str, str]] = []

    def hf_snapshot(self, repo: str, revision: str) -> Path:
        self.snapshots.append((repo, revision))
        target = self.hub / repo.replace("/", "--") / revision
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"w" * 2048)
        return target

    def delete_hub_revisions(self, revisions: list[str]) -> tuple[list[str], int]:
        """Deletes the fake snapshot directories this fetcher created, and reports their size."""
        deleted, freed = [], 0
        for revision in revisions:
            for candidate in self.hub.glob(f"*/{revision}"):
                freed += pkg._tree_bytes(candidate)
                for item in sorted(candidate.rglob("*"), reverse=True):
                    item.unlink() if item.is_file() else item.rmdir()
                candidate.rmdir()
                deleted.append(revision)
        return deleted, freed

    def url_file(self, url: str, sha256: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"corrupt" if self.corrupt else b"onnx-bytes")
        if not self.corrupt:
            # Stand in for a verified download without needing the real 2.3 MB artifact.
            pkg.sha256_file(target)
        return target


@pytest.fixture
def provisioner(tmp_path):
    return pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))


def test_an_empty_root_reports_everything_absent() -> None:
    report = pkg.list_report()
    assert report["packages"] == []
    assert set(report["environments"].values()) == {"absent"}
    assert report["total_known_bytes"] == 0


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


def test_pull_warns_that_a_declared_license_is_not_a_reviewed_one(provisioner) -> None:
    receipt = provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    warning = receipt["warnings"][0]
    assert warning["code"] == "license_unreviewed"
    assert warning["blocking"] is False
    assert warning["packages"] == ["qwen3-asr-1.7b-8bit"]

    # And says nothing where the terms were actually read.
    other = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(Path("/tmp")))
    assert other.pull(pkg.select(["speaker-diarization-coreml"]))["warnings"] == []


def test_a_crashed_pull_does_not_read_as_provisioned(tmp_path) -> None:
    """The failure mode this state machine exists for: exit 3 must still fire afterwards."""

    class ExplodingFetcher(FakeFetcher):
        def hf_snapshot(self, repo: str, revision: str) -> Path:
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
        def hf_snapshot(self, repo: str, revision: str) -> Path:
            calls["n"] += 1
            if calls["n"] == 1:
                raise pkg.ProvisioningError("download_failed", "first attempt died")
            return super().hf_snapshot(repo, revision)

    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FlakyFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")


def test_the_registry_is_written_atomically(provisioner, isolated_root) -> None:
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    leftovers = list(isolated_root.glob(".registry.json*"))
    assert leftovers == [], f"temporary registry files survived: {leftovers}"
    assert json.loads(paths.registry_path().read_text())["schema_version"] == 1


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
    assert report["hub_revisions_deleted"] == ["d0c9efdb8d614685062c04425d91e01b6f37d944"]
    assert "may be shared with other tools" in report["hub_cache_note"]


def test_remove_deletes_the_checkout_and_the_revision_it_materialized(provisioner) -> None:
    """The recorded revision is deleted; the shared cache around it is not touched."""
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    checkout = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
    snapshot = Path(pkg.load_registry()["packages"]["vibevoice-asr-7b"]["materialized"]["path"])
    sibling = snapshot.parent / "another-revision"
    sibling.mkdir(parents=True)
    (sibling / "weights").write_bytes(b"someone else's")
    assert checkout.is_dir() and snapshot.is_dir()

    report = provisioner.remove(["vibevoice-asr-7b"])
    assert not checkout.exists()
    assert not snapshot.exists(), "the revision this tool materialized should be reclaimed"
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
    assert [item["code"] for item in failure] == ["patch_not_applied"]
    assert failure[0]["fix"].startswith("audio packages pull --repair")


def test_verify_reports_a_drifted_environment_and_repairs_it(tmp_path) -> None:
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))

    drifted = pkg.Provisioner(
        toolchain=FakeToolchain(drift={"mlx": "0.31.0"}),
        fetcher=FakeFetcher(tmp_path),
    )
    report = drifted.verify()
    assert report["environments"]["mlx"] == "drifted"
    assert report["failed"][0]["code"] == "environment_drifted"
    assert report["failed"][0]["examples"]["mlx"] == {"locked": "0.32.0", "installed": "0.31.0"}
    assert report["failed"][0]["fix"] == "audio packages verify --repair"

    repaired = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))
    assert repaired.verify(repair=True)["environments"]["mlx"] == "ok"


def test_verify_reports_a_corrupted_single_file_artifact(tmp_path) -> None:
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(),
                                 fetcher=FakeFetcher(tmp_path, corrupt=True))
    provisioner.pull(pkg.select(["silero-vad"]))
    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]


def test_verify_reports_the_private_api_guard_as_unchecked_without_the_environment(
    provisioner,
) -> None:
    """No mlx environment means no verdict, not a passing one."""
    report = provisioner.verify()
    assert report["mlx_audio_private_api_matches_expected"] is None
    assert report["mlx_audio_private_api_expected_source_hash"] == (
        "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250"
    )


def test_selecting_by_stack_covers_every_package_that_stack_can_use() -> None:
    chosen = {package.id for package in pkg.select(stack="qwen-1.7b")}
    assert chosen == {"silero-vad", "qwen3-asr-1.7b-8bit", "qwen3-forcedaligner",
                      "fluidaudio", "speaker-diarization-coreml"}


def test_unknown_names_fail_with_the_menu_rather_than_a_guess() -> None:
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.select(["whisper-large"])
    assert caught.value.code == "package_unknown"
    assert "qwen3-asr-1.7b-8bit" in caught.value.payload["allowed"]

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.select(stack="whisper")
    assert caught.value.payload["allowed"] == ["firered", "qwen-0.6b", "qwen-1.7b", "vibevoice"]


def test_a_missing_required_tool_blocks_only_the_package_that_needs_it(tmp_path) -> None:
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(missing=("swift",)),
                                 fetcher=FakeFetcher(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["fluidaudio"]))
    assert caught.value.code == "toolchain_missing"
    assert caught.value.payload["requires_tool"] == ["swift"]

    # The mlx package is unaffected, which is the point of reporting rather than failing.
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")


def test_doctor_reports_provisioning_state_and_what_a_missing_tool_blocks() -> None:
    report = pkg.doctor(toolchain=FakeToolchain(missing=("swift",)))
    assert report["tools"]["swift"]["present"] is False
    assert report["environments"]["swift"]["blocked_by_missing_tool"] == ["swift"]
    assert report["environments"]["mlx"]["blocked_by_missing_tool"] == []
    assert report["environments"]["torch-vibevoice"]["provisional"] is True
    assert report["packages"]["firered-asr2s"] == "absent"


def test_registry_with_a_future_schema_version_is_refused(isolated_root) -> None:
    """A newer tool's registry must not be silently reinterpreted by an older one."""
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"


def test_cli_exit_codes(capsys, provisioner) -> None:
    assert main(["packages", "list"]) == 0
    assert json.loads(capsys.readouterr().out)["packages"] == []

    assert main(["packages", "path"]) == 0
    capsys.readouterr()

    # verify with nothing provisioned has nothing to fail on
    assert main(["packages", "verify"]) == 0
    capsys.readouterr()

    assert main(["packages", "remove", "firered-asr2s"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "package_not_provisioned"

    assert main(["packages", "pull", "--want", "diarization"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "stack_required"

    assert main(["doctor"]) == 0
    assert "environments" in json.loads(capsys.readouterr().out)


def test_verify_exits_three_when_a_check_fails(capsys, provisioner) -> None:
    provisioner.pull(pkg.select(["vibevoice-asr-7b"]))
    patched = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b") / "vibevoice" / "modular" / \
        "modeling_vibevoice_asr.py"
    patched.write_text("original\n")
    assert main(["packages", "verify"]) == 3
    codes = {item["code"] for item in json.loads(capsys.readouterr().out)["failed"]}
    assert "patch_not_applied" in codes


def test_the_patch_ships_inside_the_package() -> None:
    """It is applied at pull time on a user's machine, so it cannot live in model_tests/."""
    patch = env.HERE / "patches" / "vibevoice-logits-to-keep.patch"
    assert patch.is_file()
    assert "modeling_vibevoice_asr.py" in patch.read_text()
