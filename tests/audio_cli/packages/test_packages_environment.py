"""Environment drift, managed-root, and repair behavior."""

from __future__ import annotations

from tests.audio_cli.packages.package_test_support import (
    FakeFetcher,
    FakeToolchain,
    Path,
    env,
    package_registry,
    package_requirements,
    paths,
    pkg,
    pytest,
    types,
)
from tests.audio_cli.packages.package_test_support import (
    isolated_root as isolated_root,
)


def test_verify_treats_an_unlocked_extra_distribution_as_repairable_drift(
    tmp_path,
) -> None:
    pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)).pull(
        pkg.select(["qwen3-asr-1.7b-8bit"])
    )
    direct_reference = "@ file:///external/startup-hook"
    toolchain = FakeToolchain(drift={"startup-hook": direct_reference})
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))

    report = provisioner.verify()

    assert report["environments"]["mlx"] == "drifted"
    assert report["failed"][0]["examples"]["startup-hook"] == {
        "locked": None,
        "installed": direct_reference,
    }
    repaired = provisioner.verify(repair=True)
    assert repaired["environments"]["mlx"] == "ok"
    assert repaired["failed"] == []
    assert toolchain.created == ["mlx"]
    assert provisioner.verify()["environments"]["mlx"] == "ok"


def test_freeze_parser_keeps_direct_editable_and_unknown_installed_lines(
    monkeypatch,
) -> None:
    toolchain = pkg.Toolchain()
    stdout = "\n".join(
        (
            "locked_package==1.2.3",
            "rogue @ file:///tmp/rogue",
            "-e file:///tmp/editable#egg=editable_hook",
            "-e http://[invalid",
            "future-freeze-syntax",
        )
    )
    monkeypatch.setattr(
        toolchain,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )

    assert toolchain.frozen_packages(Path("/unused/python")) == {
        "locked-package": "1.2.3",
        "rogue": "@ file:///tmp/rogue",
        "editable-hook": "-e file:///tmp/editable#egg=editable_hook",
        "<editable:http://[invalid>": "-e http://[invalid",
        "<unparsed:future-freeze-syntax>": "future-freeze-syntax",
    }


def test_environment_drift_allows_only_manifest_owned_direct_checkout_paths(
    tmp_path,
) -> None:
    managed = tmp_path / "managed"
    external = tmp_path / "external"
    drift = package_requirements._environment_drift(
        {},
        {
            "native-backend": f"@ {managed.as_uri()}",
            "startup-hook": f"@ {external.as_uri()}",
        },
        {"native-backend": managed},
    )

    assert drift == {
        "startup-hook": (None, f"@ {external.as_uri()}"),
    }


@pytest.mark.parametrize(
    "frozen",
    [
        {},
        {"wrong-name": "@ file:///managed/checkout"},
        {"native-backend": "@ file:///external/checkout"},
    ],
)
def test_environment_drift_requires_the_exact_named_checkout_install(
    frozen: dict[str, str],
) -> None:
    required = Path("/managed/checkout")
    drift = package_requirements._environment_drift({}, frozen, {"native-backend": required})
    assert drift["native-backend"] == (f"@ {required.as_uri()}", frozen.get("native-backend"))


@pytest.mark.parametrize("package_id", ["vibevoice-asr-7b", "firered-asr2s"])
def test_verify_repair_reinstalls_ready_checkout_after_lock_sync(
    tmp_path: Path,
    package_id: str,
) -> None:
    toolchain = FakeToolchain()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    package = env.packages()[package_id]
    provisioner.pull(pkg.select([package_id]))
    environment_name = package.environment
    distribution = package.checkout["distribution"]
    assert distribution in toolchain._direct_installs[environment_name]

    # Model a lock change plus uv sync: the direct install exists before repair,
    # create_environment removes it, and install_checkout must restore it.
    locked_name = next(
        iter(package_requirements._locked_versions(env.environments()[environment_name]))
    )
    toolchain._synced.discard(environment_name)
    toolchain._drift[locked_name] = "0.invalid"
    document = pkg.load_registry()
    document["environments"][environment_name]["lock_sha256"] = "stale"
    pkg.save_registry(document)
    toolchain.calls.clear()
    toolchain.created.clear()

    repaired = provisioner.verify(repair=True)

    assert repaired["environments"][environment_name] == "ok"
    assert repaired["failed"] == []
    assert toolchain.created == [environment_name]
    assert [call[0] for call in toolchain.calls if call[0] == "install"] == ["install"]
    assert distribution in toolchain._direct_installs[environment_name]
    registry = pkg.load_registry()
    assert registry["environments"][environment_name]["state"] == "ready"
    assert registry["environments"][environment_name]["lock_sha256"] == (
        pkg.sha256_file(env.environments()[environment_name].lock)
    )
    assert provisioner.verify()["environments"][environment_name] == "ok"

    created_before = list(toolchain.created)
    provisioner.pull(pkg.select([package_id]))
    assert toolchain.created == created_before


def test_verify_repair_never_installs_a_tampered_ready_checkout(
    tmp_path: Path,
) -> None:
    toolchain = FakeToolchain()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    package_id = "vibevoice-asr-7b"
    package = env.packages()[package_id]
    provisioner.pull(pkg.select([package_id]))
    patched = (
        paths.checkout_dir(package.environment, package.id)
        / "vibevoice/modular/modeling_vibevoice_asr.py"
    )
    patched.write_text("tampered\n", encoding="utf-8")
    locked_name = next(
        iter(package_requirements._locked_versions(env.environments()[package.environment]))
    )
    toolchain._synced.discard(package.environment)
    toolchain._drift[locked_name] = "0.invalid"
    toolchain.calls.clear()
    toolchain.created.clear()

    report = provisioner.verify(repair=True)

    assert report["environments"][package.environment] == "drifted"
    assert any(
        item.get("package") == package_id and item["code"] == "package_integrity_failed"
        for item in report["failed"]
    )
    assert toolchain.created == []
    assert not [call for call in toolchain.calls if call[0] == "install"]


def test_verify_refuses_a_symlinked_environment_root_even_when_freeze_matches(
    tmp_path,
) -> None:
    toolchain = FakeToolchain(private_api_hash="would-execute-external-python")
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    managed = paths.env_dir("mlx")
    external = tmp_path / "external-mlx"
    managed.rename(external)
    managed.symlink_to(external, target_is_directory=True)
    marker = external / "owned-elsewhere"
    marker.write_text("keep\n", encoding="utf-8")
    toolchain.calls.clear()

    report = provisioner.verify()

    assert report["environments"]["mlx"] == "drifted"
    assert report["failed"][0]["code"] == "environment_drifted"
    assert "environment path is a symlink" in report["failed"][0]["detail"]
    assert report["mlx_audio_private_api_matches_expected"] is False
    assert not any(
        any(Path(str(argument)).name == "runtime_probe.py" for argument in call)
        for call in toolchain.calls
    )
    repaired = provisioner.verify(repair=True)
    assert repaired["environments"]["mlx"] == "drifted"
    assert marker.read_text(encoding="utf-8") == "keep\n"

    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["qwen3-forcedaligner"]))
    assert caught.value.code == "environment_drifted"
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_pull_refuses_a_symlinked_provisioning_root_before_skip_or_fetch(
    tmp_path,
) -> None:
    package_id = "qwen3-asr-1.7b-8bit"
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select([package_id]))
    root = paths.root()
    external = tmp_path / "external-provisioning-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)
    marker = external / "owned-elsewhere"
    marker.write_text("keep\n", encoding="utf-8")

    class NoFetch(FakeFetcher):
        def cached_revisions(self) -> set[str]:
            raise AssertionError("symlinked provisioning root reached cache inspection")

    toolchain = FakeToolchain()
    guarded = pkg.Provisioner(toolchain=toolchain, fetcher=NoFetch(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        # This ready package formerly took the skip fast path and reported success.
        guarded.pull(pkg.select([package_id]))

    assert caught.value.code == "environment_drifted"
    assert caught.value.message == f"provisioning root is a symlink: {root}"
    assert toolchain.calls == []
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_pull_refuses_a_symlinked_root_before_loading_its_registry(
    tmp_path,
    monkeypatch,
) -> None:
    root = paths.root()
    external = tmp_path / "external-provisioning-root"
    external.mkdir()
    root.symlink_to(external, target_is_directory=True)

    def forbidden_registry_read():
        raise AssertionError("symlinked provisioning root reached registry loading")

    monkeypatch.setattr(package_registry, "load_registry", forbidden_registry_read)
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path)).pull(
            pkg.select(["qwen3-asr-0.6b-8bit"])
        )

    assert caught.value.code == "environment_drifted"
    assert caught.value.message == f"provisioning root is a symlink: {root}"


def test_verify_never_probes_below_a_symlinked_provisioning_root(tmp_path) -> None:
    package_id = "vibevoice-asr-7b"
    toolchain = FakeToolchain(private_api_hash="would-execute-external-python")
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select([package_id]))
    root = paths.root()
    external = tmp_path / "external-provisioning-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)
    marker = external / "owned-elsewhere"
    marker.write_text("keep\n", encoding="utf-8")
    toolchain.calls.clear()
    toolchain.created.clear()

    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.verify(repair=True)

    assert caught.value.code == "registry_unreadable"
    assert str(root) in caught.value.message
    assert toolchain.calls == []
    assert toolchain.created == []
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_every_managed_path_rejects_a_symlinked_provisioning_root(tmp_path) -> None:
    root = paths.root()
    environment = paths.env_dir("torch-vibevoice")
    checkout = paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
    artifact = paths.models_dir() / env.packages()["silero-vad"].source["filename"]
    checkout.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")
    external = tmp_path / "external-provisioning-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)
    expected_issue = f"provisioning root is a symlink: {root}"

    assert pkg.managed_environment_path("torch-vibevoice") == (
        environment,
        expected_issue,
    )
    assert pkg.managed_checkout_path(env.packages()["vibevoice-asr-7b"], checkout) == (
        checkout,
        f"package environment is not managed: {expected_issue}",
    )
    assert pkg.managed_url_artifact_path(env.packages()["silero-vad"], artifact) == (
        artifact,
        expected_issue,
    )


def test_verify_refuses_an_in_root_symlinked_environments_parent(tmp_path) -> None:
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    envs = paths.envs_dir()
    alternate = paths.root() / "alternate-envs"
    envs.rename(alternate)
    envs.symlink_to(alternate, target_is_directory=True)

    report = provisioner.verify()

    assert report["environments"]["mlx"] == "drifted"
    assert "environment parent is a symlink" in report["failed"][0]["detail"]
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["qwen3-forcedaligner"]))
    assert caught.value.code == "environment_drifted"


def test_verify_does_not_inspect_or_launch_below_a_redirected_environment_parent(
    tmp_path,
) -> None:
    class ProbeTrap(FakeToolchain):
        armed = False

        def inspect_checkout(self, checkout: Path) -> pkg.CheckoutState:
            if self.armed:
                raise AssertionError(f"verify inspected redirected checkout {checkout}")
            return super().inspect_checkout(checkout)

        def built_product_runs(self, executable: Path) -> bool:
            if self.armed:
                raise AssertionError(f"verify launched redirected product {executable}")
            return True

    toolchain = ProbeTrap()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["vibevoice-asr-7b", "fluidaudio"]))
    envs = paths.envs_dir()
    alternate = paths.root() / "alternate-envs"
    envs.rename(alternate)
    envs.symlink_to(alternate, target_is_directory=True)
    toolchain.armed = True

    report = provisioner.verify()

    assert report["verified"] == []
    assert {item["environment"] for item in report["failed"]} == {
        "torch-vibevoice",
        "swift",
    }
    assert all(item["code"] == "environment_drifted" for item in report["failed"])


def test_verify_fails_closed_when_ready_packages_have_no_ready_environment(
    tmp_path,
) -> None:
    class ProbeTrap(FakeToolchain):
        armed = False

        def inspect_checkout(self, checkout: Path) -> pkg.CheckoutState:
            if self.armed:
                raise AssertionError(f"verify inspected checkout without environment {checkout}")
            return super().inspect_checkout(checkout)

        def built_product_runs(self, executable: Path) -> bool:
            if self.armed:
                raise AssertionError(f"verify launched product without environment {executable}")
            return True

    toolchain = ProbeTrap()
    provisioner = pkg.Provisioner(toolchain=toolchain, fetcher=FakeFetcher(tmp_path))
    provisioner.pull(pkg.select(["vibevoice-asr-7b", "fluidaudio"]))
    document = pkg.load_registry()
    document["environments"].pop("torch-vibevoice")
    document["environments"]["swift"]["state"] = "creating"
    pkg.save_registry(document)
    toolchain.armed = True

    report = provisioner.verify()

    assert report["verified"] == []
    assert report["environments"]["torch-vibevoice"] == "absent"
    assert report["environments"]["swift"] == "absent"
    failures = {item["environment"]: item for item in report["failed"]}
    assert set(failures) == {"torch-vibevoice", "swift"}
    assert failures["torch-vibevoice"]["code"] == "environment_not_ready"
    assert failures["swift"]["code"] == "environment_not_ready"
    assert failures["torch-vibevoice"]["packages"] == ["vibevoice-asr-7b"]
    assert failures["swift"]["packages"] == ["fluidaudio"]


def test_verify_reports_a_corrupted_single_file_artifact(tmp_path) -> None:
    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path, corrupt=True)
    )
    provisioner.pull(pkg.select(["silero-vad"]))
    failure = provisioner.verify()["failed"]
    assert [item["code"] for item in failure] == ["package_integrity_failed"]
