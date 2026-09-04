"""Package selection, registry validation, CLI, and shipped patch behavior."""

from __future__ import annotations

from package_test_support import (
    QWEN_REPO,
    QWEN_REVISION,
    FakeFetcher,
    FakeToolchain,
    Path,
    env,
    json,
    main,
    package_registry,
    paths,
    pkg,
    pytest,
    shutil,
    subprocess,
)
from package_test_support import (
    isolated_root as isolated_root,
)
from package_test_support import (
    provisioner as provisioner,
)


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
    assert chosen == {
        "silero-vad",
        "qwen3-asr-1.7b-8bit",
        "qwen3-forcedaligner",
        "fluidaudio",
        "speaker-diarization-coreml",
    }


def test_named_selection_stably_deduplicates_repeated_ids(provisioner) -> None:
    selection = pkg.select(
        [
            "qwen3-asr-1.7b-8bit",
            "qwen3-forcedaligner",
            "qwen3-asr-1.7b-8bit",
        ]
    )
    assert [package.id for package in selection] == [
        "qwen3-asr-1.7b-8bit",
        "qwen3-forcedaligner",
    ]

    receipt = provisioner.pull(
        pkg.select(
            [
                "qwen3-asr-1.7b-8bit",
                "qwen3-asr-1.7b-8bit",
            ]
        )
    )
    assert [item["package"] for item in receipt["pulled"]] == ["qwen3-asr-1.7b-8bit"]
    assert receipt["skipped"] == []


def test_repair_deduplicates_before_forcing_a_download(tmp_path) -> None:
    fetcher = FakeFetcher(tmp_path)
    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=fetcher)
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))

    receipt = provisioner.pull(
        pkg.select(
            [
                "qwen3-asr-1.7b-8bit",
                "qwen3-asr-1.7b-8bit",
            ]
        ),
        repair=True,
    )

    assert fetcher.forced == [(QWEN_REPO, QWEN_REVISION)]
    assert [item["package"] for item in receipt["pulled"]] == ["qwen3-asr-1.7b-8bit"]


def test_unknown_names_fail_with_the_menu_rather_than_a_guess() -> None:
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.select(["whisper-large"])
    assert caught.value.code == "package_unknown"
    assert "qwen3-asr-1.7b-8bit" in caught.value.payload["allowed"]

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.select(stack="whisper")
    assert caught.value.payload["allowed"] == ["firered", "qwen-0.6b", "qwen-1.7b", "vibevoice"]


def test_a_missing_required_tool_blocks_only_the_package_that_needs_it(tmp_path) -> None:
    provisioner = pkg.Provisioner(
        toolchain=FakeToolchain(missing=("swift",)), fetcher=FakeFetcher(tmp_path)
    )
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull(pkg.select(["fluidaudio"]))
    assert caught.value.code == "toolchain_missing"
    assert caught.value.payload["requires_tool"] == ["swift"]

    # The mlx package is unaffected, which is the point of reporting rather than failing.
    provisioner.pull(pkg.select(["qwen3-asr-1.7b-8bit"]))
    assert pkg.is_ready(pkg.load_registry(), "qwen3-asr-1.7b-8bit")


def test_doctor_separates_pull_toolchains_from_runtime_requirements() -> None:
    report = pkg.doctor(toolchain=FakeToolchain(missing=("swift",)))
    assert report["tools"]["swift"]["present"] is False
    assert report["environments"]["swift"]["requires_tool"] == ["swift"]
    assert report["environments"]["swift"]["blocked_by_missing_tool"] == ["swift"]
    assert report["environments"]["mlx"]["blocked_by_missing_tool"] == []
    assert report["environments"]["torch-vibevoice"]["provisional"] is True
    assert report["packages"]["firered-asr2s"] == "absent"
    assert env.packages()["fluidaudio"].requires_tool == ("swift",)


def test_registry_with_a_future_schema_version_is_refused(isolated_root) -> None:
    """A newer tool's registry must not be silently reinterpreted by an older one."""
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"


@pytest.mark.parametrize("foreign_kind", ["copy", "symlink"])
@pytest.mark.parametrize("teardown", ["remove", "purge"])
def test_foreign_registry_never_authorizes_hub_deletion(
    tmp_path,
    foreign_kind: str,
    teardown: str,
) -> None:
    """Hub ownership belongs to one provisioning root, not to a movable receipt."""
    foreign_root = tmp_path / "foreign-root"
    foreign_registry = foreign_root / "registry.json"
    foreign_root.mkdir()
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    document = pkg.blank_registry()
    document["root"] = str(foreign_root)
    document["packages"][package.id] = {
        "state": "ready",
        "environment": package.environment,
        "materialized": {
            "hub_revisions": [package.source["revision"]],
            "bytes": 0,
        },
    }
    foreign_registry.write_text(json.dumps(document), encoding="utf-8")
    target = paths.registry_path()
    target.parent.mkdir(parents=True)
    if foreign_kind == "copy":
        shutil.copyfile(foreign_registry, target)
    else:
        target.symlink_to(foreign_registry)

    class DeletionTripwire(FakeFetcher):
        def delete_hub_revisions(self, revisions: list[str]) -> tuple[list[str], int]:
            raise AssertionError(f"foreign registry authorized deletion of {revisions}")

    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=DeletionTripwire(tmp_path))
    with pytest.raises(pkg.ProvisioningError) as caught:
        if teardown == "remove":
            provisioner.remove([package.id])
        else:
            provisioner.purge(dry_run=False)

    assert caught.value.code == "registry_unreadable"


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"schema_version": 1, "packages": []},
        {
            "schema_version": 1,
            "packages": {"broken": []},
        },
    ],
)
def test_registry_container_shapes_are_validated(document) -> None:
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"


def test_registry_duplicate_keys_are_refused_before_last_wins_semantics(
    capsys,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_root = json.dumps(str(paths.root()))
    target.write_text(
        "{"
        '"schema_version":1,'
        '"tool_version":"test",'
        '"root":"attacker-controlled",'
        f'"root":{expected_root},'
        '"environments":{},'
        '"packages":{}'
        "}",
        encoding="utf-8",
    )

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"
    assert "duplicate JSON object key 'root'" in caught.value.message

    assert main(["packages", "list"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"


@pytest.mark.parametrize("command", ["list", "path", "verify"])
def test_cli_readers_refuse_a_nonfile_registry(command, capsys) -> None:
    paths.registry_path().mkdir(parents=True)

    assert main(["packages", command]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"
    assert "not a regular file" in error["detail"]


@pytest.mark.parametrize("revisions_key", ["hub_revisions", "hub_revisions_pre_existing"])
def test_registry_rejects_nonstring_revision_ownership_receipts(
    revisions_key,
    capsys,
) -> None:
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    document = pkg.blank_registry()
    document["packages"][package.id] = {
        "state": "pulling",
        "materialized": {revisions_key: [package.source["revision"], {}]},
    }
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()
    assert caught.value.code == "registry_unreadable"

    assert main(["packages", "pull", "--repair", package.id]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"


def test_repair_refuses_a_malformed_top_level_retry_ownership_ledger(capsys) -> None:
    package = env.packages()["qwen3-asr-0.6b-8bit"]
    document = pkg.blank_registry()
    document["packages"][package.id] = {
        "state": "pulling",
        "hub_revisions_pre_existing": [package.source["revision"], {}],
    }
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")

    assert main(["packages", "pull", "--repair", package.id]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"
    assert "hub_revisions_pre_existing" in error["detail"]


def test_registry_read_error_is_a_machine_readable_cli_failure(
    monkeypatch,
    capsys,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")
    original = package_registry.os.open

    def unreadable(path, flags, mode=0o777, *, dir_fd=None):
        if path == target.name and dir_fd is not None:
            raise PermissionError("denied")
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(package_registry.os, "open", unreadable)
    assert main(["packages", "verify"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"
    assert "Restore access" in error["fix"]


def test_registry_loader_never_follows_a_leaf_substituted_before_open(
    tmp_path,
    monkeypatch,
) -> None:
    target = paths.registry_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(pkg.blank_registry()), encoding="utf-8")
    victim = tmp_path / "external-registry.json"
    victim.write_text("external bytes must not be read\n", encoding="utf-8")
    original = package_registry.os.open
    substituted = False

    def substitute(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal substituted
        if path == target.name and dir_fd is not None and not substituted:
            substituted = True
            target.unlink()
            target.symlink_to(victim)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(package_registry.os, "open", substitute)

    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.load_registry()

    assert caught.value.code == "registry_unreadable"
    assert substituted is True
    assert target.is_symlink()
    assert victim.read_text(encoding="utf-8") == "external bytes must not be read\n"


@pytest.mark.parametrize("command", ["list", "path", "verify"])
def test_cli_readers_refuse_nonobject_materialized_registry_entry(
    command,
    capsys,
) -> None:
    document = pkg.blank_registry()
    document["packages"]["silero-vad"] = {
        "state": "ready",
        "materialized": "not-an-object",
    }
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")

    assert main(["packages", command]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"


@pytest.mark.parametrize("command", ["list", "path", "verify"])
def test_cli_readers_refuse_nonnumeric_materialized_bytes(command, capsys) -> None:
    document = pkg.blank_registry()
    document["packages"]["silero-vad"] = {
        "state": "ready",
        "materialized": {"bytes": "not-a-number"},
    }
    paths.registry_path().parent.mkdir(parents=True, exist_ok=True)
    paths.registry_path().write_text(json.dumps(document), encoding="utf-8")

    assert main(["packages", command]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_unreadable"


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
    patched = (
        paths.checkout_dir("torch-vibevoice", "vibevoice-asr-7b")
        / "vibevoice"
        / "modular"
        / "modeling_vibevoice_asr.py"
    )
    patched.write_text("original\n")
    assert main(["packages", "verify"]) == 3
    codes = {item["code"] for item in json.loads(capsys.readouterr().out)["failed"]}
    assert "package_integrity_failed" in codes


def test_the_patch_ships_inside_the_package() -> None:
    """It is applied at pull time on a user's machine, so it cannot live in model_tests/."""
    patch = env.HERE / "patches" / "vibevoice-logits-to-keep.patch"
    assert patch.is_file()
    assert "modeling_vibevoice_asr.py" in patch.read_text()


def test_fluidaudio_patch_applies_to_exact_pinned_source_and_forces_offline_models(
    tmp_path: Path,
) -> None:
    """Exercise the real patch; the provisioning fake cannot prove Swift semantics."""
    repository = Path(__file__).resolve().parents[1]
    fixture = repository / "tests/fixtures/fluidaudio/ProcessCommand.swift"
    assert pkg.sha256_file(fixture) == (
        "2a90c1f8848b21a89a18361458ac3c58fc785fb110d93e706cc2af9b0f01dc41"
    ), "fixture drifted from FluidAudio 19600a485baa4998812e4654b70d2bab8f2c9949"
    target = tmp_path / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
    target.parent.mkdir(parents=True)
    shutil.copyfile(fixture, target)
    patch = env.HERE / "patches/fluidaudio-pinned-model-dir.patch"

    check = subprocess.run(
        ["git", "apply", "--check", str(patch)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True)

    assert pkg.sha256_file(target) == (
        "cba176cb6a612f347a8fcc72be2af9ce766246a781338a27e5e668c862f9e683"
    )
    patched = target.read_text(encoding="utf-8")
    model_argument = patched.index("args.modelDirectory")
    offline_gate = patched.index("ModelHub.offlineMode = true")
    model_load = patched.index("OfflineDiarizerModels.load(from: modelDir)")
    assert model_argument < offline_gate < model_load
    assert "OfflineDiarizerModels.defaultModelsDirectory()" not in patched
