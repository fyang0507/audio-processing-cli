from __future__ import annotations

from audio_cli.transcribe.execution import runtime as orchestrator_runtime
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    InputMetadata,
    Path,
    StageOutcome,
    StageTransport,
    audio_paths,
    build_plan,
    full_registry,
    orchestrator,
    pytest,
    refusals,
    registry,
    request,
    resolve_request,
    shutil,
    wave,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)


def test_preflight_never_launches_fluidaudio_from_external_receipt_path(
    tmp_path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 16_000, 1)
    document = full_registry(tmp_path)
    marker = tmp_path / "launched"
    external = tmp_path / "external-fluid"
    product = external / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    product.chmod(0o755)
    document["packages"]["fluidaudio"]["materialized"]["path"] = str(external)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("external product reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=NoDecode(),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "built_checkout_path" in {item["check"] for item in raised.value.payload["failed"]}
    assert not marker.exists()


def test_preflight_rejects_fluidaudio_product_symlink_escape(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 16_000, 1)
    document = full_registry(tmp_path)
    checkout = audio_paths.checkout_dir("swift", "fluidaudio")
    product = checkout / ".build" / "release" / "fluidaudiocli"
    marker = tmp_path / "launched"
    external = tmp_path / "external-product"
    external.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    external.chmod(0o755)
    product.unlink()
    product.symlink_to(external)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("symlinked product reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=NoDecode(),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "built_product_executable" in {item["check"] for item in raised.value.payload["failed"]}
    assert not marker.exists()


def test_preflight_rejects_same_revision_basename_outside_hub_cache_index(
    tmp_path,
) -> None:
    resolved, metadata, _run_range = request(tmp_path)
    document = registry(tmp_path)
    materialized = document["packages"]["qwen3-asr-0.6b-8bit"]["materialized"]
    snapshot = Path(materialized["path"])
    external = tmp_path / "attacker-controlled" / snapshot.name
    shutil.copytree(snapshot, external)
    materialized["path"] = str(external)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("external snapshot reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=NoDecode(),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "hub_snapshot_integrity" in {item["check"] for item in raised.value.payload["failed"]}


def test_preflight_uses_manifest_pin_not_mutable_hub_receipt_revision(
    tmp_path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    document["packages"]["qwen3-asr-0.6b-8bit"]["materialized"]["revision"] = "tampered-history"
    plan = build_plan(
        resolved,
        metadata,
        provisioned_packages={"qwen3-asr-0.6b-8bit"},
    )

    selected = orchestrator.preflight(
        plan,
        document,
        python_runtime_probe=lambda _path: True,
    )

    assert selected["qwen3-asr-0.6b-8bit"] is document["packages"]["qwen3-asr-0.6b-8bit"]


def test_preflight_rejects_a_hub_file_target_outside_its_repository_cache(
    tmp_path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    materialized = document["packages"]["qwen3-asr-0.6b-8bit"]["materialized"]
    snapshot = Path(materialized["path"])
    external = tmp_path / "external-model.safetensors"
    external.write_bytes(b"x" * 4096)
    (snapshot / "model.safetensors").symlink_to(external)
    materialized["bytes"] = 4096
    plan = build_plan(
        resolved,
        metadata,
        provisioned_packages={"qwen3-asr-0.6b-8bit"},
    )

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(
            plan,
            document,
            python_runtime_probe=lambda _path: True,
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    hub_failure = next(
        item for item in raised.value.payload["failed"] if item["check"] == "hub_snapshot_integrity"
    )
    assert "outside its repository cache" in " ".join(hub_failure["actual"])


def test_preflight_live_probes_managed_python_before_decode(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    interpreter = tmp_path / "runtime" / "envs" / "mlx" / "bin" / "python"
    interpreter.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    interpreter.chmod(0o755)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("nonlaunching interpreter reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=NoDecode(),
        )
    assert raised.value.exit_code == 3
    assert raised.value.payload["code"] == "package_integrity_failed"
    assert raised.value.payload["failed"] == [
        {
            "package": "qwen3-asr-0.6b-8bit",
            "check": "environment_mlx_python_runs",
            "expected": True,
            "actual": False,
        }
    ]


def test_preflight_refuses_a_symlinked_environment_root_before_decode(
    tmp_path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    managed = audio_paths.env_dir("mlx")
    external = tmp_path / "external-mlx"
    managed.rename(external)
    managed.symlink_to(external, target_is_directory=True)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("redirected environment reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=NoDecode(),
        )

    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [
        {
            "package": "qwen3-asr-0.6b-8bit",
            "check": "environment_mlx_managed_root",
            "expected": str(managed),
            "actual": f"managed environment path is a symlink: {managed}",
        }
    ]


def test_preflight_refuses_a_symlinked_provisioning_root_before_decode(
    tmp_path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    root = audio_paths.root()
    external = tmp_path / "external-runtime-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("symlinked provisioning root reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=NoDecode(),
        )

    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [
        {
            "package": "qwen3-asr-0.6b-8bit",
            "check": "environment_mlx_managed_root",
            "expected": str(audio_paths.env_dir("mlx")),
            "actual": f"provisioning root is a symlink: {root}",
        }
    ]


def test_preflight_refuses_an_in_root_symlinked_environments_parent_before_decode(
    tmp_path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    envs = audio_paths.envs_dir()
    alternate = audio_paths.root() / "alternate-envs"
    envs.rename(alternate)
    envs.symlink_to(alternate, target_is_directory=True)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("redirected environments parent reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=NoDecode(),
        )

    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [
        {
            "package": "qwen3-asr-0.6b-8bit",
            "check": "environment_mlx_managed_root",
            "expected": str(audio_paths.env_dir("mlx")),
            "actual": f"managed environment parent is a symlink: {envs}",
        }
    ]


def test_python_stage_rechecks_environment_after_decode_before_runner_launch(
    tmp_path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)

    class NoLaunchRunner:
        def run(self, command):
            raise AssertionError(f"redirected interpreter was launched: {command}")

    class MutatingTransport(StageTransport):
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            managed = audio_paths.env_dir("mlx")
            external = tmp_path / "external-mlx"
            managed.rename(external)
            managed.symlink_to(external, target_is_directory=True)
            return StageOutcome("decode", "ffmpeg", {}, 1.0)

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=MutatingTransport(NoLaunchRunner()),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "environment_mlx_managed_root" in {
        item["check"] for item in raised.value.payload["failed"]
    }


def test_preflight_does_not_probe_below_a_redirected_environments_parent(
    tmp_path,
    monkeypatch,
) -> None:
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=tmp_path / "source.wav",
        wants=("diarization",),
    )
    metadata = InputMetadata("source.wav", 2.0, "wav", 16_000, 1)
    document = full_registry(tmp_path)
    plan = build_plan(
        resolved,
        metadata,
        provisioned_packages=set(document["packages"]),
    )
    envs = audio_paths.envs_dir()
    alternate = audio_paths.root() / "alternate-envs"
    envs.rename(alternate)
    envs.symlink_to(alternate, target_is_directory=True)

    def fail_probe(path: Path) -> bool:
        raise AssertionError(f"preflight launched redirected runtime {path}")

    def fail_inspection(path: Path) -> orchestrator_runtime._CheckoutState:
        raise AssertionError(f"preflight inspected redirected checkout {path}")

    monkeypatch.setattr(orchestrator_runtime, "_inspect_checkout", fail_inspection)

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(
            plan,
            document,
            built_product_probe=fail_probe,
            python_runtime_probe=fail_probe,
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert {item["check"] for item in raised.value.payload["failed"]} == {
        "environment_mlx_managed_root",
        "environment_swift_managed_root",
    }
