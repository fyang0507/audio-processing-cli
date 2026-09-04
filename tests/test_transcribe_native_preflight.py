from __future__ import annotations

from transcribe_native_test_support import (
    InputMetadata,
    Path,
    _ready_multi_package,
    _runtime,
    build_plan,
    env,
    hash_file,
    orchestrator,
    pkg,
    pytest,
    refusals,
    resolve_request,
    shutil,
)
from transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)

from audio_cli.transcribe import _orchestrator_runtime as orchestrator_runtime


@pytest.mark.parametrize(
    ("stack_id", "environment", "mutation", "expected_check"),
    [
        ("firered", "torch-firered", "missing", "checkout_directory_exists"),
        ("vibevoice", "torch-vibevoice", "commit", "checkout_commit"),
        ("vibevoice", "torch-vibevoice", "patch", "checkout_patch_integrity"),
        ("firered", "torch-firered", "head", "checkout_head"),
        ("firered", "torch-firered", "head_prefix", "checkout_head"),
        ("firered", "torch-firered", "tracked", "checkout_tracked_changes"),
        ("vibevoice", "torch-vibevoice", "untracked", "checkout_untracked_files"),
        ("vibevoice", "torch-vibevoice", "not_git", "checkout_git_state"),
        ("vibevoice", "torch-vibevoice", "tokenizer_missing", "hub_snapshot_integrity"),
        ("vibevoice", "torch-vibevoice", "snapshot_file", "hub_snapshot_integrity"),
        ("vibevoice", "torch-vibevoice", "snapshot_revision", "hub_snapshot_integrity"),
    ],
)
def test_native_preflight_rejects_untrusted_source_checkout_before_decode(
    tmp_path: Path,
    monkeypatch,
    stack_id: str,
    environment: str,
    mutation: str,
    expected_check: str,
) -> None:
    _runtime(tmp_path, monkeypatch, environment)
    package_id = "firered-asr2s" if stack_id == "firered" else "vibevoice-asr-7b"
    entry = _ready_multi_package(tmp_path, package_id)
    materialized = entry["materialized"]
    if mutation == "missing":
        checkout = Path(materialized["checkout"])
        checkout.rename(checkout.with_name("moved-checkout"))
    elif mutation == "commit":
        materialized["checkout_commit"] = "wrong"
    elif mutation == "commit_prefix":
        materialized["checkout_commit"] = env.packages()[package_id].checkout["commit"]
    elif mutation == "patch":
        patched_name = next(iter(materialized["patched_file_digests"]))
        (Path(materialized["checkout"]) / patched_name).write_text(
            "tampered\n", encoding="utf-8"
        )
    elif mutation in {"tokenizer_missing", "snapshot_file", "snapshot_revision"}:
        tokenizer = Path(
            materialized["paths"]["Qwen/Qwen2.5-7B"]
        )
        if mutation == "tokenizer_missing":
            (tokenizer / "tokenizer.json").unlink()
        else:
            if mutation == "snapshot_file":
                shutil.rmtree(tokenizer)
                tokenizer.write_bytes(b"not a snapshot directory")
            else:
                moved = tmp_path / "attacker-controlled" / tokenizer.name
                shutil.copytree(tokenizer, moved)
                materialized["paths"]["Qwen/Qwen2.5-7B"] = str(moved)

    package = env.packages()[package_id]
    _patches, expected_modified, _expected_digests = pkg.checkout_patch_expectation(package)
    if mutation in {"head", "head_prefix", "tracked", "untracked", "not_git"}:
        if mutation == "not_git":
            def inspect(_checkout: Path) -> orchestrator_runtime._CheckoutState:
                raise ValueError("not a Git checkout")
        else:
            state = orchestrator_runtime._CheckoutState(
                head=(
                    "0" * 40 if mutation == "head"
                    else package.checkout["commit"] if mutation == "head_prefix"
                    else package.checkout["resolved_commit"]
                ),
                modified=(
                    (*expected_modified, "unexpected.py")
                    if mutation == "tracked"
                    else expected_modified
                ),
                untracked=(("rogue.py",) if mutation == "untracked" else ()),
            )

            def inspect(_checkout: Path) -> orchestrator_runtime._CheckoutState:
                return state

        monkeypatch.setattr(orchestrator_runtime, "_inspect_checkout", inspect)

    source = tmp_path / f"{stack_id}.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id=stack_id, input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("preflight reached decode")

    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {environment: {"state": "ready"}},
                "packages": {package_id: entry},
            },
            transport=NoDecode(),
        )
    refusal = raised.value
    assert getattr(refusal, "exit_code") == 3
    assert expected_check in {
        item["check"] for item in refusal.payload["failed"]
    }


def test_native_preflight_accepts_legacy_short_receipt_but_requires_full_live_head(
    tmp_path: Path, monkeypatch,
) -> None:
    environment = "torch-vibevoice"
    package_id = "vibevoice-asr-7b"
    _runtime(tmp_path, monkeypatch, environment)
    entry = _ready_multi_package(tmp_path, package_id)
    package = env.packages()[package_id]
    entry["materialized"]["checkout_commit"] = package.checkout["commit"]
    source = tmp_path / "vibe.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)
    plan = build_plan(request, metadata, provisioned_packages={package_id})

    selected = orchestrator.preflight(
        plan,
        {
            "environments": {environment: {"state": "ready"}},
            "packages": {package_id: entry},
        },
        python_runtime_probe=lambda _path: True,
    )

    assert selected[package_id] is entry


def test_native_preflight_refuses_a_missing_checkout_install_before_decode(
    tmp_path: Path, monkeypatch,
) -> None:
    environment = "torch-vibevoice"
    package_id = "vibevoice-asr-7b"
    _runtime(tmp_path, monkeypatch, environment)
    entry = _ready_multi_package(tmp_path, package_id)
    source = tmp_path / "vibe.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("missing checkout install reached decode")

    monkeypatch.setattr(orchestrator_runtime, "_frozen_packages", lambda _path: {})
    monkeypatch.setattr(
        orchestrator_runtime, "_python_runtime_runs", lambda _path: True
    )
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {environment: {"state": "ready"}},
                "packages": {package_id: entry},
            },
            transport=NoDecode(),
        )

    assert raised.value.exit_code == 3
    assert f"environment_{environment}_checkout_install" in {
        item["check"] for item in raised.value.payload["failed"]
    }


def test_native_preflight_never_inspects_an_external_checkout_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    environment = "torch-vibevoice"
    package_id = "vibevoice-asr-7b"
    _runtime(tmp_path, monkeypatch, environment)
    entry = _ready_multi_package(tmp_path, package_id)
    external = tmp_path / "external-checkout"
    external.mkdir()
    entry["materialized"]["checkout"] = str(external)

    def fail_inspection(path: Path) -> orchestrator_runtime._CheckoutState:
        raise AssertionError(f"preflight inspected external checkout {path}")

    def fail_hash(path: Path) -> str:
        raise AssertionError(f"preflight hashed external checkout file {path}")

    monkeypatch.setattr(orchestrator_runtime, "_inspect_checkout", fail_inspection)
    monkeypatch.setattr(orchestrator_runtime, "_checkout_file_digest", fail_hash)
    source = tmp_path / "vibe.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)
    plan = build_plan(request, metadata, provisioned_packages={package_id})

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(
            plan,
            {
                "environments": {environment: {"state": "ready"}},
                "packages": {package_id: entry},
            },
            python_runtime_probe=lambda _path: True,
        )

    assert raised.value.exit_code == 3
    assert "checkout_directory_exists" in {
        item["check"] for item in raised.value.payload["failed"]
    }


def test_native_preflight_uses_manifest_pins_not_mutable_hub_receipt_revisions(
    tmp_path: Path, monkeypatch,
) -> None:
    environment = "torch-vibevoice"
    package_id = "vibevoice-asr-7b"
    _runtime(tmp_path, monkeypatch, environment)
    entry = _ready_multi_package(tmp_path, package_id)
    entry["materialized"]["revisions"] = ["tampered-history"]
    source = tmp_path / "vibe.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)
    plan = build_plan(request, metadata, provisioned_packages={package_id})

    selected = orchestrator.preflight(
        plan,
        {
            "environments": {environment: {"state": "ready"}},
            "packages": {package_id: entry},
        },
        python_runtime_probe=lambda _path: True,
    )

    assert selected[package_id] is entry


@pytest.mark.parametrize("package_id", ["vibevoice-asr-7b", "firered-asr2s"])
def test_native_preflight_derives_exact_patch_state_independently_of_receipt(
    tmp_path: Path, monkeypatch, package_id: str,
) -> None:
    package = env.packages()[package_id]
    environment = package.environment
    _runtime(tmp_path, monkeypatch, environment)
    entry = _ready_multi_package(tmp_path, package_id)
    materialized = entry["materialized"]
    checkout = Path(materialized["checkout"])
    _patches, names, _expected_digests = pkg.checkout_patch_expectation(package)
    if names:
        legitimate = checkout / names[0]
        legitimate.write_text("attacker-controlled\n", encoding="utf-8")
        materialized["patched_file_digests"][names[0]] = hash_file(legitimate)
    evil = checkout / "evil.py"
    evil.write_text("attacker-controlled\n", encoding="utf-8")
    materialized.setdefault("patches_applied", []).append("made-up.patch")
    materialized.setdefault("patched_file_digests", {})["evil.py"] = hash_file(evil)

    monkeypatch.setattr(
        orchestrator_runtime,
        "_inspect_checkout",
        lambda _checkout: orchestrator_runtime._CheckoutState(
            head=package.checkout["resolved_commit"],
            modified=(*names, "evil.py"),
            untracked=(),
        ),
    )
    source = tmp_path / f"{package_id}.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice" if package_id.startswith("vibevoice") else "firered",
        input_path=source,
        wants=(),
    )
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("forged patch receipt reached decode")

    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {environment: {"state": "ready"}},
                "packages": {package_id: entry},
            },
            transport=NoDecode(),
        )
    assert getattr(raised.value, "exit_code") == 3
    checks = {item["check"] for item in raised.value.payload["failed"]}
    expected = {
        "checkout_patch_applied",
        "patched_file_digests",
        "checkout_tracked_changes",
    }
    if names:
        expected.add("checkout_patch_integrity")
    assert expected <= checks


def test_native_preflight_hashes_existing_auto_fetch_artifact_before_decode(
    tmp_path: Path, monkeypatch,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    monkeypatch.delenv("AUDIO_PROCESSING_VAD_MODEL", raising=False)
    silero_source = env.packages()["silero-vad"].source
    artifact = tmp_path / "runtime" / "models" / silero_source["filename"]
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"corrupt")
    source = tmp_path / "vibe.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice", input_path=source, wants="vad"
    )
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("preflight reached decode")

    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-vibevoice": {"state": "ready"}},
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(
                        tmp_path, "vibevoice-asr-7b"
                    ),
                },
            },
            transport=NoDecode(),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert raised.value.payload["failed"] == [{
        "package": "silero-vad",
        "check": "url_artifact_sha256",
        "expected": silero_source["sha256"],
        "actual": "missing, not a file, or digest changed",
    }]


def test_native_preflight_hashes_explicit_silero_override_before_decode(
    tmp_path: Path, monkeypatch,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    override = tmp_path / "arbitrary.onnx"
    override.write_bytes(b"arbitrary model")
    monkeypatch.setenv("AUDIO_PROCESSING_VAD_MODEL", str(override))
    source = tmp_path / "vibe.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice", input_path=source, wants="vad"
    )

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("preflight reached decode")

    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request,
            InputMetadata(str(source), 1.0, "wav", 16_000, 1),
            registry={
                "environments": {"torch-vibevoice": {"state": "ready"}},
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(
                        tmp_path, "vibevoice-asr-7b"
                    ),
                },
            },
            transport=NoDecode(),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert raised.value.payload["failed"][0]["check"] == "url_artifact_sha256"
