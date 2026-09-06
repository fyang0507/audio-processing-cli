from __future__ import annotations

from pathlib import Path

from audio_cli.transcribe.execution import preflight as orchestrator_preflight
from audio_cli.transcribe.execution import runtime as orchestrator_runtime
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    FakeVad,
    FullFakeTransport,
    InputMetadata,
    SileroOnnxVad,
    audio_paths,
    build_plan,
    env,
    full_registry,
    inspect,
    orchestrator,
    pytest,
    refusals,
    registry,
    request,
    resolve_request,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)


def test_preflight_refuses_missing_package_before_transport_exists(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(resolved, metadata, registry={"packages": {}})
    assert raised.value.exit_code == 3
    assert raised.value.payload["missing"] == [
        {
            "package": "qwen3-asr-0.6b-8bit",
            "kind": "weights",
            "bytes": 1010773761,
        }
    ]


def test_preflight_refuses_missing_runtime_before_decode(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    document["environments"]["mlx"]["state"] = "absent"
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(resolved, metadata, registry=document)
    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [
        {
            "package": "qwen3-asr-0.6b-8bit",
            "check": "environment_mlx_ready",
            "expected": "ready",
            "actual": "absent",
        }
    ]


def test_preflight_refuses_missing_swift_product_before_decode(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    document = full_registry(tmp_path)
    product = (
        audio_paths.checkout_dir("swift", "fluidaudio") / ".build" / "release" / "fluidaudiocli"
    )
    product.unlink()
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(resolved, metadata, registry=document)
    assert raised.value.exit_code == 3
    assert any(
        item["check"] == "built_product_executable" for item in raised.value.payload["failed"]
    )


def test_vad_plan_config_is_exactly_the_detector_keyword_surface(tmp_path) -> None:
    source = tmp_path / "source.wav"
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("vad",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 16_000, 1)
    plan = build_plan(
        resolved,
        metadata,
        provisioned_packages={"qwen3-asr-0.6b-8bit"},
    )
    parameters = inspect.signature(SileroOnnxVad.detect).parameters
    keyword_only = {
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert set(plan.roles["vad"]["config"]) == keyword_only


def test_qwen_add_ons_share_full_timeline_and_capability_gated_output(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization", "overlapped_speech", "word_timestamps", "vad"),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    fake = FullFakeTransport()
    monkeypatch.setattr(orchestrator_runtime, "_self_peak_rss", lambda: 120)
    product = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=fake,
        vad_detector=FakeVad(),
        run_range=orchestrator.parse_range("0.5:"),
    )
    assert fake.diarizer_calls == 1
    assert fake.overlap is True
    assert fake.units == [
        {"unit_id": "turn_0", "speaker": "S1", "start": 0.0, "end": 0.8},
        {"unit_id": "turn_1", "speaker": "S2", "start": 1.0, "end": 1.8},
    ]
    assert product.payload["turns"] == [
        {"turn_id": "turn_0", "speaker": "S1", "start": 0.0, "end": 0.8},
        {"turn_id": "turn_1", "speaker": "S2", "start": 1.0, "end": 1.8},
    ]
    assert product.payload["overlapped_speech"] == [
        {"overlap_id": "overlap_0", "start": 0.8, "end": 1.0},
    ]
    assert product.payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "overlap",
            "start": 0.8,
            "end": 1.0,
        }
    ]
    assert product.payload["vad_regions"] == [{"start": 0.1, "end": 1.9}]
    assert product.payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.5, 2.0],
        "selected_unit_scope": [0.0, 1.8],
    }
    assert all("start" not in item and "end" not in item for item in product.payload["segments"])
    assert product.payload["segments"][0]["words"] == [
        {
            "word_id": "w_0",
            "text": "Hello",
            "start": 0.0,
            "end": 0.8,
        }
    ]
    observed = product.payload["provenance"]["observed"]
    assert observed["peak_rss_bytes"] == 200
    assert observed["peak_rss_bytes_by_stage"] == {
        "decode": 50,
        "diarizer": 80,
        "vad": 120,
        "asr": 100,
        "aligner": 200,
    }
    assert observed["peak_mps_live_bytes"] == 150


def test_preflight_uses_bound_live_product_not_legacy_boolean_receipts(tmp_path) -> None:
    """Boolean pull history cannot overrule the bound live executable."""
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=tmp_path / "source.wav",
        wants=("diarization",),
    )
    metadata = InputMetadata("source.wav", 2.0, "wav", 16_000, 1)
    document = full_registry(tmp_path)
    document["packages"]["fluidaudio"]["materialized"]["built"] = False
    document["packages"]["fluidaudio"]["materialized"]["product_runs"] = False
    plan = build_plan(
        resolved,
        metadata,
        provisioned_packages=set(document["packages"]),
    )
    selected = orchestrator.preflight(plan, document)
    assert "fluidaudio" in selected


def test_preflight_rejects_replaced_fluidaudio_product_bytes(tmp_path) -> None:
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=tmp_path / "source.wav",
        wants=("diarization",),
    )
    metadata = InputMetadata("source.wav", 2.0, "wav", 16_000, 1)
    document = full_registry(tmp_path)
    product = next(
        audio_paths.checkout_dir("swift", "fluidaudio").glob(".build/**/release/fluidaudiocli")
    )
    product.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    product.chmod(0o755)
    plan = build_plan(resolved, metadata, provisioned_packages=set(document["packages"]))

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(plan, document)

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "built_product_digest" in {item["check"] for item in raised.value.payload["failed"]}


def test_preflight_refuses_default_silero_through_a_symlinked_models_parent(
    tmp_path: Path,
) -> None:
    package = env.packages()["silero-vad"]
    outside = tmp_path / "outside"
    outside.mkdir()
    model = outside / package.source["filename"]
    model.write_bytes(b"model")
    models = audio_paths.models_dir()
    models.parent.mkdir(parents=True, exist_ok=True)
    models.symlink_to(outside, target_is_directory=True)
    plan = type(
        "Plan",
        (),
        {
            "stack": "qwen-0.6b",
            "packages": ({"package": "silero-vad", "auto_fetch": True},),
        },
    )()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(plan, {"packages": {}, "environments": {}})

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "managed artifact parent is a symlink" in str(raised.value.payload["failed"])


def test_preflight_refuses_missing_default_silero_below_a_symlinked_root(
    tmp_path: Path,
) -> None:
    package = env.packages()["silero-vad"]
    root = audio_paths.root()
    external = tmp_path / "external-runtime-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)
    plan = type(
        "Plan",
        (),
        {
            "stack": "qwen-0.6b",
            "packages": ({"package": "silero-vad", "auto_fetch": True},),
        },
    )()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(plan, {"packages": {}, "environments": {}})

    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [
        {
            "package": "silero-vad",
            "check": "url_artifact_sha256",
            "expected": package.source["sha256"],
            "actual": f"provisioning root is a symlink: {root}",
        }
    ]


def test_preflight_types_an_unreadable_silero_digest_as_package_integrity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = env.packages()["silero-vad"]
    model = audio_paths.models_dir() / package.source["filename"]
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    monkeypatch.setattr(
        orchestrator_preflight,
        "hash_file",
        lambda _path: (_ for _ in ()).throw(PermissionError("unreadable")),
    )
    plan = type(
        "Plan",
        (),
        {
            "stack": "qwen-0.6b",
            "packages": ({"package": "silero-vad", "auto_fetch": True},),
        },
    )()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(plan, {"packages": {}, "environments": {}})

    assert raised.value.exit_code == 3
    assert raised.value.payload["code"] == "package_integrity_failed"
    failure = raised.value.payload["failed"][0]
    assert failure["check"] == "url_artifact_sha256"
    assert "could not hash managed artifact" in failure["actual"]


def test_preflight_rejects_changed_product_digest_before_decode(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 16_000, 1)
    document = full_registry(tmp_path)
    product = (
        audio_paths.checkout_dir("swift", "fluidaudio") / ".build" / "release" / "fluidaudiocli"
    )
    product.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("nonlaunching product reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=document,
            transport=NoDecode(),
        )
    assert raised.value.exit_code == 3
    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "built_product_digest" in {item["check"] for item in raised.value.payload["failed"]}
