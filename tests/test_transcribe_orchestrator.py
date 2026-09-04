from __future__ import annotations

import inspect
import json
import os
import shlex
import shutil
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli import paths as audio_paths
from audio_cli.export import UnsafeOutputError, export_documents
from audio_cli.transcribe import orchestrator, refusals
from audio_cli.transcribe.catalog import InputMetadata
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.transport import StageOutcome, StageTransport
from audio_cli.vad import SileroOnnxVad

REVISION = "89e96d92ba34aca20b3e29fb10cc284097d1219f"
ALIGNER_REVISION = "0e1a68e91d815300c7c9754b2a7639378b23db15"
FLUID_REVISION = "19600a485baa4998812e4654b70d2bab8f2c9949"
SPEAKER_REVISION = "1ed7a662fdc7109e36d822db793ee6eebdaf8594"


@pytest.fixture(autouse=True)
def provisioned_runtime_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "runtime"
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(root))
    interpreter = root / "envs" / "mlx" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)

    def snapshot_index() -> dict[tuple[str, str], Path]:
        found = {}
        for package in env.packages().values():
            source = package.source
            repositories = (
                source["repos"] if source["type"] == "huggingface_multi"
                else [source] if source["type"] == "huggingface"
                else []
            )
            for repository in repositories:
                found[(repository["repo"], repository["revision"])] = (
                    tmp_path / "hub" / f"models--{repository['repo'].replace('/', '--')}"
                    / "snapshots" / repository["revision"]
                )
        return found

    monkeypatch.setattr(pkg, "_hub_snapshot_index", snapshot_index)
    monkeypatch.setattr(
        orchestrator,
        "_inspect_checkout",
        lambda _checkout: orchestrator._CheckoutState(
            head=FLUID_REVISION,
            modified=pkg.checkout_patch_expectation(
                env.packages()["fluidaudio"]
            )[1],
            untracked=(),
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_checkout_file_digest",
        lambda path: (
            next(iter(env.packages()["fluidaudio"].source["patched_file_sha256"].values()))
            if path.name == "ProcessCommand.swift" and path.read_bytes() == b"patched\n"
            else pkg.sha256_file(path)
        ),
    )


def request(tmp_path: Path, *, wants=(), run_range=None):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=wants,
    )
    metadata = InputMetadata(str(source), 361.0, "wav", 48_000, 2)
    return resolved, metadata, run_range


def registry(tmp_path: Path) -> dict:
    source = env.packages()["qwen3-asr-0.6b-8bit"].source
    model = (
        tmp_path / "hub" / f"models--{source['repo'].replace('/', '--')}"
        / "snapshots" / REVISION
    )
    model.mkdir(parents=True, exist_ok=True)
    return {"environments": {"mlx": {"state": "ready"}}, "packages": {
        "qwen3-asr-0.6b-8bit": {
        "state": "ready",
        "materialized": {"path": str(model), "revision": REVISION, "bytes": 0},
    }}}


class FakeTransport:
    def __init__(self, *, partial: bool = False) -> None:
        self.partial = partial
        self.units = []

    def decode(self, source, target):
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * (361 * 16_000))
        return StageOutcome("decode", "ffmpeg", {}, 2.0, peak_rss_bytes=50)

    def qwen(self, *, units, **kwargs):
        self.units = units
        raw = []
        for index, item in enumerate(units):
            raw.append({
                "unit_id": item["unit_id"],
                "processed": not self.partial or index == 0,
                "text": "language English<asr_text>Hello.",
            })
        return StageOutcome(
            "asr", "qwen3-asr-0.6b-8bit", {"units": raw}, 3.0,
            returncode=4 if self.partial else 0,
            peak_rss_bytes=100,
        )


class FullFakeTransport(FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.diarizer_calls = 0
        self.overlap = None

    def decode(self, source, target):
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 32_000)
        return StageOutcome("decode", "ffmpeg", {}, 1.0, peak_rss_bytes=50)

    def diarize(self, *, overlap, **kwargs):
        self.diarizer_calls += 1
        self.overlap = overlap
        return StageOutcome("diarizer", "fluidaudio", {"segments": [
            {"startTimeSeconds": 0.0, "endTimeSeconds": 1.0, "speakerId": "S1",
             "embedding": [0.0] * 256},
            {"startTimeSeconds": 0.8, "endTimeSeconds": 1.8, "speakerId": "S2",
             "embedding": [1.0] * 256},
        ]}, 2.0, peak_rss_bytes=80)

    def align(self, *, segments, **kwargs):
        return StageOutcome("aligner", "qwen3-forcedaligner", {"segments": [{
            "unit_id": item["unit_id"],
            "words": [{"text": "Hello", "start": item["start"], "end": item["end"]}],
        } for item in segments]}, 4.0, peak_rss_bytes=200, peak_mps_live_bytes=150)


class FakeVad:
    def detect(self, samples, rate, **config):
        assert rate == 16_000
        assert config["min_silence_ms"] == 300
        return [{"start": 0.1, "end": 1.9, "mean_probability": 0.8}]


def full_registry(tmp_path: Path) -> dict:
    packages = {}
    for identifier, revision in (
        ("qwen3-asr-0.6b-8bit", REVISION),
        ("qwen3-forcedaligner", ALIGNER_REVISION),
        ("speaker-diarization-coreml", SPEAKER_REVISION),
    ):
        source = env.packages()[identifier].source
        target = (
            tmp_path / "hub" / f"models--{source['repo'].replace('/', '--')}"
            / "snapshots" / revision
        )
        target.mkdir(parents=True, exist_ok=True)
        for pattern in source.get("allow_patterns", ()):
            marker = target / (
                f"{pattern[:-3]}/model.mil" if pattern.endswith("/**") else pattern
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_bytes(b"")
        packages[identifier] = {
            "state": "ready",
            "materialized": {
                "path": str(target), "revision": revision, "bytes": 0,
            },
        }
    fluid = audio_paths.checkout_dir("swift", "fluidaudio")
    fluid.mkdir(parents=True, exist_ok=True)
    product = fluid / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_text("#!/bin/sh\n", encoding="utf-8")
    product.chmod(0o755)
    patched = fluid / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
    patched.parent.mkdir(parents=True, exist_ok=True)
    patched.write_bytes(b"patched\n")
    patch_name, patch_paths, patch_digests = pkg.checkout_patch_expectation(
        env.packages()["fluidaudio"]
    )
    packages["fluidaudio"] = {
        "state": "ready",
        "materialized": {
            "path": str(fluid), "revision": FLUID_REVISION,
            "built": True, "product_runs": True,
            "product_path": product.relative_to(fluid).as_posix(),
            "product_sha256": pkg.sha256_file(product),
            "patches_applied": list(patch_name),
            "patched_file_digests": patch_digests,
        },
    }
    return {
        "environments": {"mlx": {"state": "ready"}, "swift": {"state": "ready"}},
        "packages": packages,
    }


def test_qwen_floor_run_uses_fixed_units_and_never_publishes_container_bounds(tmp_path) -> None:
    resolved, metadata, run_range = request(tmp_path)
    fake = FakeTransport()
    product = orchestrator.run(
        resolved, metadata, registry=registry(tmp_path), transport=fake,
        run_range=run_range,
    )
    assert [(item["start"], item["end"]) for item in fake.units] == [
        (0.0, 180.0), (180.0, 360.0), (360.0, 361.0),
    ]
    assert product.payload["segments"] == [
        {"segment_id": "seg_0", "text": "Hello."},
        {"segment_id": "seg_1", "text": "Hello."},
        {"segment_id": "seg_2", "text": "Hello."},
    ]
    assert product.payload["provenance"]["observed"]["peak_rss_bytes"] == 100
    assert product.payload["provenance"]["observed"]["total_wall_seconds"] == 5.0


def test_run_captures_source_identity_before_decode_and_export_protects_it(
    tmp_path,
) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    link = tmp_path / "input.wav"
    first.write_bytes(b"first-source")
    second.write_bytes(b"second-source")
    link.symlink_to(first)
    request_ = resolve_request(stack_id="qwen-0.6b", input_path=link, wants=())
    metadata = InputMetadata(str(link), 2.0, "wav", 16_000, 1)

    class RetargetingTransport(FullFakeTransport):
        def decode(self, source, target):
            assert Path(source) == first.resolve()
            assert Path(source).read_bytes() == b"first-source"
            link.unlink()
            link.symlink_to(second)
            return super().decode(source, target)

    product = orchestrator.run(
        request_,
        metadata,
        registry=registry(tmp_path),
        transport=RetargetingTransport(),
    )

    assert product.payload["source"]["path"] == str(first.resolve())
    transcript = tmp_path / "result.json"
    transcript.write_text(json.dumps(product.payload), encoding="utf-8")
    with pytest.raises(UnsafeOutputError):
        export_documents([transcript], "txt", output=first, force=True)
    assert first.read_bytes() == b"first-source"


def test_partial_qwen_run_writes_conforming_result_and_exact_resume_ledger(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "requested.md"
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path),
            transport=FakeTransport(partial=True), output=output,
        )
    assert raised.value.exit_code == 4
    assert raised.value.payload["code"] == "run_incomplete"
    partial = tmp_path / "requested.partial.json"
    assert raised.value.payload["output"] == str(partial)
    assert not output.exists()
    payload = json.loads(partial.read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert payload["coverage"] == raised.value.payload["coverage"]
    assert payload["coverage"]["covered_intervals"] == [[0.0, 180.0]]
    assert payload["coverage"]["missing_intervals"] == [[180.0, 361.0]]
    assert payload["coverage"]["scope_intervals"] == [[0.0, 361.0]]
    assert "--range 180.0:" in raised.value.payload["fix"]


@pytest.mark.parametrize("malformation", ["absent_units", "missing_row", "non_bool"])
def test_qwen_malformed_partial_ledger_is_backend_failure_without_publication(
    tmp_path: Path,
    malformation: str,
) -> None:
    class MalformedLedgerTransport(FakeTransport):
        def qwen(self, *, units, **kwargs):
            if malformation == "absent_units":
                payload = {}
            else:
                rows = [{
                    "unit_id": item["unit_id"],
                    "processed": index == 0,
                    "text": "language English<asr_text>Hello.",
                } for index, item in enumerate(units)]
                if malformation == "missing_row":
                    rows.pop()
                else:
                    rows[0]["processed"] = 1
                payload = {"units": rows}
            return StageOutcome(
                "asr",
                "qwen3-asr-0.6b-8bit",
                payload,
                3.0,
                returncode=4,
            )

    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "malformed.json"
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=MalformedLedgerTransport(),
            output=output,
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["backend"] == "qwen3-asr-0.6b-8bit"
    assert not output.exists()
    assert not (tmp_path / "malformed.partial.json").exists()


def test_repeated_incomplete_resumes_never_overwrite_an_earlier_partial(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as first:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
            output=tmp_path / "meeting.timed.json",
        )
    first_partial = Path(first.value.payload["output"])
    first_bytes = first_partial.read_bytes()
    assert first_partial.name == "meeting.timed.partial.json"

    with pytest.raises(refusals.Refusal) as second:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
            output=tmp_path / "meeting.timed.rest.json",
            run_range=orchestrator.parse_range("180:"),
        )
    second_partial = Path(second.value.payload["output"])
    assert second_partial.name == "meeting.timed.rest.partial.json"
    assert first_partial.read_bytes() == first_bytes
    assert first_partial.is_file() and second_partial.is_file()


def test_repeated_implicit_partial_outputs_choose_unused_siblings(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as first:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
        )
    first_partial = Path(first.value.payload["output"])
    first_bytes = first_partial.read_bytes()

    with pytest.raises(refusals.Refusal) as second:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
        )
    second_partial = Path(second.value.payload["output"])
    assert first_partial.name == "source.partial.json"
    assert second_partial.name == "source.partial.2.json"
    assert first_partial.read_bytes() == first_bytes
    assert second_partial.is_file()


def test_implicit_partial_skips_a_broken_symlink_destination(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    first_candidate = tmp_path / "source.partial.json"
    first_candidate.symlink_to("missing-partial.json")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
        )

    assert raised.value.payload["code"] == "run_incomplete"
    assert Path(raised.value.payload["output"]).name == "source.partial.2.json"
    assert first_candidate.is_symlink()
    assert first_candidate.readlink() == Path("missing-partial.json")


def test_bounded_partial_resume_preserves_the_requested_end(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
            output=tmp_path / "bounded.json",
            run_range=orchestrator.parse_range("100:300"),
        )
    assert "--range 180.0:300.0" in raised.value.payload["fix"]


def test_resume_builder_defensively_quotes_an_option_like_internal_value(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = replace(
        resolve_request(
            stack_id="qwen-0.6b",
            input_path=source,
            wants=(),
            language="English",
        ),
        language="--stack",
    )
    fix = orchestrator._resume_command(
        resolved,
        {
            "units_completed": 1,
            "covered_intervals": [[0.0, 180.0]],
            "covered_through_seconds": 180.0,
        },
        tmp_path / "source.partial.json",
        None,
    )

    arguments = shlex.split(fix)
    parsed = cli._parser().parse_args(arguments[1:])
    assert parsed.language == "--stack"
    assert parsed.stack == "qwen-0.6b"
    assert parsed.run_range == "180.0:"


def test_zero_completed_units_write_a_partial_without_a_replaying_fix(tmp_path) -> None:
    class ExhaustedTransport(FakeTransport):
        def qwen(self, *, units, **kwargs):
            self.units = units
            return StageOutcome("asr", "qwen3-asr-0.6b-8bit", {"units": [{
                "unit_id": item["unit_id"], "processed": False, "text": "",
            } for item in units]}, 3.0, returncode=4)

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=ExhaustedTransport(),
        )
    assert raised.value.exit_code == 4
    assert raised.value.payload["code"] == "run_incomplete"
    coverage = raised.value.payload["coverage"]
    assert coverage["covered_intervals"] == []
    assert coverage["missing_intervals"] == [[0.0, 361.0]]
    assert coverage["covered_fraction"] == 0.0
    assert coverage["units_completed"] == 0
    assert raised.value.payload["fix"] == (
        "no processing unit completed; --range would repeat the same deterministic work, "
        "so inspect the first unit or backend budget before retrying"
    )
    partial = json.loads(
        Path(raised.value.payload["output"]).read_text(encoding="utf-8")
    )
    assert partial["complete"] is False
    assert partial["segments"] == []


def test_zero_prefix_partial_does_not_claim_diarizer_abstained(tmp_path) -> None:
    class ExhaustedDiarizedTransport(FullFakeTransport):
        def qwen(self, *, units, **kwargs):
            return StageOutcome("asr", "qwen3-asr-0.6b-8bit", {"units": [{
                "unit_id": item["unit_id"], "processed": False, "text": "",
            } for item in units]}, 1.0, returncode=4)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=ExhaustedDiarizedTransport(),
        )
    partial = json.loads(
        Path(raised.value.payload["output"]).read_text(encoding="utf-8")
    )
    assert partial["provenance"]["outcomes"]["diarization"] == "produced"


def test_preflight_refuses_missing_package_before_transport_exists(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(resolved, metadata, registry={"packages": {}})
    assert raised.value.exit_code == 3
    assert raised.value.payload["missing"] == [{
        "package": "qwen3-asr-0.6b-8bit",
        "kind": "weights",
        "bytes": 1010773761,
    }]


def test_preflight_refuses_missing_runtime_before_decode(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    document["environments"]["mlx"]["state"] = "absent"
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(resolved, metadata, registry=document)
    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [{
        "package": "qwen3-asr-0.6b-8bit",
        "check": "environment_mlx_ready",
        "expected": "ready",
        "actual": "absent",
    }]


def test_preflight_refuses_missing_swift_product_before_decode(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    document = full_registry(tmp_path)
    product = (
        audio_paths.checkout_dir("swift", "fluidaudio")
        / ".build" / "release" / "fluidaudiocli"
    )
    product.unlink()
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(resolved, metadata, registry=document)
    assert raised.value.exit_code == 3
    assert any(
        item["check"] == "built_product_executable"
        for item in raised.value.payload["failed"]
    )


def test_vad_plan_config_is_exactly_the_detector_keyword_surface(tmp_path) -> None:
    source = tmp_path / "source.wav"
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("vad",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 16_000, 1)
    plan = build_plan(
        resolved, metadata, provisioned_packages={"qwen3-asr-0.6b-8bit"},
    )
    parameters = inspect.signature(SileroOnnxVad.detect).parameters
    keyword_only = {
        name for name, parameter in parameters.items()
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
    monkeypatch.setattr(orchestrator, "_self_peak_rss", lambda: 120)
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
    assert product.payload["abstentions"] == [{
        "abstention_id": "ab_0", "reason": "overlap", "start": 0.8, "end": 1.0,
    }]
    assert product.payload["vad_regions"] == [{"start": 0.1, "end": 1.9}]
    assert product.payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.5, 2.0],
        "selected_unit_scope": [0.0, 1.8],
    }
    assert all("start" not in item and "end" not in item for item in product.payload["segments"])
    assert product.payload["segments"][0]["words"] == [{
        "word_id": "w_0", "text": "Hello", "start": 0.0, "end": 0.8,
    }]
    observed = product.payload["provenance"]["observed"]
    assert observed["peak_rss_bytes"] == 200
    assert observed["peak_rss_bytes_by_stage"] == {
        "decode": 50, "diarizer": 80, "vad": 120, "asr": 100, "aligner": 200,
    }
    assert observed["peak_mps_live_bytes"] == 150


def test_preflight_uses_bound_live_product_not_legacy_boolean_receipts(tmp_path) -> None:
    """Boolean pull history cannot overrule the bound live executable."""
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=tmp_path / "source.wav", wants=("diarization",),
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
        stack_id="qwen-0.6b", input_path=tmp_path / "source.wav",
        wants=("diarization",),
    )
    metadata = InputMetadata("source.wav", 2.0, "wav", 16_000, 1)
    document = full_registry(tmp_path)
    product = next(
        audio_paths.checkout_dir("swift", "fluidaudio").glob(
            ".build/**/release/fluidaudiocli"
        )
    )
    product.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    product.chmod(0o755)
    plan = build_plan(
        resolved, metadata, provisioned_packages=set(document["packages"])
    )

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(plan, document)

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "built_product_digest" in {
        item["check"] for item in raised.value.payload["failed"]
    }


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
    plan = type("Plan", (), {
        "stack": "qwen-0.6b",
        "packages": ({"package": "silero-vad", "auto_fetch": True},),
    })()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(plan, {"packages": {}, "environments": {}})

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "managed artifact parent is a symlink" in str(
        raised.value.payload["failed"]
    )


def test_preflight_refuses_missing_default_silero_below_a_symlinked_root(
    tmp_path: Path,
) -> None:
    package = env.packages()["silero-vad"]
    root = audio_paths.root()
    external = tmp_path / "external-runtime-root"
    root.rename(external)
    root.symlink_to(external, target_is_directory=True)
    plan = type("Plan", (), {
        "stack": "qwen-0.6b",
        "packages": ({"package": "silero-vad", "auto_fetch": True},),
    })()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.preflight(plan, {"packages": {}, "environments": {}})

    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [{
        "package": "silero-vad",
        "check": "url_artifact_sha256",
        "expected": package.source["sha256"],
        "actual": f"provisioning root is a symlink: {root}",
    }]


def test_preflight_types_an_unreadable_silero_digest_as_package_integrity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = env.packages()["silero-vad"]
    model = audio_paths.models_dir() / package.source["filename"]
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    monkeypatch.setattr(
        orchestrator,
        "hash_file",
        lambda _path: (_ for _ in ()).throw(PermissionError("unreadable")),
    )
    plan = type("Plan", (), {
        "stack": "qwen-0.6b",
        "packages": ({"package": "silero-vad", "auto_fetch": True},),
    })()

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
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 16_000, 1)
    document = full_registry(tmp_path)
    product = (
        audio_paths.checkout_dir("swift", "fluidaudio")
        / ".build" / "release" / "fluidaudiocli"
    )
    product.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("nonlaunching product reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved, metadata, registry=document, transport=NoDecode(),
        )
    assert raised.value.exit_code == 3
    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "built_product_digest" in {
        item["check"] for item in raised.value.payload["failed"]
    }


def test_preflight_never_launches_fluidaudio_from_external_receipt_path(
    tmp_path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
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
            resolved, metadata, registry=document, transport=NoDecode(),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "built_checkout_path" in {
        item["check"] for item in raised.value.payload["failed"]
    }
    assert not marker.exists()


def test_preflight_rejects_fluidaudio_product_symlink_escape(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
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
            resolved, metadata, registry=document, transport=NoDecode(),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "built_product_executable" in {
        item["check"] for item in raised.value.payload["failed"]
    }
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
            resolved, metadata, registry=document, transport=NoDecode(),
        )

    assert raised.value.payload["code"] == "package_integrity_failed"
    assert "hub_snapshot_integrity" in {
        item["check"] for item in raised.value.payload["failed"]
    }


def test_preflight_uses_manifest_pin_not_mutable_hub_receipt_revision(
    tmp_path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    document["packages"]["qwen3-asr-0.6b-8bit"]["materialized"][
        "revision"
    ] = "tampered-history"
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

    assert selected["qwen3-asr-0.6b-8bit"] is document["packages"][
        "qwen3-asr-0.6b-8bit"
    ]


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
        item for item in raised.value.payload["failed"]
        if item["check"] == "hub_snapshot_integrity"
    )
    assert "outside its repository cache" in " ".join(hub_failure["actual"])


def test_preflight_live_probes_managed_python_before_decode(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    document = registry(tmp_path)
    interpreter = (
        tmp_path / "runtime" / "envs" / "mlx" / "bin" / "python"
    )
    interpreter.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    interpreter.chmod(0o755)

    class NoDecode:
        def decode(self, *_args, **_kwargs):
            raise AssertionError("nonlaunching interpreter reached decode")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved, metadata, registry=document, transport=NoDecode(),
        )
    assert raised.value.exit_code == 3
    assert raised.value.payload["code"] == "package_integrity_failed"
    assert raised.value.payload["failed"] == [{
        "package": "qwen3-asr-0.6b-8bit",
        "check": "environment_mlx_python_runs",
        "expected": True,
        "actual": False,
    }]


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
            resolved, metadata, registry=document, transport=NoDecode(),
        )

    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [{
        "package": "qwen3-asr-0.6b-8bit",
        "check": "environment_mlx_managed_root",
        "expected": str(managed),
        "actual": f"managed environment path is a symlink: {managed}",
    }]


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
            resolved, metadata, registry=document, transport=NoDecode(),
        )

    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [{
        "package": "qwen3-asr-0.6b-8bit",
        "check": "environment_mlx_managed_root",
        "expected": str(audio_paths.env_dir("mlx")),
        "actual": f"provisioning root is a symlink: {root}",
    }]


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
            resolved, metadata, registry=document, transport=NoDecode(),
        )

    assert raised.value.exit_code == 3
    assert raised.value.payload["failed"] == [{
        "package": "qwen3-asr-0.6b-8bit",
        "check": "environment_mlx_managed_root",
        "expected": str(audio_paths.env_dir("mlx")),
        "actual": f"managed environment parent is a symlink: {envs}",
    }]


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
    tmp_path, monkeypatch,
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

    def fail_inspection(path: Path) -> orchestrator._CheckoutState:
        raise AssertionError(f"preflight inspected redirected checkout {path}")

    monkeypatch.setattr(orchestrator, "_inspect_checkout", fail_inspection)

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


def test_nonintersecting_range_refuses_after_decode_but_before_model(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)

    class DecodeOnly(FakeTransport):
        decode_calls = 0

        def decode(self, source, target):
            self.decode_calls += 1
            return super().decode(source, target)

        def qwen(self, **kwargs):
            raise AssertionError("range validation reached model transport")

    transport = DecodeOnly()
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            run_range=orchestrator.parse_range("400:"),
        )
    assert transport.decode_calls == 1
    assert raised.value.exit_code == 2
    assert raised.value.payload["code"] == "range_invalid"
    assert raised.value.payload["provided"] == "400:"


def test_open_range_end_resolves_against_canonical_not_shorter_probe(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(stack_id="qwen-0.6b", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 2)
    transport = FullFakeTransport()
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=registry(tmp_path),
        transport=transport,
        run_range=orchestrator.parse_range("0.5:"),
    ).payload
    assert transport.units == [{"unit_id": "unit_0", "start": 0.0, "end": 2.0}]
    assert payload["provenance"]["plan"]["execution"]["range"]["requested"] == [
        0.5, 2.0,
    ]


def test_fixed_units_match_the_catalog_rule_even_for_a_subsecond_tail(tmp_path) -> None:
    resolved, _, _ = request(tmp_path)
    assert orchestrator._fixed_units(360.0000625, resolved) == [
        {"unit_id": "unit_0", "start": 0.0, "end": 180.0},
        {"unit_id": "unit_1", "start": 180.0, "end": 360.0},
        {"unit_id": "unit_2", "start": 360.0, "end": 360.000063},
    ]


class NoncontiguousTransport(FullFakeTransport):
    def __init__(self, *, partial: bool) -> None:
        super().__init__()
        self.partial = partial

    def diarize(self, **kwargs):
        return StageOutcome("diarizer", "fluidaudio", {"segments": [
            {"startTimeSeconds": 0.0, "endTimeSeconds": 0.6, "speakerId": "S1"},
            {"startTimeSeconds": 0.6, "endTimeSeconds": 1.3, "speakerId": "S2"},
            {"startTimeSeconds": 1.35, "endTimeSeconds": 1.45, "speakerId": "S4"},
            {"startTimeSeconds": 1.5, "endTimeSeconds": 2.0, "speakerId": "S3"},
        ]}, 1.0)

    def qwen(self, *, units, **kwargs):
        self.units = list(units)
        return StageOutcome("asr", "qwen3-asr-0.6b-8bit", {"units": [
            {
                "unit_id": item["unit_id"],
                "processed": not self.partial or item["unit_id"] != "turn_1",
                "text": f"Text {item['unit_id']}.",
            }
            for item in units
        ]}, 1.0, returncode=4 if self.partial else 0)


def test_noncontiguous_partial_resume_reuses_whole_turn_bounds_and_labels(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    first_transport = NoncontiguousTransport(partial=True)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=first_transport,
        )
    partial = json.loads(
        Path(raised.value.payload["output"]).read_text(encoding="utf-8")
    )
    assert partial["coverage"]["covered_intervals"] == [[0.0, 0.6]]
    assert partial["coverage"]["missing_intervals"] == [[0.6, 2.0]]
    assert partial["turns"] == [
        {"turn_id": "turn_0", "speaker": "S1", "start": 0.0, "end": 0.6},
    ]
    assert partial["abstentions"] == []

    resume_transport = NoncontiguousTransport(partial=False)
    resumed = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=resume_transport,
        run_range=orchestrator.parse_range("0.6:"),
    ).payload
    assert resumed["turns"] == [
        {"turn_id": "turn_1", "speaker": "S2", "start": 0.6, "end": 1.3},
        {"turn_id": "turn_2", "speaker": "S3", "start": 1.5, "end": 2.0},
    ]
    assert resumed["abstentions"] == [{
        "abstention_id": "ab_0",
        "reason": "raw_fragment",
        "start": 1.35,
        "end": 1.45,
    }]
    assert resumed["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.6, 2.0],
        "selected_unit_scope": [0.6, 2.0],
    }


def test_open_resume_owns_auxiliary_evidence_after_the_last_selected_turn(tmp_path) -> None:
    class TailEvidenceTransport(NoncontiguousTransport):
        def decode(self, source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * (3 * 16_000))
            return StageOutcome("decode", "ffmpeg", {}, 1.0)

        def diarize(self, **kwargs):
            base = super().diarize(**kwargs).payload["segments"]
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                *base,
                {"startTimeSeconds": 2.5, "endTimeSeconds": 2.6, "speakerId": "S5"},
            ]}, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    payload = orchestrator.run(
        resolved,
        InputMetadata(str(source), 3.0, "wav", 48_000, 2),
        registry=full_registry(tmp_path),
        transport=TailEvidenceTransport(partial=False),
        run_range=orchestrator.parse_range("0.6:"),
    ).payload
    assert payload["abstentions"][-1] == {
        "abstention_id": "ab_1",
        "reason": "raw_fragment",
        "start": 2.5,
        "end": 2.6,
    }


def test_incomplete_hand_range_owns_evidence_before_its_first_selected_turn(tmp_path) -> None:
    class GapExhaustedTransport(NoncontiguousTransport):
        def diarize(self, **kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"startTimeSeconds": 0.0, "endTimeSeconds": 1.3, "speakerId": "S1"},
                {"startTimeSeconds": 1.42, "endTimeSeconds": 1.45, "speakerId": "S4"},
                {"startTimeSeconds": 1.5, "endTimeSeconds": 2.0, "speakerId": "S2"},
            ]}, 1.0)

        def qwen(self, *, units, **kwargs):
            self.units = list(units)
            return StageOutcome("asr", "qwen3-asr-0.6b-8bit", {"units": [{
                "unit_id": item["unit_id"], "processed": False, "text": "",
            } for item in units]}, 1.0, returncode=4)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    transport = GapExhaustedTransport(partial=True)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            InputMetadata(str(source), 2.0, "wav", 48_000, 2),
            registry=full_registry(tmp_path),
            transport=transport,
            run_range=orchestrator.parse_range("1.4:"),
        )
    partial = json.loads(
        Path(raised.value.payload["output"]).read_text(encoding="utf-8")
    )
    assert transport.units == [{
        "unit_id": "turn_1", "speaker": "S2", "start": 1.5, "end": 2.0,
    }]
    assert partial["coverage"]["scope_intervals"] == [[1.5, 2.0]]
    assert partial["abstentions"] == [{
        "abstention_id": "ab_0",
        "reason": "raw_fragment",
        "start": 1.42,
        "end": 1.45,
    }]


def test_run_refuses_existing_outputs_before_decode_and_force_is_explicit(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("keep", encoding="utf-8")
    transport = FakeTransport()
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path), transport=transport,
            output=output,
        )
    assert raised.value.payload["code"] == "output_exists"
    assert raised.value.payload["fix"].endswith(f"-o {output} --force")
    assert transport.units == []
    assert output.read_text(encoding="utf-8") == "keep"

    payload = orchestrator.run(
        resolved, metadata, registry=registry(tmp_path), transport=FakeTransport(),
        output=output, force=True,
    ).payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_run_treats_broken_symlinks_as_existing_before_decode(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "broken.json"
    output.symlink_to("missing.json")
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
        )

    assert raised.value.payload["code"] == "output_exists"
    assert transport.units == []
    assert output.is_symlink()
    assert output.readlink() == Path("missing.json")


def test_run_treats_broken_derived_partial_as_existing_before_decode(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    partial = tmp_path / "result.partial.json"
    partial.symlink_to("missing-partial.json")
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
        )

    assert raised.value.payload["code"] == "output_exists"
    assert raised.value.payload["existing"] == str(partial)
    assert transport.units == []
    assert partial.is_symlink()


def test_force_refuses_an_output_symlink_loop_as_an_invalid_path(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "loop.json"
    output.symlink_to(output.name)
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
            force=True,
        )

    assert raised.value.payload["code"] == "output_path_invalid"
    assert raised.value.payload["target"] == str(output)
    assert transport.units == []
    assert output.is_symlink()


@pytest.mark.parametrize("partial_target", [False, True])
def test_force_refuses_a_directory_output_before_decode(
    tmp_path, partial_target: bool
) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    target = tmp_path / "result.partial.json" if partial_target else output
    target.mkdir()
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
            force=True,
        )

    assert raised.value.payload["code"] == "output_path_invalid"
    assert raised.value.payload["target"] == str(target)
    assert transport.units == []
    assert target.is_dir()


@pytest.mark.parametrize("partial", [False, True])
def test_late_output_collision_is_never_clobbered(tmp_path, partial: bool) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    collision = (
        tmp_path / "result.partial.json" if partial else output
    )

    class LateCollisionTransport(FakeTransport):
        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            collision.write_text("racer-owned\n", encoding="utf-8")
            return outcome

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=LateCollisionTransport(partial=partial),
            output=output,
        )

    assert raised.value.payload["code"] == "output_exists"
    assert raised.value.payload["existing"] == str(collision)
    assert collision.read_text(encoding="utf-8") == "racer-owned\n"
    if partial:
        assert not output.exists()


def test_late_directory_collision_is_a_typed_refusal(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"

    class LateDirectoryTransport(FakeTransport):
        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            output.mkdir()
            return outcome

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=LateDirectoryTransport(),
            output=output,
        )

    assert raised.value.payload["code"] == "output_path_invalid"
    assert raised.value.payload["target"] == str(output)
    assert output.is_dir()


@pytest.mark.parametrize("partial", [False, True])
def test_late_unwritable_destination_is_a_typed_refusal(
    tmp_path: Path,
    monkeypatch,
    partial: bool,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    transport = FakeTransport(partial=partial)

    def refuse_write(*_args, **_kwargs):
        raise PermissionError("destination is not writable")

    monkeypatch.setattr(orchestrator, "atomic_write_json", refuse_write)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
        )

    target = tmp_path / "result.partial.json" if partial else output
    assert raised.value.exit_code == 2
    assert raised.value.payload["code"] == "output_path_invalid"
    assert raised.value.payload["provided"] == str(output)
    assert raised.value.payload["target"] == str(target)
    assert "not writable" in raised.value.payload["reason"]


def test_run_never_allows_output_to_resolve_to_canonical_input(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path), transport=FakeTransport(),
            output=resolved.input_path, force=True,
        )
    assert raised.value.payload["code"] == "output_is_canonical_input"
    assert resolved.input_path.read_bytes() == b"source"

    derived_source = tmp_path / "meeting.partial.json"
    derived_source.write_bytes(b"canonical")
    derived_request = resolve_request(
        stack_id="qwen-0.6b", input_path=derived_source, wants=(),
    )
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            derived_request,
            InputMetadata(str(derived_source), 361.0, "json", 48_000, 2),
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
            output=tmp_path / "meeting.json",
            force=True,
        )
    assert raised.value.payload["code"] == "output_is_canonical_input"
    assert raised.value.payload["resolved_target"] == str(derived_source)
    assert derived_source.read_bytes() == b"canonical"


def test_qwen_publication_preserves_source_renamed_to_forced_output_after_decode(
    tmp_path: Path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    output.write_bytes(b"replaceable")

    class SourceMovingTransport(FakeTransport):
        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            os.replace(resolved.input_path, output)
            return outcome

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=SourceMovingTransport(),
            output=output,
            force=True,
        )

    assert raised.value.payload["code"] == "output_is_canonical_input"
    assert not resolved.input_path.exists()
    assert output.read_bytes() == b"source"


def test_diarizer_exclusions_survive_without_requesting_diarization(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("overlapped_speech",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=NoncontiguousTransport(partial=False),
    ).payload
    assert "turns" not in payload
    assert payload["abstentions"] == [{
        "abstention_id": "ab_0",
        "reason": "raw_fragment",
        "start": 1.35,
        "end": 1.45,
    }]


def test_overlap_abstention_survives_without_optional_overlap_array(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=FullFakeTransport(),
    ).payload
    assert "overlapped_speech" not in payload
    assert payload["abstentions"] == [{
        "abstention_id": "ab_0", "reason": "overlap", "start": 0.8, "end": 1.0,
    }]


def test_backend_failure_fix_does_not_promise_an_unrelated_command(tmp_path) -> None:
    class BrokenTransport(FakeTransport):
        def qwen(self, **kwargs):
            from audio_cli.transcribe.transport import StageFailure

            raise StageFailure("asr", "qwen3-asr-0.6b-8bit", "out of memory")

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path), transport=BrokenTransport(),
        )
    assert raised.value.exit_code == 1
    assert raised.value.payload["role"] == "asr"
    assert raised.value.payload["backend"] == "qwen3-asr-0.6b-8bit"
    assert not raised.value.payload["fix"].startswith("audio ")


@pytest.mark.parametrize(
    ("partial_payload", "returncode"),
    [(False, 4), (True, 0)],
)
def test_qwen_exit_status_must_match_the_unfinished_unit_ledger(
    tmp_path: Path,
    partial_payload: bool,
    returncode: int,
) -> None:
    class ContradictoryTransport(FakeTransport):
        def __init__(self) -> None:
            super().__init__(partial=partial_payload)

        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            return StageOutcome(
                outcome.role,
                outcome.backend,
                outcome.payload,
                outcome.wall_seconds,
                returncode=returncode,
                peak_rss_bytes=outcome.peak_rss_bytes,
            )

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=ContradictoryTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "asr"
    assert "exit status disagrees with unfinished unit ledger" in (
        raised.value.payload["detail"]
    )


def test_invalid_internal_stage_walls_are_a_typed_backend_refusal(tmp_path) -> None:
    class InvalidMetricsTransport(FakeTransport):
        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            return StageOutcome(
                outcome.role,
                outcome.backend,
                outcome.payload,
                outcome.wall_seconds,
                returncode=outcome.returncode,
                peak_rss_bytes=outcome.peak_rss_bytes,
                wall_seconds_by_stage={"asr": outcome.wall_seconds + 1.0},
            )

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=InvalidMetricsTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "asr"
    assert "internal wall metrics exceed its process wall" in (
        raised.value.payload["detail"]
    )


@pytest.mark.parametrize(
    "canonical_bytes",
    [b"", b"not a wave", bytes.fromhex("52494646a8eddd645741564545e52057cfbfd3cf")],
)
def test_malformed_canonical_wav_is_a_typed_decode_failure(
    tmp_path,
    canonical_bytes: bytes,
) -> None:
    class MalformedDecodeTransport(FakeTransport):
        def decode(self, source, target):
            target.write_bytes(canonical_bytes)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=MalformedDecodeTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "decode"
    assert raised.value.payload["backend"] == "ffmpeg"
    assert isinstance(raised.value.payload["detail"], str)


def test_backend_derived_result_error_is_one_json_refusal(tmp_path) -> None:
    class NonLabelTransport(FullFakeTransport):
        def diarize(self, **kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [{
                "startTimeSeconds": 0.0,
                "endTimeSeconds": 1.0,
                "speakerId": "N/A",
            }]}, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=NonLabelTransport(),
        )
    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert "must be absent" in raised.value.payload["detail"]


@pytest.mark.parametrize(
    "invalid_segment",
    [
        {
            "startTimeSeconds": False,
            "endTimeSeconds": 1.0,
            "speakerId": "S1",
        },
        {
            "startTimeSeconds": 0.0,
            "endTimeSeconds": 1.0,
            "speakerId": [],
        },
    ],
)
def test_invalid_diarizer_artifact_is_a_typed_backend_refusal(
    tmp_path,
    invalid_segment: dict[str, object],
) -> None:
    class InvalidDiarizerTransport(FullFakeTransport):
        def diarize(self, **kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {"segments": [invalid_segment]},
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=InvalidDiarizerTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "diarizer"
    assert raised.value.payload["backend"] == "fluidaudio"


def test_invalid_vad_artifact_is_a_typed_backend_refusal(tmp_path) -> None:
    class OverlappingVad:
        def detect(self, samples, rate, **config):
            return [
                {"start": 0.0, "end": 0.8},
                {"start": 0.7, "end": 1.0},
            ]

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("vad",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=FullFakeTransport(),
            vad_detector=OverlappingVad(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "vad"
    assert raised.value.payload["backend"] == "silero-vad"
    assert "overlaps" in raised.value.payload["detail"]


def test_abstentions_are_reidentified_in_source_order(tmp_path) -> None:
    class MixedEvidenceTransport(FullFakeTransport):
        def diarize(self, **kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"startTimeSeconds": 0.0, "endTimeSeconds": 0.8, "speakerId": "S1"},
                {"startTimeSeconds": 0.4, "endTimeSeconds": 1.0, "speakerId": "S2"},
                {"startTimeSeconds": 1.2, "endTimeSeconds": 1.3, "speakerId": "S3"},
                {"startTimeSeconds": 1.4, "endTimeSeconds": 2.0, "speakerId": "S4"},
            ]}, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization", "overlapped_speech"),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=MixedEvidenceTransport(),
    ).payload
    starts = [item["start"] for item in payload["abstentions"]]
    assert starts == sorted(starts)
    assert [item["abstention_id"] for item in payload["abstentions"]] == [
        f"ab_{index}" for index in range(len(starts))
    ]


def test_empty_transcript_keeps_diarizer_abstention_evidence(tmp_path) -> None:
    class RawOnlyTransport(FullFakeTransport):
        def diarize(self, **kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [{
                "startTimeSeconds": 0.2,
                "endTimeSeconds": 0.3,
                "speakerId": "S1",
            }]}, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=RawOnlyTransport(),
    ).payload
    assert payload["complete"] is True
    assert payload["segments"] == []
    assert payload["turns"] == []
    assert payload["abstentions"] == [{
        "abstention_id": "ab_0",
        "reason": "raw_fragment",
        "start": 0.2,
        "end": 0.3,
    }]
    assert payload["provenance"]["outcomes"]["diarization"] == "produced"


def test_alignment_abstention_is_observable_and_does_not_invent_words(tmp_path) -> None:
    class AbstainingAligner(FullFakeTransport):
        def align(self, *, segments, **kwargs):
            return StageOutcome("aligner", "qwen3-forcedaligner", {"segments": [{
                "unit_id": item["unit_id"], "words": None,
            } for item in segments]}, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("word_timestamps",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=AbstainingAligner(),
    ).payload
    assert all("words" not in item for item in payload["segments"])
    assert payload["provenance"]["outcomes"]["word_timestamps"] == "abstained"
    assert payload["provenance"]["observed"]["segments_without_words"] == 1
    assert payload["abstentions"] == [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.0,
        "end": 2.0,
    }]


@pytest.mark.parametrize(
    "aligner_payload",
    [
        {},
        {"segments": []},
        {"segments": [{"unit_id": "turn_0"}]},
    ],
)
def test_qwen_rejects_incomplete_aligner_ledgers_without_publication(
    tmp_path: Path,
    aligner_payload: dict,
) -> None:
    class MalformedAligner(FullFakeTransport):
        def align(self, **kwargs):
            return StageOutcome(
                "aligner", "qwen3-forcedaligner", aligner_payload, 1.0
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("word_timestamps",),
    )
    output = tmp_path / "result.json"
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            InputMetadata(str(source), 2.0, "wav", 48_000, 2),
            registry=full_registry(tmp_path),
            transport=MalformedAligner(),
            output=output,
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["backend"] == "qwen3-forcedaligner"
    assert not output.exists()


def test_alignment_text_mismatch_records_the_attempted_unit_bounds(tmp_path) -> None:
    class MismatchedAligner(FullFakeTransport):
        def align(self, *, segments, **kwargs):
            return StageOutcome("aligner", "qwen3-forcedaligner", {"segments": [{
                "unit_id": item["unit_id"],
                "words": [{"text": "Goodbye", "start": 0.0, "end": 1.0}],
            } for item in segments]}, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("word_timestamps",),
    )
    payload = orchestrator.run(
        resolved,
        InputMetadata(str(source), 2.0, "wav", 48_000, 2),
        registry=full_registry(tmp_path),
        transport=MismatchedAligner(),
    ).payload

    assert payload["abstentions"] == [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.0,
        "end": 2.0,
    }]
    assert all("words" not in item for item in payload["segments"])


def test_partial_qwen_alignment_abstention_covers_only_the_completed_prefix(
    tmp_path,
) -> None:
    class PartialAbstainingAligner(FakeTransport):
        def align(self, *, segments, **kwargs):
            return StageOutcome("aligner", "qwen3-forcedaligner", {"segments": [{
                "unit_id": item["unit_id"], "words": None,
            } for item in segments]}, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("word_timestamps",),
    )
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            InputMetadata(str(source), 361.0, "wav", 48_000, 2),
            registry=full_registry(tmp_path),
            transport=PartialAbstainingAligner(partial=True),
        )

    payload = json.loads(
        Path(raised.value.payload["output"]).read_text(encoding="utf-8")
    )
    assert payload["abstentions"] == [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.0,
        "end": 180.0,
    }]


def test_punctuation_only_alignment_with_empty_words_is_produced_not_abstained(
    tmp_path,
) -> None:
    class PunctuationTransport(FakeTransport):
        def qwen(self, *, units, **kwargs):
            return StageOutcome("asr", "qwen3-asr-0.6b-8bit", {"units": [{
                "unit_id": item["unit_id"], "processed": True, "text": "……？！",
            } for item in units]}, 1.0)

        def align(self, *, segments, **kwargs):
            return StageOutcome("aligner", "qwen3-forcedaligner", {"segments": [{
                "unit_id": item["unit_id"], "words": [],
            } for item in segments]}, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("word_timestamps",),
    )
    payload = orchestrator.run(
        resolved,
        InputMetadata(str(source), 361.0, "wav", 48_000, 2),
        registry=full_registry(tmp_path),
        transport=PunctuationTransport(),
    ).payload
    assert all(item["words"] == [] for item in payload["segments"])
    assert payload["provenance"]["outcomes"]["word_timestamps"] == "produced"
    assert payload["provenance"]["observed"]["segments_without_words"] == 0


@pytest.mark.parametrize(
    ("returncode", "transport_failure", "detail"),
    [(1, True, "out of memory"), (4, False, "unsupported exit 4")],
)
def test_whole_aligner_failure_is_a_backend_error_and_writes_no_result(
    tmp_path, returncode: int, transport_failure: bool, detail: str
) -> None:
    class FailingAligner(FullFakeTransport):
        def align(self, **kwargs):
            from audio_cli.transcribe.transport import StageFailure

            if not transport_failure:
                successful = super().align(**kwargs)
                return StageOutcome(
                    successful.role,
                    successful.backend,
                    successful.payload,
                    successful.wall_seconds,
                    returncode=returncode,
                )
            outcome = StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {"error": {"message": "out of memory"}},
                2.5,
                returncode=returncode,
                peak_rss_bytes=250,
            )
            raise StageFailure(
                "aligner",
                "qwen3-forcedaligner",
                "out of memory",
                outcome=outcome,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("word_timestamps",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=FailingAligner(),
            output=output,
        )
    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "aligner"
    assert caught.value.payload["backend"] == "qwen3-forcedaligner"
    assert detail in caught.value.payload["detail"]
    assert not output.exists()


def test_canonical_wav_header_owns_duration_and_fixed_unit_bounds(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    metadata = InputMetadata(metadata.source["path"], 999.0, "wav", 48_000, 2)
    transport = FullFakeTransport()
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=registry(tmp_path),
        transport=transport,
    ).payload
    assert payload["source"] == {
        "path": str(resolved.input_path),
        "duration_seconds": 2.0,
        "timebase": "seconds",
    }
    assert transport.units == [{"unit_id": "unit_0", "start": 0.0, "end": 2.0}]


def test_range_filters_auxiliary_observations_to_selected_processing_scope(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization", "overlapped_speech", "vad"),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=FullFakeTransport(),
        vad_detector=FakeVad(),
        run_range=orchestrator.parse_range("1.2:1.5"),
    ).payload
    assert payload["turns"] == [{
        "turn_id": "turn_1", "speaker": "S2", "start": 1.0, "end": 1.8,
    }]
    assert payload["overlapped_speech"] == []
    assert payload["vad_regions"] == []
    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [1.2, 1.5], "selected_unit_scope": [1.0, 1.8],
    }


@pytest.mark.parametrize("wants", [
    (),
    ("languages",),
    ("verbatim",),
    ("diarization",),
    ("overlapped_speech",),
    ("vad",),
    ("word_timestamps",),
    ("languages", "verbatim", "diarization", "overlapped_speech", "vad",
     "word_timestamps"),
])
def test_real_qwen_result_shape_matches_its_plan_sample_over_derivation_cells(
    tmp_path, wants
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=wants,
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    document = full_registry(tmp_path)
    ready = set(document["packages"])
    sample = serialize_plan(build_plan(
        resolved, metadata, provisioned_packages=ready,
    ))["sample_output"]
    actual = orchestrator.run(
        resolved, metadata, registry=document, transport=FullFakeTransport(),
        vad_detector=FakeVad(),
    ).payload

    assert set(actual) == set(sample) - {"sample", "note"}
    assert set(actual["provenance"]) == set(sample["provenance"])
    assert set(actual["segments"][0]) == set(sample["segments"][0])
    assert set(actual["provenance"]["outcomes"]) == set(wants)
    for capability, field in {
        "diarization": "turns",
        "overlapped_speech": "overlapped_speech",
        "vad": "vad_regions",
    }.items():
        assert (field in actual) is (capability in wants)
