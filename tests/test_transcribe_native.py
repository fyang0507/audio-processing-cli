from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli import paths as audio_paths
from audio_cli.export import export_documents, load_result_document
from audio_cli.media import hash_file
from audio_cli.transcribe import native as native_module
from audio_cli.transcribe import orchestrator, refusals
from audio_cli.transcribe.adapters import normalize_vibevoice_result
from audio_cli.transcribe.catalog import InputMetadata
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.transport import StageOutcome


_REAL_INSPECT_CHECKOUT = orchestrator._inspect_checkout


def _complete_vibe_payload(
    segments: list[dict], **extra: object,
) -> dict[str, object]:
    """Build the two equivalent streams emitted by the pinned post-processor."""

    generated = []
    for segment in segments:
        item = {
            "Start": segment["start_time"],
            "End": segment["end_time"],
            "Content": segment["text"],
        }
        if "speaker_id" in segment:
            item["Speaker"] = segment["speaker_id"]
        generated.append(item)
    return {
        "raw_text": json.dumps(generated),
        "segments": segments,
        "hit_max_new_tokens": False,
        **extra,
    }


@pytest.fixture(autouse=True)
def trusted_checkout_probe(tmp_path: Path, monkeypatch) -> None:
    """Unit fixtures model provisioned checkouts without creating nested Git repos."""

    def inspect(checkout: Path) -> orchestrator._CheckoutState:
        if "fluidaudio" in str(checkout):
            package = env.packages()["fluidaudio"]
            _patches, modified, _digests = pkg.checkout_patch_expectation(package)
            return orchestrator._CheckoutState(
                head=package.source["commit"], modified=modified, untracked=()
            )
        package_id = (
            "vibevoice-asr-7b"
            if "vibevoice-asr-7b" in str(checkout)
            else "firered-asr2s"
        )
        package = env.packages()[package_id]
        _patches, modified, _digests = pkg.checkout_patch_expectation(package)
        return orchestrator._CheckoutState(
            head=package.checkout["resolved_commit"],
            modified=modified,
            untracked=(),
        )

    monkeypatch.setattr(orchestrator, "_inspect_checkout", inspect)

    def digest(path: Path) -> str:
        fluid = env.packages()["fluidaudio"].source.get("patched_file_sha256", {})
        for name, expected in fluid.items():
            if path.as_posix().endswith(f"/{name}") and path.read_bytes() == b"patched\n":
                return expected
        package = env.packages()["vibevoice-asr-7b"]
        for name, expected in package.checkout["patched_file_sha256"].items():
            if path.as_posix().endswith(f"/{name}") and path.read_bytes() == b"patched\n":
                return expected
        return hash_file(path)

    monkeypatch.setattr(orchestrator, "_checkout_file_digest", digest)

    def frozen(interpreter: Path) -> dict[str, str]:
        environment_name = interpreter.parent.parent.name
        installed: dict[str, str] = {}
        for package in env.packages().values():
            if package.environment != environment_name or package.checkout is None:
                continue
            checkout = audio_paths.checkout_dir(package.environment, package.id)
            installed[package.checkout["distribution"]] = (
                f"@ {checkout.resolve(strict=False).as_uri()}"
            )
        return installed

    monkeypatch.setattr(orchestrator, "_frozen_packages", frozen)
    monkeypatch.setattr(orchestrator, "_python_runtime_runs", lambda _path: True)

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


def test_checkout_probe_reads_live_head_tracked_and_untracked_state(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=checkout,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return completed.stdout.strip()

    git("init", "--quiet")
    (checkout / ".gitignore").write_text(
        "ignored.tmp\n__pycache__/\n", encoding="utf-8"
    )
    (checkout / "tracked.py").write_text("original\n", encoding="utf-8")
    git("add", ".gitignore", "tracked.py")
    git(
        "-c",
        "user.name=Audio Test",
        "-c",
        "user.email=audio@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    git("config", "core.abbrev", "12")
    git("config", "diff.noprefix", "true")
    git("config", "diff.interHunkContext", "100")
    git("config", "diff.suppressBlankEmpty", "true")

    clean = _REAL_INSPECT_CHECKOUT(checkout)
    assert clean.head == git("rev-parse", "HEAD")
    assert clean.modified == ()
    assert clean.untracked == ()

    (checkout / "tracked.py").write_text("changed\n", encoding="utf-8")
    (checkout / "rogue.py").write_text("rogue\n", encoding="utf-8")
    (checkout / "ignored.tmp").write_text("ignored\n", encoding="utf-8")
    bytecode = checkout / "__pycache__" / "tracked.cpython-312.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"forged bytecode")
    changed = _REAL_INSPECT_CHECKOUT(checkout)
    assert changed.modified == ("tracked.py",)
    assert changed.untracked == (
        "__pycache__/tracked.cpython-312.pyc",
        "ignored.tmp",
        "rogue.py",
    )


def _ready_multi_package(tmp_path: Path, package_id: str) -> dict:
    package = env.packages()[package_id]
    locations = {}
    for repository in package.source["repos"]:
        target = (
            tmp_path / "hub" / f"models--{repository['repo'].replace('/', '--')}"
            / "snapshots" / repository["revision"]
        )
        target.mkdir(parents=True, exist_ok=True)
        for pattern in repository.get("allow_patterns", ()):
            destination = target / pattern
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"")
        locations[repository["repo"]] = str(target)
    checkout = audio_paths.checkout_dir(package.environment, package.id)
    checkout.mkdir(parents=True)
    materialized = {
        "paths": locations,
        "revisions": [item["revision"] for item in package.source["repos"]],
        "bytes": 0,
        "checkout": str(checkout),
        "checkout_commit": package.checkout["resolved_commit"],
    }
    patch_name = package.checkout.get("patch")
    if patch_name:
        _patches, names, expected_digests = pkg.checkout_patch_expectation(package)
        patched = checkout / names[0]
        patched.parent.mkdir(parents=True, exist_ok=True)
        patched.write_text("patched\n", encoding="utf-8")
        materialized.update({
            "patches_applied": [Path(patch_name).name],
            "patched_file_digests": expected_digests,
        })
    return {
        "state": "ready",
        "materialized": materialized,
    }


def _runtime(tmp_path: Path, monkeypatch, name: str) -> None:
    root = tmp_path / "runtime"
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(root))
    interpreter = root / "envs" / name / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)


@pytest.mark.parametrize(
    "canonical_bytes",
    [b"", b"not a wave", bytes.fromhex("52494646a8eddd645741564545e52057cfbfd3cf")],
)
def test_firered_malformed_canonical_wav_is_a_typed_decode_failure(
    tmp_path: Path,
    monkeypatch,
    canonical_bytes: bytes,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "field.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants=(),
    )
    metadata = InputMetadata(str(source), 5.0, "wav", 48_000, 1)

    class MalformedDecodeTransport:
        def decode(self, _source, target):
            target.write_bytes(canonical_bytes)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {
            "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")
        },
    }
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=MalformedDecodeTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "decode"
    assert raised.value.payload["backend"] == "ffmpeg"
    assert isinstance(raised.value.payload["detail"], str)


def _ready_single_package(tmp_path: Path, package_id: str) -> dict:
    package = env.packages()[package_id]
    target = (
        tmp_path / "hub" / f"models--{package.source['repo'].replace('/', '--')}"
        / "snapshots" / package.source["revision"]
    )
    target.mkdir(parents=True, exist_ok=True)
    for pattern in package.source.get("allow_patterns", ()):
        marker = target / (
            f"{pattern[:-3]}/model.mil" if pattern.endswith("/**") else pattern
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"")
    return {
        "state": "ready",
        "materialized": {
            "path": str(target), "revision": package.source["revision"], "bytes": 0,
        },
    }


def _ready_fluidaudio(tmp_path: Path) -> dict:
    package = env.packages()["fluidaudio"]
    target = audio_paths.checkout_dir(package.environment, package.id)
    target.mkdir(parents=True, exist_ok=True)
    product = target / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_text("#!/bin/sh\n", encoding="utf-8")
    product.chmod(0o755)
    patched = target / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
    patched.parent.mkdir(parents=True)
    patched.write_bytes(b"patched\n")
    patch_names, _modified, patch_digests = pkg.checkout_patch_expectation(package)
    return {
        "state": "ready",
        "materialized": {
            "path": str(target),
            "revision": env.packages()["fluidaudio"].source["commit"],
            "built": True,
            "product_runs": True,
            "product_path": product.relative_to(target).as_posix(),
            "product_sha256": pkg.sha256_file(product),
            "patches_applied": list(patch_names),
            "patched_file_digests": patch_digests,
        },
    }


@pytest.mark.parametrize(
    ("stack_id", "runtime", "package_id", "vad"),
    [
        ("firered", "torch-firered", "firered-asr2s", "silero-vad"),
        ("vibevoice", "torch-vibevoice", "vibevoice-asr-7b", None),
    ],
)
def test_native_silero_bounds_are_typed_before_model_work(
    tmp_path: Path,
    monkeypatch,
    stack_id: str,
    runtime: str,
    package_id: str,
    vad: str | None,
) -> None:
    _runtime(tmp_path, monkeypatch, runtime)
    source = tmp_path / f"{stack_id}-invalid-vad.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id=stack_id,
        input_path=source,
        wants="vad",
        vad=vad,
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return [{"start": 0.25, "end": 3.0}]

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **_kwargs):
            raise AssertionError("FireRed must not receive invalid Silero bounds")

        def vibevoice(self, **_kwargs):
            raise AssertionError("VibeVoice must not run after invalid Silero bounds")

    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {runtime: {"state": "ready"}},
                "packages": {
                    package_id: _ready_multi_package(tmp_path, package_id),
                },
            },
            transport=Transport(),
            vad_detector=Detector(),
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "vad"
    assert caught.value.payload["backend"] == "silero-vad"
    assert "exceeds canonical source duration" in caught.value.payload["detail"]


def test_firered_run_crosses_transport_and_adapter_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "field.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="verbatim,word_timestamps,vad,segment_timestamps,lid",
    )
    metadata = InputMetadata(str(source), 5.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 80_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1, peak_rss_bytes=10)

        def firered(self, **kwargs):
            assert kwargs["lid_enabled"] is True
            assert kwargs["vad_regions"] is None
            assert kwargs["asr_config"]["beam_size"] == 3
            assert set(kwargs["models"]) == {"vad", "lid", "asr", "punctuator"}
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [{
                            "start_ms": 100,
                            "end_ms": 900,
                            "text": "Hello!",
                            "lang": "en",
                            "lang_confidence": 0.9,
                        }],
                        "words": [{
                            "start_ms": 120, "end_ms": 800, "text": "hello",
                        }],
                        "vad_segments_ms": [[100, 1000]],
                    },
                    "regions": [{
                        "region_id": "vad_0", "start": 0.1, "end": 1.0,
                        "processed": True,
                    }],
                },
                30.0,
                peak_rss_bytes=100,
                wall_seconds_by_stage={
                    "vad": 0.2, "lid": 0.3, "asr": 2.0, "punctuator": 0.5,
                },
            )

    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
    }
    payload = orchestrator.run(
        request, metadata, registry=registry, transport=Transport()
    ).payload
    assert payload["segments"] == [{
        "segment_id": "seg_0",
        "text": "Hello!",
        "start": 0.1,
        "end": 0.9,
        "words": [{
            "word_id": "w_0", "text": "hello", "start": 0.12, "end": 0.8,
        }],
    }]
    assert payload["vad_regions"] == [{"start": 0.1, "end": 1.0}]
    assert payload["lid_regions"] == [{
        "start": 0.1, "end": 1.0, "language": "en", "confidence": 0.9,
    }]
    observed = payload["provenance"]["observed"]
    assert observed["stage_wall_seconds"] == {
        "decode": 0.1,
        "vad": 0.2,
        "lid": 0.3,
        "asr": 2.0,
        "punctuator": 0.5,
        "firered_process_overhead": 27.0,
    }
    assert observed["total_wall_seconds"] == 30.1
    assert observed["peak_rss_bytes_by_stage"] == {
        "decode": 10, "firered_process": 100,
    }
    assert observed["punctuation_invariant_checked"] is True


def test_vibevoice_native_events_keep_bounds_but_not_speaker_or_words(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "demo.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants=(
            "verbatim,diarization,segment_timestamps,word_timestamps,"
            "overlapped_speech"
        ),
    )
    metadata = InputMetadata(str(source), 10.0, "wav", 48_000, 2)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 160_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **kwargs):
            assert kwargs["model"].name == (
                "d0c9efdb8d614685062c04425d91e01b6f37d944"
            )
            assert kwargs["tokenizer"].name == (
                "d149729398750b98c0af14eb82c78cfe92750796"
            )
            assert kwargs["config"] == {
                "device": "mps",
                "dtype": "bfloat16",
                "attention": "sdpa",
                "seed": 1234,
                "max_new_tokens": 16384,
            }
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {"start_time": 0.0, "end_time": 2.0, "speaker_id": 0,
                         "text": "Hello."},
                        {"start_time": 2.0, "end_time": 3.0,
                         "text": "[Environmental Sounds]"},
                        {"start_time": 3.0, "end_time": 5.0, "speaker_id": 0,
                         "text": "Again."},
                    ],
                    generated_tokens=20,
                    eos_observed=True,
                ),
                3.0,
                peak_mps_live_bytes=100,
            )

        def align(self, *, segments, **kwargs):
            assert kwargs["model"].name == (
                "0e1a68e91d815300c7c9754b2a7639378b23db15"
            )
            assert [item["unit_id"] for item in segments] == ["native_0", "native_1"]
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {"segments": [
                    {"unit_id": "native_0", "words": [
                        {"text": "Hello", "start": 0.2, "end": 1.8},
                    ]},
                    {"unit_id": "native_1", "words": [
                        {"text": "Again", "start": 3.2, "end": 4.8},
                    ]},
                ]},
                1.0,
            )

        def diarize(self, **_kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"startTimeSeconds": 0.0, "endTimeSeconds": 2.0,
                 "speakerId": "S0"},
                {"startTimeSeconds": 1.0, "endTimeSeconds": 1.5,
                 "speakerId": "S1"},
            ]}, 0.2)

    registry = {
        "environments": {
            "torch-vibevoice": {"state": "ready"},
            "mlx": {"state": "ready"},
            "swift": {"state": "ready"},
        },
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
            "qwen3-forcedaligner": _ready_single_package(
                tmp_path, "qwen3-forcedaligner"
            ),
            "fluidaudio": _ready_fluidaudio(tmp_path),
            "speaker-diarization-coreml": _ready_single_package(
                tmp_path, "speaker-diarization-coreml"
            ),
        },
    }
    payload = orchestrator.run(
        request, metadata, registry=registry, transport=Transport()
    ).payload
    assert payload["segments"][1] == {
        "segment_id": "seg_1",
        "text": "[Environmental Sounds]",
        "start": 2.0,
        "end": 3.0,
    }
    assert "speaker" not in payload["segments"][0]
    assert payload["segments"][2]["speaker"] == "0"
    assert payload["overlapped_speech"] == [{
        "overlap_id": "overlap_0", "start": 1.0, "end": 1.5,
    }]
    assert payload["turns"] == [
        {"turn_id": "turn_0", "speaker": "0", "start": 0.0, "end": 2.0},
        {"turn_id": "turn_1", "speaker": "0", "start": 3.0, "end": 5.0},
    ]
    assert payload["provenance"]["outcomes"]["word_timestamps"] == "produced"
    assert payload["provenance"]["observed"]["segments_without_words"] == 1
    assert payload["abstentions"] == [{
        "abstention_id": "ab_0", "reason": "overlap",
        "start": 1.0, "end": 1.5,
    }]
    assert "N/A" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("segments", "detail"),
    [
        (
            [{"start_time": 0.0, "end_time": 1.0, "text": "Unlabelled speech."}],
            "requires a speaker label",
        ),
        (
            [{
                "start_time": 0.0,
                "end_time": 1.0,
                "speaker_id": 0,
                "text": "[Music]",
            }],
            "must not carry a speaker",
        ),
    ],
)
def test_vibevoice_native_diarization_rejects_uncoupled_segment_labels(
    tmp_path: Path,
    monkeypatch,
    segments: list[dict],
    detail: str,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "invalid-native-diarization.wav"
    source.write_bytes(b"source")
    output = tmp_path / "result.json"

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(segments),
                1.0,
            )

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolve_request(
                stack_id="vibevoice",
                input_path=source,
                wants="diarization,segment_timestamps",
            ),
            InputMetadata(str(source), 1.0, "wav", 48_000, 1),
            registry={
                "environments": {"torch-vibevoice": {"state": "ready"}},
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(
                        tmp_path, "vibevoice-asr-7b"
                    )
                },
            },
            transport=Transport(),
            output=output,
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["backend"] == "vibevoice-asr-7b"
    assert detail in raised.value.payload["detail"]
    assert not output.exists()


def test_native_publication_preserves_source_renamed_to_forced_output_after_decode(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    output = tmp_path / "result.json"
    output.write_bytes(b"replaceable")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            os.replace(source, output)
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload([
                    {
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "speaker_id": 0,
                        "text": "Hello.",
                    }
                ]),
                1.0,
            )

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-vibevoice": {"state": "ready"}},
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(
                        tmp_path, "vibevoice-asr-7b"
                    )
                },
            },
            transport=Transport(),
            output=output,
            force=True,
        )

    assert raised.value.payload["code"] == "output_is_canonical_input"
    assert not source.exists()
    assert output.read_bytes() == b"source"


def test_vibevoice_native_turns_preserve_the_recorded_same_speaker_gap() -> None:
    fixture = json.loads((
        Path(__file__).parents[1]
        / "tests/fixtures/vibevoice_multispeaker_excerpt.json"
    ).read_text(encoding="utf-8"))
    normalized = normalize_vibevoice_result({
        "segments": fixture["segments"],
        "raw_text": json.dumps(fixture["segments"]),
        "hit_max_new_tokens": False,
    }, clip_duration_seconds=60.0)

    turns = native_module._native_turns(normalized.segments)
    assert turns[:2] == [
        {"turn_id": "turn_0", "speaker": "1", "start": 16.83, "end": 32.05},
        {"turn_id": "turn_1", "speaker": "1", "start": 33.64, "end": 37.84},
    ]
    assert turns[1]["start"] - turns[0]["end"] == pytest.approx(1.59)


@pytest.mark.parametrize(
    "words",
    [
        [],
        [{"text": "Goodbye", "start": 0.2, "end": 1.8}],
        [
            {"text": "Hel", "start": 1.0, "end": 1.4},
            {"text": "lo", "start": 0.2, "end": 0.8},
        ],
        [
            {"text": "Hel", "start": 0.2, "end": 1.2},
            {"text": "lo", "start": 1.0, "end": 1.8},
        ],
        [
            {"text": "", "start": 0.2, "end": 0.3},
            {"text": "Hello", "start": 0.3, "end": 1.8},
        ],
    ],
)
def test_vibevoice_nonconforming_speech_alignment_is_a_segment_abstention(
    tmp_path: Path,
    monkeypatch,
    words: list[dict],
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "bad-alignment.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice", input_path=source, wants="word_timestamps"
    )
    metadata = InputMetadata(str(source), 5.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 80_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome("asr", "vibevoice-asr-7b", _complete_vibe_payload(
                [
                    {
                        "start_time": 0.0,
                        "end_time": 2.0,
                        "speaker_id": 0,
                        "text": "Hello.",
                    },
                    {
                        "start_time": 2.0,
                        "end_time": 3.0,
                        "text": "[Environmental Sounds]",
                    },
                    {
                        "start_time": 3.0,
                        "end_time": 5.0,
                        "speaker_id": 0,
                        "text": "Again.",
                    },
                ],
                generated_tokens=4,
                eos_observed=True,
            ), 1.0)

        def align(self, **_kwargs):
            return StageOutcome("aligner", "qwen3-forcedaligner", {
                "segments": [
                    {"unit_id": "native_0", "words": words},
                    {"unit_id": "native_1", "words": None},
                ],
            }, 1.0)

    registry = {
        "environments": {
            "torch-vibevoice": {"state": "ready"},
            "mlx": {"state": "ready"},
        },
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(
                tmp_path, "vibevoice-asr-7b"
            ),
            "qwen3-forcedaligner": _ready_single_package(
                tmp_path, "qwen3-forcedaligner"
            ),
        },
    }
    output = tmp_path / "alignment-abstention.json"
    payload = orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=Transport(),
        output=output,
    ).payload

    assert payload["segments"] == [
        {"segment_id": "seg_0", "text": "Hello."},
        {"segment_id": "seg_1", "text": "[Environmental Sounds]"},
        {"segment_id": "seg_2", "text": "Again."},
    ]
    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "start": 0.0,
            "end": 2.0,
        },
        {
            "abstention_id": "ab_1",
            "reason": "alignment_unavailable",
            "start": 3.0,
            "end": 5.0,
        },
    ]
    assert payload["provenance"]["outcomes"]["word_timestamps"] == "abstained"
    assert payload["provenance"]["observed"]["segments_without_words"] == 3
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    ("aligner_segments", "expected_outcome", "expected_words"),
    [
        (
            [{"unit_id": "native_0", "words": []}],
            "produced",
            [],
        ),
        ([{"unit_id": "native_0", "words": None}], "abstained", None),
    ],
)
def test_vibevoice_punctuation_only_empty_mapping_differs_from_absence(
    tmp_path: Path,
    monkeypatch,
    aligner_segments: list[dict],
    expected_outcome: str,
    expected_words: list | None,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "punctuation-only.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice", input_path=source, wants="word_timestamps"
    )
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome("asr", "vibevoice-asr-7b", _complete_vibe_payload([{
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "speaker_id": 0,
                    "text": "……？！",
                }]), 1.0)

        def align(self, **_kwargs):
            return StageOutcome("aligner", "qwen3-forcedaligner", {
                "segments": aligner_segments,
            }, 1.0)

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {
                "torch-vibevoice": {"state": "ready"},
                "mlx": {"state": "ready"},
            },
            "packages": {
                "vibevoice-asr-7b": _ready_multi_package(
                    tmp_path, "vibevoice-asr-7b"
                ),
                "qwen3-forcedaligner": _ready_single_package(
                    tmp_path, "qwen3-forcedaligner"
                ),
            },
        },
        transport=Transport(),
    ).payload

    segment = payload["segments"][0]
    if expected_words is None:
        assert "words" not in segment
        assert payload["provenance"]["observed"]["segments_without_words"] == 1
        assert payload["abstentions"] == [{
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "start": 0.0,
            "end": 1.0,
        }]
    else:
        assert segment["words"] == expected_words
        assert payload["provenance"]["observed"]["segments_without_words"] == 0
        assert payload["abstentions"] == []
    assert payload["provenance"]["outcomes"]["word_timestamps"] == expected_outcome


@pytest.mark.parametrize(
    "aligner_payload",
    [
        {},
        {"segments": []},
        {"segments": [{"unit_id": "native_0"}]},
    ],
)
def test_vibevoice_rejects_incomplete_aligner_ledgers_without_publication(
    tmp_path: Path,
    monkeypatch,
    aligner_payload: dict,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "malformed-aligner-ledger.wav"
    source.write_bytes(b"source")
    output = tmp_path / "result.json"

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload([{
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "speaker_id": 0,
                    "text": "Hello.",
                }]),
                1.0,
            )

        def align(self, **_kwargs):
            return StageOutcome(
                "aligner", "qwen3-forcedaligner", aligner_payload, 1.0
            )

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolve_request(
                stack_id="vibevoice",
                input_path=source,
                wants="word_timestamps",
            ),
            InputMetadata(str(source), 1.0, "wav", 48_000, 1),
            registry={
                "environments": {
                    "torch-vibevoice": {"state": "ready"},
                    "mlx": {"state": "ready"},
                },
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(
                        tmp_path, "vibevoice-asr-7b"
                    ),
                    "qwen3-forcedaligner": _ready_single_package(
                        tmp_path, "qwen3-forcedaligner"
                    ),
                },
            },
            transport=Transport(),
            output=output,
        )

    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["backend"] == "qwen3-forcedaligner"
    assert not output.exists()


def test_vibevoice_rejects_nonzero_nonpartial_exit_without_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "failed-complete-vibe.wav"
    source.write_bytes(b"source")
    output = tmp_path / "result.json"

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload([{
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "speaker_id": 0,
                    "text": "Must not publish.",
                }]),
                1.0,
                returncode=1,
            )

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolve_request(stack_id="vibevoice", input_path=source, wants=()),
            InputMetadata(str(source), 1.0, "wav", 48_000, 1),
            registry={
                "environments": {"torch-vibevoice": {"state": "ready"}},
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(
                        tmp_path, "vibevoice-asr-7b"
                    )
                },
            },
            transport=Transport(),
            output=output,
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["backend"] == "vibevoice-asr-7b"
    assert "unsupported exit 1" in raised.value.payload["detail"]
    assert not output.exists()


def test_vibevoice_overflowing_stage_number_is_a_typed_backend_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "overflow.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome("asr", "vibevoice-asr-7b", _complete_vibe_payload([{
                    "start_time": 10**400,
                    "end_time": 1.0,
                    "text": "Impossible.",
                }]), 1.0)

    output = tmp_path / "overflow-result.json"
    with pytest.raises(refusals.Refusal) as caught:
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
            transport=Transport(),
            output=output,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "asr"
    assert caught.value.payload["backend"] == "vibevoice-asr-7b"
    assert "must be finite and non-negative" in caught.value.payload["detail"]
    assert not output.exists()


def test_vibevoice_complete_result_is_bound_to_generated_text(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "mismatched-postprocess.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome("asr", "vibevoice-asr-7b", {
                "raw_text": json.dumps([{
                    "Start": 0,
                    "End": 1,
                    "Speaker": 0,
                    "Content": "model said raw",
                }]),
                "segments": [{
                    "start_time": 0,
                    "end_time": 1,
                    "speaker_id": 0,
                    "text": "postprocessor fabricated",
                }],
                "hit_max_new_tokens": False,
            }, 1.0)

    output = tmp_path / "must-not-exist.json"
    with pytest.raises(refusals.Refusal) as caught:
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
            transport=Transport(),
            output=output,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "asr"
    assert "differs from generated JSON" in caught.value.payload["detail"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ("timestamp", "must be a finite non-negative number"),
        ("confidence", "must be between 0 and 1"),
    ],
)
def test_firered_overflowing_stage_number_is_a_typed_backend_failure(
    tmp_path: Path, monkeypatch, mutation: str, detail: str
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / f"firered-{mutation}-overflow.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,lid",
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **_kwargs):
            raw = {
                "sentences": [{
                    "start_ms": 0,
                    "end_ms": 1000,
                    "text": "Done.",
                    "lang": "en",
                    "lang_confidence": 0.9,
                }],
                "words": [{
                    "start_ms": 100,
                    "end_ms": 900,
                    "text": "done",
                }],
                "vad_segments_ms": [[0, 1000]],
            }
            if mutation == "timestamp":
                raw["words"][0]["start_ms"] = 10**400
            else:
                raw["sentences"][0]["lang_confidence"] = 10**400
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": raw,
                    "regions": [{
                        "region_id": "vad_0",
                        "start": 0.0,
                        "end": 1.0,
                        "processed": True,
                    }],
                },
                1.0,
            )

    output = tmp_path / f"firered-{mutation}-overflow.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-firered": {"state": "ready"}},
                "packages": {
                    "firered-asr2s": _ready_multi_package(
                        tmp_path, "firered-asr2s"
                    ),
                },
            },
            transport=Transport(),
            output=output,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "firered_process"
    assert caught.value.payload["backend"] == "firered-asr2s"
    assert detail in caught.value.payload["detail"]
    assert not output.exists()


@pytest.mark.parametrize(
    "provided",
    ["0.00003:0.000063", "0.0000629:0.000063"],
)
def test_vibevoice_subsample_eof_range_is_typed_before_model_work(
    tmp_path: Path, monkeypatch, provided: str
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "one-frame.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 0.000063, "wav", 16_000, 1)
    calls: list[str] = []

    class Transport:
        def decode(self, _source, target):
            calls.append("decode")
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0")
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            calls.append("vibevoice")
            raise AssertionError("sub-sample range reached model work")

    output = tmp_path / "one-frame-result.json"
    with pytest.raises(refusals.Refusal) as caught:
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
            transport=Transport(),
            output=output,
            run_range=orchestrator.parse_range(provided),
        )

    assert caught.value.exit_code == 2
    assert caught.value.payload["code"] == "range_invalid"
    assert caught.value.payload["provided"] == provided
    assert "no complete sample" in caught.value.payload["reason"]
    assert calls == ["decode"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("returncode", "transport_failure", "detail"),
    [(1, True, "synthetic failure"), (4, False, "unsupported exit 4")],
)
def test_vibevoice_failed_aligner_is_a_backend_error_and_writes_no_result(
    tmp_path: Path,
    monkeypatch,
    returncode: int,
    transport_failure: bool,
    detail: str,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "aligner-failure.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice", input_path=source, wants="word_timestamps"
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome("asr", "vibevoice-asr-7b", _complete_vibe_payload([{
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "speaker_id": 0,
                    "text": "Hello.",
                }]), 1.0)

        def align(self, **_kwargs):
            outcome = StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {"error": {"message": "synthetic failure"}},
                2.5,
                returncode=returncode,
                peak_rss_bytes=250,
            )
            if not transport_failure:
                return outcome
            raise orchestrator.StageFailure(
                "aligner",
                "qwen3-forcedaligner",
                "synthetic failure",
                outcome=outcome,
            )

    registry = {
        "environments": {
            "torch-vibevoice": {"state": "ready"},
            "mlx": {"state": "ready"},
        },
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(
                tmp_path, "vibevoice-asr-7b"
            ),
            "qwen3-forcedaligner": _ready_single_package(
                tmp_path, "qwen3-forcedaligner"
            ),
        },
    }
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(Exception) as caught:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=Transport(),
            output=output,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "aligner"
    assert caught.value.payload["backend"] == "qwen3-forcedaligner"
    assert detail in caught.value.payload["detail"]
    assert not output.exists()


def test_vibevoice_cap_salvage_has_honest_zero_of_one_units_and_runnable_resume(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "long.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps,vad,overlapped_speech",
    )
    metadata = InputMetadata(str(source), 10.0, "wav", 48_000, 1)

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return [
                {"start": 0.5, "end": 1.0},
                {"start": 4.0, "end": 5.0},
            ]

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 160_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                {
                    "raw_text": (
                        'assistant\n[{"Start":0,"End":2,"Speaker":0,'
                        '"Content":"Complete."},{"Start":2,"End":8,'
                        '"Speaker":1,"Content":"cut'
                    ),
                    "segments": [],
                    "hit_max_new_tokens": True,
                    "generated_tokens": 16384,
                    "eos_observed": False,
                },
                3.0,
                returncode=4,
            )

        def diarize(self, **_kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"startTimeSeconds": 0.2, "endTimeSeconds": 1.2,
                 "speakerId": "S0"},
                {"startTimeSeconds": 4.0, "endTimeSeconds": 6.0,
                 "speakerId": "S1"},
                {"startTimeSeconds": 5.0, "endTimeSeconds": 7.0,
                 "speakerId": "S2"},
            ]}, 0.2)

    registry = {
        "environments": {
            "torch-vibevoice": {"state": "ready"},
            "swift": {"state": "ready"},
        },
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
            "fluidaudio": _ready_fluidaudio(tmp_path),
            "speaker-diarization-coreml": _ready_single_package(
                tmp_path, "speaker-diarization-coreml"
            ),
        },
    }
    output = tmp_path / "long.json"
    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=Transport(),
            vad_detector=Detector(),
            output=output,
        )
    refusal = raised.value
    assert getattr(refusal, "exit_code") == 4
    assert refusal.payload["coverage"] == {
        "scope_intervals": [[0.0, 10.0]],
        "covered_through_seconds": 2.0,
        "covered_fraction": 0.2,
        "covered_intervals": [[0.0, 2.0]],
        "missing_intervals": [[2.0, 10.0]],
        "units_total": 1,
        "units_completed": 0,
    }
    assert "--range 2.0:" in refusal.payload["fix"]
    partial = json.loads((tmp_path / "long.partial.json").read_text(encoding="utf-8"))
    assert partial["complete"] is False
    assert partial["segments"] == [{
        "segment_id": "seg_0", "text": "Complete.", "start": 0.0, "end": 2.0,
    }]
    assert partial["vad_regions"] == [{"start": 0.5, "end": 1.0}]
    assert partial["overlapped_speech"] == []
    assert all(item["start"] < 2.0 for item in partial["abstentions"])


def test_vibevoice_complete_range_restores_canonical_source_timeline(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "original.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps",
    )
    metadata = InputMetadata(str(source), 10.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 160_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, *, audio, **_kwargs):
            assert audio.name == "vibevoice-range.wav"
            with wave.open(str(audio), "rb") as handle:
                assert handle.getnframes() == 4 * 16_000
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {
                            "start_time": 0.25,
                            "end_time": 1.5,
                            "speaker_id": 0,
                            "text": "Range first.",
                        },
                        {
                            "start_time": 3.0,
                            "end_time": 4.0,
                            "speaker_id": 1,
                            "text": "Range last.",
                        },
                    ],
                    generated_tokens=20,
                    eos_observed=True,
                ),
                1.0,
            )

    payload = orchestrator.run(
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
        transport=Transport(),
        run_range=orchestrator.parse_range("3:7"),
    ).payload

    assert payload["source"] == {
        "path": str(source.resolve()),
        "duration_seconds": 10.0,
        "timebase": "seconds",
    }
    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [3.0, 7.0],
        "selected_unit_scope": [3.0, 7.0],
    }
    assert [
        (segment["start"], segment["end"])
        for segment in payload["segments"]
    ] == [(3.25, 4.5), (6.0, 7.0)]


def test_vibevoice_non_frame_aligned_range_uses_actual_clip_bounds(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "fractional.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps",
    )
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 1_601)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, *, audio, **_kwargs):
            with wave.open(str(audio), "rb") as handle:
                assert handle.getnframes() == 1_600
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload([{
                        "start_time": 0.0,
                        # The pinned post-processor may be one microsecond over
                        # the exact selected clip. The adapter clamps only that
                        # accepted tolerance before restoring source time.
                        "end_time": (1_600 / 16_000) + 1e-6,
                        "speaker_id": 0,
                        "text": "Sample aligned.",
                    }], generated_tokens=4, eos_observed=True),
                1.0,
            )

    saved = tmp_path / "fractional-result.json"
    payload = orchestrator.run(
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
        transport=Transport(),
        run_range=orchestrator.parse_range("0.00003:0.10004"),
        output=saved,
    ).payload

    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.00003, 0.10004],
        "selected_unit_scope": [0.000063, 0.100062],
    }
    assert payload["segments"][0]["start"] == 0.000063
    # Both durable fields use the schema's six-decimal source timeline; the
    # adapter was validated against the exact 1601/16000 boundary above.
    assert payload["segments"][0]["end"] == 0.100062
    assert load_result_document(saved).owned_intervals == ((0.000063, 0.100062),)
    assert export_documents([saved], "txt").content == "Sample aligned.\n"


def test_vibevoice_non_frame_partial_and_resume_merge_without_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "resume.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps",
    )
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 1)
    registry = {
        "environments": {"torch-vibevoice": {"state": "ready"}},
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(
                tmp_path, "vibevoice-asr-7b"
            ),
        },
    }

    class BaseTransport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

    class PartialTransport(BaseTransport):
        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                {
                    "raw_text": (
                        'assistant\n[{"Start":0,"End":0.0999775,"Speaker":0,'
                        '"Content":"First."},{"Start":0.2,"End":0.4,'
                        '"Content":"cut'
                    ),
                    "segments": [],
                    "hit_max_new_tokens": True,
                    "generated_tokens": 16_384,
                    "eos_observed": False,
                },
                1.0,
                returncode=4,
            )

    class ResumeTransport(BaseTransport):
        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload([{
                        "start_time": 0.0,
                        "end_time": 0.1,
                        "speaker_id": 0,
                        "text": "Second.",
                    }], generated_tokens=4, eos_observed=True),
                1.0,
            )

    first_output = tmp_path / "first.json"
    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=PartialTransport(),
            output=first_output,
            run_range=orchestrator.parse_range("0.00003:0.50004"),
        )
    assert getattr(raised.value, "exit_code") == 4
    partial = Path(raised.value.payload["output"])
    assert raised.value.payload["coverage"]["covered_intervals"] == [
        [0.000063, 0.10004]
    ]

    rest = tmp_path / "rest.json"
    orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=ResumeTransport(),
        output=rest,
        run_range=orchestrator.parse_range("0.10004:"),
    )

    assert load_result_document(partial).owned_intervals == ((0.000063, 0.10004),)
    assert load_result_document(rest).owned_intervals == ((0.100062, 1.0),)
    product = export_documents([partial, rest], "txt")
    assert product.content == "First.\nSecond.\n"


def test_firered_silero_range_selects_intersections_and_owns_expanded_auxiliary_scope(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "range.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="vad,segment_timestamps,diarization,overlapped_speech",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return [
                {"start": 0.0, "end": 1.0},
                {"start": 1.4, "end": 2.4},
                {"start": 2.6, "end": 2.9},
            ]

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == [
                {"start": 0.0, "end": 1.0},
                {"start": 1.4, "end": 2.4},
            ]
            assert (kwargs["range_start"], kwargs["range_end"]) == (0.5, 1.5)
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {"start_ms": 100, "end_ms": 800, "text": "One.",
                             "lang": None, "lang_confidence": 0},
                            {"start_ms": 1500, "end_ms": 2200, "text": "Two.",
                             "lang": None, "lang_confidence": 0},
                        ],
                        "words": [
                            {"start_ms": 100, "end_ms": 800, "text": "one"},
                            {"start_ms": 1500, "end_ms": 2200, "text": "two"},
                        ],
                        "vad_segments_ms": [[0, 1000], [1400, 2400]],
                    },
                    "regions": [
                        {"region_id": "vad_0", "start": 0.0, "end": 1.0,
                         "processed": True},
                        {"region_id": "vad_1", "start": 1.4, "end": 2.4,
                         "processed": True},
                    ],
                },
                1.0,
            )

        def diarize(self, **_kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"startTimeSeconds": 0.1, "endTimeSeconds": 0.9,
                 "speakerId": "S0", "embedding": [0.0] * 256},
                {"startTimeSeconds": 0.6, "endTimeSeconds": 0.9,
                 "speakerId": "S2", "embedding": [2.0] * 256},
                {"startTimeSeconds": 1.5, "endTimeSeconds": 2.2,
                 "speakerId": "S1", "embedding": [1.0] * 256},
            ]}, 0.2)

    registry = {
        "environments": {
            "torch-firered": {"state": "ready"},
            "swift": {"state": "ready"},
        },
        "packages": {
            "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s"),
            "fluidaudio": _ready_fluidaudio(tmp_path),
            "speaker-diarization-coreml": _ready_single_package(
                tmp_path, "speaker-diarization-coreml"
            ),
        },
    }
    payload = orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=Transport(),
        vad_detector=Detector(),
        run_range=orchestrator.parse_range("0.5:1.5"),
    ).payload
    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.5, 1.5],
        "selected_unit_scope": [0.0, 2.4],
    }
    assert payload["vad_regions"] == [
        {"start": 0.0, "end": 1.0},
        {"start": 1.4, "end": 2.4},
    ]
    assert payload["turns"][0]["start"] == 0.1
    assert "speaker" not in payload["segments"][0]
    assert payload["segments"][1]["speaker"] == "S1"
    assert payload["overlapped_speech"] == [{
        "overlap_id": "overlap_0", "start": 0.6, "end": 0.9,
    }]


def test_firered_native_range_records_expanded_whole_vad_scope(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "native-range.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,diarization",
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] is None
            return StageOutcome(
                "firered_process", "firered-asr2s", {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {"start_ms": 200, "end_ms": 900, "text": "One.",
                             "lang": None, "lang_confidence": 0},
                            {"start_ms": 1400, "end_ms": 2100, "text": "Two.",
                             "lang": None, "lang_confidence": 0},
                        ],
                        "words": [
                            {"start_ms": 200, "end_ms": 900, "text": "one"},
                            {"start_ms": 1400, "end_ms": 2100, "text": "two"},
                        ],
                        "vad_segments_ms": [[100, 1000], [1300, 2300]],
                    },
                    "regions": [
                        {"region_id": "vad_0", "start": 0.1, "end": 1.0,
                         "processed": True},
                        {"region_id": "vad_1", "start": 1.3, "end": 2.3,
                         "processed": True},
                    ],
                }, 1.0,
            )

        def diarize(self, **_kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"startTimeSeconds": 0.01, "endTimeSeconds": 0.7,
                 "speakerId": "S0"},
                {"startTimeSeconds": 2.4, "endTimeSeconds": 2.95,
                 "speakerId": "S1"},
            ]}, 0.2)

    registry = {
        "environments": {
            "torch-firered": {"state": "ready"},
            "swift": {"state": "ready"},
        },
        "packages": {
            "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s"),
            "fluidaudio": _ready_fluidaudio(tmp_path),
            "speaker-diarization-coreml": _ready_single_package(
                tmp_path, "speaker-diarization-coreml"
            ),
        },
    }
    payload = orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=Transport(),
        run_range=orchestrator.parse_range("0:3"),
    ).payload
    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.0, 3.0],
        "selected_unit_scope": [0.1, 2.3],
    }
    assert [turn["start"] for turn in payload["turns"]] == [0.01, 2.4]


def test_firered_range_masks_cross_boundary_overlap_without_republishing_it(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "firered-cross-boundary.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="vad,segment_timestamps,diarization,overlapped_speech",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 12.0, "wav", 48_000, 1)

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return [{"start": 10.0, "end": 11.0}]

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 192_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == [{"start": 10.0, "end": 11.0}]
            assert (kwargs["range_start"], kwargs["range_end"]) == (10.0, 12.0)
            return StageOutcome("firered_process", "firered-asr2s", {
                "complete": True,
                "result": {
                    "sentences": [{
                        "start_ms": 10_000,
                        "end_ms": 11_000,
                        "text": "Cross.",
                        "lang": None,
                        "lang_confidence": 0,
                    }],
                    "words": [{
                        "start_ms": 10_050,
                        "end_ms": 10_900,
                        "text": "cross",
                    }],
                    "vad_segments_ms": [[10_000, 11_000]],
                },
                    "regions": [{
                        "region_id": "vad_0",
                    "start": 10.0,
                    "end": 11.0,
                    "processed": True,
                }],
            }, 1.0)

        def diarize(self, **_kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"startTimeSeconds": 9.0, "endTimeSeconds": 11.0,
                 "speakerId": "S0"},
                {"startTimeSeconds": 9.8, "endTimeSeconds": 10.2,
                 "speakerId": "S1"},
            ]}, 0.2)

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {
                "torch-firered": {"state": "ready"},
                "swift": {"state": "ready"},
            },
            "packages": {
                "firered-asr2s": _ready_multi_package(
                    tmp_path, "firered-asr2s"
                ),
                "fluidaudio": _ready_fluidaudio(tmp_path),
                "speaker-diarization-coreml": _ready_single_package(
                    tmp_path, "speaker-diarization-coreml"
                ),
            },
        },
        transport=Transport(),
        vad_detector=Detector(),
        run_range=orchestrator.parse_range("10:12"),
    ).payload

    assert payload["segments"] == [{
        "segment_id": "seg_0", "text": "Cross.", "start": 10.0, "end": 11.0,
    }]
    # The second S0 single-speaker span proves a label would otherwise be chosen.
    assert payload["turns"] == [{
        "turn_id": "turn_1", "speaker": "S0", "start": 10.2, "end": 11.0,
    }]
    assert payload["overlapped_speech"] == []
    assert payload["abstentions"] == []


def test_vibevoice_range_masks_cross_boundary_overlap_without_republishing_it(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "vibe-cross-boundary.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps,diarization,overlapped_speech",
    )
    metadata = InputMetadata(str(source), 12.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 192_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome("asr", "vibevoice-asr-7b", _complete_vibe_payload([{
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "speaker_id": 0,
                    "text": "Cross.",
                }]), 1.0)

        def diarize(self, **_kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"startTimeSeconds": 9.0, "endTimeSeconds": 11.0,
                 "speakerId": "S0"},
                {"startTimeSeconds": 9.8, "endTimeSeconds": 10.2,
                 "speakerId": "S1"},
            ]}, 0.2)

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {
                "torch-vibevoice": {"state": "ready"},
                "swift": {"state": "ready"},
            },
            "packages": {
                "vibevoice-asr-7b": _ready_multi_package(
                    tmp_path, "vibevoice-asr-7b"
                ),
                "fluidaudio": _ready_fluidaudio(tmp_path),
                "speaker-diarization-coreml": _ready_single_package(
                    tmp_path, "speaker-diarization-coreml"
                ),
            },
        },
        transport=Transport(),
        run_range=orchestrator.parse_range("10:12"),
    ).payload

    assert payload["segments"] == [{
        "segment_id": "seg_0", "text": "Cross.", "start": 10.0, "end": 11.0,
    }]
    assert payload["turns"] == [{
        "turn_id": "turn_0", "speaker": "0", "start": 10.0, "end": 11.0,
    }]
    assert payload["overlapped_speech"] == []
    assert payload["abstentions"] == []


def test_firered_partial_writes_prefix_coverage_and_runnable_resume(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "partial.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered", input_path=source, wants="segment_timestamps,vad"
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **_kwargs):
            return StageOutcome(
                "firered_process", "firered-asr2s", {
                    "complete": False,
                    "result": {
                        "sentences": [{"start_ms": 300, "end_ms": 900,
                                       "text": "Done.", "lang": None,
                                       "lang_confidence": 0}],
                        "words": [{"start_ms": 300, "end_ms": 900,
                                   "text": "done"}],
                        "vad_segments_ms": [[200, 1000]],
                    },
                    "regions": [
                        {"region_id": "vad_0", "start": 0.2, "end": 1.0,
                         "processed": True},
                        {"region_id": "vad_1", "start": 1.5, "end": 2.5,
                         "processed": False},
                    ],
                    "error": {"type": "RuntimeError", "message": "batch failed"},
                }, 1.0, returncode=4,
            )

    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {
            "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")
        },
    }
    output = tmp_path / "partial.json"
    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request, metadata, registry=registry, transport=Transport(), output=output
        )
    refusal = raised.value
    assert getattr(refusal, "exit_code") == 4
    assert refusal.payload["coverage"] == {
        "scope_intervals": [[0.2, 2.5]],
        "covered_through_seconds": 1.5,
        "covered_fraction": 0.565217,
        "covered_intervals": [[0.2, 1.5]],
        "missing_intervals": [[1.5, 2.5]],
        "units_total": 2,
        "units_completed": 1,
    }
    assert "--range 1.5:" in refusal.payload["fix"]
    partial = json.loads(
        (tmp_path / "partial.partial.json").read_text(encoding="utf-8")
    )
    assert partial["complete"] is False
    assert partial["segments"] == [{
        "segment_id": "seg_0", "text": "Done.", "start": 0.3, "end": 0.9,
    }]


@pytest.mark.parametrize(
    ("regions", "detail"),
    [
        (
            [
                {"region_id": "same", "start": 0.1, "end": 0.5,
                 "processed": True},
                {"region_id": "same", "start": 0.6, "end": 1.0,
                 "processed": False},
            ],
            "unique non-empty strings",
        ),
        (
            [{"region_id": "vad_0", "start": True, "end": 0.5,
              "processed": True}],
            "start must be a number",
        ),
        (
            [{"region_id": "vad_0", "start": 10**400, "end": 1.0,
              "processed": True}],
            "start must be finite",
        ),
        (
            [{"region_id": "vad_0", "start": 0.5, "end": 0.5,
              "processed": True}],
            "positive bounds within the source timeline",
        ),
        (
            [
                {"region_id": "vad_0", "start": 0.2, "end": 0.8,
                 "processed": True},
                {"region_id": "vad_1", "start": 0.7, "end": 1.0,
                 "processed": False},
            ],
            "chronological and non-overlapping",
        ),
        (
            [{"region_id": "vad_0", "start": 0.2, "end": 2.1,
              "processed": True}],
            "within the source timeline",
        ),
        (
            [{"region_id": "vad_0", "start": 0.2, "end": 0.8,
              "processed": True}],
            "does not intersect the requested processing range",
        ),
        (
            [
                {"region_id": "vad_0", "start": 0.2, "end": 0.8,
                 "processed": False},
                {"region_id": "vad_1", "start": 0.9, "end": 1.2,
                 "processed": True},
            ],
            "processed regions must form a prefix",
        ),
    ],
)
def test_firered_region_ledger_rejects_untrusted_stage_mutations(
    regions: list[dict], detail: str
) -> None:
    scope = (1.0, 2.0) if "does not intersect" in detail else (0.0, 2.0)
    with pytest.raises((TypeError, ValueError), match=detail):
        native_module._firered_region_ledger(
            regions,
            source_duration=2.0,
            requested_scope=scope,
        )


def test_firered_ledger_comparison_uses_the_stage_millisecond_truncation() -> None:
    ledger = native_module._firered_published_region_ledger([
        {
            "region_id": "vad_0",
            "start": 0.2009,
            "end": 0.9999,
            "processed": True,
        },
        {
            "region_id": "vad_1",
            "start": 1.1119,
            "end": 1.5559,
            "processed": False,
        },
    ])
    assert native_module._firered_published_vad_prefix(ledger) == (
        {"start": 0.2, "end": 0.999},
    )


@pytest.mark.parametrize("mutation", ["dropped", "reordered_ids", "raw_bounds"])
def test_firered_external_silero_stage_ledger_must_preserve_every_selected_unit(
    tmp_path: Path, monkeypatch, mutation: str,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / f"external-ledger-{mutation}.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,vad",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)
    selected = [
        {"start": 0.2009, "end": 0.8009},
        {"start": 1.2009, "end": 1.8009},
    ]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == selected
            regions = [
                {"region_id": "vad_0", **selected[0], "processed": True},
                {"region_id": "vad_1", **selected[1], "processed": True},
            ]
            if mutation == "dropped":
                regions.pop()
            elif mutation == "reordered_ids":
                regions[0]["region_id"], regions[1]["region_id"] = (
                    regions[1]["region_id"], regions[0]["region_id"]
                )
            else:
                # This remains the same 200 ms public FireRed bound, so only
                # exact binding to the selected Silero ledger can detect it.
                regions[0]["start"] = 0.2008
            count = len(regions)
            sentences = [
                {
                    "start_ms": 300,
                    "end_ms": 700,
                    "text": "First.",
                    "lang": None,
                    "lang_confidence": 0,
                },
                {
                    "start_ms": 1300,
                    "end_ms": 1700,
                    "text": "Second.",
                    "lang": None,
                    "lang_confidence": 0,
                },
            ][:count]
            words = [
                {"start_ms": 300, "end_ms": 700, "text": "first"},
                {"start_ms": 1300, "end_ms": 1700, "text": "second"},
            ][:count]
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": sentences,
                        "words": words,
                        "vad_segments_ms": [
                            [int(item["start"] * 1000), int(item["end"] * 1000)]
                            for item in regions
                        ],
                    },
                    "regions": regions,
                },
                1.0,
            )

    output = tmp_path / "must-not-exist.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-firered": {"state": "ready"}},
                "packages": {
                    "firered-asr2s": _ready_multi_package(
                        tmp_path, "firered-asr2s"
                    )
                },
            },
            transport=Transport(),
            vad_detector=Detector(),
            output=output,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "firered_process"
    assert "supplied Silero VAD" in caught.value.payload["detail"]
    assert not output.exists()
    assert not (tmp_path / "must-not-exist.partial.json").exists()


def test_firered_silero_preserves_exact_fractional_vad_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "fractional-silero.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,vad",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)
    selected = [
        {"start": 0.2009, "end": 0.9999},
        {"start": 1.5009, "end": 2.5009},
    ]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == selected
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 300,
                                "end_ms": 900,
                                "text": "First.",
                                "lang": None,
                                "lang_confidence": 0,
                            },
                            {
                                "start_ms": 1600,
                                "end_ms": 2400,
                                "text": "Second.",
                                "lang": None,
                                "lang_confidence": 0,
                            },
                        ],
                        "words": [
                            {"start_ms": 300, "end_ms": 900, "text": "first"},
                            {
                                "start_ms": 1600,
                                "end_ms": 2400,
                                "text": "second",
                            },
                        ],
                        # FireRed's result is validated on its integer-ms grid.
                        "vad_segments_ms": [[200, 999], [1500, 2500]],
                    },
                    # The stage ledger retains the exact selected Silero spans.
                    "regions": [
                        {"region_id": "vad_0", **selected[0], "processed": True},
                        {"region_id": "vad_1", **selected[1], "processed": True},
                    ],
                },
                1.0,
            )

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {"torch-firered": {"state": "ready"}},
            "packages": {
                "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")
            },
        },
        transport=Transport(),
        vad_detector=Detector(),
    ).payload

    assert payload["vad_regions"] == selected


def test_firered_partial_silero_preserves_exact_processed_fractional_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "fractional-silero-partial.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,vad",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)
    selected = [
        {"start": 0.2009, "end": 0.9999},
        {"start": 1.5009, "end": 2.5009},
    ]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == selected
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": False,
                    "result": {
                        "sentences": [{
                            "start_ms": 300,
                            "end_ms": 900,
                            "text": "First.",
                            "lang": None,
                            "lang_confidence": 0,
                        }],
                        "words": [{
                            "start_ms": 300,
                            "end_ms": 900,
                            "text": "first",
                        }],
                        "vad_segments_ms": [[200, 999]],
                    },
                    "regions": [
                        {"region_id": "vad_0", **selected[0], "processed": True},
                        {"region_id": "vad_1", **selected[1], "processed": False},
                    ],
                    "error": {"type": "RuntimeError", "message": "batch failed"},
                },
                1.0,
                returncode=4,
            )

    output = tmp_path / "fractional-silero-partial.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-firered": {"state": "ready"}},
                "packages": {
                    "firered-asr2s": _ready_multi_package(
                        tmp_path, "firered-asr2s"
                    )
                },
            },
            transport=Transport(),
            vad_detector=Detector(),
            output=output,
        )

    assert caught.value.exit_code == 4
    partial = json.loads(
        (tmp_path / "fractional-silero-partial.partial.json").read_text(
            encoding="utf-8"
        )
    )
    assert partial["vad_regions"] == [selected[0]]


def test_firered_fractional_public_scope_is_loadable_and_exportable(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "fractional-owned.wav"
    source.write_bytes(b"source")
    selected = [{"start": 0.0009, "end": 1.0009}]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == selected
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [{
                            "start_ms": 0,
                            "end_ms": 1000,
                            "text": "Whole unit.",
                            "lang": None,
                            "lang_confidence": 0,
                        }],
                        "words": [{
                            "start_ms": 100,
                            "end_ms": 900,
                            "text": "whole unit",
                        }],
                        "vad_segments_ms": [[0, 1000]],
                    },
                    "regions": [{
                        "region_id": "vad_0",
                        **selected[0],
                        "processed": True,
                    }],
                },
                1.0,
            )

    output = tmp_path / "fractional-owned.json"
    payload = orchestrator.run(
        resolve_request(
            stack_id="firered",
            input_path=source,
            wants="vad,segment_timestamps,word_timestamps",
            vad="silero-vad",
        ),
        InputMetadata(str(source), 2.0, "wav", 48_000, 1),
        registry={
            "environments": {"torch-firered": {"state": "ready"}},
            "packages": {
                "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")
            },
        },
        transport=Transport(),
        vad_detector=Detector(),
        run_range=orchestrator.parse_range("0.0005:0.5"),
        output=output,
    ).payload

    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.0005, 0.5],
        "selected_unit_scope": [0.0, 1.0],
    }
    assert load_result_document(output).owned_intervals == ((0.0, 1.0),)
    assert "Whole unit." in export_documents([output], "srt").content


def test_firered_fractional_partial_resume_uses_public_ownership_once(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "fractional-resume.wav"
    source.write_bytes(b"source")
    selected = [
        {"start": 0.0009, "end": 1.0008},
        {"start": 1.0009, "end": 2.0},
    ]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class BaseTransport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

    class PartialTransport(BaseTransport):
        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == selected
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": False,
                    "result": {
                        "sentences": [{
                            "start_ms": 100,
                            "end_ms": 900,
                            "text": "First.",
                            "lang": None,
                            "lang_confidence": 0,
                        }],
                        "words": [{
                            "start_ms": 100,
                            "end_ms": 900,
                            "text": "first",
                        }],
                        "vad_segments_ms": [[0, 1000]],
                    },
                    "regions": [
                        {"region_id": "vad_0", **selected[0], "processed": True},
                        {"region_id": "vad_1", **selected[1], "processed": False},
                    ],
                    "error": {"type": "RuntimeError", "message": "batch failed"},
                },
                1.0,
                returncode=4,
            )

    class ResumeTransport(BaseTransport):
        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == [selected[1]]
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [{
                            "start_ms": 1100,
                            "end_ms": 1900,
                            "text": "Second.",
                            "lang": None,
                            "lang_confidence": 0,
                        }],
                        "words": [{
                            "start_ms": 1100,
                            "end_ms": 1900,
                            "text": "second",
                        }],
                        "vad_segments_ms": [[1000, 2000]],
                    },
                    "regions": [{
                        "region_id": "vad_0",
                        **selected[1],
                        "processed": True,
                    }],
                },
                1.0,
            )

    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="vad,segment_timestamps",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)
    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {
            "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")
        },
    }
    requested = tmp_path / "first.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=PartialTransport(),
            vad_detector=Detector(),
            output=requested,
        )
    partial = Path(caught.value.payload["output"])
    assert caught.value.payload["coverage"]["covered_through_seconds"] == 1.0
    assert "--range 1.0:" in caught.value.payload["fix"]

    rest = tmp_path / "rest.json"
    orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=ResumeTransport(),
        vad_detector=Detector(),
        run_range=orchestrator.parse_range("1.0:"),
        output=rest,
    )

    assert load_result_document(partial).owned_intervals == ((0.0, 1.0),)
    assert load_result_document(rest).owned_intervals == ((1.0, 2.0),)
    assert export_documents([partial, rest], "txt").content == "First.\nSecond.\n"


def test_firered_rejects_same_length_vad_bound_mutation_without_replacing_output(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "ledger-mutation.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered", input_path=source, wants="segment_timestamps,vad"
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **_kwargs):
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [{
                            "start_ms": 300,
                            "end_ms": 900,
                            "text": "Done.",
                            "lang": None,
                            "lang_confidence": 0,
                        }],
                        "words": [{
                            "start_ms": 300,
                            "end_ms": 900,
                            "text": "done",
                        }],
                        # Same cardinality as the processed ledger, but a mutated
                        # start bound that the former length-only check accepted.
                        "vad_segments_ms": [[250, 1000]],
                    },
                    "regions": [{
                        "region_id": "vad_0",
                        "start": 0.2,
                        "end": 1.0,
                        "processed": True,
                    }],
                },
                1.0,
            )

    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {
            "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")
        },
    }
    output = tmp_path / "existing.json"
    original = b"preexisting output must survive"
    output.write_bytes(original)

    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=Transport(),
            output=output,
            force=True,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "firered_process"
    assert caught.value.payload["backend"] == "firered-asr2s"
    assert "exactly match the processed region prefix" in caught.value.payload["detail"]
    assert output.read_bytes() == original


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
            def inspect(_checkout: Path) -> orchestrator._CheckoutState:
                raise ValueError("not a Git checkout")
        else:
            state = orchestrator._CheckoutState(
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

            def inspect(_checkout: Path) -> orchestrator._CheckoutState:
                return state

        monkeypatch.setattr(orchestrator, "_inspect_checkout", inspect)

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

    monkeypatch.setattr(orchestrator, "_frozen_packages", lambda _path: {})
    monkeypatch.setattr(orchestrator, "_python_runtime_runs", lambda _path: True)
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

    def fail_inspection(path: Path) -> orchestrator._CheckoutState:
        raise AssertionError(f"preflight inspected external checkout {path}")

    def fail_hash(path: Path) -> str:
        raise AssertionError(f"preflight hashed external checkout file {path}")

    monkeypatch.setattr(orchestrator, "_inspect_checkout", fail_inspection)
    monkeypatch.setattr(orchestrator, "_checkout_file_digest", fail_hash)
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
        orchestrator,
        "_inspect_checkout",
        lambda _checkout: orchestrator._CheckoutState(
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
