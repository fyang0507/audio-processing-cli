"""Shipped transcription run shapes against the happy-path specification."""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import pytest
from shipped_command_test_support import (
    assert_documented_shape,
    configure_isolated_root,
    documented_block,
)

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.transcribe import refusals
from audio_cli.transcribe.catalog import InputMetadata
from audio_cli.transcribe.orchestrator import run
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.transport import StageOutcome


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    configure_isolated_root(tmp_path, monkeypatch)


def test_qwen_run_emits_the_shape_happy_path_publishes(tmp_path) -> None:
    interpreter = tmp_path / "root" / "envs" / "mlx" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)

    class Transport:
        def decode(self, source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * (120 * 16_000))
            return StageOutcome("decode", "ffmpeg", {}, 1.0, peak_rss_bytes=50)

        def diarize(self, **kwargs):
            return StageOutcome("diarizer", "fluidaudio", {"segments": [
                {"start_s": 2.28, "end_s": 4.79, "speaker": "S1"},
                {"start_s": 5.06, "end_s": 7.51, "speaker": "S2"},
                {"start_s": 41.86, "end_s": 42.07, "speaker": "S1"},
                {"start_s": 118.44, "end_s": 118.79, "speaker": "S2"},
            ]}, 2.0, peak_rss_bytes=100)

        def qwen(self, *, units, **kwargs):
            texts = [
                "language Chinese<asr_text>好，我們今天想聊一下你的工作。",
                "language Chinese<asr_text>嗯，好啊，我做咗五年設計。",
            ]
            return StageOutcome("asr", "qwen3-asr-1.7b-8bit", {"units": [{
                "unit_id": unit["unit_id"], "processed": True, "text": texts[index],
            } for index, unit in enumerate(units)]}, 3.0, peak_rss_bytes=300,
                                peak_mps_live_bytes=300)

        def align(self, *, segments, **kwargs):
            return StageOutcome("aligner", "qwen3-forcedaligner", {"segments": [{
                "unit_id": segment["unit_id"], "words": [{
                    "text": re.sub(r"\W", "", segment["text"]),
                    "start": segment["start"], "end": segment["end"],
                }],
            } for segment in segments]}, 4.0, peak_rss_bytes=200,
                                  peak_mps_live_bytes=200)

    revisions = {
        "qwen3-asr-1.7b-8bit": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
        "qwen3-forcedaligner": "0e1a68e91d815300c7c9754b2a7639378b23db15",
        "speaker-diarization-coreml": "1ed7a662fdc7109e36d822db793ee6eebdaf8594",
        "fluidaudio": "19600a485baa4998812e4654b70d2bab8f2c9949",
    }
    entries = {}
    for identifier, revision in revisions.items():
        package = env.packages()[identifier]
        target = (
            paths.checkout_dir(package.environment, package.id)
            if identifier == "fluidaudio"
            else tmp_path / "hub"
            / f"models--{package.source['repo'].replace('/', '--')}"
            / "snapshots" / revision
        )
        target.mkdir(parents=True, exist_ok=True)
        for pattern in package.source.get("allow_patterns", ()):
            marker = target / (
                f"{pattern[:-3]}/model.mil" if pattern.endswith("/**") else pattern
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_bytes(b"")
        materialized = {"path": str(target), "revision": revision, "bytes": 0}
        if identifier == "fluidaudio":
            materialized.update({"built": True, "product_runs": True})
            product = target / ".build" / "release" / "fluidaudiocli"
            product.parent.mkdir(parents=True)
            product.write_text("#!/bin/sh\n", encoding="utf-8")
            product.chmod(0o755)
            patched = (
                target / "Sources" / "FluidAudioCLI" / "Commands"
                / "ProcessCommand.swift"
            )
            patched.parent.mkdir(parents=True)
            patched.write_bytes(b"patched\n")
            patch_names, _modified, patch_digests = pkg.checkout_patch_expectation(package)
            materialized.update({
                "product_path": product.relative_to(target).as_posix(),
                "product_sha256": pkg.sha256_file(product),
                "patches_applied": list(patch_names),
                "patched_file_digests": patch_digests,
            })
        entries[identifier] = {"state": "ready", "materialized": materialized}

    request = resolve_request(
        stack_id="qwen-1.7b", input_path=Path("meeting.m4a"),
        wants="diarization,word_timestamps", language="Cantonese",
    )
    metadata = InputMetadata("meeting.m4a", 1794.2, "m4a", 44_100, 2)
    registry = {
        "environments": {"mlx": {"state": "ready"}, "swift": {"state": "ready"}},
        "packages": entries,
    }
    actual = run(
        request, metadata, registry=registry, transport=Transport()
    ).payload
    expected_plan = serialize_plan(build_plan(
        request, metadata, provisioned_packages=set(entries),
    ))
    expected_plan.pop("sample_output")
    assert actual["provenance"]["plan"] == expected_plan
    documented = documented_block("audio transcribe run --input meeting.m4a \\")
    # §1.2 publishes the resolved plan in full; §1.4 intentionally omits that repeated object.
    documented["provenance"]["plan"] = expected_plan
    assert_documented_shape(actual, documented, "audio transcribe run --stack qwen-1.7b")


def _native_package_entry(tmp_path: Path, identifier: str) -> dict:
    package = env.packages()[identifier]
    source = package.source
    checkout = paths.checkout_dir(package.environment, package.id)
    checkout.mkdir(parents=True)
    if source["type"] == "huggingface_multi":
        locations = {}
        for repository in source["repos"]:
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
        materialized = {
            "paths": locations,
            "revisions": [item["revision"] for item in source["repos"]],
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
    else:
        target = (
            tmp_path / "hub" / f"models--{source['repo'].replace('/', '--')}"
            / "snapshots" / source["revision"]
        )
        target.mkdir(parents=True, exist_ok=True)
        materialized = {
            "path": str(target), "revision": source["revision"], "bytes": 0,
        }
    return {"state": "ready", "materialized": materialized}


def _native_interpreter(tmp_path: Path, environment: str) -> None:
    interpreter = tmp_path / "root" / "envs" / environment / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)


def test_firered_run_emits_the_shape_happy_path_publishes(tmp_path) -> None:
    _native_interpreter(tmp_path, "torch-firered")

    class Transport:
        def decode(self, source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * round(27.8 * 16_000))
            return StageOutcome("decode", "ffmpeg", {}, 0.09)

        def firered(self, **kwargs):
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                    "sentences": [
                        {"start_ms": 380, "end_ms": 1620,
                         "text": "This is a测试。", "lang": "en",
                         "lang_confidence": 0.724},
                        {"start_ms": 3280, "end_ms": 4490,
                         "text": "我们要来看哈，", "lang": "zh",
                         "lang_confidence": 0.961},
                    ],
                    "words": [
                        {"start_ms": 410, "end_ms": 620, "text": "this"},
                        {"start_ms": 620, "end_ms": 740, "text": "is"},
                        {"start_ms": 740, "end_ms": 810, "text": "a"},
                        {"start_ms": 1020, "end_ms": 1240, "text": "测"},
                        {"start_ms": 1240, "end_ms": 1480, "text": "试"},
                        {"start_ms": 3310, "end_ms": 3440, "text": "我"},
                        {"start_ms": 3440, "end_ms": 3580, "text": "们"},
                        {"start_ms": 3580, "end_ms": 3720, "text": "要"},
                        {"start_ms": 3720, "end_ms": 3860, "text": "来"},
                        {"start_ms": 3860, "end_ms": 4030, "text": "看"},
                        {"start_ms": 4030, "end_ms": 4290, "text": "哈"},
                    ],
                    "vad_segments_ms": [[380, 1660], [3280, 4520]],
                    },
                    "regions": [
                        {"region_id": "vad_0", "start": 0.38, "end": 1.66,
                         "processed": True},
                        {"region_id": "vad_1", "start": 3.28, "end": 4.52,
                         "processed": True},
                    ],
                },
                20.65,
                peak_rss_bytes=13_169_377_280,
                wall_seconds_by_stage={
                    "vad": 0.61, "lid": 8.83, "asr": 9.14, "punctuator": 1.07,
                },
            )

    entries = {"firered-asr2s": _native_package_entry(tmp_path, "firered-asr2s")}
    request = resolve_request(
        stack_id="firered",
        input_path=Path("field.wav"),
        wants="verbatim,word_timestamps,vad,segment_timestamps,lid",
    )
    metadata = InputMetadata("field.wav", 27.8, "wav", 48_000, 1)
    actual = run(
        request,
        metadata,
        registry={
            "environments": {"torch-firered": {"state": "ready"}},
            "packages": entries,
        },
        transport=Transport(),
    ).payload
    expected_plan = serialize_plan(build_plan(
        request, metadata, provisioned_packages=set(entries)
    ))
    expected_plan.pop("sample_output")
    documented = documented_block(
        "audio transcribe run --input field.wav --stack firered \\\n"
    )
    documented["provenance"]["plan"] = expected_plan
    assert_documented_shape(actual, documented, "audio transcribe run --stack firered")


def test_vibevoice_run_emits_the_shape_happy_path_publishes(tmp_path) -> None:
    _native_interpreter(tmp_path, "torch-vibevoice")
    _native_interpreter(tmp_path, "mlx")

    class Transport:
        def decode(self, source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * round(112.4 * 16_000))
            return StageOutcome("decode", "ffmpeg", {}, 0.44)

        def vibevoice(self, **kwargs):
            segments = [
                {"start_time": 0.0, "end_time": 4.52,
                 "speaker_id": 0, "text": "So, um, this is the new editor."},
                {"start_time": 4.52, "end_time": 6.08,
                 "text": "[Environmental Sounds]"},
                {"start_time": 6.08, "end_time": 9.41,
                 "speaker_id": 1, "text": "And it renders straight away?"},
            ]
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                {
                    "raw_text": json.dumps(segments),
                    "segments": segments,
                    "hit_max_new_tokens": False,
                    "generated_tokens": 100,
                    "eos_observed": True,
                },
                53.16,
                peak_mps_live_bytes=19_983_452_160,
            )

        def align(self, *, segments, **kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {"segments": [
                    {"unit_id": segments[0]["unit_id"], "words": [
                        {"text": "So", "start": 0.31, "end": 0.48},
                        {"text": "um", "start": 0.50, "end": 0.65},
                        {"text": "this", "start": 0.70, "end": 0.90},
                        {"text": "is", "start": 0.95, "end": 1.05},
                        {"text": "the", "start": 1.10, "end": 1.25},
                        {"text": "new", "start": 1.30, "end": 1.55},
                        {"text": "editor", "start": 1.60, "end": 2.00},
                    ]},
                    {"unit_id": segments[1]["unit_id"], "words": [
                        {"text": "And", "start": 6.22, "end": 6.39},
                        {"text": "it", "start": 6.40, "end": 6.50},
                        {"text": "renders", "start": 6.55, "end": 6.90},
                        {"text": "straight", "start": 6.95, "end": 7.30},
                        {"text": "away", "start": 7.35, "end": 7.70},
                    ]},
                ]},
                3.72,
                peak_mps_live_bytes=2_210_398_208,
            )

    entries = {
        "vibevoice-asr-7b": _native_package_entry(tmp_path, "vibevoice-asr-7b"),
        "qwen3-forcedaligner": _native_package_entry(tmp_path, "qwen3-forcedaligner"),
    }
    request = resolve_request(
        stack_id="vibevoice",
        input_path=Path("demo.mp4"),
        wants="verbatim,diarization,segment_timestamps,word_timestamps",
    )
    metadata = InputMetadata("demo.mp4", 112.4, "mp4", 48_000, 2)
    actual = run(
        request,
        metadata,
        registry={
            "environments": {
                "torch-vibevoice": {"state": "ready"},
                "mlx": {"state": "ready"},
            },
            "packages": entries,
        },
        transport=Transport(),
    ).payload
    expected_plan = serialize_plan(build_plan(
        request, metadata, provisioned_packages=set(entries)
    ))
    expected_plan.pop("sample_output")
    documented = documented_block(
        "audio transcribe run --input demo.mp4 --stack vibevoice \\\n"
    )
    documented["provenance"]["plan"] = expected_plan
    assert_documented_shape(actual, documented, "audio transcribe run --stack vibevoice")


def test_qwen_incomplete_refusal_emits_the_shape_happy_path_publishes(tmp_path) -> None:
    interpreter = tmp_path / "root" / "envs" / "mlx" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    revision = "89e96d92ba34aca20b3e29fb10cc284097d1219f"
    source = env.packages()["qwen3-asr-0.6b-8bit"].source
    model = (
        tmp_path / "hub" / f"models--{source['repo'].replace('/', '--')}"
        / "snapshots" / revision
    )
    model.mkdir(parents=True)

    class PartialTransport:
        def decode(self, source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * (361 * 16_000))
            return StageOutcome("decode", "ffmpeg", {}, 1.0, peak_rss_bytes=50)

        def qwen(self, *, units, **kwargs):
            return StageOutcome("asr", "qwen3-asr-0.6b-8bit", {"units": [{
                "unit_id": item["unit_id"],
                "processed": index == 0,
                "text": "Hello." if index == 0 else "",
            } for index, item in enumerate(units)]}, 2.0, returncode=4)

    request = resolve_request(
        stack_id="qwen-0.6b", input_path=tmp_path / "meeting.m4a", wants=(),
    )
    metadata = InputMetadata(str(request.input_path), 361.0, "m4a", 44_100, 2)
    registry = {
        "environments": {"mlx": {"state": "ready"}},
        "packages": {"qwen3-asr-0.6b-8bit": {
            "state": "ready",
            "materialized": {
                    "path": str(model),
                        "revision": revision,
                    "bytes": 0,
                },
        }},
    }
    with pytest.raises(refusals.Refusal) as raised:
        run(
            request,
            metadata,
            registry=registry,
            transport=PartialTransport(),
            output=tmp_path / "meeting.timed.json",
        )
    documented = documented_block("A Qwen budget stop emits")
    assert_documented_shape(
        raised.value.payload,
        documented,
        "audio transcribe run incomplete refusal",
    )
