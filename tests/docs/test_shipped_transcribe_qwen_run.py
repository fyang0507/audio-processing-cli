"""Shipped transcription run shapes against the happy-path specification."""

from __future__ import annotations

import re
import wave
from pathlib import Path

import pytest

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.transcribe import refusals
from audio_cli.transcribe.catalog import InputMetadata
from audio_cli.transcribe.orchestrator import run
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.transport import StageOutcome
from tests.docs.shipped_command_test_support import (
    assert_documented_shape,
    configure_isolated_root,
    documented_block,
)


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
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {"start_s": 2.28, "end_s": 4.79, "speaker": "S1"},
                        {"start_s": 5.06, "end_s": 7.51, "speaker": "S2"},
                        {"start_s": 41.86, "end_s": 42.07, "speaker": "S1"},
                        {"start_s": 118.44, "end_s": 118.79, "speaker": "S2"},
                    ]
                },
                2.0,
                peak_rss_bytes=100,
            )

        def qwen(self, *, units, **kwargs):
            texts = [
                "language Chinese<asr_text>好，我們今天想聊一下你的工作。",
                "language Chinese<asr_text>嗯，好啊，我做咗五年設計。",
            ]
            return StageOutcome(
                "asr",
                "qwen3-asr-1.7b-8bit",
                {
                    "units": [
                        {
                            "unit_id": unit["unit_id"],
                            "processed": True,
                            "text": texts[index],
                        }
                        for index, unit in enumerate(units)
                    ]
                },
                3.0,
                peak_rss_bytes=300,
                peak_mps_live_bytes=300,
            )

        def align(self, *, segments, **kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {
                            "unit_id": segment["unit_id"],
                            "words": [
                                {
                                    "text": re.sub(r"\W", "", segment["text"]),
                                    "start": segment["start"],
                                    "end": segment["end"],
                                }
                            ],
                        }
                        for segment in segments
                    ]
                },
                4.0,
                peak_rss_bytes=200,
                peak_mps_live_bytes=200,
            )

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
            else tmp_path
            / "hub"
            / f"models--{package.source['repo'].replace('/', '--')}"
            / "snapshots"
            / revision
        )
        target.mkdir(parents=True, exist_ok=True)
        for pattern in package.source.get("allow_patterns", ()):
            marker = target / (f"{pattern[:-3]}/model.mil" if pattern.endswith("/**") else pattern)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_bytes(b"")
        materialized = {"path": str(target), "revision": revision, "bytes": 0}
        if identifier == "fluidaudio":
            materialized.update({"built": True, "product_runs": True})
            product = target / ".build" / "release" / "fluidaudiocli"
            product.parent.mkdir(parents=True)
            product.write_text("#!/bin/sh\n", encoding="utf-8")
            product.chmod(0o755)
            patched = target / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
            patched.parent.mkdir(parents=True)
            patched.write_bytes(b"patched\n")
            patch_names, _modified, patch_digests = pkg.checkout_patch_expectation(package)
            materialized.update(
                {
                    "product_path": product.relative_to(target).as_posix(),
                    "product_sha256": pkg.sha256_file(product),
                    "patches_applied": list(patch_names),
                    "patched_file_digests": patch_digests,
                }
            )
        entries[identifier] = {"state": "ready", "materialized": materialized}

    request = resolve_request(
        stack_id="qwen-1.7b",
        input_path=Path("meeting.m4a"),
        wants="diarization,word_timestamps",
        language="Cantonese",
    )
    metadata = InputMetadata("meeting.m4a", 1794.2, "m4a", 44_100, 2)
    registry = {
        "environments": {"mlx": {"state": "ready"}, "swift": {"state": "ready"}},
        "packages": entries,
    }
    actual = run(request, metadata, registry=registry, transport=Transport()).payload
    expected_plan = serialize_plan(
        build_plan(
            request,
            metadata,
            provisioned_packages=set(entries),
        )
    )
    expected_plan.pop("sample_output")
    assert actual["provenance"]["plan"] == expected_plan
    documented = documented_block("audio transcribe run --input meeting.m4a \\")
    # §1.2 publishes the resolved plan in full; §1.4 intentionally omits that repeated object.
    documented["provenance"]["plan"] = expected_plan
    assert_documented_shape(actual, documented, "audio transcribe run --stack qwen-1.7b")


def test_qwen_incomplete_refusal_emits_the_shape_happy_path_publishes(tmp_path) -> None:
    interpreter = tmp_path / "root" / "envs" / "mlx" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    revision = "89e96d92ba34aca20b3e29fb10cc284097d1219f"
    source = env.packages()["qwen3-asr-0.6b-8bit"].source
    model = (
        tmp_path / "hub" / f"models--{source['repo'].replace('/', '--')}" / "snapshots" / revision
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
            return StageOutcome(
                "asr",
                "qwen3-asr-0.6b-8bit",
                {
                    "units": [
                        {
                            "unit_id": item["unit_id"],
                            "processed": index == 0,
                            "text": "Hello." if index == 0 else "",
                        }
                        for index, item in enumerate(units)
                    ]
                },
                2.0,
                returncode=4,
            )

    request = resolve_request(
        stack_id="qwen-0.6b",
        input_path=tmp_path / "meeting.m4a",
        wants=(),
    )
    metadata = InputMetadata(str(request.input_path), 361.0, "m4a", 44_100, 2)
    registry = {
        "environments": {"mlx": {"state": "ready"}},
        "packages": {
            "qwen3-asr-0.6b-8bit": {
                "state": "ready",
                "materialized": {
                    "path": str(model),
                    "revision": revision,
                    "bytes": 0,
                },
            }
        },
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
