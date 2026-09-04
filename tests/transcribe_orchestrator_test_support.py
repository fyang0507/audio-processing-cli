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
from audio_cli.packages import integrity as package_integrity
from audio_cli.transcribe import _orchestrator_runtime as orchestrator_runtime
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

    monkeypatch.setattr(package_integrity, "_hub_snapshot_index", snapshot_index)
    monkeypatch.setattr(
        orchestrator_runtime,
        "_inspect_checkout",
        lambda _checkout: orchestrator_runtime._CheckoutState(
            head=FLUID_REVISION,
            modified=pkg.checkout_patch_expectation(
                env.packages()["fluidaudio"]
            )[1],
            untracked=(),
        ),
    )
    monkeypatch.setattr(
        orchestrator_runtime,
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


__all__ = [
    "FakeTransport",
    "FakeVad",
    "FullFakeTransport",
    "InputMetadata",
    "NoncontiguousTransport",
    "Path",
    "SileroOnnxVad",
    "StageOutcome",
    "StageTransport",
    "UnsafeOutputError",
    "audio_paths",
    "build_plan",
    "cli",
    "env",
    "export_documents",
    "full_registry",
    "inspect",
    "json",
    "orchestrator",
    "os",
    "provisioned_runtime_root",
    "pytest",
    "refusals",
    "registry",
    "replace",
    "request",
    "resolve_request",
    "serialize_plan",
    "shlex",
    "shutil",
    "wave",
]
