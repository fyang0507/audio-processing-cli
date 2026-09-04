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

__all__ = [name for name in globals() if not name.startswith("__")]
