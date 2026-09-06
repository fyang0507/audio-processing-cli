"""Shared setup and document-shape helpers for shipped-command acceptance tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.media import hash_file
from audio_cli.packages import integrity as package_integrity
from audio_cli.transcribe.execution import runtime as orchestrator_runtime
from audio_cli.transcribe.result import NormalizedResult, serialize_result
from tests.docs.spec_document_loader import read_spec_document

REPO = Path(__file__).resolve().parents[2]
HAPPY_PATH = REPO / "docs" / "TRANSCRIBE_HAPPY_PATH.md"
CONTRACT = REPO / "docs" / "TRANSCRIBE_CONTRACT.md"


def write_export_result(
    path: Path,
    *,
    source: str | Path,
    segments: list[dict],
    outcomes: dict[str, str],
    language: str | None,
) -> None:
    optional_arrays = {"turns": []} if "diarization" in outcomes else {}
    payload = serialize_result(
        NormalizedResult(
            source={
                "path": str(source),
                "duration_seconds": 10.0,
                "timebase": "seconds",
            },
            segments=segments,
            abstentions=[],
            provenance={
                "stack": "qwen-1.7b",
                "outcomes": outcomes,
                "observed": {},
                "plan": {
                    "roles": {"asr": {"config": {"language": language}}},
                },
            },
            requested_capabilities=frozenset(outcomes),
            **optional_arrays,
        )
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def configure_isolated_root(tmp_path, monkeypatch):
    """The document shows an unprovisioned machine, and so must this."""
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(tmp_path / "root"))

    def inspect(checkout: Path) -> orchestrator_runtime._CheckoutState:
        if "fluidaudio" in str(checkout):
            package = env.packages()["fluidaudio"]
            _patches, modified, _digests = pkg.checkout_patch_expectation(package)
            return orchestrator_runtime._CheckoutState(
                head=package.source["commit"], modified=modified, untracked=()
            )
        package_id = "vibevoice-asr-7b" if "vibevoice-asr-7b" in str(checkout) else "firered-asr2s"
        package = env.packages()[package_id]
        _patches, modified, _digests = pkg.checkout_patch_expectation(package)
        return orchestrator_runtime._CheckoutState(
            head=package.checkout["resolved_commit"],
            modified=modified,
            untracked=(),
        )

    monkeypatch.setattr(orchestrator_runtime, "_inspect_checkout", inspect)

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

    monkeypatch.setattr(orchestrator_runtime, "_checkout_file_digest", digest)

    def frozen(interpreter: Path) -> dict[str, str]:
        environment_name = interpreter.parent.parent.name
        installed: dict[str, str] = {}
        for package in env.packages().values():
            if package.environment != environment_name or package.checkout is None:
                continue
            checkout = paths.checkout_dir(package.environment, package.id)
            installed[package.checkout["distribution"]] = (
                f"@ {checkout.resolve(strict=False).as_uri()}"
            )
        return installed

    monkeypatch.setattr(orchestrator_runtime, "_frozen_packages", frozen)

    def snapshot_index() -> dict[tuple[str, str], Path]:
        found = {}
        for package in env.packages().values():
            source = package.source
            repositories = (
                source["repos"]
                if source["type"] == "huggingface_multi"
                else [source]
                if source["type"] == "huggingface"
                else []
            )
            for repository in repositories:
                found[(repository["repo"], repository["revision"])] = (
                    tmp_path
                    / "hub"
                    / f"models--{repository['repo'].replace('/', '--')}"
                    / "snapshots"
                    / repository["revision"]
                )
        return found

    monkeypatch.setattr(package_integrity, "_hub_snapshot_index", snapshot_index)


def _native_package_entry(tmp_path: Path, identifier: str) -> dict:
    package = env.packages()[identifier]
    source = package.source
    checkout = paths.checkout_dir(package.environment, package.id)
    checkout.mkdir(parents=True)
    if source["type"] == "huggingface_multi":
        locations = {}
        for repository in source["repos"]:
            target = (
                tmp_path
                / "hub"
                / f"models--{repository['repo'].replace('/', '--')}"
                / "snapshots"
                / repository["revision"]
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
            materialized.update(
                {
                    "patches_applied": [Path(patch_name).name],
                    "patched_file_digests": expected_digests,
                }
            )
    else:
        target = (
            tmp_path
            / "hub"
            / f"models--{source['repo'].replace('/', '--')}"
            / "snapshots"
            / source["revision"]
        )
        target.mkdir(parents=True, exist_ok=True)
        materialized = {
            "path": str(target),
            "revision": source["revision"],
            "bytes": 0,
        }
    return {"state": "ready", "materialized": materialized}


def _native_interpreter(tmp_path: Path, environment: str) -> None:
    interpreter = tmp_path / "root" / "envs" / environment / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)


class StubToolchain(pkg.Toolchain):
    """Present tools, no subprocesses. §0 is the everything-installed case."""

    def which(self, tool: str) -> str | None:
        return f"/usr/bin/{tool}"


def documented_doctor() -> dict:
    """The one JSON block under §0."""
    text = read_spec_document(HAPPY_PATH)
    section = text.index("## 0. Once per machine")
    block = re.search(r"```json\n(.*?)```", text[section:], re.S)
    assert block is not None, "TRANSCRIBE_HAPPY_PATH.md §0 no longer publishes a JSON block"
    return json.loads(block.group(1))


def shape(node, trail: str = "") -> dict[str, str]:
    """Every leaf's path mapped to its JSON type. Dict keys are part of the path."""
    if isinstance(node, dict):
        found: dict[str, str] = {}
        for key, value in node.items():
            found.update(shape(value, f"{trail}.{key}" if trail else key))
        return found
    if isinstance(node, list):
        # Cardinality is never promised; the element shape is. An empty list is its own leaf,
        # because there is no element to describe.
        #
        # Elements are unioned, not intersected, because they legitimately differ -- `fluidaudio`
        # carries `bytes: null` where a weights package carries a count, and a failed `verify`
        # entry carries different keys than a passing one. The cost of that choice, stated so it
        # is not mistaken for coverage: a key dropped from *some* elements of a documented array
        # stays invisible. Only a key dropped from all of them fails.
        if not node:
            return {f"{trail}[]": "empty"}
        merged: dict[str, str] = {}
        for item in node:
            merged.update(shape(item, f"{trail}[]"))
        return merged
    return {trail: type(node).__name__}


COMPATIBLE = {"empty", "list"}


def provisioned_like_the_document(tmp_path: Path) -> pkg.Provisioner:
    """A root holding every package, so `list` and `verify` print their populated shape.

    `doctor` reports the same keys whatever is provisioned, so it needs no fixture. These two do
    not: an empty root prints an empty `packages` array, and an array with no element promises no
    element shape. The document's blocks depict a fully-provisioned machine, so the fixture has to
    be one.
    """
    from tests.audio_cli.packages.test_packages import (  # sibling module, same rootdir
        FakeFetcher,
        FakeToolchain,
    )

    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))
    provisioner.pull(list(pkg.select(stack="qwen-1.7b")))
    provisioner.pull(list(pkg.select(stack="firered")))
    provisioner.pull(list(pkg.select(stack="vibevoice")))
    return provisioner


def documented_block(anchor: str, *, document: Path = HAPPY_PATH) -> dict:
    text = read_spec_document(document)
    block = re.search(r"```json\n(.*?)```", text[text.index(anchor) :], re.S)
    assert block is not None, f"{document.name} no longer publishes a JSON block at {anchor!r}"
    return json.loads(block.group(1))


def documented_fenced_block(
    anchor: str,
    language: str,
    *,
    document: Path = HAPPY_PATH,
) -> str:
    text = read_spec_document(document)
    block = re.search(
        rf"```{re.escape(language)}\n(.*?)```",
        text[text.index(anchor) :],
        re.S,
    )
    assert block is not None, (
        f"{document.name} no longer publishes a {language} block at {anchor!r}"
    )
    return block.group(1)


def documented_text_block(anchor: str, *, document: Path = HAPPY_PATH) -> str:
    return documented_fenced_block(anchor, "text", document=document)


def assert_documented_shape(actual: dict, documented: dict, label: str) -> None:
    actual_shape = shape(actual)
    documented_shape = shape(documented)
    assert sorted(set(actual_shape) - set(documented_shape)) == [], (
        f"{label} emits undocumented fields: {sorted(set(actual_shape) - set(documented_shape))}"
    )
    assert sorted(set(documented_shape) - set(actual_shape)) == [], (
        f"{label} omits documented fields: {sorted(set(documented_shape) - set(actual_shape))}"
    )
    for trail, expected in documented_shape.items():
        found = actual_shape[trail]
        if "NoneType" in (expected, found):
            continue
        if trail.endswith("[]") and "empty" in (expected, found):
            # Cardinality is input and registry state, not part of an array's shape.
            continue
        if {expected, found} <= COMPATIBLE:
            continue
        assert found == expected, f"{label} {trail}: expected {expected}, found {found}"
