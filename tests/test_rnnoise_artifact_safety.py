"""Git identity, transfer failures, and descriptor-bound artifact handoff defenses."""

from __future__ import annotations

import hashlib
import os
import types
from dataclasses import replace
from pathlib import Path

import pytest
from package_test_support import isolated_root as isolated_root
from rnnoise_test_support import BLOB_SHA1, PAYLOAD, Response
from rnnoise_test_support import rnnoise as rnnoise

from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.media import files as media_files
from audio_cli.packages import artifacts, catalog, fetcher


@pytest.fixture(params=["rnnoise-voice", "silero-vad"])
def single_file(request, rnnoise, monkeypatch):
    package, provisioner, requested = rnnoise
    if request.param == "silero-vad":
        declared = pkg.packages()[request.param]
        package = replace(
            declared,
            bytes=len(PAYLOAD),
            source={**declared.source, "sha256": hashlib.sha256(PAYLOAD).hexdigest()},
        )
        declared_packages = {**pkg.packages(), package.id: package}
        monkeypatch.setattr(catalog, "packages", lambda: declared_packages)
    provisioner.pull([package])
    return package, provisioner, requested


@pytest.mark.parametrize("surface", ["helper", "verify"])
@pytest.mark.parametrize("redirect", [False, True])
def test_receipt_parent_traversal_is_refused_before_normalization(single_file, surface, redirect):
    package, provisioner, requested = single_file
    expected = paths.models_dir() / package.source["filename"]
    intermediate = expected.parent / "redirect"
    if redirect:
        elsewhere = paths.root() / "elsewhere"
        child = elsewhere / "child"
        child.mkdir(parents=True)
        (elsewhere / expected.name).write_bytes(b"different bytes")
        intermediate.symlink_to(child, target_is_directory=True)
    else:
        intermediate.mkdir()
    hostile = intermediate / ".." / expected.name
    assert Path(os.path.abspath(hostile)) == expected
    assert hostile.read_bytes() == (b"different bytes" if redirect else PAYLOAD)
    document = pkg.load_registry()
    document["packages"][package.id]["materialized"]["path"] = str(hostile)
    pkg.save_registry(document)

    if surface == "helper":
        with pytest.raises(pkg.ProvisioningError) as caught:
            pkg.verified_artifact(package.id)
        failure = caught.value.as_dict()
    else:
        report = provisioner.verify()
        assert report["verified"] == []
        assert len(report["failed"]) == 1
        failure = report["failed"][0]
    assert failure["code"] == "package_integrity_failed"
    assert "parent traversal" in failure["detail"]
    assert failure["fix"] == f"audio packages pull --repair {package.id}"
    assert expected.read_bytes() == PAYLOAD
    assert len(requested) == 1


@pytest.mark.parametrize("relative_receipt", [False, True])
def test_valid_receipt_returns_manifest_path_and_matching_bytes(
    single_file, monkeypatch, relative_receipt
):
    package, provisioner, requested = single_file
    expected = paths.models_dir() / package.source["filename"]
    if relative_receipt:
        monkeypatch.chdir(paths.root())
        document = pkg.load_registry()
        document["packages"][package.id]["materialized"]["path"] = str(
            expected.relative_to(paths.root())
        )
        pkg.save_registry(document)
    location, provenance = pkg.verified_artifact(package.id)
    assert location == expected
    assert provenance["sha256"] == hashlib.sha256(location.read_bytes()).hexdigest()
    report = provisioner.verify()
    assert report["failed"] == []
    if package.source["type"] == "url":
        assert report["verified"] == [{"package": package.id, "digest": "ok"}]
    else:
        assert report["verified"] == [
            {
                "package": package.id,
                "revision": package.source["revision"],
                "git_blob_sha1": BLOB_SHA1,
                "bytes": len(PAYLOAD),
            }
        ]
    assert len(requested) == 1


@pytest.mark.parametrize("payload", [b"jello\n", b"hello", b"hello\nextra"])
def test_failed_download_never_publishes_bad_bytes_and_leaves_retryable_state(
    rnnoise, monkeypatch, payload
):
    package, provisioner, _requested = rnnoise
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *_a, **_k: Response(payload))
    with pytest.raises(pkg.ProvisioningError) as caught:
        provisioner.pull([package])
    assert caught.value.code == "package_integrity_failed"
    assert caught.value.payload["fix"] == f"audio packages pull --repair {package.id}"
    assert pkg.load_registry()["packages"][package.id]["state"] == "pulling"
    assert list(paths.models_dir().iterdir()) == []
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *_a, **_k: Response(PAYLOAD))
    provisioner.pull([package])
    assert pkg.verified_artifact(package.id)[1]["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()


def test_repair_checksum_failure_preserves_previous_cache_entry(rnnoise, monkeypatch):
    package, provisioner, _requested = rnnoise
    provisioner.pull([package])
    path, _provenance = pkg.verified_artifact(package.id)
    path.write_bytes(b"previous")
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *_a, **_k: Response(b"jello\n"))
    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull([package], repair=True)
    assert path.read_bytes() == b"previous"
    assert list(path.parent.glob(".audio-download-*")) == []


def test_git_download_refuses_non_https_response(rnnoise, monkeypatch):
    package, provisioner, _requested = rnnoise
    response = Response(PAYLOAD)
    response.geturl = lambda: "http://example.invalid/insecure"
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *_a, **_k: response)
    with pytest.raises(pkg.ProvisioningError, match="HTTPS"):
        provisioner.pull([package])
    assert list(paths.models_dir().iterdir()) == []


@pytest.mark.parametrize("redirect", ["leaf", "models", "root"])
def test_resolver_and_verify_refuse_redirected_paths(rnnoise, tmp_path, redirect):
    package, provisioner, requested = rnnoise
    provisioner.pull([package])
    path, _provenance = pkg.verified_artifact(package.id)
    target = {"leaf": path, "models": path.parent, "root": paths.root()}[redirect]
    moved = tmp_path / f"moved-{redirect}"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=redirect != "leaf")
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.verified_artifact(package.id)
    assert caught.value.code == "package_integrity_failed"
    if redirect == "root":
        with pytest.raises(pkg.ProvisioningError):
            provisioner.verify()
    else:
        assert provisioner.verify()["failed"][0]["code"] == "package_integrity_failed"
    if redirect == "leaf":
        provisioner.pull([package], repair=True)
        assert not path.is_symlink()
        assert path.read_bytes() == moved.read_bytes() == PAYLOAD
        assert len(requested) == 2
    else:
        with pytest.raises(pkg.ProvisioningError):
            provisioner.pull([package], repair=True)
        with pytest.raises(pkg.ProvisioningError):
            provisioner.remove([package.id])
        assert len(requested) == 1


def test_parent_swap_after_hashing_refuses_handoff(rnnoise, tmp_path, monkeypatch):
    package, provisioner, _requested = rnnoise
    provisioner.pull([package])
    models = paths.models_dir()
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / package.source["filename"]
    external.write_bytes(b"elsewhere")
    real_hash = artifacts.regular_file_digests_at

    def swap_after_hash(*args, **kwargs):
        found = real_hash(*args, **kwargs)
        models.rename(paths.root() / "original-models")
        models.symlink_to(outside, target_is_directory=True)
        return found

    monkeypatch.setattr(artifacts, "regular_file_digests_at", swap_after_hash)
    with pytest.raises(pkg.ProvisioningError, match="directory identity changed"):
        pkg.verified_artifact(package.id)
    assert external.read_bytes() == b"elsewhere"


def test_temporary_symlink_cannot_redirect_git_download(rnnoise, tmp_path, monkeypatch):
    package, provisioner, _requested = rnnoise
    paths.models_dir().mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep")
    monkeypatch.setattr(fetcher.uuid, "uuid4", lambda: types.SimpleNamespace(hex="fixed"))
    temporary = paths.models_dir() / f".audio-download-{os.getpid()}-fixed.part"
    temporary.symlink_to(victim)
    with pytest.raises(pkg.ProvisioningError):
        provisioner.pull([package])
    assert temporary.is_symlink()
    assert victim.read_bytes() == b"keep"


def test_same_read_pass_produces_git_blob_identity_and_local_sha256(rnnoise, monkeypatch):
    package, provisioner, _requested = rnnoise
    provisioner.pull([package])
    real_read = media_files.os.read
    chunks = []

    def read_small(descriptor, _size):
        chunk = real_read(descriptor, 2)
        chunks.append(chunk)
        return chunk

    monkeypatch.setattr(media_files.os, "read", read_small)
    _path, provenance = pkg.verified_artifact(package.id)
    assert chunks == [b"he", b"ll", b"o\n", b""]
    assert provenance["source"]["git_blob_sha1"] == BLOB_SHA1
    assert provenance["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert provenance["bytes"] == 6


@pytest.mark.parametrize("mutation", ["append", "replace", "rewrite"])
def test_hash_rejects_file_changes_during_read(rnnoise, monkeypatch, mutation):
    package, provisioner, _requested = rnnoise
    provisioner.pull([package])
    path, _provenance = pkg.verified_artifact(package.id)
    real_read = media_files.os.read
    changed = False

    def mutate_after_read(descriptor, size):
        nonlocal changed
        chunk = real_read(descriptor, size)
        if chunk and not changed:
            changed = True
            if mutation == "append":
                with path.open("ab") as handle:
                    handle.write(b"extra")
            elif mutation == "replace":
                path.rename(path.with_suffix(".old"))
                path.write_bytes(PAYLOAD)
            else:
                path.write_bytes(b"jello\n")
        return chunk

    monkeypatch.setattr(media_files.os, "read", mutate_after_read)
    with pytest.raises(pkg.ProvisioningError, match="changed while hashing"):
        pkg.verified_artifact(package.id)
    assert changed
