"""Optional enhancement-only package lifecycle against a tiny synthetic Git blob."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.cli import main
from audio_cli.packages import catalog
from tests.audio_cli.packages.package_test_support import isolated_root as isolated_root
from tests.audio_cli.packages.rnnoise_test_support import BLOB_SHA1, PAYLOAD
from tests.audio_cli.packages.rnnoise_test_support import rnnoise as rnnoise


def test_rnnoise_declaration_has_exact_upstream_identity_and_no_transcription_role():
    package = env.packages()["rnnoise-voice"]
    assert package.environment == "core"
    assert not env.environments()[package.environment].provisioned
    assert package.bytes == 299693
    assert package.roles == package.stacks == ()
    assert not package.auto_fetch
    assert package.requires_tool == ()
    assert "rnnoise-voice" not in env.backends()
    assert package.source == {
        "type": "git-blob",
        "repo": "GregorR/rnnoise-models",
        "revision": "3eee541a283fd3b8f81b85b1748e3b9ccbefa04d",
        "path": "beguiling-drafter-2018-08-30/bd.rnnn",
        "url": "https://raw.githubusercontent.com/GregorR/rnnoise-models/"
        "3eee541a283fd3b8f81b85b1748e3b9ccbefa04d/beguiling-drafter-2018-08-30/bd.rnnn",
        "filename": "rnnoise-voice-bd-2018-08-30.rnnn",
        "git_blob_sha1": "0173a18664f5905dbdc4543d3528741fb32cd69f",
    }
    assert env.validate() == []
    for stack in {stack for p in env.packages().values() for stack in p.stacks}:
        assert package not in pkg.select(stack=stack)


@pytest.mark.parametrize(
    ("change", "size", "auto_fetch"),
    [
        ({"url": "http://example.invalid/model"}, 6, False),
        ({"revision": "main"}, 6, False),
        ({"git_blob_sha1": "a" * 64}, 6, False),
        ({"filename": "../escape"}, 6, False),
        ({"path": "../escape"}, 6, False),
        ({"repo": "../escape"}, 6, False),
        ({"sha256": "a" * 64}, 6, False),
        ({}, None, False),
        ({}, True, False),
        ({}, 0, False),
        ({}, 6, True),
    ],
)
def test_git_source_validation_rejects_unsafe_or_false_declarations(change, size, auto_fetch):
    source = env.packages()["rnnoise-voice"].source
    assert env.git_blob_source_problems({**source, **change}, size, auto_fetch=auto_fetch)


def test_explicit_cli_pull_reports_git_identity_and_helper_returns_local_digest(rnnoise, capsys):
    package, _provisioner, requested = rnnoise
    assert main(["packages", "pull", package.id]) == 0
    receipt = json.loads(capsys.readouterr().out)
    expected = {
        "package": package.id,
        "environment": "core",
        "bytes": len(PAYLOAD),
        "revision": package.source["revision"],
        "git_blob_sha1": BLOB_SHA1,
    }
    assert receipt["pulled"] == [expected]
    assert receipt["environments_created"] == []
    assert requested == [package.source["url"]]
    document = pkg.load_registry()
    materialized = document["packages"][package.id]["materialized"]
    assert set(materialized) == {"path", "bytes", "revision", "git_blob_sha1"}
    path, provenance = pkg.verified_artifact(package.id)
    assert path == paths.models_dir() / package.source["filename"]
    assert provenance == {
        "package": package.id,
        "source": package.source,
        "bytes": len(PAYLOAD),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    }
    assert document["environments"] == {}
    assert not paths.envs_dir().exists()
    assert main(["packages", "verify"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] == [{k: v for k, v in expected.items() if k != "environment"}]
    documented = json.loads(
        re.search(
            r"```json\n(.*?)```",
            (Path(__file__).resolve().parents[3] / "docs/packages/rnnoise.md").read_text(),
            re.S,
        )[1]
    )
    declared = env.packages()[package.id]
    assert documented == {
        "package": package.id,
        "revision": declared.source["revision"],
        "git_blob_sha1": declared.source["git_blob_sha1"],
        "bytes": declared.bytes,
    }
    assert report["verified"][0] == {
        **documented,
        "bytes": len(PAYLOAD),
        "git_blob_sha1": BLOB_SHA1,
    }
    assert report["failed"] == []
    assert pkg.list_report()["packages"][0]["used_by_stacks"] == []
    assert pkg.path_report()["packages"][package.id]["location"] == str(path)
    assert requested == [package.source["url"]], "read-only operations fetched weights"


def test_pull_cache_hit_and_repair_use_actual_git_content(rnnoise):
    package, provisioner, requested = rnnoise
    provisioner.pull([package])
    path, _provenance = pkg.verified_artifact(package.id)
    assert provisioner.pull([package])["skipped"] == [package.id]
    assert provisioner.pull([package], repair=True)["pulled"][0]["git_blob_sha1"] == BLOB_SHA1
    assert len(requested) == 1
    path.write_bytes(b"jello\n")
    provisioner.pull([package], repair=True)
    assert len(requested) == 2
    assert path.read_bytes() == PAYLOAD
    assert provisioner.verify()["failed"] == []


@pytest.mark.parametrize("payload", [b"jello\n", b"hello", b"hello\nextra"])
def test_helper_and_verify_reject_live_byte_and_size_drift(rnnoise, payload):
    package, provisioner, requested = rnnoise
    provisioner.pull([package])
    path, _provenance = pkg.verified_artifact(package.id)
    path.write_bytes(payload)
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.verified_artifact(package.id)
    assert caught.value.code == "package_integrity_failed"
    assert caught.value.payload["fix"] == f"audio packages pull --repair {package.id}"
    report = provisioner.verify()
    assert report["verified"] == []
    assert report["failed"][0]["code"] == "package_integrity_failed"
    assert len(requested) == 1


def test_helper_rechecks_manifest_size_even_with_correct_blob_identity(rnnoise, monkeypatch):
    package, provisioner, _requested = rnnoise
    provisioner.pull([package])
    monkeypatch.setattr(catalog, "packages", lambda: {package.id: replace(package, bytes=7)})
    with pytest.raises(pkg.ProvisioningError, match="byte count"):
        pkg.verified_artifact(package.id)


def test_missing_file_and_forged_digest_receipt_do_not_earn_a_verified_path(rnnoise):
    package, provisioner, requested = rnnoise
    provisioner.pull([package])
    path, _provenance = pkg.verified_artifact(package.id)
    document = pkg.load_registry()
    materialized = document["packages"][package.id]["materialized"]
    materialized.update(sha256=hashlib.sha256(PAYLOAD).hexdigest(), digest_verified=True)
    pkg.save_registry(document)
    path.unlink()
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.verified_artifact(package.id)
    assert caught.value.code == "package_integrity_failed"
    assert caught.value.payload["fix"] == f"audio packages pull --repair {package.id}"
    assert provisioner.verify()["verified"] == []
    assert len(requested) == 1


def test_resolver_refuses_unknown_and_nonsingle_file_packages():
    for identifier in ("unknown-model", "vibevoice-asr-7b"):
        with pytest.raises(pkg.ProvisioningError) as caught:
            pkg.verified_artifact(identifier)
        assert caught.value.exit_code == 2
        assert caught.value.payload["fix"] == "audio packages list"


def test_download_rejects_wrong_declared_size_even_when_git_hash_matches(rnnoise):
    package, provisioner, requested = rnnoise
    provisioner.pull([package])
    path, _provenance = pkg.verified_artifact(package.id)
    with pytest.raises(pkg.ProvisioningError, match="size mismatch"):
        provisioner.pull([replace(package, bytes=7)], repair=True)
    assert len(requested) == 2, "wrong declared size must not earn a cache hit"
    assert path.read_bytes() == PAYLOAD


@pytest.mark.parametrize("state", [None, "pulling"])
def test_missing_or_incomplete_package_refuses_without_fetching(rnnoise, state):
    package, _provisioner, requested = rnnoise
    if state:
        document = pkg.blank_registry()
        document["packages"][package.id] = {"state": state}
        pkg.save_registry(document)
    with pytest.raises(pkg.ProvisioningError) as caught:
        pkg.verified_artifact(package.id)
    assert caught.value.code == "package_not_provisioned"
    assert caught.value.exit_code == 3
    assert caught.value.payload["fix"] == f"audio packages pull {package.id}"
    assert requested == []
    assert not paths.models_dir().exists()


def test_forged_receipt_cannot_substitute_an_external_artifact_and_remove_owns_only_manifest_path(
    rnnoise, tmp_path
):
    package, provisioner, _requested = rnnoise
    provisioner.pull([package])
    path, _provenance = pkg.verified_artifact(package.id)
    external = tmp_path / "external.rnnn"
    external.write_bytes(PAYLOAD)
    document = pkg.load_registry()
    document["packages"][package.id]["materialized"]["path"] = str(external)
    pkg.save_registry(document)
    with pytest.raises(pkg.ProvisioningError, match="not managed path"):
        pkg.verified_artifact(package.id)
    assert provisioner.verify()["verified"] == []
    removed = provisioner.remove([package.id])
    assert removed["removed"] == [package.id]
    assert removed["reclaimed_bytes"] == len(PAYLOAD)
    assert not path.exists()
    assert external.read_bytes() == PAYLOAD


def test_removal_and_purge_use_selected_root_and_leave_unrelated_files(rnnoise, tmp_path):
    package, provisioner, _requested = rnnoise
    provisioner.pull([package])
    unrelated = paths.models_dir() / "unrelated.rnnn"
    unrelated.write_bytes(b"keep")
    external = tmp_path / "other-root" / "models" / package.source["filename"]
    external.parent.mkdir(parents=True)
    external.write_bytes(PAYLOAD)
    report = provisioner.purge(dry_run=True)
    assert report["would_remove"]["packages"] == [package.id]
    assert report["reclaimable_known_bytes"] == len(PAYLOAD)
    assert report["would_remove"]["hub_revisions"] == []
    assert provisioner.purge(dry_run=False)["removed"]["packages"] == [package.id]
    assert pkg.load_registry()["packages"] == {}
    assert unrelated.read_bytes() == b"keep"
    assert external.read_bytes() == PAYLOAD
