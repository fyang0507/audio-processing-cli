"""Invariants over the provisioned-environment manifest and the documents that quote it.

The layout in `src/audio_cli/environments/manifest.json` was derived from a resolver run
(`model_tests/benchmark/results/2026-08-17-environment-partition.json`), not chosen. These
tests exist so it cannot drift back by hand: an environment name typed into a spec document
that the manifest does not define, a lock that stops matching its input, a package pointing at
an environment nobody provisions.

Each assertion below corresponds to something that was actually wrong before this pass, or
that the spec documents asserted incorrectly for the two weeks they existed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from audio_cli import environments as env

REPO = Path(__file__).resolve().parents[1]
SPEC_DOCS = (REPO / "TRANSCRIBE_CONTRACT.md", REPO / "TRANSCRIBE_HAPPY_PATH.md")
PARTITION = REPO / "model_tests/benchmark/results/2026-08-17-environment-partition.json"


def test_manifest_is_self_consistent() -> None:
    """Roles, kinds, environments, and the multi-repo byte sums all have to agree."""
    assert env.validate() == []


def test_every_provisioned_environment_has_a_lock_that_exists() -> None:
    for environment in env.environments().values():
        if not (environment.provisioned and environment.has_interpreter):
            continue
        assert environment.lock is not None and environment.lock.is_file()
        assert environment.requirements is not None and environment.requirements.is_file()
        assert env.lock_digest(environment.name)


def test_locks_are_hash_pinned() -> None:
    """--generate-hashes is what makes the installer verify each wheel, so it is not optional."""
    for environment in env.environments().values():
        if environment.lock is None:
            continue
        body = environment.lock.read_text()
        pinned = re.findall(r"^([A-Za-z0-9._-]+)==", body, re.M)
        assert pinned, f"{environment.lock.name} pins nothing"
        assert body.count("--hash=sha256:") >= len(pinned), (
            f"{environment.lock.name} has fewer hashes than pins; regenerate it with "
            "--generate-hashes"
        )


def test_lock_headers_name_the_input_they_were_compiled_from() -> None:
    """A lock whose input has moved on is worse than no lock: it looks authoritative."""
    for environment in env.environments().values():
        if environment.lock is None or environment.requirements is None:
            continue
        header = environment.lock.read_text()[:400]
        assert environment.requirements.name in header, (
            f"{environment.lock.name} does not record {environment.requirements.name} as its "
            "input"
        )


def test_locks_agree_with_the_partition_the_resolver_derived() -> None:
    """The environment count and grouping come from the probe, not from preference."""
    partition = json.loads(PARTITION.read_text())
    assert partition["minimal_python_partition_is_unique"], (
        "the partition stopped being unique; the layout now contains an undocumented choice"
    )
    python_environments = [
        name for name, environment in env.environments().items()
        if environment.provisioned and environment.has_interpreter
    ]
    assert len(python_environments) == len(partition["minimal_python_partition"]), (
        f"manifest has {len(python_environments)} Python environments, the resolver derived "
        f"{len(partition['minimal_python_partition'])}"
    )
    assert partition["conflict_axis"]["conflicts_naming_transformers"] == \
        partition["conflict_axis"]["groups_conflicting"], (
            "not every conflict is about transformers any more; re-read the matrix before "
            "describing the split that way"
        )


def test_mlx_stays_torch_free() -> None:
    """The fast path's whole point. A torch pin here would be a silent regression."""
    body = env.environments()["mlx"].lock.read_text()
    assert not re.search(r"^torch(vision|audio)?==", body, re.M), (
        "torch appeared in the mlx lock; that environment is torch-free by intent and in fact"
    )


def test_the_private_api_guard_is_declared_where_it_is_checkable() -> None:
    """`verify` needs the hash and the signature, and both must be weight-free to check."""
    guards = {guard["kind"]: guard for guard in env.environments()["mlx"].guards}
    assert set(guards) == {"source_hash", "signature"}
    assert guards["source_hash"]["sha256"] == (
        "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250"
    )
    assert all(guard["checkable_without_weights"] for guard in guards.values())


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_spec_documents_only_name_environments_that_exist(path: Path) -> None:
    """The defect this catches: `"environment": "torch"` outliving the environment named torch."""
    known = set(env.environments())
    for index, body in enumerate(re.findall(r"```json\n(.*?)```", path.read_text(), re.S),
                                 start=1):
        document = json.loads(body)
        for name in _environment_values(document):
            assert name in known, (
                f"{path.name} block {index}: environment {name!r} is not in the manifest "
                f"(known: {sorted(known)})"
            )


def _environment_values(node) -> list[str]:
    """Every environment name a payload mentions, however it is nested."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "environment" and isinstance(value, str):
                found.append(value)
            elif key in {"environments_spanned", "environments_created", "environments_kept",
                         "environments_removed"} and isinstance(value, list):
                found.extend(item for item in value if isinstance(item, str))
            elif key == "environments" and isinstance(value, dict):
                found.extend(value)
            elif key == "environments" and isinstance(value, list):
                found.extend(item for item in value if isinstance(item, str))
            else:
                found.extend(_environment_values(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_environment_values(item))
    return found


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_spec_documents_only_name_packages_that_exist(path: Path) -> None:
    known = set(env.packages())
    for index, body in enumerate(re.findall(r"```json\n(.*?)```", path.read_text(), re.S),
                                 start=1):
        document = json.loads(body)
        for name in _package_values(document):
            assert name in known, (
                f"{path.name} block {index}: package {name!r} is not in the manifest"
            )


def _package_values(node) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "package" and isinstance(value, str):
                found.append(value)
            elif key in {"missing", "pulled", "verified"} and isinstance(value, list):
                found.extend(item["package"] for item in value
                             if isinstance(item, dict) and "package" in item)
            else:
                found.extend(_package_values(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_package_values(item))
    return found


def test_a_license_is_never_reported_as_reviewed_without_a_source() -> None:
    """A card's declaration is not a clearance, so the two must not collapse into one field."""
    raw = json.loads((Path(env.MANIFEST)).read_text())
    for identifier, body in raw["packages"].items():
        if body.get("license_reviewed"):
            assert body.get("license_source"), (
                f"{identifier}: license_reviewed is true with no license_source to back it"
            )


def test_download_totals_report_unsized_packages_separately() -> None:
    """A total that quietly omitted an unsized package would understate a download."""
    selection = env.packages_for("qwen-1.7b", {
        "decode": "ffmpeg", "asr": "qwen3-asr-1.7b-8bit",
        "aligner": "qwen3-forcedaligner", "diarizer": "fluidaudio",
    })
    known, unsized = env.download_bytes(selection)
    assert unsized == ["fluidaudio"]
    assert known == sum(p.bytes for p in selection if p.sized)
    assert env.environments_spanned(selection) == ["mlx", "swift"]


def test_packages_for_rejects_a_backend_that_does_not_fill_the_role() -> None:
    """The mapping is only useful if a wrong pairing fails loudly rather than silently."""
    with pytest.raises(env.ManifestError, match="does not fill"):
        env.packages_for("qwen-1.7b", {"asr": "qwen3-forcedaligner"})
    with pytest.raises(env.ManifestError, match="no package supplies"):
        env.packages_for("qwen-1.7b", {"asr": "whisper-large"})


def test_manifest_byte_counts_are_not_the_illustrative_ones_the_specs_used() -> None:
    """The spec figures were placeholders and several were wrong; regressing to them is a bug."""
    retired = {2463307541, 18253611008, 9878424576, 1932735283, 84279296}
    for package in env.packages().values():
        assert package.bytes not in retired, (
            f"{package.id} carries {package.bytes}, one of the illustrative figures the spec "
            "documents used before real sizes were read from the Hub"
        )


def test_silero_digest_matches_the_shipped_backend() -> None:
    """One package is described in two places, so the two must not drift."""
    from audio_cli import vad

    silero = env.packages()["silero-vad"]
    assert silero.source["sha256"] == vad.MODEL_SHA256
    assert silero.source["version"] in vad.MODEL_VERSION
    assert silero.auto_fetch, "silero-vad is the one package that may auto-fetch"
    cached = Path(vad._cache_root()) / vad.MODEL_FILENAME
    if cached.is_file():
        assert hashlib.sha256(cached.read_bytes()).hexdigest() == silero.source["sha256"]
        assert cached.stat().st_size == silero.bytes
