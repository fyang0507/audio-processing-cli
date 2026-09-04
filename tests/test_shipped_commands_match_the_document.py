"""The commands that already ship, against the shapes `TRANSCRIBE_HAPPY_PATH.md` publishes.

Most blocks in that document mock a command that does not exist yet, so `tests/test_spec_docs.py`
can only check the documents against each other. `doctor`, `packages list`, and `packages verify`
ship, which makes them the first places the document can be *wrong* rather than merely unbuilt.

All three were. §0 described `external_tools` carrying version strings, flat `"mlx": "absent"`
environment states, `packages` as a `{provisioned, count}` pair, and a top-level `warnings` array,
none of which `doctor` emits. §5 omitted the `state` `list` reports per package. §1.3 published
`matches_expected: true` without the expected hash `verify` prints beside it.

Nothing caught any of it. `test_spec_docs.py` compares the documents to each other and
`test_packages.py` compares each command to its own output, so the one comparison that mattered
existed nowhere. This is that comparison, and it is the pattern the `transcribe` acceptance
criterion needs later: a plan's `sample_output` key set against a real run's.

Structure only. The documents' paths, versions, and counters are illustrative by their own
declaration; their key sets and nesting are the part that must match.
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli import environments as env
from audio_cli.media import hash_file
from audio_cli import packages as pkg
from audio_cli import paths
from audio_cli.transcribe import orchestrator as transcribe_orchestrator
from audio_cli.transcribe import refusals
from audio_cli.transcribe.catalog import InputMetadata, build_catalog
from audio_cli.transcribe.orchestrator import run
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.result import NormalizedResult, serialize_result
from audio_cli.transcribe.stacks import get_stack
from audio_cli.transcribe.transport import StageOutcome

REPO = Path(__file__).resolve().parents[1]
HAPPY_PATH = REPO / "TRANSCRIBE_HAPPY_PATH.md"
CONTRACT = REPO / "TRANSCRIBE_CONTRACT.md"


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    """The document shows an unprovisioned machine, and so must this."""
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(tmp_path / "root"))

    def inspect(checkout: Path) -> transcribe_orchestrator._CheckoutState:
        if "fluidaudio" in str(checkout):
            package = env.packages()["fluidaudio"]
            _patches, modified, _digests = pkg.checkout_patch_expectation(package)
            return transcribe_orchestrator._CheckoutState(
                head=package.source["commit"], modified=modified, untracked=()
            )
        package_id = (
            "vibevoice-asr-7b"
            if "vibevoice-asr-7b" in str(checkout)
            else "firered-asr2s"
        )
        package = env.packages()[package_id]
        _patches, modified, _digests = pkg.checkout_patch_expectation(package)
        return transcribe_orchestrator._CheckoutState(
            head=package.checkout["resolved_commit"],
            modified=modified,
            untracked=(),
        )

    monkeypatch.setattr(transcribe_orchestrator, "_inspect_checkout", inspect)

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

    monkeypatch.setattr(transcribe_orchestrator, "_checkout_file_digest", digest)

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

    monkeypatch.setattr(transcribe_orchestrator, "_frozen_packages", frozen)

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


class StubToolchain(pkg.Toolchain):
    """Present tools, no subprocesses. §0 is the everything-installed case."""

    def which(self, tool: str) -> str | None:
        return f"/usr/bin/{tool}"


def documented_doctor() -> dict:
    """The one JSON block under §0."""
    text = HAPPY_PATH.read_text()
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


def test_doctor_emits_the_shape_the_document_publishes() -> None:
    """The defect this catches: §0 describing a doctor payload that doctor does not emit."""
    documented = shape(documented_doctor())
    actual = shape(pkg.doctor(toolchain=StubToolchain()))

    undocumented = sorted(set(actual) - set(documented))
    unimplemented = sorted(set(documented) - set(actual))
    assert not undocumented, (
        "audio doctor emits fields TRANSCRIBE_HAPPY_PATH.md §0 does not publish: "
        f"{undocumented}"
    )
    assert not unimplemented, (
        "TRANSCRIBE_HAPPY_PATH.md §0 publishes fields audio doctor does not emit: "
        f"{unimplemented}"
    )

    # A nullable field is legitimately null on one side and populated on the other, so NoneType
    # agrees with anything. Everything else has to agree on type, which is what keeps a count
    # from being documented as a string.
    for trail, documented_type in sorted(documented.items()):
        actual_type = actual[trail]
        if "NoneType" in (documented_type, actual_type):
            continue
        if {documented_type, actual_type} <= COMPATIBLE:
            continue
        assert documented_type == actual_type, (
            f"{trail}: §0 publishes {documented_type}, audio doctor emits {actual_type}"
        )


def test_the_document_names_every_environment_and_package_doctor_reports() -> None:
    """Key-for-key on the two maps that grow: a new package must be documented to land."""
    documented = documented_doctor()
    actual = pkg.doctor(toolchain=StubToolchain())
    for field in ("environments", "packages", "tools"):
        assert sorted(documented[field]) == sorted(actual[field]), (
            f"§0's {field} names {sorted(documented[field])}; "
            f"doctor reports {sorted(actual[field])}"
        )


def provisioned_like_the_document(tmp_path: Path) -> pkg.Provisioner:
    """A root holding every package, so `list` and `verify` print their populated shape.

    `doctor` reports the same keys whatever is provisioned, so it needs no fixture. These two do
    not: an empty root prints an empty `packages` array, and an array with no element promises no
    element shape. The document's blocks depict a fully-provisioned machine, so the fixture has to
    be one.
    """
    from test_packages import FakeFetcher, FakeToolchain  # sibling module, same rootdir

    provisioner = pkg.Provisioner(toolchain=FakeToolchain(), fetcher=FakeFetcher(tmp_path))
    provisioner.pull(list(pkg.select(stack="qwen-1.7b")))
    provisioner.pull(list(pkg.select(stack="firered")))
    provisioner.pull(list(pkg.select(stack="vibevoice")))
    return provisioner


def documented_block(anchor: str, *, document: Path = HAPPY_PATH) -> dict:
    text = document.read_text()
    block = re.search(r"```json\n(.*?)```", text[text.index(anchor):], re.S)
    assert block is not None, (
        f"{document.name} no longer publishes a JSON block at {anchor!r}"
    )
    return json.loads(block.group(1))


def documented_fenced_block(
    anchor: str,
    language: str,
    *,
    document: Path = HAPPY_PATH,
) -> str:
    text = document.read_text()
    block = re.search(
        rf"```{re.escape(language)}\n(.*?)```",
        text[text.index(anchor):],
        re.S,
    )
    assert block is not None, (
        f"{document.name} no longer publishes a {language} block at {anchor!r}"
    )
    return block.group(1)


def documented_text_block(anchor: str, *, document: Path = HAPPY_PATH) -> str:
    return documented_fenced_block(anchor, "text", document=document)


def test_packages_list_emits_exactly_the_shape_the_document_publishes(tmp_path) -> None:
    """The defect this catches: the same drift as `doctor`, one command over.

    It was there: `list` emits a `state` per package that §5 did not publish.
    """
    provisioned_like_the_document(tmp_path)
    documented = shape(documented_block("## 5. Teardown"))
    actual = shape(pkg.list_report())

    assert sorted(set(actual) - set(documented)) == [], (
        "audio packages list emits fields §5 does not publish: "
        f"{sorted(set(actual) - set(documented))}"
    )
    assert sorted(set(documented) - set(actual)) == [], (
        "§5 publishes fields audio packages list does not emit: "
        f"{sorted(set(documented) - set(actual))}"
    )


# Keys `verify` emits only when a check fails or cannot run. §1.3 is the happy path, so it
# legitimately shows none of them, and a fake fetcher legitimately triggers them. Each is listed
# with the reason it is exempt, so the exemption is a decision rather than a hole -- a genuinely
# new key is still a failure.
VERIFY_CONDITIONAL = {
    # Shape disputed: TRANSCRIBE_CONTRACT.md §64 declares (package, check, expected, actual)
    # and the implementation emits (package, code, detail, fix). Until that is ruled on, this
    # test does not ratify either side by asserting on it.
    "failed[].package", "failed[].code", "failed[].detail", "failed[].fix",
    # Absence is meaningful: set when the interpreter yields no verdict, so that
    # `matches_expected: null` cannot read as a passing check.
    "mlx_audio_private_api_error",
    # Only a package spanning several repositories emits the plural, and §1.3's stack has none:
    # `firered-asr2s` is the one, and it belongs to §3. The singular `revision` beside it *is*
    # published, so the claim itself is documented -- this exempts its four-repo spelling.
    "verified[].revisions[]",
}


def test_packages_verify_emits_no_field_the_document_does_not_publish(tmp_path) -> None:
    """The defect this catches: `verify` publishing a claim §1.3 never described.

    It was there twice. `verify` emits the expected private-API hash beside the measured one,
    where §1.3 published only the measured -- so `matches_expected: true` was unreproducible by
    eye. And the guard's *passing* verdict adds `signature_ok` and `target`, which stayed
    invisible here for as long as the only reachable path was the one that yields no verdict.

    So both guard paths are compared, unioned. §1.3 depicts a machine where the guard passes, and
    a payload shape that only one fixture can reach is a payload shape nothing checks.
    """
    from test_packages import FakeToolchain  # sibling module, same rootdir

    documented = shape(documented_block("audio packages verify"))
    silent = provisioned_like_the_document(tmp_path)
    emitted = set(shape(silent.verify()))

    expected_hash = {guard["kind"]: guard
                     for guard in env.environments()["mlx"].guards}["source_hash"]["sha256"]
    # Keep the same double that provisioned the source checkouts: live verification now reads
    # their tracked baselines as well as answering the private-API probe.
    assert isinstance(silent.toolchain, FakeToolchain)
    silent.toolchain.private_api_hash = expected_hash
    emitted |= set(shape(silent.verify()))

    undocumented = sorted(emitted - set(documented) - VERIFY_CONDITIONAL)
    assert not undocumented, (
        f"audio packages verify emits fields §1.3 does not publish: {undocumented}"
    )


def test_the_verify_exemptions_are_all_still_reachable(tmp_path) -> None:
    """An exemption for a key that can no longer appear is dead weight; make it prove itself."""
    provisioner = provisioned_like_the_document(tmp_path)
    reachable = set(shape(provisioner.verify()))
    stale = sorted(VERIFY_CONDITIONAL - reachable)
    assert not stale, (
        f"VERIFY_CONDITIONAL exempts keys verify no longer emits: {stale} -- drop them"
    )


def test_the_comparison_can_fail() -> None:
    """The floor this repository learned the hard way: an invariant that cannot fail is inert.

    The punctuation floor spent four review passes asserting something about a stage that emits
    no such output, and its test passed vacuously. So prove this one bites.
    """
    baseline = pkg.doctor(toolchain=StubToolchain())

    drifted = dict(baseline)
    drifted["warnings"] = []
    assert set(shape(drifted)) - set(shape(baseline)) == {"warnings[]"}

    retyped = json.loads(json.dumps(baseline))
    retyped["root_exists"] = "true"
    assert shape(retyped)["root_exists"] == "str"
    assert shape(baseline)["root_exists"] == "bool"

    renamed = json.loads(json.dumps(baseline))
    renamed["external_tools"] = renamed.pop("tools")
    assert "tools.ffmpeg.present" not in shape(renamed)


def assert_documented_shape(actual: dict, documented: dict, label: str) -> None:
    actual_shape = shape(actual)
    documented_shape = shape(documented)
    assert sorted(set(actual_shape) - set(documented_shape)) == [], (
        f"{label} emits undocumented fields: "
        f"{sorted(set(actual_shape) - set(documented_shape))}"
    )
    assert sorted(set(documented_shape) - set(actual_shape)) == [], (
        f"{label} omits documented fields: "
        f"{sorted(set(documented_shape) - set(actual_shape))}"
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


def test_transcribe_capabilities_emits_the_shape_happy_path_publishes() -> None:
    metadata = InputMetadata("meeting.m4a", 1794.2, "m4a", 44100, 2)
    actual = build_catalog(get_stack("qwen-1.7b"), metadata)
    documented = documented_block(
        "audio transcribe capabilities --stack qwen-1.7b --input meeting.m4a"
    )
    assert_documented_shape(actual, documented, "audio transcribe capabilities")


@pytest.mark.parametrize(
    ("anchor", "input_name", "duration", "container", "sample_rate", "channels",
     "stack", "wants", "language", "provisioned"),
    [
        (
            "audio transcribe plan --input meeting.m4a \\",
            "meeting.m4a", 1794.2, "m4a", 44100, 2, "qwen-1.7b",
            "diarization,word_timestamps", "Cantonese", (),
        ),
        (
            "audio transcribe plan --input demo.mp4 --stack vibevoice \\",
            "demo.mp4", 112.4, "mp4", 48000, 2, "vibevoice",
            "verbatim,diarization,segment_timestamps,word_timestamps", None,
            ("qwen3-forcedaligner",),
        ),
        (
            "audio transcribe plan --input field.wav --stack firered \\",
            "field.wav", 27.8, "wav", 48000, 1, "firered",
            "verbatim,word_timestamps,vad,segment_timestamps,lid", None, (),
        ),
    ],
)
def test_transcribe_plan_emits_the_shape_happy_path_publishes(
    anchor: str,
    input_name: str,
    duration: float,
    container: str,
    sample_rate: int,
    channels: int,
    stack: str,
    wants: str,
    language: str | None,
    provisioned: tuple[str, ...],
) -> None:
    metadata = InputMetadata(input_name, duration, container, sample_rate, channels)
    request = resolve_request(
        stack_id=stack,
        input_path=Path(input_name),
        wants=wants,
        language=language,
    )
    actual = serialize_plan(
        build_plan(request, metadata, provisioned_packages=provisioned)
    )
    documented = documented_block(anchor)

    # The prose explicitly elides this recursive-sized object as a string. Phase A guarantees
    # and tests its real structure; replace only its value so every surrounding plan/result key
    # is still diffed against the published example.
    assert set(actual["sample_output"]["provenance"]) == {
        "stack", "outcomes", "observed", "plan",
    }
    assert actual["sample_output"]["provenance"]["outcomes"] == {}
    actual["sample_output"]["provenance"] = documented["sample_output"]["provenance"]
    assert_documented_shape(actual, documented, f"audio transcribe plan --stack {stack}")
    if stack == "firered":
        assert actual["execution"]["note"] == documented["execution"]["note"]
        assert (
            actual["capabilities"]["vad"]["note"]
            == documented["capabilities"]["vad"]["note"]
        )


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


def _write_export_result(
    path: Path,
    *,
    source: str | Path,
    segments: list[dict],
    outcomes: dict[str, str],
    language: str | None,
) -> None:
    optional_arrays = {"turns": []} if "diarization" in outcomes else {}
    payload = serialize_result(NormalizedResult(
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
    ))
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_success_summary_matches_the_happy_path(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"source")
    _write_export_result(
        Path("meeting.timed.json"),
        source=source,
        segments=[
            {
                "segment_id": "seg_0",
                "text": "好，我們今天想聊一下你的工作。",
                "words": [{
                    "word_id": "w_0",
                    "text": "好我們今天想聊一下你的工作",
                    "start": 2.31,
                    "end": 4.71,
                }],
            },
            {
                "segment_id": "seg_1",
                "text": "嗯，好啊，我做咗五年設計。",
                "words": [{
                    "word_id": "w_1",
                    "text": "嗯好啊我做咗五年設計",
                    "start": 5.12,
                    "end": 7.44,
                }],
            },
        ],
        outcomes={"word_timestamps": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "meeting.timed.json", "--format", "srt",
        "-o", "meeting.srt",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    actual = json.loads(captured.out)
    documented = documented_block("### 1.5 Export subtitles")

    assert_documented_shape(actual, documented, "audio export --format srt")
    assert actual["warnings"][0]["detail"] == documented["warnings"][0]["detail"]
    assert Path("meeting.srt").read_text(encoding="utf-8") == documented_text_block(
        "Exit 0. `meeting.srt`:"
    )


def test_export_vtt_matches_the_documented_voice_tag_render(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"source")
    _write_export_result(
        Path("demo.transcript.json"),
        source=source,
        segments=[
            {
                "segment_id": "seg_0",
                "speaker": "0",
                "start": 0.0,
                "end": 4.52,
                "text": (
                    "So, um, this is the new editor. "
                    "You can, like, drag a clip here."
                ),
                "words": [
                    {
                        "word_id": "w_0",
                        "text": "So um this is the new editor",
                        "start": 0.31,
                        "end": 2.21,
                    },
                    {
                        "word_id": "w_1",
                        "text": "You can like drag a clip here",
                        "start": 2.58,
                        "end": 4.52,
                    },
                ],
            },
            {
                "segment_id": "seg_1",
                "start": 4.52,
                "end": 6.08,
                "text": "[Environmental Sounds]",
            },
            {
                "segment_id": "seg_2",
                "speaker": "1",
                "start": 6.08,
                "end": 9.41,
                "text": "And it renders straight away?",
                "words": [{
                    "word_id": "w_2",
                    "text": "And it renders straight away",
                    "start": 6.22,
                    "end": 7.86,
                }],
            },
        ],
        outcomes={
            "verbatim": "produced",
            "diarization": "produced",
            "segment_timestamps": "produced",
            "word_timestamps": "produced",
        },
        language=None,
    )

    assert cli.main([
        "export", "--input", "demo.transcript.json", "--format", "vtt",
        "-o", "demo.vtt",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["speaker_labels_rendered"] is True
    assert summary["cues"] == 3
    assert Path("demo.vtt").read_text(encoding="utf-8") == documented_text_block(
        "Exit 0. `demo.vtt`:"
    )


@pytest.mark.parametrize(
    ("output_format", "output_name", "document_anchor", "fence_language"),
    [
        ("txt", "meeting.txt", "Exit 0, `meeting.txt`:", "text"),
        ("md", "meeting.md", "Exit 0, `meeting.md`:", "markdown"),
        ("jsonl", "meeting.jsonl", "Exit 0, `meeting.jsonl`:", "jsonl"),
    ],
)
def test_untimed_file_exports_match_the_happy_path(
    output_format: str,
    output_name: str,
    document_anchor: str,
    fence_language: str,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"source")
    _write_export_result(
        Path("meeting.transcript.json"),
        source=source,
        segments=[
            {"segment_id": "seg_0", "text": "First."},
            {"segment_id": "seg_1", "text": "Second."},
        ],
        outcomes={"verbatim": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "meeting.transcript.json", "--format", output_format,
        "-o", output_name,
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    actual = json.loads(captured.out)
    documented = documented_block(
        "An on-disk untimed format reports segments"
    )

    assert set(actual) == set(documented)
    assert actual["input"] == documented["input"]
    assert actual["segments"] == documented["segments"]
    assert actual["format"] == output_format
    assert actual["output"] == output_name
    assert Path(output_name).read_text(encoding="utf-8") == documented_fenced_block(
        document_anchor, fence_language
    )


def test_multi_input_export_summary_and_merged_order_match_the_happy_path(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"source")
    base_plan = {
        "roles": {"asr": {"config": {"language": "Cantonese"}}},
        "execution": {},
    }
    outcomes = {"segment_timestamps": "produced"}

    partial = serialize_result(NormalizedResult(
        source={
            "path": str(source),
            "duration_seconds": 10.0,
            "timebase": "seconds",
        },
        segments=[
            {"segment_id": "seg_0", "text": "First.", "start": 1.0, "end": 2.0},
            {"segment_id": "seg_1", "text": "Second.", "start": 3.0, "end": 4.0},
        ],
        abstentions=[],
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": outcomes,
            "observed": {},
            "plan": base_plan,
        },
        requested_capabilities=frozenset(outcomes),
        complete=False,
        coverage={
            "scope_intervals": [[0.0, 10.0]],
            "covered_through_seconds": 5.0,
            "covered_fraction": 0.5,
            "covered_intervals": [[0.0, 5.0]],
            "missing_intervals": [[5.0, 10.0]],
            "units_total": 4,
            "units_completed": 2,
        },
    ))
    Path("meeting.timed.partial.json").write_text(
        json.dumps(partial), encoding="utf-8"
    )

    rest_plan = json.loads(json.dumps(base_plan))
    rest_plan["execution"]["range"] = {
        "requested": [5.0, 10.0],
        "selected_unit_scope": [5.0, 10.0],
    }
    rest = serialize_result(NormalizedResult(
        source=partial["source"],
        segments=[
            {"segment_id": "seg_0", "text": "Third.", "start": 6.0, "end": 7.0},
            {"segment_id": "seg_1", "text": "Fourth.", "start": 8.0, "end": 9.0},
        ],
        abstentions=[],
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": outcomes,
            "observed": {},
            "plan": rest_plan,
        },
        requested_capabilities=frozenset(outcomes),
    ))
    Path("meeting.timed.rest.json").write_text(json.dumps(rest), encoding="utf-8")

    assert cli.main([
        "export",
        "--input", "meeting.timed.partial.json",
        "--input", "meeting.timed.rest.json",
        "--format", "jsonl",
        "-o", "meeting.timed.merged.jsonl",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    actual = json.loads(captured.out)
    documented = documented_block("After the ranged resume finishes")
    assert actual == documented

    merged = [
        json.loads(line)
        for line in Path("meeting.timed.merged.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [segment["start"] for segment in merged] == [1.0, 3.0, 6.0, 8.0]
    segment_ids = [segment["segment_id"] for segment in merged]
    assert segment_ids == ["seg_0", "seg_1", "seg_2", "seg_3"]
    assert len(segment_ids) == len(set(segment_ids))


def test_export_timing_refusal_matches_the_contract(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "recordings" / "meeting.m4a"
    source.parent.mkdir()
    source.write_bytes(b"source")
    _write_export_result(
        Path("meeting.transcript.json"),
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"diarization": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "meeting.transcript.json", "--format", "srt",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "Subtitle formats require word timing", document=CONTRACT,
    )
    documented["fix"] = documented["fix"].replace(
        "/Users/you/recordings/meeting.m4a", str(source)
    )

    assert_documented_shape(actual, documented, "audio export timing refusal")
    for field in (
        "code", "field", "provided", "requires_capability", "note", "fix",
    ):
        assert actual[field] == documented[field]


def test_export_recorded_timing_without_words_matches_the_contract(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_export_result(
        Path("ordinary.sentences.json"),
        source=Path("/Users/you/recordings/meeting.m4a"),
        segments=[{
            "segment_id": "seg_0",
            "text": "Ordinary sentence text with no word stream.",
        }],
        outcomes={"word_timestamps": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "ordinary.sentences.json", "--format", "srt",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "A contradictory legacy or hand-edited result", document=CONTRACT,
    )

    assert actual == documented


def test_export_legacy_source_timing_refusal_matches_the_contract(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_export_result(
        Path("legacy.transcript.json"),
        source=Path("recordings/meeting.m4a"),
        segments=[{"segment_id": "seg_0", "text": "Legacy sentence."}],
        outcomes={"verbatim": "produced"},
        language="Cantonese",
    )

    assert cli.main([
        "export", "--input", "legacy.transcript.json", "--format", "srt",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    actual = json.loads(captured.err)
    documented = documented_block(
        "A result written by an older CLI", document=CONTRACT,
    )
    assert actual == documented
