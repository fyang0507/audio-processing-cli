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

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli.transcribe import refusals
from audio_cli.transcribe.catalog import InputMetadata, build_catalog
from audio_cli.transcribe.orchestrator import run
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.stacks import get_stack
from audio_cli.transcribe.transport import StageOutcome

REPO = Path(__file__).resolve().parents[1]
HAPPY_PATH = REPO / "TRANSCRIBE_HAPPY_PATH.md"


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    """The document shows an unprovisioned machine, and so must this."""
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(tmp_path / "root"))


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


def documented_block(anchor: str) -> dict:
    text = HAPPY_PATH.read_text()
    block = re.search(r"```json\n(.*?)```", text[text.index(anchor):], re.S)
    assert block is not None, f"the document no longer publishes a JSON block at {anchor!r}"
    return json.loads(block.group(1))


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
    answering = pkg.Provisioner(toolchain=FakeToolchain(private_api_hash=expected_hash),
                                fetcher=silent.fetcher)
    emitted |= set(shape(answering.verify()))

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
        target = tmp_path / identifier
        target.mkdir()
        materialized = {"path": str(target), "revision": revision}
        if identifier == "fluidaudio":
            materialized.update({"built": True, "product_runs": True})
            product = target / ".build" / "release" / "fluidaudiocli"
            product.parent.mkdir(parents=True)
            product.write_text("#!/bin/sh\n", encoding="utf-8")
            product.chmod(0o755)
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


def test_qwen_incomplete_refusal_emits_the_shape_happy_path_publishes(tmp_path) -> None:
    interpreter = tmp_path / "root" / "envs" / "mlx" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    model = tmp_path / "qwen-model"
    model.mkdir()

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
                "revision": "89e96d92ba34aca20b3e29fb10cc284097d1219f",
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
