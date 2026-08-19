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
from pathlib import Path

import pytest

from audio_cli import environments as env
from audio_cli import packages as pkg

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
