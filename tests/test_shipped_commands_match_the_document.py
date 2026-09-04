"""Shipped doctor, package, capability, and plan shapes against the specification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shipped_command_test_support import (
    COMPATIBLE,
    StubToolchain,
    assert_documented_shape,
    configure_isolated_root,
    documented_block,
    documented_doctor,
    provisioned_like_the_document,
    shape,
)

from audio_cli import environments as env
from audio_cli import packages as pkg
from audio_cli.transcribe.catalog import InputMetadata, build_catalog
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.stacks import get_stack


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    configure_isolated_root(tmp_path, monkeypatch)


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
