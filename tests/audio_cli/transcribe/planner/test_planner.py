from __future__ import annotations

from pathlib import Path

import pytest

from audio_cli.transcribe import refusals, stacks
from audio_cli.transcribe.catalog import InputMetadata, build_catalog
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request

METADATA = InputMetadata("sample.wav", 361.0, "wav", 48000, 2)


def plan_for(stack_id: str, wants: str | None = None, **options):
    request = resolve_request(
        stack_id=stack_id,
        input_path=Path("sample.wav"),
        wants=wants,
        **options,
    )
    return serialize_plan(build_plan(request, METADATA))


def test_omitted_want_is_a_floors_only_request() -> None:
    emitted = plan_for("qwen-1.7b")
    assert set(emitted["roles"]) == {"decode", "asr"}
    assert emitted["capabilities"] == {}
    assert set(emitted["sample_output"]) == {
        "sample",
        "note",
        "schema_version",
        "complete",
        "source",
        "segments",
        "abstentions",
        "provenance",
    }
    assert emitted["sample_output"]["segments"] == [{"segment_id": "seg_0", "text": None}]
    assert "outcomes" not in emitted
    assert emitted["sample_output"]["provenance"]["outcomes"] == {}


@pytest.mark.parametrize(
    ("stack_id", "capability"),
    [
        (stack_id, capability)
        for stack_id, definition in stacks.stack_definitions().items()
        for capability, cell in definition.capabilities.items()
        if cell["resolution"] in {"native", "native_stage", "add_on"}
    ],
)
def test_every_satisfiable_cell_drives_plan_and_sample_shape(
    stack_id: str, capability: str
) -> None:
    definition = stacks.get_stack(stack_id)
    cell = definition.capabilities[capability]
    emitted = plan_for(stack_id, capability)
    planned = emitted["capabilities"][capability]
    expected = "derived" if cell["resolution"] == "add_on" else "native"
    assert planned["satisfaction"] == expected
    assert "outcome" not in planned
    if cell["resolution"] == "native_stage":
        assert planned["stage"] == cell["stage"]
    if cell["resolution"] == "add_on":
        assert planned["backend"] == cell["package"]

    sample = emitted["sample_output"]
    segment = sample["segments"][0]
    top_level = {
        "diarization": "turns",
        "overlapped_speech": "overlapped_speech",
        "vad": "vad_regions",
        "lid": "lid_regions",
    }
    segment_key = {
        "diarization": "speaker",
        "word_timestamps": "words",
        "segment_timestamps": "start",
    }
    if capability in top_level:
        assert top_level[capability] in sample
    if capability in segment_key:
        assert segment_key[capability] in segment
    if capability in {"languages", "verbatim"}:
        assert capability not in sample
    if capability == "overlapped_speech":
        assert sample["overlapped_speech"][0]["overlap_id"] == "overlap_0"


@pytest.mark.parametrize(
    ("stack_id", "capability", "code"),
    [
        (
            stack_id,
            capability,
            "capability_unsupported"
            if cell["resolution"] == "unsupported"
            else "capability_unsatisfiable_on_stack",
        )
        for stack_id, definition in stacks.stack_definitions().items()
        for capability, cell in definition.capabilities.items()
        if cell["resolution"] in {"unsupported", "unsatisfiable_on_stack"}
    ],
)
def test_every_impossible_cell_drives_the_distinct_refusal(
    stack_id: str, capability: str, code: str
) -> None:
    with pytest.raises(refusals.Refusal) as caught:
        resolve_request(
            stack_id=stack_id,
            input_path=Path("sample.wav"),
            wants=capability,
        )
    assert caught.value.payload["code"] == code
    allowed = caught.value.payload["allowed"]
    assert bool(allowed) is (code == "capability_unsatisfiable_on_stack")


def test_language_is_canonicalized_without_reaching_the_aligner() -> None:
    emitted = plan_for("qwen-1.7b", "word_timestamps", language="english")
    assert emitted["roles"]["asr"]["config"]["language"] == "English"
    rule = emitted["roles"]["aligner"]["config"]["language_rule"]
    assert "ASR --language hint is never forwarded" in rule
    assert "English" in rule and "Chinese" in rule


def test_native_vad_can_be_explicitly_substituted_with_the_shipped_pin() -> None:
    emitted = plan_for("firered", "vad", vad="silero-vad")
    assert emitted["roles"]["vad"]["backend"] == "silero-vad"
    assert emitted["roles"]["vad"]["selected_by"] == "pin:--vad"
    assert emitted["capabilities"]["vad"] == {
        "satisfaction": "derived",
        "backend": "silero-vad",
        "evidence": {"interface": "verified", "quality": "measured"},
        "note": stacks.get_stack("qwen-1.7b").capabilities["vad"]["plan_note"],
    }


def test_vibevoice_refuses_a_diarizer_pin_for_its_native_capability() -> None:
    with pytest.raises(refusals.Refusal) as caught:
        resolve_request(
            stack_id="vibevoice",
            input_path=Path("sample.wav"),
            wants="diarization",
            diarizer="fluidaudio",
        )
    assert caught.value.exit_code == 2
    assert caught.value.payload["code"] == "pin_conflicts_with_native_capability"


def test_firered_timing_is_native_and_punctuation_is_a_floor() -> None:
    emitted = plan_for("firered", "segment_timestamps,word_timestamps")
    assert emitted["roles"]["punctuator"]["selected_by"] == (
        "floor:punctuated_sentence_segmented_text"
    )
    assert emitted["capabilities"]["segment_timestamps"]["satisfaction"] == "native"
    assert emitted["capabilities"]["word_timestamps"]["satisfaction"] == "native"
    assert [item["package"] for item in emitted["packages"]] == ["firered-asr2s"]
    assert "aligner" not in emitted["roles"]


def test_sample_provenance_embeds_the_executable_plan_without_recursing() -> None:
    emitted = plan_for("vibevoice", "diarization,word_timestamps")
    core = {key: value for key, value in emitted.items() if key not in {"sample_output", "next"}}
    assert emitted["sample_output"]["provenance"]["plan"] == core
    assert "sample_output" not in core
    assert "next" not in core
    assert emitted["sample_output"]["abstentions"] == []


def test_vibevoice_plan_records_both_executed_hub_revisions() -> None:
    emitted = plan_for("vibevoice")
    assert emitted["roles"]["asr"]["revision"] == ("d0c9efdb8d614685062c04425d91e01b6f37d944")
    assert emitted["roles"]["asr"]["tokenizer"] == {
        "materialized_role": "tokenizer",
        "repository": "Qwen/Qwen2.5-7B",
        "revision": "d149729398750b98c0af14eb82c78cfe92750796",
    }
    assert emitted["packages"] == [
        {
            "package": "vibevoice-asr-7b",
            "environment": "torch-vibevoice",
            "kind": "weights",
            "bytes": 17_361_048_135,
            "provisioned": False,
        }
    ]
    assert emitted["total_known_download_bytes"] == 17_361_048_135


@pytest.mark.parametrize("stack_id", ["qwen-1.7b", "qwen-0.6b", "vibevoice"])
def test_word_timing_plan_records_the_executed_aligner_revision(stack_id: str) -> None:
    emitted = plan_for(stack_id, "word_timestamps")
    assert emitted["roles"]["aligner"]["revision"] == ("0e1a68e91d815300c7c9754b2a7639378b23db15")


def test_firered_plan_records_every_weight_revision_and_checkout_commit() -> None:
    emitted = plan_for("firered", "lid")
    roles = emitted["roles"]
    assert {role: roles[role]["revision"] for role in ("vad", "lid", "asr", "punctuator")} == {
        "vad": "7990aaccc6b7aec1e527743bd30201f2c4a03b8c",
        "lid": "1bb4d285c8456429385d9c0810300df4297bc11b",
        "asr": "2304afed56eacfee6256dee5937ed22ffa0b64ec",
        "punctuator": "e448fd967f44182a1c323cc30f5d89f2400c28da",
    }
    assert {roles[role]["source_commit"] for role in ("vad", "lid", "asr", "punctuator")} == {
        "4e7d9aaf4482a47cec1724807026b9b151926eb5"
    }


def test_firered_determinism_policy_matches_the_recorded_artifact() -> None:
    asr = plan_for("firered")["roles"]["asr"]
    assert asr["determinism_tolerance_ms"] == 2.0
    assert asr["determinism_basis"] == (
        "exact-repeat 60-minute fixture repeated the text and speaker-null sequences; "
        "maximum rebased timestamp drift was 1.0000000000002037 ms, within the frozen "
        "2.0 ms tolerance, so normalized segments were not byte-equal"
    )


def test_vibevoice_catalog_scopes_the_generation_cap_projection() -> None:
    catalog = build_catalog(stacks.get_stack("vibevoice"), METADATA)
    assert catalog["failure_recovery"] == {
        "partial_results": "prefix_only",
        "note": (
            "Generation truncation can preserve complete decoded segments before the cut and "
            "--range addresses the remainder; a failure before the first complete segment "
            "leaves nothing. On spice-30min-participant, 11,345 generated tokens over 1,800 "
            "seconds is 6.30 tokens/second; at that observed rate, the declared 16,384-token "
            "cap projects to about 43 minutes of comparable audio. This is a rate "
            "extrapolation, not an observed truncation."
        ),
    }


def test_ready_packages_do_not_count_as_downloads() -> None:
    request = resolve_request(
        stack_id="qwen-1.7b",
        input_path=Path("sample.wav"),
        wants="diarization,word_timestamps",
    )
    first = build_plan(request, METADATA)
    ready = [item["package"] for item in first.packages]
    emitted = serialize_plan(build_plan(request, METADATA, provisioned_packages=ready))
    assert emitted["total_known_download_bytes"] == 0
    assert emitted["unsized_packages"] == []
    assert all(item["provisioned"] for item in emitted["packages"])
