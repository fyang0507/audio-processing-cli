from __future__ import annotations

from pathlib import Path

import pytest

from audio_cli.transcribe import refusals, stacks
from audio_cli.transcribe.catalog import InputMetadata
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
        "sample", "note", "schema_version", "complete", "source", "segments",
        "abstentions", "provenance",
    }
    assert emitted["sample_output"]["segments"] == [
        {"segment_id": "seg_0", "text": None}
    ]
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
        (stack_id, capability,
         "capability_unsupported" if cell["resolution"] == "unsupported"
         else "capability_unsatisfiable_on_stack")
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


def test_sample_provenance_embeds_the_executable_plan_without_recursing() -> None:
    emitted = plan_for("vibevoice", "diarization,word_timestamps")
    core = {key: value for key, value in emitted.items() if key != "sample_output"}
    assert emitted["sample_output"]["provenance"]["plan"] == core
    assert "sample_output" not in core
    assert emitted["sample_output"]["abstentions"] == []


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
