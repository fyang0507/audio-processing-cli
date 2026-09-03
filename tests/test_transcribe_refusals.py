from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from audio_cli.transcribe import refusals, stacks
from audio_cli.transcribe.planner import resolve_request

HAPPY_PATH = Path(__file__).resolve().parents[1] / "TRANSCRIBE_HAPPY_PATH.md"


def documented_errors() -> dict[str, dict]:
    blocks = re.findall(r"```json\n(.*?)```", HAPPY_PATH.read_text(), re.S)
    return {
        value["code"]: value
        for block in blocks
        if isinstance((value := json.loads(block)), dict) and "code" in value
    }


@pytest.mark.parametrize(
    "request_args",
    [
        {"stack_id": None, "input_path": Path("meeting.m4a"), "wants": "diarization"},
        {"stack_id": "qwen-1.7b", "input_path": None, "wants": "diarization"},
        {"stack_id": "qwen-1.7b", "input_path": Path("meeting.m4a"),
         "wants": "word_timing"},
        {"stack_id": "qwen-1.7b", "input_path": Path("meeting.m4a"),
         "wants": "segment_timestamps"},
        {"stack_id": "firered", "input_path": Path("meeting.m4a"),
         "wants": "token_lid"},
        {"stack_id": "vibevoice", "input_path": Path("demo.mp4"), "wants": "verbatim",
         "language": "Cantonese"},
        {"stack_id": "qwen-1.7b", "input_path": Path("meeting.m4a"), "wants": None,
         "language": "EN"},
        {"stack_id": "vibevoice", "input_path": Path("demo.mp4"),
         "wants": "diarization", "diarizer": "fluidaudio"},
    ],
)
def test_every_worked_request_refusal_is_field_for_field(request_args: dict) -> None:
    with pytest.raises(refusals.Refusal) as caught:
        resolve_request(**request_args)
    documented = documented_errors()[caught.value.payload["code"]]
    assert caught.value.exit_code == 2
    assert caught.value.payload == documented


def test_lowercase_language_is_accepted_but_abbreviation_is_only_a_suggestion() -> None:
    accepted = resolve_request(
        stack_id="qwen-1.7b",
        input_path=Path("meeting.m4a"),
        wants=None,
        language="english",
    )
    assert accepted.language == "English"

    with pytest.raises(refusals.Refusal) as caught:
        resolve_request(
            stack_id="qwen-1.7b",
            input_path=Path("meeting.m4a"),
            wants=None,
            language="EN",
        )
    assert caught.value.payload["did_you_mean"] == "English"


def test_unknown_name_without_a_suggestion_gets_a_runnable_floors_or_valid_fix() -> None:
    with pytest.raises(refusals.Refusal) as caught:
        resolve_request(
            stack_id="qwen-1.7b",
            input_path=Path("meeting.m4a"),
            wants="vad,banana",
        )
    assert "banana" not in caught.value.payload["fix"]
    assert caught.value.payload["fix"].endswith("--want vad")


def test_invalid_pin_value_fix_adds_the_capability_the_pin_needs() -> None:
    with pytest.raises(refusals.Refusal) as caught:
        resolve_request(
            stack_id="qwen-1.7b",
            input_path=Path("meeting.m4a"),
            wants=None,
            vad="silero",
        )
    assert caught.value.payload["code"] == "option_value_unsupported"
    assert caught.value.payload["fix"].endswith("--want vad --vad silero-vad")


def test_unsupported_reason_and_fix_come_from_the_selected_table_cell(monkeypatch) -> None:
    definition = stacks.get_stack("firered")
    capabilities = deepcopy(definition.capabilities)
    capabilities["token_lid"]["reason"] = "recorded_custom_reason"
    capabilities["token_lid"]["refusal_fix"] = "use the separately named region output"
    changed = replace(definition, capabilities=capabilities)
    definitions = dict(stacks.stack_definitions())
    definitions["firered"] = changed
    monkeypatch.setattr(stacks, "stack_definitions", lambda: definitions)
    monkeypatch.setattr(stacks, "get_stack", lambda identifier: definitions[identifier])

    with pytest.raises(refusals.Refusal) as caught:
        resolve_request(
            stack_id="firered",
            input_path=Path("meeting.m4a"),
            wants="token_lid",
        )
    assert caught.value.payload["reason"] == "recorded_custom_reason"
    assert caught.value.payload["fix"] == "use the separately named region output"


def test_empty_unsatisfiable_alternative_fails_as_table_drift(monkeypatch) -> None:
    monkeypatch.setattr(stacks, "allowed_stacks", lambda capability: [])
    with pytest.raises(stacks.StackTableError, match="has no alternative"):
        refusals.capability_unsatisfiable_on_stack(
            stacks.get_stack("qwen-1.7b"),
            "meeting.m4a",
            "segment_timestamps",
            ("segment_timestamps",),
        )


EXPECTED_KEYS = {
    "stack_required": {"code", "field", "allowed", "stacks", "fix"},
    "input_required": {"code", "field", "note", "fix"},
    "capability_unknown": {
        "code", "field", "provided", "did_you_mean", "available_on_stack", "fix",
    },
    "option_unsupported_on_stack": {
        "code", "field", "provided", "allowed", "stacks_accepting", "fix",
    },
    "option_value_unsupported": {
        "code", "field", "provided", "allowed", "did_you_mean", "fix",
    },
    "capability_unsatisfiable_on_stack": {
        "code", "capability", "allowed", "available_on_stack", "fix",
    },
    "capability_unsupported": {"code", "capability", "allowed", "reason", "fix"},
    "pin_conflicts_with_native_capability": {
        "code", "field", "provided", "allowed", "capability", "fix",
    },
    "timing_required_for_format": {
        "code", "field", "provided", "requires_capability", "found", "note", "fix",
    },
    "packages_not_provisioned": {
        "code", "missing", "total_known_download_bytes", "unsized_packages", "fix",
    },
    "package_integrity_failed": {"code", "failed", "fix"},
    "package_build_unusable": {"code", "package", "product", "built", "fix"},
    "backend_failed": {"code", "role", "backend", "detail", "fix"},
    "run_incomplete": {
        "code", "role", "backend", "detail", "coverage", "output", "fix",
    },
}


def all_refusal_examples() -> list[refusals.Refusal]:
    qwen = stacks.get_stack("qwen-1.7b")
    vibe = stacks.get_stack("vibevoice")
    coverage = {
        "covered_through_seconds": 1.0,
        "covered_fraction": 0.5,
        "covered_intervals": [[0.0, 1.0]],
        "missing_intervals": [[1.0, 2.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    return [
        refusals.stack_required("meeting.m4a", ("diarization",)),
        refusals.input_required("qwen-1.7b", ("diarization",)),
        refusals.capability_unknown(
            qwen, "meeting.m4a", "word_timing", ("word_timing",), "word_timestamps"
        ),
        refusals.option_unsupported_on_stack(
            vibe, "demo.mp4", "--language", "Cantonese", ("verbatim",)
        ),
        refusals.option_value_unsupported(
            qwen, "meeting.m4a", "--language", "EN",
            stacks.language_vocabulary("qwen"), (), "English",
        ),
        refusals.capability_unsatisfiable_on_stack(
            qwen, "meeting.m4a", "segment_timestamps", ("segment_timestamps",)
        ),
        refusals.capability_unsupported(
            "token_lid", "no_backend_declares",
            "no stack or add-on satisfies this; use region-level lid instead",
        ),
        refusals.pin_conflicts_with_native_capability(
            vibe, "demo.mp4", "--diarizer", "fluidaudio", "diarization",
            ("diarization",),
        ),
        refusals.timing_required_for_format(
            "meeting.json", "srt", (), "qwen-1.7b", ("verbatim",)
        ),
        refusals.packages_not_provisioned(
            "qwen-1.7b", ("qwen3-asr-1.7b-8bit",), 123, ()
        ),
        refusals.package_integrity_failed(({
            "package": "qwen3-forcedaligner", "check": "weight_digest",
            "expected": "a", "actual": "b",
        },)),
        refusals.package_build_unusable("fluidaudio", "fluidaudiocli"),
        refusals.backend_failed("asr", "vibevoice-asr-7b", "oom", "free memory"),
        refusals.run_incomplete(
            "asr", "qwen3-asr-1.7b-8bit", "budget", coverage,
            "partial.json", "audio transcribe run --range 1.0:",
        ),
    ]


def test_every_current_contract_refusal_has_one_fixed_shape_builder() -> None:
    examples = all_refusal_examples()
    assert len(examples) == 14
    assert {item.payload["code"] for item in examples} == set(EXPECTED_KEYS)
    for item in examples:
        assert set(item.payload) == EXPECTED_KEYS[item.payload["code"]]
        assert item.payload["fix"]
