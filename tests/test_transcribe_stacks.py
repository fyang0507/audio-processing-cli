from __future__ import annotations

import json
from dataclasses import replace
from copy import deepcopy
from pathlib import Path

import pytest

from audio_cli import packages as pkg
from audio_cli.transcribe import stacks


EXPECTED = {
    "qwen-1.7b": {
        "languages": "native", "verbatim": "native", "diarization": "add_on",
        "overlapped_speech": "add_on", "vad": "add_on",
        "word_timestamps": "add_on",
        "segment_timestamps": "unsatisfiable_on_stack",
        "lid": "unsatisfiable_on_stack", "token_lid": "unsupported",
    },
    "qwen-0.6b": {
        "languages": "native", "verbatim": "native", "diarization": "add_on",
        "overlapped_speech": "add_on", "vad": "add_on",
        "word_timestamps": "add_on",
        "segment_timestamps": "unsatisfiable_on_stack",
        "lid": "unsatisfiable_on_stack", "token_lid": "unsupported",
    },
    "vibevoice": {
        "languages": "native", "verbatim": "native", "diarization": "native",
        "overlapped_speech": "add_on", "vad": "add_on",
        "word_timestamps": "add_on", "segment_timestamps": "native",
        "lid": "unsatisfiable_on_stack", "token_lid": "unsupported",
    },
    "firered": {
        "languages": "native", "verbatim": "native", "diarization": "add_on",
        "overlapped_speech": "add_on", "vad": "native_stage",
        "word_timestamps": "native", "segment_timestamps": "native",
        "lid": "native_stage", "token_lid": "unsupported",
    },
}


def test_stack_table_is_complete_and_cross_checked_with_the_manifest() -> None:
    assert stacks.validate() == []
    assert set(stacks.stack_ids()) == set(EXPECTED)
    assert len(stacks.capability_order()) == 9
    assert sum(len(item.capabilities) for item in stacks.stack_definitions().values()) == 36
    for definition in stacks.stack_definitions().values():
        for cell in definition.capabilities.values():
            assert cell["evidence_source"].startswith("model_tests/")


@pytest.mark.parametrize(
    ("stack_id", "capability", "resolution"),
    [
        (stack_id, capability, resolution)
        for stack_id, capabilities in EXPECTED.items()
        for capability, resolution in capabilities.items()
    ],
)
def test_every_derivation_cell_is_data(
    stack_id: str, capability: str, resolution: str
) -> None:
    assert stacks.get_stack(stack_id).capabilities[capability]["resolution"] == resolution


def test_qwen_language_vocabulary_is_the_closed_thirty_name_list() -> None:
    languages = stacks.language_vocabulary("qwen")
    assert len(languages) == 30
    assert languages[:3] == ("Chinese", "English", "Cantonese")
    assert languages[-3:] == ("Romanian", "Hungarian", "Macedonian")
    assert len({language.casefold() for language in languages}) == 30


def test_unsatisfiable_cells_must_have_an_available_alternative(monkeypatch) -> None:
    definitions = dict(stacks.stack_definitions())
    firered = definitions["firered"]
    capabilities = deepcopy(firered.capabilities)
    capabilities["lid"] = {
        "resolution": "unsatisfiable_on_stack",
        "reason": "no_backend_declares_on_stack",
        "quality": "unmeasured",
        "catalog_note": capabilities["lid"]["catalog_note"],
        "evidence_source": capabilities["lid"]["evidence_source"],
    }
    definitions["firered"] = replace(firered, capabilities=capabilities)
    monkeypatch.setattr(stacks, "stack_definitions", lambda: definitions)
    assert any(
        "lid: unsatisfiable_on_stack requires a non-empty alternative" in problem
        for problem in stacks.validate()
    )


def test_provisioned_qwen_configs_publish_the_same_language_vocabulary() -> None:
    registry = pkg.load_registry()
    checked = 0
    for package_id in ("qwen3-asr-1.7b-8bit", "qwen3-asr-0.6b-8bit"):
        entry = registry.get("packages", {}).get(package_id, {})
        location = entry.get("materialized", {}).get("path")
        if entry.get("state") != "ready" or not location:
            continue
        config_path = Path(location) / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        assert tuple(config["support_languages"]) == stacks.language_vocabulary("qwen")
        checked += 1
    if checked == 0:
        pytest.skip("neither pinned Qwen checkpoint is provisioned")
