"""Core invariants over the transcription specification documents.

Their JSON examples are load-bearing: `TRANSCRIBE_HAPPY_PATH.md` is diffed against real
implementations, and every payload in `TRANSCRIBE_CONTRACT.md` is an exact contract. These
tests check the documents, not the product; shipped-command tests separately cover real output.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.docs.spec_document_loader import read_spec_document
from tests.docs.spec_document_test_support import (
    CAPABILITY_NAMES,
    CONTRACT,
    HAPPY_PATH,
    QWEN_LANGUAGES,
    RETIRED_KEYS,
    SPEC_DOCS,
    is_catalog,
    json_blocks,
    walk,
)


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_every_json_block_parses(path: Path) -> None:
    """A malformed example is worse than no example: it cannot be diffed against."""
    assert json_blocks(path), f"{path.name} has no JSON examples"


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_no_retired_key_reappears(path: Path) -> None:
    for index, doc in json_blocks(path):
        for pointer, key in walk(doc):
            assert key not in RETIRED_KEYS, (
                f"{path.name} block {index}: retired key {key!r} at {pointer}. "
                "See VOCABULARY.md's retired words for why it was removed."
            )


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_catalog_entries_are_availability_plus_prose(path: Path) -> None:
    """A capabilities report is read to make a choice, so it carries no nested objects."""
    for index, doc in json_blocks(path):
        if not is_catalog(doc):
            continue
        for name, entry in doc["capabilities"].items():
            assert name in CAPABILITY_NAMES, (
                f"{path.name} block {index}: unknown capability {name!r}"
            )
            assert set(entry) <= {"availability", "note", "reason"}, (
                f"{path.name} block {index}: {name} carries {sorted(set(entry))}"
            )
            assert entry["availability"] in {"native", "requires_add_on", "impossible"}
            assert len(entry.get("note", "")) > 20, (
                f"{path.name} block {index}: {name} has no substantive note"
            )


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_capability_errors_publish_the_whole_menu(path: Path) -> None:
    """A caller correcting --want must not need a second command to find the options."""
    for index, doc in json_blocks(path):
        available = isinstance(doc, dict) and doc.get("available_on_stack")
        if not available:
            continue
        partition = (
            set(available["native"])
            | set(available["requires_add_on"])
            | set(available["impossible"])
        )
        assert partition == CAPABILITY_NAMES, (
            f"{path.name} block {index}: available_on_stack misses "
            f"{sorted(CAPABILITY_NAMES - partition)} and adds {sorted(partition - CAPABILITY_NAMES)}"
        )


def test_every_declared_error_code_is_shown_with_a_fix() -> None:
    """An error code documented but never rendered is a code nobody has had to design."""
    declared = set(
        re.findall(
            r"\| `([a-z_]+)` \| [0-4] \|",
            read_spec_document(CONTRACT),
        )
    )
    assert len(declared) >= 12, "the error-code table lost rows"
    shown = {}
    for path in SPEC_DOCS:
        for _, doc in json_blocks(path):
            if isinstance(doc, dict) and "code" in doc:
                assert "fix" in doc, f"{doc['code']} payload has no fix"
                shown[doc["code"]] = doc["fix"]
    assert declared <= set(shown), f"declared but never shown: {sorted(declared - set(shown))}"


def test_most_fixes_are_runnable_commands() -> None:
    """`fix` is a command wherever one would work, and a sentence only where none would."""
    sentence_fixes = set()
    for path in SPEC_DOCS:
        for _, doc in json_blocks(path):
            if isinstance(doc, dict) and "code" in doc and not doc["fix"].startswith("audio "):
                sentence_fixes.add(doc["code"])
    assert sentence_fixes == {
        "stack_required",
        "capability_unknown",
        "capability_unsatisfiable_on_stack",
        "option_unsupported_on_stack",
        "option_value_unsupported",
        "pin_conflicts_with_native_capability",
        "range_invalid",
        "input_required",
        "want_not_implemented",
        "timing_required_for_timestamps",
        "timestamps_unsupported_for_format",
        "provenance_unsupported_for_format",
        "receipt_options_invalid",
        "capability_unsupported",
        "backend_failed",
        "output_is_canonical_input",
        "output_path_invalid",
        "output_required_for_force",
        "export_input_invalid",
        "export_inputs_incompatible",
        "timing_required_for_format",
    }, (
        f"unexpected sentence-only fixes: {sorted(sentence_fixes)}. Every other code has a "
        "configuration that works, so its fix must be copy-pasteable."
    )


def test_language_option_errors_keep_distinct_exact_shapes() -> None:
    """A bad value and an unsupported option are different corrections."""
    errors = {
        doc["code"]: doc
        for _, doc in json_blocks(HAPPY_PATH)
        if isinstance(doc, dict) and "code" in doc
    }
    unsupported = errors["option_unsupported_on_stack"]
    assert set(unsupported) == {
        "code",
        "field",
        "provided",
        "allowed",
        "stacks_accepting",
        "fix",
    }
    assert unsupported["allowed"] == []

    bad_value = errors["option_value_unsupported"]
    assert set(bad_value) == {
        "code",
        "field",
        "provided",
        "allowed",
        "did_you_mean",
        "fix",
    }
    assert tuple(bad_value["allowed"]) == QWEN_LANGUAGES
    assert bad_value["did_you_mean"] == "English"
