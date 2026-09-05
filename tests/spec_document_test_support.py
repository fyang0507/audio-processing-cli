"""Shared paths, constants, and predicates for the transcription spec tests."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from spec_document_loader import read_spec_document

__all__ = [
    "AGENT_GUIDANCE",
    "CAPABILITY_ARRAYS",
    "CAPABILITY_NAMES",
    "CONTRACT",
    "HAPPY_PATH",
    "QWEN_LANGUAGES",
    "REPO",
    "RETIRED_KEYS",
    "SAMPLE_META",
    "SEGMENT_KEYS",
    "SHIPPED_SKILL_GUIDANCE",
    "SPEC_DOCS",
    "VOCABULARY",
    "is_catalog",
    "is_plan",
    "is_result",
    "json_blocks",
    "strip_punctuation",
    "valid_completion_shape",
    "walk",
]

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "TRANSCRIBE_CONTRACT.md"
HAPPY_PATH = REPO / "TRANSCRIBE_HAPPY_PATH.md"
VOCABULARY = REPO / "VOCABULARY.md"
SPEC_DOCS = (CONTRACT, HAPPY_PATH)
AGENT_GUIDANCE = REPO / "AGENTS.md"
AUDIO_SKILL = REPO / ".agents" / "skills" / "audio-cli"
SHIPPED_SKILL_GUIDANCE = (
    AUDIO_SKILL / "SKILL.md",
    AUDIO_SKILL / "references" / "model-packages.md",
    AUDIO_SKILL / "references" / "transcribe.md",
)

# The capability namespace, as published in VOCABULARY.md.
CAPABILITY_NAMES = frozenset(
    {
        "languages",
        "verbatim",
        "diarization",
        "overlapped_speech",
        "vad",
        "word_timestamps",
        "segment_timestamps",
        "lid",
        "token_lid",
    }
)

# Result-body arrays and the capability that must have been requested to produce them.
CAPABILITY_ARRAYS = {
    "diarization": "turns",
    "vad": "vad_regions",
    "lid": "lid_regions",
    "overlapped_speech": "overlapped_speech",
}

# Segment-level keys and the capability that must have been requested to produce them.
SEGMENT_KEYS = {
    "speaker": "diarization",
    "words": "word_timestamps",
    "start": "segment_timestamps",
    "end": "segment_timestamps",
}

# Names removed from the payload during spec review. Each was retired for a stated reason;
# a reappearance is a regression, not a stylistic choice.
RETIRED_KEYS = frozenset(
    {
        # replaced by field-standard capability names
        "speaker_attribution",
        "turn_bounds",
        "overlap_intervals",
        "speech_bounds",
        "word_bounds",
        "segment_bounds",
        "region_language",
        "token_language",
        # collapsed into one prose note per capability
        "measured_limit",
        "observed_limit",
        "interface_basis",
        "timing_precision",
        "alternative",
        "shares_stage_with",
        "add_on_cost",
        "stage_cost",
        "measured_envelope",
        "produces",
        # removed outright: no caller can act on them
        "floors",
        "policy",
        "determinism",
        "provenance_only",
        "container_bounds",
        "container_language",
        "language_input",
        "roles_included",
        "roles_conditional",
        # citations and versions that will not exist in shipped output
        "record",
        "determinism_record",
        "plan_version",
        "catalog_version",
        # a plan echoing the request back, and a printing note dressed as output
        "request",
        "elided",
        # restated by the capabilities report
        "measured",
        # deterministic glue is not a role
        "reconciler",
        # contradicted the fixed processing.unit_count field
        "unit_count_known_at_plan_time",
    }
)

SAMPLE_META = frozenset(
    {
        "sample",
        "note",
        "schema_version",
        "complete",
        "source",
        "segments",
        "abstentions",
        "provenance",
    }
)

QWEN_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)


def json_blocks(path: Path) -> list[tuple[int, object]]:
    bodies = re.findall(r"```json\n(.*?)```", read_spec_document(path), re.S)
    out = []
    for index, body in enumerate(bodies, start=1):
        out.append((index, json.loads(body)))
    return out


def walk(node, path: str = ""):
    """Yield (json-pointer-ish path, key) for every mapping key in a document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield path + "/" + key, key
            yield from walk(value, path + "/" + key)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from walk(item, f"{path}[{i}]")


def strip_punctuation(text: str) -> str:
    without = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", "", without).lower()


def is_catalog(doc) -> bool:
    return isinstance(doc, dict) and "capabilities" in doc and "processing" in doc


def is_result(doc) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("provenance"), dict)


def is_plan(doc) -> bool:
    return isinstance(doc, dict) and "sample_output" in doc


def valid_completion_shape(doc: dict) -> bool:
    """A result says whether it finished, and explains its gaps only when it did not."""
    return isinstance(doc.get("complete"), bool) and ("coverage" in doc) is (not doc["complete"])
