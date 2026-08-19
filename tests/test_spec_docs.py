"""Invariants over the transcription spec documents.

The `transcribe` surface is specified in three tracked documents before any of it is
implemented. Their JSON examples are load-bearing: `TRANSCRIBE_HAPPY_PATH.md` exists to be
diffed against a real implementation, and every payload in `TRANSCRIBE_CONTRACT.md` is a
contract an implementer is expected to satisfy exactly.

These tests exist because reviewing those documents by eye repeatedly missed defects that a
five-line check catches: a leaked speaker id, an arithmetic sum that did not add up, a
retired field name that came back, a capability key present for a capability nobody
requested. Each assertion below corresponds to a defect that was actually found this way.

They check the documents, not the product. When the CLI is implemented, the same key-set and
absence rules must be asserted against real command output; these tests do not replace that.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "TRANSCRIBE_CONTRACT.md"
HAPPY_PATH = REPO / "TRANSCRIBE_HAPPY_PATH.md"
VOCABULARY = REPO / "VOCABULARY.md"
SPEC_DOCS = (CONTRACT, HAPPY_PATH)

# The capability namespace, as published in VOCABULARY.md.
CAPABILITY_NAMES = frozenset({
    "languages", "verbatim", "diarization", "overlapped_speech", "vad",
    "word_timestamps", "segment_timestamps", "lid", "token_lid",
})

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
RETIRED_KEYS = frozenset({
    # replaced by field-standard capability names
    "speaker_attribution", "turn_bounds", "overlap_intervals", "speech_bounds",
    "word_bounds", "segment_bounds", "region_language", "token_language",
    # collapsed into one prose note per capability
    "measured_limit", "observed_limit", "interface_basis", "timing_precision",
    "alternative", "shares_stage_with", "add_on_cost", "stage_cost",
    "measured_envelope", "produces",
    # removed outright: no caller can act on them
    "floors", "policy", "determinism", "provenance_only", "container_bounds",
    "container_language", "language_input", "roles_included", "roles_conditional",
    # citations and versions that will not exist in shipped output
    "record", "determinism_record", "plan_version", "catalog_version",
    # a plan echoing the request back, and a printing note dressed as output
    "request", "elided",
    # restated by the capabilities report
    "measured",
    # deterministic glue is not a role
    "reconciler",
})

SAMPLE_META = frozenset({"sample", "note", "schema_version", "source", "segments",
                         "abstentions", "provenance"})


def json_blocks(path: Path) -> list[tuple[int, object]]:
    bodies = re.findall(r"```json\n(.*?)```", path.read_text(), re.S)
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
            assert name in CAPABILITY_NAMES, f"{path.name} block {index}: unknown capability {name!r}"
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
        partition = (set(available["native"]) | set(available["requires_add_on"])
                     | set(available["impossible"]))
        assert partition == CAPABILITY_NAMES, (
            f"{path.name} block {index}: available_on_stack misses "
            f"{sorted(CAPABILITY_NAMES - partition)} and adds {sorted(partition - CAPABILITY_NAMES)}"
        )


def test_every_declared_error_code_is_shown_with_a_fix() -> None:
    """An error code documented but never rendered is a code nobody has had to design."""
    declared = set(re.findall(r"\| `([a-z_]+)` \| [0-4] \|", CONTRACT.read_text()))
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
    assert sentence_fixes == {"capability_unsupported", "backend_failed"}, (
        f"unexpected sentence-only fixes: {sorted(sentence_fixes)}. Every other code has a "
        "configuration that works, so its fix must be copy-pasteable."
    )


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_results_carry_no_key_for_an_unrequested_capability(path: Path) -> None:
    """The anti-fabrication guarantee: absence is meaningful, so it must be exact."""
    for index, doc in json_blocks(path):
        if not is_result(doc):
            continue
        requested = set(doc["provenance"]["outcomes"])
        body = set(doc) - {"schema_version", "source", "segments", "abstentions", "provenance"}
        for capability, array in CAPABILITY_ARRAYS.items():
            if array in body:
                assert capability in requested, (
                    f"{path.name} block {index}: {array} present but {capability} not requested"
                )
            if capability in requested:
                assert array in body, (
                    f"{path.name} block {index}: {capability} requested but {array} absent"
                )
        for segment in doc["segments"]:
            for key, capability in SEGMENT_KEYS.items():
                if key in segment:
                    assert capability in requested, (
                        f"{path.name} block {index}: {segment['segment_id']} has {key!r} "
                        f"without {capability}"
                    )


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_results_never_publish_a_non_label_as_a_speaker(path: Path) -> None:
    """VibeVoice emits Speaker "N/A" on non-speech; the adapter must drop it, not forward it."""
    for index, doc in json_blocks(path):
        if not isinstance(doc, dict):
            continue
        # Anywhere, not just on segments: turns carry speaker labels too, and the first
        # version of this check walked only segments and let an injected turn through.
        for pointer, key in walk(doc):
            if key != "speaker":
                continue
            value = doc
            for part in pointer.strip("/").replace("]", "").split("/"):
                if "[" in part:
                    name, idx = part.split("[")
                    value = value[name][int(idx)]
                else:
                    value = value[part]
            assert value != "N/A", (
                f"{path.name} block {index}: {pointer} publishes 'N/A' as a speaker. "
                "VibeVoice emits it on non-speech segments and the adapter must drop the key."
            )


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_result_observations_are_self_consistent(path: Path) -> None:
    """Counts, wall times, and peaks are checkable arithmetic, so check them."""
    for index, doc in json_blocks(path):
        if not is_result(doc):
            continue
        provenance = doc["provenance"]
        assert set(provenance) == {"stack", "outcomes", "observed"}, (
            f"{path.name} block {index}: provenance carries {sorted(provenance)}; it embeds the "
            "executed plan and adds only what running revealed"
        )
        observed = provenance["observed"]
        assert len(doc["segments"]) == observed["segments"]
        assert sum(len(s.get("words", [])) for s in doc["segments"]) == observed["words"]
        if "turns" in doc:
            assert len(doc["turns"]) == observed["turns"]
        walls = observed["stage_wall_seconds"]
        assert round(sum(walls.values()), 2) == pytest.approx(observed["total_wall_seconds"]), (
            f"{path.name} block {index}: stage walls do not sum to the total"
        )
        for by_stage in ("peak_rss_bytes_by_stage", "peak_mps_live_bytes_by_stage"):
            if by_stage in observed:
                total_key = by_stage.replace("_by_stage", "")
                assert max(observed[by_stage].values()) == observed[total_key], (
                    f"{path.name} block {index}: {total_key} must be the maximum of the "
                    "per-stage peaks, not their sum — stages do not overlap"
                )


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_printed_word_arrays_satisfy_the_punctuation_invariant(path: Path) -> None:
    """A sentence's text, stripped of punctuation, is its word texts joined.

    Case-insensitively: FireRedPunc lowercases its input and then re-capitalizes sentence
    starts, so sentence text and word text differ in case by construction.
    """
    for index, doc in json_blocks(path):
        if not is_result(doc):
            continue
        for segment in doc["segments"]:
            words = segment.get("words")
            if not words:
                continue
            joined = strip_punctuation("".join(w["text"] for w in words))
            assert joined == strip_punctuation(segment["text"]), (
                f"{path.name} block {index}: {segment['segment_id']} breaks the punctuation "
                f"invariant\n  words: {joined!r}\n  text:  {strip_punctuation(segment['text'])!r}"
            )


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_bounds_are_monotonic_and_inside_the_source(path: Path) -> None:
    for index, doc in json_blocks(path):
        if not is_result(doc):
            continue
        duration = doc["source"]["duration_seconds"]
        for segment in doc["segments"]:
            for earlier, later in zip(segment.get("words") or [], (segment.get("words") or [])[1:]):
                assert later["start"] >= earlier["end"] - 1e-9, (
                    f"{path.name} block {index}: {segment['segment_id']} word bounds go backwards"
                )
            if "start" in segment and segment.get("words"):
                first, last = segment["words"][0], segment["words"][-1]
                assert segment["start"] - 1e-9 <= first["start"] and last["end"] <= segment["end"] + 1e-9, (
                    f"{path.name} block {index}: {segment['segment_id']} words fall outside it"
                )
        for array in ("turns", "vad_regions", "lid_regions"):
            for span in doc.get(array, []):
                if span.get("start") is None:
                    continue
                assert 0 <= span["start"] < span["end"] <= duration, (
                    f"{path.name} block {index}: {array} span outside the source"
                )


def test_want_arguments_in_examples_use_real_capability_names() -> None:
    """A stale name in a shell example is as misleading as one in a payload."""
    intentionally_invalid = {"word_timing"}  # the capability_unknown demonstration
    for path in SPEC_DOCS:
        for line in path.read_text().splitlines():
            match = re.search(r"--want ([a-z_][a-z_,]*)", line)
            if not match:
                continue  # `--want <capabilities>` placeholders carry no names to check
            for name in filter(None, match.group(1).split(",")):
                assert name in CAPABILITY_NAMES or name in intentionally_invalid, (
                    f"{path.name}: --want {name!r} is not a capability name"
                )


def test_vocabulary_publishes_the_namespace_the_examples_use() -> None:
    """The naming contract and the worked examples must not drift apart."""
    text = VOCABULARY.read_text()
    for name in CAPABILITY_NAMES:
        assert f"`{name}`" in text, f"VOCABULARY.md does not define {name!r}"
