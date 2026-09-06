"""Result-shape invariants over the transcription specification examples."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from tests.docs.spec_document_test_support import (
    CAPABILITY_ARRAYS,
    SEGMENT_KEYS,
    SPEC_DOCS,
    is_plan,
    is_result,
    json_blocks,
    strip_punctuation,
    valid_completion_shape,
    walk,
)


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_results_and_samples_declare_completeness(path: Path) -> None:
    """Saved output must carry its completion state after the exit code is gone."""
    for index, doc in json_blocks(path):
        candidates = []
        if is_result(doc):
            candidates.append(("result", doc))
        if is_plan(doc):
            candidates.append(("sample_output", doc["sample_output"]))
        for kind, candidate in candidates:
            assert valid_completion_shape(candidate), (
                f"{path.name} block {index}: {kind} must carry coverage iff complete is false"
            )


@pytest.mark.parametrize(
    "candidate",
    [
        {"complete": True, "coverage": {}},
        {"complete": False},
        {"coverage": {}},
    ],
)
def test_completion_shape_rejects_each_invalid_state(candidate: dict) -> None:
    """Both directions of the iff must be reachable even before an incomplete sample ships."""
    assert not valid_completion_shape(candidate)


@pytest.mark.parametrize(
    "candidate",
    [
        {"complete": True},
        {"complete": False, "coverage": {}},
    ],
)
def test_completion_shape_accepts_each_valid_state(candidate: dict) -> None:
    """The complete and incomplete forms are both legal, not just rejectable mutations."""
    assert valid_completion_shape(candidate)


@pytest.mark.parametrize("path", SPEC_DOCS, ids=lambda p: p.name)
def test_results_carry_no_key_for_an_unrequested_capability(path: Path) -> None:
    """The anti-fabrication guarantee: absence is meaningful, so it must be exact."""
    for index, doc in json_blocks(path):
        if not is_result(doc):
            continue
        requested = set(doc["provenance"]["outcomes"])
        body = set(doc) - {
            "schema_version",
            "complete",
            "coverage",
            "source",
            "segments",
            "abstentions",
            "provenance",
        }
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
            for earlier, later in pairwise(segment.get("words") or []):
                assert later["start"] >= earlier["end"] - 1e-9, (
                    f"{path.name} block {index}: {segment['segment_id']} word bounds go backwards"
                )
            if "start" in segment and segment.get("words"):
                first, last = segment["words"][0], segment["words"][-1]
                assert (
                    segment["start"] - 1e-9 <= first["start"]
                    and last["end"] <= segment["end"] + 1e-9
                ), f"{path.name} block {index}: {segment['segment_id']} words fall outside it"
        for array in ("turns", "vad_regions", "lid_regions"):
            for span in doc.get(array, []):
                if span.get("start") is None:
                    continue
                assert 0 <= span["start"] < span["end"] <= duration, (
                    f"{path.name} block {index}: {array} span outside the source"
                )
