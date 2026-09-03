from __future__ import annotations

from dataclasses import replace

import pytest

from audio_cli.transcribe import sample
from audio_cli.transcribe.result import CAPABILITY_NAMES, ResultError, serialize_result

SOURCE = {"path": "meeting.m4a", "duration_seconds": 1794.2, "timebase": "seconds"}


def build(*capabilities: str) -> dict:
    return sample.build_sample_output(
        source=SOURCE,
        stack="qwen-1.7b",
        requested_capabilities=capabilities,
        plan={"roles": {}},
    )


def test_sample_uses_the_production_serializer(monkeypatch) -> None:
    captured = []

    def recording_serializer(value):
        captured.append(value)
        return {"through": "serializer"}

    monkeypatch.setattr(sample.result, "serialize_result", recording_serializer)
    assert build("diarization") == {"through": "serializer"}
    assert len(captured) == 1
    assert captured[0].sample is True


@pytest.mark.parametrize("capability", sorted(CAPABILITY_NAMES - {"token_lid"}))
def test_each_capability_produces_exactly_its_licensed_shape(capability: str) -> None:
    baseline = build()
    emitted = build(capability)
    added = set(emitted) - set(baseline)
    expected = {
        "diarization": {"turns"},
        "overlapped_speech": {"overlapped_speech"},
        "vad": {"vad_regions"},
        "lid": {"lid_regions"},
    }.get(capability, set())
    assert added == expected

    segment_added = set(emitted["segments"][0]) - set(baseline["segments"][0])
    segment_expected = {
        "diarization": {"speaker"},
        "word_timestamps": {"words"},
        "segment_timestamps": {"start", "end"},
    }.get(capability, set())
    assert segment_added == segment_expected


def test_combined_sample_contains_every_requested_shape_once() -> None:
    capabilities = (
        "languages", "verbatim", "diarization", "overlapped_speech", "vad",
        "word_timestamps", "segment_timestamps", "lid",
    )
    emitted = sample.build_sample_output(
        source=SOURCE,
        stack="qwen-1.7b",
        requested_capabilities=capabilities,
        plan={"roles": {}},
        abstention_reason="raw_fragment",
    )
    assert set(emitted) == {
        "sample", "note", "schema_version", "complete", "source", "segments", "turns",
        "vad_regions", "lid_regions", "overlapped_speech", "abstentions", "provenance",
    }
    assert emitted["complete"] is True
    assert emitted["source"] == SOURCE
    assert emitted["provenance"] == {
        "stack": "qwen-1.7b", "outcomes": {}, "observed": {}, "plan": {"roles": {}},
    }
    assert len(emitted["abstentions"]) == 1
    assert emitted["abstentions"][0]["reason"] == "raw_fragment"
    assert emitted["overlapped_speech"][0]["overlap_id"] == "overlap_0"


def test_placeholder_has_null_content_and_bounds_not_plausible_values() -> None:
    emitted = build("diarization", "word_timestamps", "segment_timestamps", "lid")
    segment = emitted["segments"][0]
    assert segment["text"] is None
    assert segment["speaker"] is None
    assert segment["start"] is None and segment["end"] is None
    assert segment["words"][0]["text"] is None
    assert segment["words"][0]["start"] is None
    assert emitted["lid_regions"][0]["confidence"] is None


def test_serializer_rejects_mutated_placeholder_content_and_zero_bound() -> None:
    original = sample.result.NormalizedResult(
        source=SOURCE,
        segments=[{"segment_id": "seg_0", "text": None, "start": None, "end": None}],
        abstentions=[{
            "abstention_id": "ab_0", "reason": "raw_fragment", "start": None, "end": None,
        }],
        provenance={"stack": "firered", "outcomes": {}, "observed": {}, "plan": {}},
        requested_capabilities=frozenset({"segment_timestamps", "diarization"}),
        turns=[],
        sample=True,
        note=sample.SAMPLE_NOTE,
    )
    with pytest.raises(ResultError, match="text placeholder must be null"):
        serialize_result(replace(
            original,
            segments=[{"segment_id": "seg_0", "text": "placeholder", "start": None,
                       "end": None}],
        ))
    with pytest.raises(ResultError, match="placeholder bounds must be null"):
        serialize_result(replace(
            original,
            segments=[{"segment_id": "seg_0", "text": None, "start": 0.0, "end": 0.0}],
        ))


def test_sample_rejects_an_unknown_capability() -> None:
    with pytest.raises(ResultError, match="unknown requested"):
        build("speaker_attribution")


def test_sample_rejects_the_globally_unsupported_capability() -> None:
    with pytest.raises(ResultError, match="unsupported by every stack"):
        build("token_lid")


def test_sample_does_not_fabricate_stack_specific_abstentions() -> None:
    assert build("diarization")["abstentions"] == []
    assert build("overlapped_speech")["abstentions"] == []
