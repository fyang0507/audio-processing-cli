from __future__ import annotations

from dataclasses import replace

import pytest

from audio_cli.transcribe.result import (
    ABSTENTION_REASONS,
    NormalizedResult,
    ResultError,
    serialize_result,
)


def base_result(**changes) -> NormalizedResult:
    found = NormalizedResult(
        source={"path": "sample.wav", "duration_seconds": 10.0, "timebase": "seconds"},
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        abstentions=[],
        provenance={"stack": "qwen-1.7b", "outcomes": {}, "observed": {}, "plan": {}},
        requested_capabilities=frozenset(),
    )
    return replace(found, **changes)


def test_result_has_the_fixed_floor_shape() -> None:
    assert serialize_result(base_result()) == {
        "schema_version": 1,
        "complete": True,
        "source": {"path": "sample.wav", "duration_seconds": 10.0, "timebase": "seconds"},
        "segments": [{"segment_id": "seg_0", "text": "Hello."}],
        "abstentions": [],
        "provenance": {"stack": "qwen-1.7b", "outcomes": {}, "observed": {}, "plan": {}},
    }


@pytest.mark.parametrize(
    ("capability", "field", "value"),
    [
        ("diarization", "turns", []),
        ("vad", "vad_regions", []),
        ("lid", "lid_regions", []),
        ("overlapped_speech", "overlapped_speech", []),
    ],
)
def test_top_level_array_exists_iff_its_capability_is_requested(
    capability: str,
    field: str,
    value: list,
) -> None:
    with pytest.raises(ResultError, match=field):
        serialize_result(base_result(**{field: value}))
    provenance = {
        "stack": "qwen-1.7b",
        "outcomes": {capability: "produced"},
        "observed": {},
        "plan": {},
    }
    with pytest.raises(ResultError, match=field):
        serialize_result(base_result(
            requested_capabilities=frozenset({capability}),
            provenance=provenance,
        ))

    emitted = serialize_result(base_result(
        requested_capabilities=frozenset({capability}),
        provenance=provenance,
        **{field: value},
    ))
    assert field in emitted


@pytest.mark.parametrize(
    ("capability", "keys", "values"),
    [
        ("diarization", {"speaker"}, {"speaker": "S1"}),
        (
            "word_timestamps",
            {"words"},
            {"words": [{"word_id": "w_0", "text": "Hello", "start": 0.0, "end": 1.0}]},
        ),
        ("segment_timestamps", {"start", "end"}, {"start": 0.0, "end": 1.0}),
    ],
)
def test_segment_key_is_refused_without_its_capability(
    capability: str,
    keys: set[str],
    values: dict,
) -> None:
    segment = {"segment_id": "seg_0", "text": "Hello.", **values}
    with pytest.raises(ResultError, match=capability):
        serialize_result(base_result(segments=[segment]))

    kwargs = {
        "requested_capabilities": frozenset({capability}),
        "provenance": {
            "stack": "qwen-1.7b",
            "outcomes": {capability: "produced"},
            "observed": {},
            "plan": {},
        },
    }
    if capability == "diarization":
        kwargs["turns"] = []
    emitted = serialize_result(base_result(segments=[segment], **kwargs))
    assert keys <= set(emitted["segments"][0])


def test_languages_and_verbatim_add_no_result_key_and_cannot_abstain() -> None:
    requested = frozenset({"languages", "verbatim"})
    result = base_result(
        requested_capabilities=requested,
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": {"languages": "produced", "verbatim": "produced"},
            "observed": {},
            "plan": {},
        },
    )
    emitted = serialize_result(result)
    assert "languages" not in emitted
    assert "verbatim" not in emitted

    bad = replace(result, provenance={
        "stack": "qwen-1.7b",
        "outcomes": {"languages": "abstained", "verbatim": "produced"},
        "observed": {},
        "plan": {},
    })
    with pytest.raises(ResultError, match="languages has no abstained"):
        serialize_result(bad)


def test_complete_and_coverage_are_bidirectional() -> None:
    coverage = {
        "scope_intervals": [[0.0, 10.0]],
        "covered_through_seconds": 4.0,
        "covered_fraction": 0.4,
        "covered_intervals": [[0.0, 4.0]],
        "missing_intervals": [[4.0, 10.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    with pytest.raises(ResultError, match="if and only if"):
        serialize_result(base_result(complete=False))
    with pytest.raises(ResultError, match="if and only if"):
        serialize_result(base_result(coverage=coverage))

    emitted = serialize_result(base_result(complete=False, coverage=coverage))
    assert emitted["complete"] is False
    assert emitted["coverage"] == coverage


def test_non_label_speaker_is_rejected_anywhere() -> None:
    requested = frozenset({"diarization"})
    provenance = {
        "stack": "vibevoice",
        "outcomes": {"diarization": "produced"},
        "observed": {},
        "plan": {},
    }
    for changes in (
        {"segments": [{"segment_id": "seg_0", "text": "noise", "speaker": "N/A"}],
         "turns": []},
        {"turns": [{"turn_id": "turn_0", "speaker": "N/A", "start": 0.0, "end": 1.0}]},
    ):
        with pytest.raises(ResultError, match="must be absent"):
            serialize_result(base_result(
                requested_capabilities=requested,
                provenance=provenance,
                **changes,
            ))


def test_real_attribution_and_lid_values_are_not_null_placeholders() -> None:
    diarization = frozenset({"diarization"})
    with pytest.raises(ResultError, match="speaker must be a string"):
        serialize_result(base_result(
            segments=[{"segment_id": "seg_0", "text": "noise", "speaker": None}],
            turns=[],
            requested_capabilities=diarization,
            provenance={
                "stack": "vibevoice",
                "outcomes": {"diarization": "produced"},
                "observed": {},
                "plan": {},
            },
        ))

    lid = frozenset({"lid"})
    with pytest.raises(ResultError, match="language must be a non-empty string"):
        serialize_result(base_result(
            lid_regions=[{"start": 0.0, "end": 1.0, "language": None, "confidence": None}],
            requested_capabilities=lid,
            provenance={
                "stack": "firered",
                "outcomes": {"lid": "produced"},
                "observed": {},
                "plan": {},
            },
        ))


def test_timed_output_cannot_leave_the_source_duration() -> None:
    with pytest.raises(ResultError, match="source duration"):
        serialize_result(base_result(
            segments=[{
                "segment_id": "seg_0",
                "text": "Hello.",
                "words": [{
                    "word_id": "w_0", "text": "Hello", "start": 9.0, "end": 11.0,
                }],
            }],
            requested_capabilities=frozenset({"word_timestamps"}),
            provenance={
                "stack": "qwen-1.7b",
                "outcomes": {"word_timestamps": "produced"},
                "observed": {},
                "plan": {},
            },
        ))


@pytest.mark.parametrize(
    ("word_start", "word_end"),
    [(0.998, 2.0), (2.0, 3.002)],
)
def test_words_must_stay_inside_explicit_segment_bounds(
    word_start: float,
    word_end: float,
) -> None:
    requested = frozenset({"segment_timestamps", "word_timestamps"})
    with pytest.raises(ResultError, match="words must fall inside the segment bounds"):
        serialize_result(base_result(
            segments=[{
                "segment_id": "seg_0",
                "text": "Hello.",
                "start": 1.0,
                "end": 3.0,
                "words": [{
                    "word_id": "w_0",
                    "text": "Hello",
                    "start": word_start,
                    "end": word_end,
                }],
            }],
            requested_capabilities=requested,
            provenance={
                "stack": "firered",
                "outcomes": {
                    "segment_timestamps": "produced",
                    "word_timestamps": "produced",
                },
                "observed": {},
                "plan": {},
            },
        ))


def test_words_preserve_the_recorded_one_millisecond_firered_end_seam() -> None:
    requested = frozenset({"segment_timestamps", "word_timestamps"})
    payload = serialize_result(base_result(
        segments=[{
            "segment_id": "seg_0",
            "text": "Hello.",
            "start": 1.0,
            "end": 3.0,
            "words": [{
                "word_id": "w_0",
                "text": "Hello",
                "start": 2.0,
                "end": 3.001,
            }],
        }],
        requested_capabilities=requested,
        provenance={
            "stack": "firered",
            "outcomes": {
                "segment_timestamps": "produced",
                "word_timestamps": "produced",
            },
            "observed": {},
            "plan": {},
        },
    ))

    assert payload["segments"][0]["end"] == 3.0
    assert payload["segments"][0]["words"][0]["end"] == 3.001


def test_overlap_id_is_a_non_empty_document_scoped_id() -> None:
    requested = frozenset({"overlapped_speech"})
    provenance = {
        "stack": "qwen-1.7b",
        "outcomes": {"overlapped_speech": "produced"},
        "observed": {},
        "plan": {},
    }
    for bad_id in ("", None, 123):
        with pytest.raises(ResultError, match="overlap_id must be a non-empty string"):
            serialize_result(base_result(
                overlapped_speech=[{
                    "overlap_id": bad_id, "start": 0.0, "end": 1.0,
                }],
                requested_capabilities=requested,
                provenance=provenance,
            ))

def test_abstention_reason_is_the_closed_recorded_enum() -> None:
    assert ABSTENTION_REASONS == {
        "alignment_unavailable", "overlap", "short_turn", "raw_fragment",
    }
    for reason in ABSTENTION_REASONS:
        if reason == "alignment_unavailable":
            capability = "word_timestamps"
            optional = {}
            outcome = "abstained"
        elif reason == "overlap":
            capability = "overlapped_speech"
            optional = {"overlapped_speech": []}
            outcome = "produced"
        else:
            capability = "diarization"
            optional = {"turns": []}
            outcome = "produced"
        emitted = serialize_result(base_result(
            abstentions=[{
                "abstention_id": "ab_0", "reason": reason, "start": 0.0, "end": 0.2,
            }],
            requested_capabilities=frozenset({capability}),
            provenance={
                "stack": "qwen-1.7b",
                "outcomes": {capability: outcome},
                "observed": {},
                "plan": {},
            },
            **optional,
        ))
        assert emitted["abstentions"][0]["reason"] == reason
    with pytest.raises(ResultError, match="reason must be one of"):
        serialize_result(base_result(abstentions=[{
            "abstention_id": "ab_0", "reason": "budget", "start": 0.0, "end": 0.2,
        }]))


@pytest.mark.parametrize("reason", sorted(ABSTENTION_REASONS))
def test_abstention_floor_is_not_gated_by_optional_capabilities(reason: str) -> None:
    emitted = serialize_result(base_result(abstentions=[{
        "abstention_id": "ab_0", "reason": reason, "start": 0.0, "end": 0.2,
    }]))
    assert emitted["abstentions"][0]["reason"] == reason


def test_unknown_capability_and_model_specific_keys_are_rejected() -> None:
    with pytest.raises(ResultError, match="unknown requested"):
        serialize_result(base_result(requested_capabilities=frozenset({"word_confidence"})))
    with pytest.raises(ResultError, match="unknown keys"):
        serialize_result(base_result(segments=[{
            "segment_id": "seg_0", "text": "Hello.", "asr_confidence": 0.9,
        }]))


def test_globally_unsupported_token_lid_cannot_reach_a_result() -> None:
    with pytest.raises(ResultError, match="unsupported by every stack"):
        serialize_result(base_result(requested_capabilities=frozenset({"token_lid"})))
