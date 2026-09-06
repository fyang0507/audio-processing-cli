"""The normalized transcription result and its only serializer.

The serializer owns capability-gated absence. Adapters provide normalized JSON-safe mappings;
they do not decide which optional keys are legal, and samples use this same function rather than
maintaining a second rendering path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .alignment import _validate_abstentions, validate_alignment_evidence
from .types import (
    _ARRAY_CAPABILITIES,
    ABSENT,
    SCHEMA_VERSION,
    NormalizedResult,
    ResultError,
)
from .validation import (
    _array,
    _capabilities,
    _reject_non_label_speakers,
    _validate_coverage,
    _validate_provenance,
    _validate_segments,
    _validate_source,
    _validate_span_array,
)


def serialize_result(result: NormalizedResult) -> dict[str, Any]:
    """Validate and return the one JSON-safe normalized result representation."""
    requested = _capabilities(result.requested_capabilities)
    if not isinstance(result.complete, bool):
        raise ResultError("complete must be a boolean")
    has_coverage = result.coverage is not ABSENT
    if has_coverage is result.complete:
        raise ResultError("coverage must be present if and only if complete is false")
    if has_coverage and not isinstance(result.coverage, Mapping):
        raise ResultError("coverage must be an object")

    duration = _validate_source(result.source, sample=result.sample)
    if has_coverage:
        assert isinstance(result.coverage, Mapping)
        _validate_coverage(result.coverage, duration=duration)
    _validate_segments(result.segments, requested, sample=result.sample, duration=duration)
    _validate_abstentions(
        result.abstentions, sample=result.sample, duration=duration, segments=result.segments
    )
    _validate_provenance(result.provenance, requested, result, sample=result.sample)
    if (
        any("alignment" in item for item in result.abstentions)
        and result.provenance["outcomes"].get("word_timestamps") != "abstained"
    ):
        raise ResultError("alignment evidence requires an abstained word_timestamps outcome")

    arrays = {
        "turns": ({"turn_id", "speaker", "start", "end"}, result.turns),
        "vad_regions": ({"start", "end"}, result.vad_regions),
        "lid_regions": ({"start", "end", "language", "confidence"}, result.lid_regions),
        "overlapped_speech": ({"overlap_id", "start", "end"}, result.overlapped_speech),
    }
    for capability, field in _ARRAY_CAPABILITIES.items():
        expected_keys, value = arrays[field]
        present = value is not ABSENT
        if present != (capability in requested):
            state = "present" if present else "absent"
            raise ResultError(
                f"{field} is {state}, but capability {capability!r} "
                f"was {'requested' if capability in requested else 'not requested'}"
            )
        if present:
            if not isinstance(value, (list, tuple)):
                raise ResultError(f"{field} must be an array")
            _validate_span_array(
                value, field, expected_keys, sample=result.sample, duration=duration
            )

    validate_alignment_evidence(result, duration=duration)

    payload: dict[str, Any] = {}
    if result.sample:
        if not result.note:
            raise ResultError("a sample result requires a note")
        payload.update({"sample": True, "note": result.note})
    elif result.note is not None:
        raise ResultError("note is sample metadata and cannot appear on a run result")
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "complete": result.complete,
            "source": dict(result.source),
            "segments": [dict(item) for item in result.segments],
        }
    )
    for field in ("turns", "vad_regions", "lid_regions", "overlapped_speech"):
        value = _array(result, field)
        if value is not ABSENT:
            payload[field] = [dict(item) for item in value]
    payload["abstentions"] = [dict(item) for item in result.abstentions]
    if has_coverage:
        assert isinstance(result.coverage, Mapping)
        payload["coverage"] = dict(result.coverage)
    payload["provenance"] = dict(result.provenance)

    _reject_non_label_speakers(payload)
    try:
        return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ResultError(f"result is not JSON-safe: {exc}") from exc
