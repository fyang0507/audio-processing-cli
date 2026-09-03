"""Placeholder results for `audio transcribe plan`."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from . import result

SAMPLE_NOTE = "shape only; values are placeholders and cardinality is unknown until run"


def build_sample_output(
    *,
    source: Mapping[str, Any],
    stack: str,
    requested_capabilities: Iterable[str],
    plan: Mapping[str, Any],
    abstention_reason: str | None = None,
) -> dict[str, Any]:
    """Build a placeholder and send it through the production result serializer."""
    requested = frozenset(requested_capabilities)
    segment: dict[str, Any] = {"segment_id": "seg_0", "text": None}
    if "diarization" in requested:
        segment["speaker"] = None
    if "segment_timestamps" in requested:
        segment.update({"start": None, "end": None})
    if "word_timestamps" in requested:
        segment["words"] = [
            {"word_id": "w_0", "text": None, "start": None, "end": None}
        ]

    values: dict[str, Any] = {}
    if "diarization" in requested:
        values["turns"] = [
            {"turn_id": "turn_0", "speaker": None, "start": None, "end": None}
        ]
    if "vad" in requested:
        values["vad_regions"] = [{"start": None, "end": None}]
    if "lid" in requested:
        values["lid_regions"] = [
            {"start": None, "end": None, "language": None, "confidence": None}
        ]
    if "overlapped_speech" in requested:
        values["overlapped_speech"] = [
            {"overlap_id": "overlap_0", "start": None, "end": None}
        ]

    abstentions = []
    if abstention_reason is not None:
        abstentions.append({
            "abstention_id": "ab_0",
            "reason": abstention_reason,
            "start": None,
            "end": None,
        })

    placeholder = result.NormalizedResult(
        source=source,
        segments=[segment],
        abstentions=abstentions,
        provenance={"stack": stack, "outcomes": {}, "observed": {}, "plan": dict(plan)},
        requested_capabilities=requested,
        sample=True,
        note=SAMPLE_NOTE,
        **values,
    )
    return result.serialize_result(placeholder)
