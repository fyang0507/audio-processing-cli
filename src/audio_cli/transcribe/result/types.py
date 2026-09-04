"""Types and closed vocabularies for normalized transcription results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SCHEMA_VERSION = 1

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

ABSTENTION_REASONS = frozenset(
    {
        "alignment_unavailable",
        "overlap",
        "short_turn",
        "raw_fragment",
    }
)

_ARRAY_CAPABILITIES = {
    "diarization": "turns",
    "vad": "vad_regions",
    "lid": "lid_regions",
    "overlapped_speech": "overlapped_speech",
}

_SEGMENT_CAPABILITIES = {
    "speaker": "diarization",
    "words": "word_timestamps",
    "start": "segment_timestamps",
    "end": "segment_timestamps",
}

# FireRed's recorded native artifacts contain six sentence/word edges where the
# millisecond-quantized final word ends exactly 1 ms after the sentence. Treat that
# measured quantization seam as internally consistent, without scaling tolerance for
# long source timelines.
_SEGMENT_WORD_EDGE_TOLERANCE_SECONDS = 0.001 + 1e-9


class ResultError(ValueError):
    """A normalized result violates the schema or its resolved capability set."""


class _Absent:
    __slots__ = ()

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT = _Absent()

JsonMapping = Mapping[str, Any]
OptionalArray = Sequence[JsonMapping] | _Absent
OptionalMapping = JsonMapping | _Absent


@dataclass(frozen=True)
class NormalizedResult:
    """Normalized values plus the capabilities that license their optional keys."""

    source: JsonMapping
    segments: Sequence[JsonMapping]
    abstentions: Sequence[JsonMapping]
    provenance: JsonMapping
    requested_capabilities: frozenset[str]
    complete: bool = True
    coverage: OptionalMapping = ABSENT
    turns: OptionalArray = ABSENT
    vad_regions: OptionalArray = ABSENT
    lid_regions: OptionalArray = ABSENT
    overlapped_speech: OptionalArray = ABSENT
    sample: bool = False
    note: str | None = None
