"""Stable facade for normalized transcription results and serialization."""

from .serialization import serialize_result
from .types import (
    ABSENT,
    ABSTENTION_REASONS,
    CAPABILITY_NAMES,
    SCHEMA_VERSION,
    JsonMapping,
    NormalizedResult,
    OptionalArray,
    OptionalMapping,
    ResultError,
)

__all__ = [
    "ABSENT",
    "ABSTENTION_REASONS",
    "CAPABILITY_NAMES",
    "SCHEMA_VERSION",
    "JsonMapping",
    "NormalizedResult",
    "OptionalArray",
    "OptionalMapping",
    "ResultError",
    "serialize_result",
]
