"""Transcription planning, execution, and normalized result primitives.

Provider-specific orchestration composes isolated model stages through the shared transport and
publishes every backend through the durable normalization boundary exposed here.
"""

from .result import (
    ABSENT,
    ABSTENTION_REASONS,
    CAPABILITY_NAMES,
    NormalizedResult,
    ResultError,
    serialize_result,
)
from .sample import build_sample_output

__all__ = [
    "ABSENT",
    "ABSTENTION_REASONS",
    "CAPABILITY_NAMES",
    "NormalizedResult",
    "ResultError",
    "build_sample_output",
    "serialize_result",
]
