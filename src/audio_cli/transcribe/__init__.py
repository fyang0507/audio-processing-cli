"""Pure transcription planning and result primitives.

Backends and command wiring land in later phases.  This package starts with the durable result
shape so every later adapter has one normalization boundary to target.
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
