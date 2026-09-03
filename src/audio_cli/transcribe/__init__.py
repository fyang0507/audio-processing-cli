"""Pure transcription planning and normalized result primitives.

The commands in this phase stop after metadata probing and plan serialization. Backends and
execution land later; every adapter will target the durable normalization boundary here.
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
