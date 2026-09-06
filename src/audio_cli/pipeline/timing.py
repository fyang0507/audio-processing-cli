"""Enhancement's duration-only publication policy and explicit alignment abstention."""

from __future__ import annotations

from fractions import Fraction

DURATION_TOLERANCE_MS = 50


def timeline_verification(*, checked: bool = False) -> dict[str, object]:
    return {
        "status": "pass" if checked else "not_run",
        "scope": "decoded_audio_duration_only",
        "tolerance_ms": DURATION_TOLERANCE_MS,
        "content_alignment": {"status": "abstained", "reason": "not_measured"},
        "av_sync": {"status": "abstained", "reason": "not_measured"},
    }


def duration_check(
    source_samples: int, source_rate: int, output_samples: int, output_rate: int
) -> tuple[bool, float]:
    """Compare sample durations before rounding, including exactly 50 ms."""
    delta_ms = 1000 * (
        Fraction(output_samples, output_rate) - Fraction(source_samples, source_rate)
    )
    return abs(delta_ms) <= DURATION_TOLERANCE_MS, float(delta_ms)
