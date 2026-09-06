"""Probed timestamp facts and decoded sample-count evidence, without sync inference."""

from __future__ import annotations

import math
from fractions import Fraction


def _number(value: object, *, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (nonnegative and number < 0):
        return None
    return round(number, 6)


def probed_duration(probe: dict[str, object]) -> dict[str, object]:
    """Prefer primary audio duration; omit an unavailable measurement and its basis."""
    for item, basis in (
        (probe.get("primary_audio_stream"), "probed_audio_stream"),
        (probe.get("format"), "probed_container"),
    ):
        if isinstance(item, dict):
            duration = _number(item.get("duration"), nonnegative=True)
            if duration is not None:
                return {"duration_seconds": duration, "duration_basis": basis}
    return {}


def _stream_timing(stream: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for raw, public in (("start_time", "start_seconds"), ("duration", "duration_seconds")):
        value = _number(stream.get(raw), nonnegative=raw == "duration")
        if value is not None:
            result[public] = value
    for key in ("index", "start_pts", "duration_ts", "initial_padding", "trailing_padding"):
        value = stream.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and (key == "start_pts" or value >= 0)
        ):
            result["stream_index" if key == "index" else key] = value
    time_base = stream.get("time_base")
    if isinstance(time_base, str):
        try:
            valid = Fraction(time_base) > 0
        except (ValueError, ZeroDivisionError):
            valid = False
        if valid:
            result["time_base"] = time_base
    return result


def stream_timing(probe: dict[str, object]) -> dict[str, object]:
    """Report all available A/V stream origins; relative starts are metadata, not content."""
    fmt = probe.get("format", {})
    result: dict[str, object] = {"basis": "probed_timestamps"}
    if isinstance(fmt, dict):
        container = _stream_timing(fmt)
        if container:
            result["container"] = container
    primary = probe.get("primary_audio_stream", {})
    audio_start = _number(primary.get("start_time")) if isinstance(primary, dict) else None
    for kind in ("audio", "video"):
        rows = []
        for stream in probe.get("streams", []):
            if not isinstance(stream, dict) or stream.get("codec_type") != kind:
                continue
            row = _stream_timing(stream)
            if kind == "video" and audio_start is not None and "start_seconds" in row:
                row["primary_audio_start_minus_video_start_seconds"] = round(
                    audio_start - row["start_seconds"], 6
                )
            rows.append(row)
        if rows:
            result[kind] = rows
    return result


def decoded_audio_timing(sample_count: int, sample_rate: int) -> dict[str, object]:
    """Describe actual decoded frames; zero is the first decoded sample, not container PTS."""
    return {
        "duration_seconds": round(sample_count / sample_rate, 6),
        "duration_basis": "decoded_pcm",
        "sample_count": sample_count,
        "sample_rate_hz": sample_rate,
        "time_origin": "first_decoded_sample",
    }
