"""Recorded source identity and decoded-duration compatibility, without opening media."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from .loading import mapping
from .metrics import number, pointed


def digest(metadata: dict[str, Any], pointer: str) -> str:
    value = metadata.get("sha256")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{pointer}/sha256 must be a recorded SHA256")
    return value.lower()


def duration(metadata: dict[str, Any], pointer: str) -> Fraction:
    decoded = mapping(metadata.get("decoded_audio"), f"{pointer}/decoded_audio")
    if (
        decoded.get("time_origin") != "first_decoded_sample"
        or decoded.get("duration_basis") != "decoded_pcm"
    ):
        raise ValueError(
            f"{pointer}/decoded_audio requires the first decoded sample origin and decoded_pcm basis"
        )
    count, rate = decoded.get("sample_count"), decoded.get("sample_rate_hz")
    if type(count) is not int or type(rate) is not int or count < 0 or rate <= 0:
        raise ValueError(
            f"{pointer}/decoded_audio requires nonnegative sample_count and positive sample_rate_hz"
        )
    result = Fraction(count, rate)
    if "duration_seconds" in decoded:
        seconds = number(decoded["duration_seconds"], f"{pointer}/decoded_audio/duration_seconds")
        if abs(result - Fraction(str(seconds))) > Fraction(1, 1_000_000):
            raise ValueError(
                f"{pointer}/decoded_audio duration contradicts sample_count/sample_rate_hz"
            )
    return result


def identity(side: str, report: dict[str, Any], key: str) -> dict[str, Any]:
    return {"side": side, **pointed(report[key], f"/{key}")}


def compatibility(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    sources = [report["source"] for report in (left, right)]
    hashes = [digest(source, "/source") for source in sources]
    durations = [duration(source, "/source") for source in sources]
    for report in (left, right):
        basis = report.get("region_basis", {})
        if basis.get("timeline", "source") != "source":
            raise ValueError("region_basis/timeline is incompatible with the source timeline")
    if hashes[0] == hashes[1]:
        if durations[0] != durations[1]:
            raise ValueError("same source SHA256 has conflicting decoded durations")
        return {
            "basis": "same_source_sha256",
            "identities": [identity("left", left, "source"), identity("right", right, "source")],
            "timeline_scope": "first_decoded_sample",
        }
    for side, other_side, enhanced, other in (
        ("left", "right", left, right),
        ("right", "left", right, left),
    ):
        if enhanced["kind"] != "audio_enhancement_report" or "output" not in enhanced:
            continue
        output = mapping(enhanced["output"], "/output")
        if digest(output, "/output") != digest(other["source"], "/source"):
            continue
        if (
            not enhanced["rendered"]
            or enhanced["dry_run"]
            or enhanced.get("timeline_preserved") is not True
        ):
            raise ValueError(
                "linked output requires a rendered report with timeline_preserved true"
            )
        verification = mapping(enhanced.get("timeline_verification"), "/timeline_verification")
        if (
            verification.get("status") != "pass"
            or verification.get("scope") != "decoded_audio_duration_only"
        ):
            raise ValueError("linked output requires a recorded decoded-duration verification pass")
        tolerance = number(verification.get("tolerance_ms"), "/timeline_verification/tolerance_ms")
        if tolerance < 0:
            raise ValueError("recorded duration tolerance must be nonnegative")
        output_duration = duration(output, "/output")
        if output_duration != duration(other["source"], "/source"):
            raise ValueError("linked output has conflicting decoded durations")
        delta = 1000 * (output_duration - duration(enhanced["source"], "/source"))
        if abs(delta) > Fraction(str(tolerance)):
            raise ValueError("linked output contradicts its recorded duration verification")
        return {
            "basis": "recorded_output_sha256",
            "identities": [
                identity(side, enhanced, "output"),
                identity(other_side, other, "source"),
            ],
            "timeline_scope": "decoded_audio_duration_only",
            "timeline_verification": {
                "side": side,
                **pointed(verification, "/timeline_verification"),
            },
        }
    raise ValueError(
        "incompatible source SHA256 values; no recorded output identity links these reports"
    )
