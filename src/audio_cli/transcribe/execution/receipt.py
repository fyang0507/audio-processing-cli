"""Opt-in command presentation for published transcription results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..refusals import request as refusals


def validate_receipt_options(output: Path | None, output_format: str) -> None:
    if output is None or output_format != "json":
        raise refusals.receipt_options_invalid(output, output_format)


def build_receipt(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Project saved facts without changing the result or supplying absent measurements."""
    counts = {"segments": len(payload["segments"]), "abstentions": len(payload["abstentions"])}
    words = [segment["words"] for segment in payload["segments"] if "words" in segment]
    if words:
        counts["words"] = sum(len(stream) for stream in words)
    for field in ("turns", "vad_regions", "lid_regions", "overlapped_speech"):
        if field in payload:
            counts[field] = len(payload[field])
    receipt = {
        "output": str(path),
        "source": dict(payload["source"]),
        "stack": payload["provenance"]["stack"],
        "complete": payload["complete"],
        "counts": counts,
    }
    if "coverage" in payload:
        receipt["coverage"] = dict(payload["coverage"])
    return receipt
