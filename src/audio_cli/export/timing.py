"""Word-timing ledger validation for timed transcript exports."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import InvalidResultError, TimingRequiredError
from .models import LoadedResult, MergedTranscript


def _found_timing(document: LoadedResult) -> tuple[str, ...]:
    found = []
    outcomes = document.payload["provenance"]["outcomes"]
    for capability in ("word_timestamps", "segment_timestamps"):
        if outcomes.get(capability) == "produced":
            found.append(capability)
    return tuple(found)


def _raise_timing_required(document: LoadedResult) -> None:
    payload = document.payload
    raise TimingRequiredError(
        document.path,
        found=_found_timing(document),
        source_path=Path(payload["source"]["path"]),
        stack=payload["provenance"]["stack"],
        wants=tuple(payload["provenance"]["outcomes"]),
        plan=payload["provenance"]["plan"],
        word_timing_outcome=payload["provenance"]["outcomes"].get(
            "word_timestamps"
        ),
    )


def _is_bounded_event(segment: Mapping[str, Any]) -> bool:
    text = segment.get("text")
    stripped = text.strip() if isinstance(text, str) else ""
    return (
        "words" not in segment
        and "speaker" not in segment
        and "start" in segment
        and "end" in segment
        and len(stripped) >= 2
        and stripped.startswith("[")
        and stripped.endswith("]")
        and bool(stripped[1:-1].strip())
        and float(segment["end"]) > float(segment["start"])
    )


def _has_lexical_text(text: str) -> bool:
    return any(
        not character.isspace()
        and not unicodedata.category(character).startswith("P")
        for character in text
    )


def _is_ordinary_wordless(segment: Mapping[str, Any]) -> bool:
    words = segment.get("words")
    if isinstance(words, (list, tuple)) and words:
        return False
    if _is_bounded_event(segment):
        return False
    text = str(segment.get("text", ""))
    # An explicit empty stream is valid for punctuation-only text: there is no
    # lexical token for the aligner to time.
    return "words" not in segment or _has_lexical_text(text)


def _alignment_abstention_bounds(document: LoadedResult) -> set[tuple[float, float]]:
    return {
        (float(item["start"]), float(item["end"]))
        for item in document.payload["abstentions"]
        if item.get("reason") == "alignment_unavailable"
    }


def _validate_word_timing_ledger(document: LoadedResult) -> None:
    """Require an explicit abstention for every ordinary wordless timing request."""

    outcome = document.payload["provenance"]["outcomes"].get("word_timestamps")
    if outcome not in {"produced", "abstained"}:
        return
    ordinary_wordless = [
        segment for segment in document.payload["segments"]
        if _is_ordinary_wordless(segment)
    ]
    if not ordinary_wordless:
        return
    alignment_bounds = _alignment_abstention_bounds(document)
    if outcome != "abstained" or not alignment_bounds:
        raise InvalidResultError(
            document.path,
            "ordinary speech without words requires an alignment_unavailable "
            "abstention and an abstained word_timestamps outcome",
        )
    for segment in ordinary_wordless:
        if "start" not in segment or "end" not in segment:
            continue
        bounds = (float(segment["start"]), float(segment["end"]))
        if bounds not in alignment_bounds:
            raise InvalidResultError(
                document.path,
                "bounded ordinary speech without words requires a same-bounds "
                "alignment_unavailable abstention",
            )


def _require_word_timing(merged: MergedTranscript) -> None:
    has_real_word_stream = False
    timing_produced_by: LoadedResult | None = None
    for document in merged.documents:
        outcome = document.payload["provenance"]["outcomes"].get(
            "word_timestamps"
        )
        if outcome == "produced" and timing_produced_by is None:
            timing_produced_by = document
        for segment in document.payload["segments"]:
            if segment.get("words"):
                has_real_word_stream = True
    # A bounded segment can be omitted only when the document explicitly binds
    # its failed alignment to those same bounds. Qwen's public segments have no
    # segment bounds, and v1 carries no segment-to-unit association; even a real
    # unit-level abstention elsewhere cannot prove which unbounded text it owns.
    for document in merged.documents:
        outcome = document.payload["provenance"]["outcomes"].get(
            "word_timestamps"
        )
        alignment_bounds = _alignment_abstention_bounds(document)
        for segment in document.payload["segments"]:
            if not _is_ordinary_wordless(segment):
                continue
            if "start" not in segment or "end" not in segment:
                _raise_timing_required(document)
            bounds = (float(segment["start"]), float(segment["end"]))
            if outcome != "abstained" or bounds not in alignment_bounds:
                _raise_timing_required(document)
    if has_real_word_stream:
        return
    all_segments = [
        segment
        for document in merged.documents
        for segment in document.payload["segments"]
    ]
    if timing_produced_by is not None and all_segments and all(
        _is_bounded_event(segment) for segment in all_segments
    ):
        return
    # A produced-timing event document cannot repair ordinary speech in another
    # range.  Prefer the ordinary document whose timing was not produced so the
    # refusal can preserve its own plan/range in a runnable rerun command.
    repairable = next((
        document
        for document in merged.documents
        if document.payload["provenance"]["outcomes"].get("word_timestamps")
        != "produced"
        and any(
            not _is_bounded_event(segment)
            for segment in document.payload["segments"]
        )
    ), None)
    _raise_timing_required(repairable or timing_produced_by or merged.documents[0])
