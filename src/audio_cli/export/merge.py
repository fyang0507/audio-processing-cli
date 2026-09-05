"""Compatibility and source-timeline merging for normalized results."""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.media import resolve_path_identity

from .errors import IncompatibleResultsError
from .models import (
    _VIBEVOICE_MULTI_INPUT_DIARIZATION_FIX,
    _VIBEVOICE_MULTI_INPUT_DIARIZATION_REASON,
    LoadedResult,
    MergedTranscript,
)


def _plan_signature(plan: Mapping[str, Any]) -> dict[str, Any]:
    signature = copy.deepcopy(dict(plan))
    execution = signature.get("execution")
    if isinstance(execution, dict):
        execution.pop("range", None)
    return signature


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _span_inside(
    start: float,
    end: float,
    intervals: Sequence[tuple[float, float]],
) -> bool:
    return any(
        interval_start <= start and end <= interval_end
        for interval_start, interval_end in intervals
    )


def _validate_document_ownership(document: LoadedResult) -> None:
    if not document.owned_intervals:
        if document.payload["segments"]:
            raise ValueError("a document with no covered interval contains segments")
        return
    for segment_index, segment in enumerate(document.payload["segments"]):
        spans: list[tuple[float, float]] = []
        words = segment.get("words")
        if isinstance(words, (list, tuple)) and words:
            spans.append((float(words[0]["start"]), float(words[-1]["end"])))
        if "start" in segment and "end" in segment:
            spans.append((float(segment["start"]), float(segment["end"])))
        for start, end in spans:
            if not _span_inside(start, end, document.owned_intervals):
                raise ValueError(
                    f"segments[{segment_index}] lies outside the document's owned intervals"
                )


def merge_documents(documents: Sequence[LoadedResult]) -> MergedTranscript:
    """Concatenate compatible documents in caller-supplied source order and re-id them."""
    docs = tuple(documents)
    if not docs:
        raise IncompatibleResultsError((), "at least one input is required")
    paths = tuple(document.path for document in docs)
    try:
        # Inputs were opened literally by ``load_result_document``.  Keep that exact
        # path semantics here: expanding a leading ``~name`` after the fact could
        # either crash for an unknown account or compare a different file identity.
        resolved_paths = [resolve_path_identity(path) for path in paths]
    except (OSError, RuntimeError) as exc:
        raise IncompatibleResultsError(
            paths, f"input path identity could not be resolved: {exc}"
        ) from exc
    same_file = any(
        _same_existing_file(left, right)
        for index, left in enumerate(paths)
        for right in paths[index + 1 :]
    )
    captured_identities = [
        (identity.device, identity.inode)
        for document in docs
        if (identity := document.file_identity) is not None
    ]
    same_captured_file = len(captured_identities) != len(set(captured_identities))
    if len(set(resolved_paths)) != len(resolved_paths) or same_file or same_captured_file:
        raise IncompatibleResultsError(paths, "the same input document was supplied twice")

    reference = docs[0]
    try:
        if (
            len(docs) > 1
            and reference.payload["provenance"]["stack"] == "vibevoice"
            and "diarization" in reference.requested_capabilities
        ):
            raise IncompatibleResultsError(
                paths,
                _VIBEVOICE_MULTI_INPUT_DIARIZATION_REASON,
                fix=_VIBEVOICE_MULTI_INPUT_DIARIZATION_FIX,
            )
        reference_plan = json.dumps(
            _plan_signature(reference.payload["provenance"]["plan"]),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        previous_interval_end = -math.inf
        for document_index, document in enumerate(docs):
            if document.payload["source"] != reference.payload["source"]:
                raise ValueError("source identity, duration, or timebase differs")
            if document.payload["provenance"]["stack"] != reference.payload["provenance"]["stack"]:
                raise ValueError("provenance.stack differs")
            if document.requested_capabilities != reference.requested_capabilities:
                raise ValueError("requested capability sets differ")
            document_plan = json.dumps(
                _plan_signature(document.payload["provenance"]["plan"]),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if document_plan != reference_plan:
                raise ValueError("executed plans differ beyond their selected ranges")
            _validate_document_ownership(document)
            for interval_index, (start, end) in enumerate(document.owned_intervals):
                if start < previous_interval_end:
                    raise ValueError(
                        f"input {document_index} owned interval {interval_index} overlaps "
                        "or precedes an earlier input"
                    )
                previous_interval_end = end
    except IncompatibleResultsError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise IncompatibleResultsError(paths, str(exc)) from exc

    segments: list[dict[str, Any]] = []
    word_index = 0
    for document in docs:
        for segment in document.payload["segments"]:
            merged = copy.deepcopy(dict(segment))
            merged["segment_id"] = f"seg_{len(segments)}"
            if "words" in merged:
                for word in merged["words"]:
                    word["word_id"] = f"w_{word_index}"
                    word_index += 1
            segments.append(merged)
    return MergedTranscript(
        inputs=paths,
        source=copy.deepcopy(dict(reference.payload["source"])),
        segments=tuple(segments),
        documents=docs,
        requested_capabilities=reference.requested_capabilities,
    )
