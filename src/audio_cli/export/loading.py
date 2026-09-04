"""Strict normalized-result loading and document ownership derivation."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.media import file_identity_from_descriptor
from audio_cli.transcribe.result import ABSENT, NormalizedResult, serialize_result

from .errors import InvalidResultError
from .merge import _plan_signature, _validate_document_ownership
from .models import LoadedResult, _reject_duplicate_json_keys


def _as_normalized(payload: Mapping[str, Any]) -> NormalizedResult:
    provenance = payload["provenance"]
    outcomes = provenance["outcomes"]
    requested = frozenset(outcomes)
    return NormalizedResult(
        source=payload["source"],
        segments=payload["segments"],
        abstentions=payload["abstentions"],
        provenance=provenance,
        requested_capabilities=requested,
        complete=payload["complete"],
        coverage=payload.get("coverage", ABSENT),
        turns=payload.get("turns", ABSENT),
        vad_regions=payload.get("vad_regions", ABSENT),
        lid_regions=payload.get("lid_regions", ABSENT),
        overlapped_speech=payload.get("overlapped_speech", ABSENT),
        sample=payload.get("sample", False),
        note=payload.get("note"),
    )


def _unique(values: Sequence[Mapping[str, Any]], key: str, field: str) -> None:
    seen: set[str] = set()
    for index, item in enumerate(values):
        identifier = item[key]
        if identifier in seen:
            raise ValueError(f"{field}[{index}].{key} duplicates {identifier!r}")
        seen.add(identifier)


def _segment_start(segment: Mapping[str, Any]) -> float | None:
    if "start" in segment:
        return float(segment["start"])
    words = segment.get("words")
    if isinstance(words, (list, tuple)) and words:
        return float(words[0]["start"])
    return None


def _validate_export_order(payload: Mapping[str, Any]) -> None:
    segments = payload["segments"]
    _unique(segments, "segment_id", "segments")
    word_ids: set[str] = set()
    previous_known_start = -math.inf
    for segment_index, segment in enumerate(segments):
        start = _segment_start(segment)
        if start is not None:
            if start < previous_known_start:
                raise ValueError("segments are not in source-timeline order")
            previous_known_start = start
        words = segment.get("words", ())
        previous_end = -math.inf
        for word_index, word in enumerate(words):
            identifier = word["word_id"]
            if identifier in word_ids:
                raise ValueError(
                    f"segments[{segment_index}].words[{word_index}].word_id "
                    f"duplicates {identifier!r}"
                )
            word_ids.add(identifier)
            start_value = float(word["start"])
            end_value = float(word["end"])
            if end_value < start_value:
                raise ValueError(
                    f"segments[{segment_index}].words[{word_index}] must have "
                    "non-negative duration"
                )
            if start_value < previous_end:
                raise ValueError(
                    f"segments[{segment_index}].words[{word_index}] overlaps the "
                    "preceding word"
                )
            previous_end = end_value
    _unique(payload["abstentions"], "abstention_id", "abstentions")
    if "turns" in payload:
        _unique(payload["turns"], "turn_id", "turns")
    if "overlapped_speech" in payload:
        _unique(payload["overlapped_speech"], "overlap_id", "overlapped_speech")


def _owned_intervals(payload: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    range_owned: tuple[tuple[float, float], ...] | None = None
    execution = payload["provenance"]["plan"].get("execution", {})
    if isinstance(execution, Mapping) and "range" in execution:
        run_range = execution["range"]
        if not isinstance(run_range, Mapping):
            raise ValueError("provenance.plan.execution.range must be an object")
        requested = run_range.get("requested")
        selected = run_range.get("selected_unit_scope")
        if not (
            isinstance(requested, (list, tuple)) and len(requested) == 2
            and isinstance(selected, (list, tuple)) and len(selected) == 2
        ):
            raise ValueError(
                "provenance.plan.execution.range must carry requested and "
                "selected_unit_scope pairs"
            )

        def range_number(value: object, field: str) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"provenance.plan.execution.range.{field} must contain numbers"
                )
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(
                    f"provenance.plan.execution.range.{field} must contain finite numbers"
                )
            return parsed

        requested_pair = (
            range_number(requested[0], "requested"),
            range_number(requested[1], "requested"),
        )
        selected_pair = (
            range_number(selected[0], "selected_unit_scope"),
            range_number(selected[1], "selected_unit_scope"),
        )
        duration = float(payload["source"]["duration_seconds"])
        for field, (start, end) in (
            ("requested", requested_pair),
            ("selected_unit_scope", selected_pair),
        ):
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and 0 <= start < end <= duration
            ):
                raise ValueError(
                    f"provenance.plan.execution.range.{field} is not a source interval"
                )
        if max(requested_pair[0], selected_pair[0]) >= min(
            requested_pair[1], selected_pair[1]
        ):
            raise ValueError(
                "requested range and selected unit scope do not intersect"
            )
        # A ranged document owns the processing units it actually selected.
        # The requested interval can begin before the first selectable unit (or
        # between PCM samples); unioning it back in creates overlap when a resume
        # begins at that same logical boundary.
        range_owned = (selected_pair,)
    if not payload["complete"]:
        coverage_owned = tuple(
            (float(start), float(end))
            for start, end in payload["coverage"]["covered_intervals"]
        )
        if range_owned is not None:
            coverage_scope = tuple(
                (float(start), float(end))
                for start, end in payload["coverage"]["scope_intervals"]
            )
            if coverage_scope != range_owned:
                raise ValueError(
                    "incomplete coverage scope must equal the selected unit scope"
                )
        return coverage_owned
    if range_owned is not None:
        return range_owned
    duration = float(payload["source"]["duration_seconds"])
    return () if duration == 0 else ((0.0, duration),)


def load_result_document(path: Path) -> LoadedResult:
    """Load one exact v1 normalized result; samples and schema drift fail closed."""
    input_path = Path(path)
    try:
        descriptor = os.open(
            input_path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            file_identity = file_identity_from_descriptor(handle.fileno(), input_path)
            if file_identity is None:
                raise ValueError("input must be a regular file")
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
        if not isinstance(payload, Mapping):
            raise ValueError("top level must be an object")
        if payload.get("sample") is True:
            raise ValueError("sample output is not an exportable run result")
        normalized = _as_normalized(payload)
        canonical = serialize_result(normalized)
        # ``json.loads`` accepts an escaped lone surrogate, but it cannot be emitted by
        # the UTF-8/no-BOM writers promised by export.  Reject it at the input boundary.
        json.dumps(canonical, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if canonical != payload:
            raise ValueError(
                "document is not the exact current normalized result shape"
            )
        _validate_export_order(payload)
        owned = _owned_intervals(payload)
        for index, (start, end) in enumerate(owned):
            if end <= start:
                raise ValueError(f"owned interval {index} is empty or reversed")
        # Merge compares plans after removing only their selected range.  Exercise that
        # exact copy operation at the guarded input boundary so a deeply nested but
        # otherwise JSON-decodable plan cannot leak ``RecursionError`` during export.
        _plan_signature(payload["provenance"]["plan"])
        loaded = LoadedResult(
            path=input_path,
            payload=dict(payload),
            requested_capabilities=normalized.requested_capabilities,
            owned_intervals=owned,
            file_identity=file_identity,
        )
        _validate_document_ownership(loaded)
        return loaded
    except InvalidResultError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise InvalidResultError(input_path, str(exc)) from exc
