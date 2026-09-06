"""Validate durable alignment rejection and boundary-correction evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from .types import (
    ABSENT,
    ABSTENTION_REASONS,
    ALIGNMENT_CODES,
    JsonMapping,
    NormalizedResult,
    ResultError,
)
from .validation import _exact_keys, _number, _validate_bounds

_BOUNDARY_KEYS = {
    "original_bounds",
    "unit_bounds",
    "start_overrun_ms",
    "end_overrun_ms",
    "max_overrun_ms",
}
_CORRECTION_KEYS = _BOUNDARY_KEYS | {
    "applied_bounds",
    "word_index",
    "unit_id",
    "segment_id",
    "word_id",
}


def _pair(value: object, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ResultError(f"{field} must be a pair of finite bounds")
    start = _number(value[0], field)
    end = _number(value[1], field)
    assert start is not None and end is not None
    if end < start:
        raise ResultError(f"{field} must not be reversed")
    return start, end


def _configured_limit(result: NormalizedResult) -> float:
    plan = result.provenance["plan"]
    try:
        limit = plan["roles"]["aligner"]["config"]["max_overrun_ms"]
    except (KeyError, TypeError) as exc:
        raise ResultError(
            "alignment boundary evidence requires the executed max_overrun_ms"
        ) from exc
    value = _number(limit, "plan.roles.aligner.config.max_overrun_ms")
    assert value is not None
    if value < 0.501:
        raise ResultError("alignment max_overrun_ms must be at least 0.501")
    return value


def _boundary(
    item: Mapping[str, Any], *, field: str, duration: float, limit: float
) -> tuple[tuple[float, float], tuple[float, float], float]:
    original = _pair(item["original_bounds"], f"{field}.original_bounds")
    unit = _pair(item["unit_bounds"], f"{field}.unit_bounds")
    if not 0 <= unit[0] < unit[1] <= duration:
        raise ResultError(f"{field}.unit_bounds must be a positive source interval")
    if original[0] >= unit[1] or original[1] <= unit[0]:
        raise ResultError(f"{field}.original_bounds must overlap the input unit")
    expected = (
        max(Decimal(0), Decimal(str(unit[0])) - Decimal(str(original[0]))) * 1000,
        max(Decimal(0), Decimal(str(original[1])) - Decimal(str(unit[1]))) * 1000,
    )
    for key, actual in zip(("start_overrun_ms", "end_overrun_ms"), expected, strict=True):
        stated = _number(item[key], f"{field}.{key}")
        assert stated is not None
        if stated < 0 or not math.isclose(stated, float(actual), rel_tol=0, abs_tol=1e-9):
            raise ResultError(f"{field}.{key} disagrees with recorded bounds")
    stated_limit = _number(item["max_overrun_ms"], f"{field}.max_overrun_ms")
    if stated_limit != limit:
        raise ResultError(f"{field}.max_overrun_ms disagrees with the executed plan")
    return original, unit, float(max(expected))


def _unit_scope(
    result: NormalizedResult, unit: tuple[float, float], field: str, duration: float
) -> None:
    scopes = [(0.0, duration)]
    if result.coverage is not ABSENT:
        scopes = result.coverage["covered_intervals"]
    execution = result.provenance["plan"].get("execution", {})
    if isinstance(execution, Mapping) and "range" in execution:
        selected = execution["range"]
        if not isinstance(selected, Mapping) or "selected_unit_scope" not in selected:
            raise ResultError(f"{field} requires a valid selected range scope")
        selected_bounds = _pair(selected["selected_unit_scope"], f"{field}.selected_unit_scope")
        if not selected_bounds[0] <= unit[0] < unit[1] <= selected_bounds[1]:
            raise ResultError(f"{field}.unit_bounds lies outside the selected scope")
    if not any(start <= unit[0] < unit[1] <= end for start, end in scopes):
        raise ResultError(f"{field}.unit_bounds lies outside the covered scope")


def validate_alignment_evidence(result: NormalizedResult, *, duration: float) -> None:
    """Check arithmetic, applied bounds and links without rerunning an aligner."""
    rejected = [item for item in result.abstentions if "boundary" in item.get("alignment", {})]
    observed = result.provenance["observed"]
    has_corrections = "alignment_corrections" in observed
    if not rejected and not has_corrections:
        return
    if result.sample or "word_timestamps" not in result.requested_capabilities:
        raise ResultError("alignment boundary evidence requires a real word-timing request")
    limit = _configured_limit(result)
    unit_scopes: dict[str, tuple[float, float]] = {}
    rejected_units: set[str] = set()
    for abstention in result.abstentions:
        alignment = abstention.get("alignment")
        if alignment is None:
            continue
        unit_id = alignment["unit_id"]
        unit = (float(abstention["start"]), float(abstention["end"]))
        if unit_id in unit_scopes and unit_scopes[unit_id] != unit:
            raise ResultError("alignment evidence gives one unit conflicting bounds")
        unit_scopes[unit_id] = unit
        rejected_units.add(unit_id)
    for abstention in rejected:
        field = f"abstention {abstention['abstention_id']}.alignment.boundary"
        boundary = abstention["alignment"]["boundary"]
        if not isinstance(boundary, Mapping):
            raise ResultError(f"{field} must be an object")
        _exact_keys(boundary, _BOUNDARY_KEYS, field)
        _, unit, overrun = _boundary(boundary, field=field, duration=duration, limit=limit)
        if unit != (float(abstention["start"]), float(abstention["end"])):
            raise ResultError(f"{field}.unit_bounds must match the abstention scope")
        if overrun <= limit:
            raise ResultError(f"{field} does not exceed the configured limit")
        _unit_scope(result, unit, field, duration)
    if not has_corrections:
        return
    corrections = observed["alignment_corrections"]
    if not isinstance(corrections, (list, tuple)) or not corrections:
        raise ResultError("alignment_corrections must be a nonempty array when present")
    by_segment = {item["segment_id"]: item for item in result.segments}
    by_word = {
        word["word_id"]: (segment, word)
        for segment in result.segments
        for word in segment.get("words", ())
    }
    total_words = sum(len(segment.get("words", ())) for segment in result.segments)
    if len(by_segment) != len(result.segments) or len(by_word) != total_words:
        raise ResultError("alignment_corrections require unique segment and word IDs")
    seen_words: set[str] = set()
    seen_indices: set[tuple[str, int]] = set()
    for index, correction in enumerate(corrections):
        field = f"alignment_corrections[{index}]"
        if not isinstance(correction, Mapping):
            raise ResultError(f"{field} must be an object")
        _exact_keys(correction, _CORRECTION_KEYS, field)
        for key in ("unit_id", "segment_id", "word_id"):
            if not isinstance(correction[key], str) or not correction[key]:
                raise ResultError(f"{field}.{key} must be a nonempty string")
        word_index = correction["word_index"]
        if isinstance(word_index, bool) or not isinstance(word_index, int) or word_index < 0:
            raise ResultError(f"{field}.word_index must be a nonnegative integer")
        if word_index >= total_words:
            raise ResultError(f"{field}.word_index exceeds the published word stream")
        identity = (correction["unit_id"], word_index)
        if correction["word_id"] in seen_words or identity in seen_indices:
            raise ResultError(f"{field} duplicates a corrected word")
        seen_words.add(correction["word_id"])
        seen_indices.add(identity)
        linked = by_word.get(correction["word_id"])
        if linked is None or linked[0]["segment_id"] != correction["segment_id"]:
            raise ResultError(f"{field} must identify a word in its supplied segment")
        segment, word = linked
        original, unit, overrun = _boundary(correction, field=field, duration=duration, limit=limit)
        unit_id = correction["unit_id"]
        if unit_id in rejected_units:
            raise ResultError(f"{field} cannot correct a rejected unit")
        if unit_id in unit_scopes and unit_scopes[unit_id] != unit:
            raise ResultError(f"{field} gives one unit conflicting bounds")
        unit_scopes[unit_id] = unit
        _unit_scope(result, unit, field, duration)
        if not 0.501 < overrun <= limit:
            raise ResultError(f"{field} must record an allowed correction beyond serialization")
        applied = _pair(correction["applied_bounds"], f"{field}.applied_bounds")
        if applied != (max(original[0], unit[0]), min(original[1], unit[1])):
            raise ResultError(f"{field}.applied_bounds must clip only the outlying endpoints")
        if applied[1] <= applied[0] or applied != (float(word["start"]), float(word["end"])):
            raise ResultError(f"{field}.applied_bounds must match a positive supplied word")
        if "start" in segment and unit != (float(segment["start"]), float(segment["end"])):
            raise ResultError(f"{field}.unit_bounds must match the native segment")


def _validate_abstentions(
    values: Sequence[JsonMapping],
    *,
    sample: bool,
    duration: float,
    segments: Sequence[JsonMapping],
) -> None:
    by_id = {item["segment_id"]: item for item in segments}
    if len(by_id) != len(segments) and any(
        isinstance(item, Mapping) and "alignment" in item for item in values
    ):
        raise ResultError("alignment evidence requires unique segment IDs")
    linked: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ResultError(f"abstentions[{index}] must be an object")
        name = f"abstentions[{index}]"
        expected = {"abstention_id", "reason", "start", "end"}
        if "alignment" in item:
            expected.add("alignment")
        _exact_keys(item, expected, name)
        _validate_bounds(item, name, sample=sample, duration=duration)
        if "alignment" in item:
            alignment = item["alignment"]
            if item.get("reason") != "alignment_unavailable" or not isinstance(alignment, Mapping):
                raise ResultError(f"{name}.alignment requires an alignment_unavailable object")
            keys = {"unit_id", "segment_ids", "code"}
            if "word_index" in alignment:
                keys.add("word_index")
                word_index = alignment["word_index"]
                if (
                    isinstance(word_index, bool)
                    or not isinstance(word_index, int)
                    or word_index < 0
                ):
                    raise ResultError(f"{name}.alignment.word_index must be a non-negative integer")
            if "boundary" in alignment:
                keys.add("boundary")
                if alignment.get("code") != "out_of_unit_bounds" or "word_index" not in alignment:
                    raise ResultError(
                        f"{name}.alignment.boundary requires a rejected word boundary"
                    )
            _exact_keys(alignment, keys, f"{name}.alignment")
            if not isinstance(alignment["unit_id"], str) or not alignment["unit_id"]:
                raise ResultError(f"{name}.alignment.unit_id must be a non-empty string")
            if not isinstance(alignment["code"], str) or alignment["code"] not in ALIGNMENT_CODES:
                raise ResultError(f"{name}.alignment.code is not a declared alignment code")
            if "word_index" in alignment and alignment["code"] in {
                "provider_unavailable",
                "text_mismatch",
                "sentence_reconciliation",
            }:
                raise ResultError(f"{name}.alignment.code cannot identify a returned word")
            identifiers = alignment["segment_ids"]
            if not isinstance(identifiers, list) or not identifiers:
                raise ResultError(f"{name}.alignment.segment_ids must be a non-empty array")
            for identifier in identifiers:
                if (
                    not isinstance(identifier, str)
                    or identifier not in by_id
                    or identifier in linked
                ):
                    raise ResultError(
                        f"{name}.alignment.segment_ids contains an unknown or repeated segment"
                    )
                if "words" in by_id[identifier]:
                    raise ResultError(f"{name}.alignment cannot reference a segment with words")
                segment = by_id[identifier]
                if "start" in segment and (
                    segment["start"] != item["start"] or segment["end"] != item["end"]
                ):
                    raise ResultError(f"{name}.alignment must match the native segment bounds")
                linked.add(identifier)
        if not isinstance(item["abstention_id"], str) or not item["abstention_id"]:
            raise ResultError(f"{name}.abstention_id must be a non-empty string")
        if item["reason"] not in ABSTENTION_REASONS:
            raise ResultError(f"{name}.reason must be one of {sorted(ABSTENTION_REASONS)}")
