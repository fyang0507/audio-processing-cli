"""Validation helpers for normalized transcription results."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

from .types import (
    _SEGMENT_CAPABILITIES,
    _SEGMENT_WORD_EDGE_TOLERANCE_SECONDS,
    ABSENT,
    ABSTENTION_REASONS,
    ALIGNMENT_CODES,
    CAPABILITY_NAMES,
    JsonMapping,
    NormalizedResult,
    OptionalArray,
    ResultError,
)


def _capabilities(values: Iterable[str]) -> frozenset[str]:
    found = frozenset(values)
    unknown = sorted(found - CAPABILITY_NAMES)
    if unknown:
        raise ResultError(f"unknown requested capabilities: {unknown}")
    if "token_lid" in found:
        raise ResultError("token_lid is unsupported by every stack and cannot appear in a result")
    return found


def _number(value: object, field: str, *, nullable: bool = False) -> float | None:
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultError(f"{field} must be a finite number")
    try:
        found = float(value)
    except OverflowError as exc:
        raise ResultError(f"{field} must be a finite number") from exc
    if not math.isfinite(found):
        raise ResultError(f"{field} must be a finite number")
    return found


def _exact_keys(item: JsonMapping, expected: set[str], field: str) -> None:
    actual = set(item)
    if actual != expected:
        raise ResultError(f"{field} has keys {sorted(actual)}, expected {sorted(expected)}")


def _validate_source(source: JsonMapping, *, sample: bool = False) -> float:
    expected = {"path", "duration_seconds", "timebase"}
    if "duration_basis" in source:
        expected.add("duration_basis")
        if source["duration_basis"] not in (
            "probed_audio_stream",
            "probed_container",
            "canonical_decoded_pcm",
        ):
            raise ResultError("source.duration_basis is not a declared duration basis")
        if not sample and source["duration_basis"] != "canonical_decoded_pcm":
            raise ResultError("run source.duration_basis must be canonical_decoded_pcm")
    _exact_keys(source, expected, "source")
    if not isinstance(source["path"], str) or not source["path"]:
        raise ResultError("source.path must be a non-empty string")
    duration = _number(source["duration_seconds"], "source.duration_seconds")
    assert duration is not None
    if duration < 0:
        raise ResultError("source.duration_seconds must not be negative")
    if source["timebase"] != "seconds":
        raise ResultError("source.timebase must be 'seconds'")
    return duration


def _validate_bounds(item: JsonMapping, field: str, *, sample: bool, duration: float) -> None:
    has_start = "start" in item
    has_end = "end" in item
    if has_start != has_end:
        raise ResultError(f"{field} must carry start and end together")
    if not has_start:
        return
    start = _number(item["start"], f"{field}.start", nullable=sample)
    end = _number(item["end"], f"{field}.end", nullable=sample)
    if sample:
        if start is not None or end is not None:
            raise ResultError(f"{field} placeholder bounds must be null")
        return
    assert start is not None and end is not None
    if start < 0 or end < start or end > duration:
        raise ResultError(f"{field} bounds must satisfy 0 <= start <= end <= source duration")


def _validate_words(words: object, field: str, *, sample: bool, duration: float) -> None:
    if not isinstance(words, (list, tuple)):
        raise ResultError(f"{field} must be an array")
    for index, word in enumerate(words):
        if not isinstance(word, Mapping):
            raise ResultError(f"{field}[{index}] must be an object")
        name = f"{field}[{index}]"
        _exact_keys(word, {"word_id", "text", "start", "end"}, name)
        if not isinstance(word["word_id"], str) or not word["word_id"]:
            raise ResultError(f"{name}.word_id must be a non-empty string")
        if sample:
            if word["text"] is not None:
                raise ResultError(f"{name}.text placeholder must be null")
        elif not isinstance(word["text"], str):
            raise ResultError(f"{name}.text must be a string")
        _validate_bounds(word, name, sample=sample, duration=duration)


def _validate_segments(
    segments: Sequence[JsonMapping],
    requested: frozenset[str],
    *,
    sample: bool,
    duration: float,
) -> None:
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ResultError(f"segments[{index}] must be an object")
        field = f"segments[{index}]"
        allowed = {"segment_id", "text"} | set(_SEGMENT_CAPABILITIES)
        extra = set(segment) - allowed
        if extra:
            raise ResultError(f"{field} carries unknown keys {sorted(extra)}")
        if not {"segment_id", "text"} <= set(segment):
            raise ResultError(f"{field} must carry segment_id and text")
        if not isinstance(segment["segment_id"], str) or not segment["segment_id"]:
            raise ResultError(f"{field}.segment_id must be a non-empty string")
        if sample:
            if segment["text"] is not None:
                raise ResultError(f"{field}.text placeholder must be null")
        elif not isinstance(segment["text"], str):
            raise ResultError(f"{field}.text must be a string")
        for key, capability in _SEGMENT_CAPABILITIES.items():
            if key in segment and capability not in requested:
                raise ResultError(f"{field}.{key} requires requested capability {capability!r}")
        if "speaker" in segment:
            if sample and segment["speaker"] is not None:
                raise ResultError(f"{field}.speaker placeholder must be null")
            if not sample and not isinstance(segment["speaker"], str):
                raise ResultError(f"{field}.speaker must be a string when present")
        if "words" in segment:
            _validate_words(segment["words"], f"{field}.words", sample=sample, duration=duration)
        _validate_bounds(segment, field, sample=sample, duration=duration)
        words = segment.get("words")
        if (
            not sample
            and isinstance(words, (list, tuple))
            and words
            and "start" in segment
            and "end" in segment
        ):
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])
            if any(
                float(word["start"]) < segment_start - _SEGMENT_WORD_EDGE_TOLERANCE_SECONDS
                or float(word["end"]) > segment_end + _SEGMENT_WORD_EDGE_TOLERANCE_SECONDS
                for word in words
            ):
                raise ResultError(f"{field}.words must fall inside the segment bounds")


def _validate_span_array(
    values: Sequence[JsonMapping],
    field: str,
    expected: set[str],
    *,
    sample: bool,
    duration: float,
) -> None:
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ResultError(f"{field}[{index}] must be an object")
        name = f"{field}[{index}]"
        _exact_keys(item, expected, name)
        _validate_bounds(item, name, sample=sample, duration=duration)
        for key in expected - {"start", "end"}:
            value = item[key]
            if sample and key not in {"turn_id", "overlap_id"} and value is not None:
                raise ResultError(f"{name}.{key} placeholder must be null")
        if field == "turns":
            if not isinstance(item["turn_id"], str) or not item["turn_id"]:
                raise ResultError(f"{name}.turn_id must be a non-empty string")
            if not sample and not isinstance(item["speaker"], str):
                raise ResultError(f"{name}.speaker must be a string when present")
        if field == "overlapped_speech" and (
            not isinstance(item["overlap_id"], str) or not item["overlap_id"]
        ):
            raise ResultError(f"{name}.overlap_id must be a non-empty string")
        if field == "lid_regions" and not sample:
            if not isinstance(item["language"], str) or not item["language"]:
                raise ResultError(f"{name}.language must be a non-empty string")
            confidence = _number(item["confidence"], f"{name}.confidence")
            assert confidence is not None
            if not 0 <= confidence <= 1:
                raise ResultError(f"{name}.confidence must be between 0 and 1")


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


def _validate_coverage(coverage: JsonMapping, *, duration: float) -> None:
    _exact_keys(
        coverage,
        {
            "covered_through_seconds",
            "covered_fraction",
            "covered_intervals",
            "missing_intervals",
            "scope_intervals",
            "units_total",
            "units_completed",
        },
        "coverage",
    )
    fraction = _number(coverage["covered_fraction"], "coverage.covered_fraction")
    watermark = _number(coverage["covered_through_seconds"], "coverage.covered_through_seconds")
    assert fraction is not None and watermark is not None
    if not 0 <= fraction <= 1 or not 0 <= watermark <= duration:
        raise ResultError("coverage fraction and watermark are outside their valid ranges")
    for key in ("units_total", "units_completed"):
        value = coverage[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResultError(f"coverage.{key} must be a non-negative integer")
    if coverage["units_completed"] > coverage["units_total"]:
        raise ResultError("coverage.units_completed exceeds coverage.units_total")
    parsed: dict[str, list[tuple[float, float]]] = {}
    for key in ("scope_intervals", "covered_intervals", "missing_intervals"):
        intervals = coverage[key]
        if not isinstance(intervals, (list, tuple)):
            raise ResultError(f"coverage.{key} must be an array")
        parsed[key] = []
        for index, interval in enumerate(intervals):
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ResultError(f"coverage.{key}[{index}] must be a [start, end] pair")
            start = _number(interval[0], f"coverage.{key}[{index}][0]")
            end = _number(interval[1], f"coverage.{key}[{index}][1]")
            assert start is not None and end is not None
            if start < 0 or end <= start or end > duration:
                raise ResultError(f"coverage.{key}[{index}] is not an ordered interval")
            if parsed[key] and start < parsed[key][-1][1]:
                raise ResultError(f"coverage.{key} must be chronological and non-overlapping")
            parsed[key].append((start, end))

    if duration == 0 or not parsed["scope_intervals"] or not parsed["missing_intervals"]:
        raise ResultError("an incomplete result must have non-empty missing intervals")
    first_missing = min(start for start, _ in parsed["missing_intervals"])
    if not math.isclose(watermark, first_missing, rel_tol=0.0, abs_tol=1e-6):
        raise ResultError("covered_through_seconds must equal the first missing interval start")

    scopes = parsed["scope_intervals"]
    for left, right in pairwise(scopes):
        if left[1] > right[0]:
            raise ResultError("coverage scope intervals must not overlap")
    tiled = sorted(parsed["covered_intervals"] + parsed["missing_intervals"])
    interval_index = 0
    for scope_start, scope_end in scopes:
        cursor = scope_start
        while interval_index < len(tiled) and tiled[interval_index][0] < scope_end:
            start, end = tiled[interval_index]
            if not math.isclose(start, cursor, rel_tol=0.0, abs_tol=1e-6) or end > scope_end:
                raise ResultError(
                    "covered and missing intervals must tile the coverage scope without gaps"
                )
            cursor = end
            interval_index += 1
        if not math.isclose(cursor, scope_end, rel_tol=0.0, abs_tol=1e-6):
            raise ResultError(
                "covered and missing intervals must tile the coverage scope without gaps"
            )
    if interval_index != len(tiled):
        raise ResultError("covered and missing intervals must stay inside the coverage scope")

    covered_seconds = sum(end - start for start, end in parsed["covered_intervals"])
    scope_seconds = sum(end - start for start, end in scopes)
    if not math.isclose(fraction, covered_seconds / scope_seconds, rel_tol=0.0, abs_tol=0.0005):
        raise ResultError("covered_fraction must equal covered duration over coverage scope")
    if coverage["units_completed"] >= coverage["units_total"]:
        raise ResultError("an incomplete result must leave at least one unit incomplete")


def _validate_observed(observed: Mapping[str, Any], result: NormalizedResult) -> None:
    allowed = {
        "stage_wall_seconds",
        "total_wall_seconds",
        "peak_rss_bytes_by_stage",
        "peak_rss_bytes",
        "peak_mps_live_bytes_by_stage",
        "peak_mps_live_bytes",
        "segments",
        "words",
        "turns",
        "segments_without_words",
        "abstentions",
        "vad_regions",
        "lid_regions",
        "overlapped_speech",
        "punctuation_invariant_checked",
        "punctuation_invariant_note",
    }
    extra = set(observed) - allowed
    if extra:
        raise ResultError(f"provenance.observed carries unknown keys {sorted(extra)}")
    walls = observed.get("stage_wall_seconds", ABSENT)
    total_wall = observed.get("total_wall_seconds", ABSENT)
    if (walls is ABSENT) != (total_wall is ABSENT):
        raise ResultError(
            "provenance.observed stage_wall_seconds and total_wall_seconds must appear together"
        )
    if walls is not ABSENT:
        if not isinstance(walls, Mapping) or not walls:
            raise ResultError("provenance.observed.stage_wall_seconds must be a non-empty object")
        stage_total = 0.0
        for stage, value in walls.items():
            found = _number(value, f"provenance.observed.stage_wall_seconds.{stage}")
            assert found is not None
            if found < 0:
                raise ResultError("stage wall seconds must not be negative")
            stage_total += found
        stated = _number(total_wall, "provenance.observed.total_wall_seconds")
        assert stated is not None
        if not math.isclose(stage_total, stated, rel_tol=0.0, abs_tol=0.005):
            raise ResultError("total_wall_seconds must equal the sum of stage_wall_seconds")

    for by_stage in ("peak_rss_bytes_by_stage", "peak_mps_live_bytes_by_stage"):
        total_key = by_stage.removesuffix("_by_stage")
        peaks = observed.get(by_stage, ABSENT)
        total_peak = observed.get(total_key, ABSENT)
        if (peaks is ABSENT) != (total_peak is ABSENT):
            raise ResultError(f"{by_stage} and {total_key} must appear together")
        if peaks is ABSENT:
            continue
        if not isinstance(peaks, Mapping) or not peaks:
            raise ResultError(f"provenance.observed.{by_stage} must be a non-empty object")
        measured = []
        for stage, value in peaks.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ResultError(f"provenance.observed.{by_stage}.{stage} must be bytes")
            measured.append(value)
        if isinstance(total_peak, bool) or not isinstance(total_peak, int):
            raise ResultError(f"provenance.observed.{total_key} must be bytes")
        if max(measured) != total_peak:
            raise ResultError(f"{total_key} must equal the maximum of {by_stage}")

    def count_array(value: OptionalArray) -> int | None:
        return None if value is ABSENT else len(value)

    expected_counts = {
        "segments": len(result.segments),
        "words": sum(len(segment.get("words", [])) for segment in result.segments),
        "turns": count_array(result.turns),
        "segments_without_words": sum(1 for segment in result.segments if "words" not in segment),
        "abstentions": len(result.abstentions),
        "vad_regions": count_array(result.vad_regions),
        "lid_regions": count_array(result.lid_regions),
        "overlapped_speech": count_array(result.overlapped_speech),
    }
    for key, expected in expected_counts.items():
        if key not in observed:
            continue
        value = observed[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ResultError(f"provenance.observed.{key} must be a non-negative integer")
        if expected is None or value != expected:
            raise ResultError(
                f"provenance.observed.{key} is {value}, but the result contains {expected}"
            )


def _validate_provenance(
    provenance: JsonMapping,
    requested: frozenset[str],
    result: NormalizedResult,
    *,
    sample: bool,
) -> None:
    _exact_keys(provenance, {"stack", "outcomes", "observed", "plan"}, "provenance")
    if not isinstance(provenance["stack"], str) or not provenance["stack"]:
        raise ResultError("provenance.stack must be a non-empty string")
    outcomes = provenance["outcomes"]
    if not isinstance(outcomes, Mapping):
        raise ResultError("provenance.outcomes must be an object")
    if sample:
        if outcomes:
            raise ResultError("sample provenance.outcomes must be empty")
    elif set(outcomes) != requested:
        raise ResultError("result provenance.outcomes must name every requested capability")
    for capability, outcome in outcomes.items():
        if capability not in requested or outcome not in {"produced", "abstained"}:
            raise ResultError(f"invalid outcome for capability {capability!r}")
        if capability in {"languages", "verbatim"} and outcome != "produced":
            raise ResultError(f"{capability} has no abstained output shape")
    if not isinstance(provenance["observed"], Mapping):
        raise ResultError("provenance.observed must be an object")
    _validate_observed(provenance["observed"], result)
    if not isinstance(provenance["plan"], Mapping):
        raise ResultError("provenance.plan must be an object")


def _reject_non_label_speakers(node: object, field: str = "") -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            child = f"{field}.{key}" if field else str(key)
            if key == "speaker" and value == "N/A":
                raise ResultError(f"{child} must be absent rather than 'N/A'")
            _reject_non_label_speakers(value, child)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _reject_non_label_speakers(value, f"{field}[{index}]")


def _array(result: NormalizedResult, field: str) -> OptionalArray:
    return getattr(result, field)
