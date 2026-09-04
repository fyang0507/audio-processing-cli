"""Strict loading, merging, rendering, and safe publication for ``audio export``."""

from __future__ import annotations

import copy
import json
import math
import os
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_cli.media import (
    ProtectedFileIdentity,
    capture_file_identity,
    file_identity_from_descriptor,
)
from audio_cli.transcribe.result import ABSENT, NormalizedResult, serialize_result

from .cues import Cue, CueError, V1_CUE_POLICY, build_cues
from .errors import (
    ExportError,
    IncompatibleResultsError,
    InvalidResultError,
    OutputExistsError,
    OutputWriteError,
    TimingRequiredError,
    UnsafeOutputError,
)
from .writers import (
    normalize_voice_annotation,
    render_jsonl,
    render_markdown,
    render_srt,
    render_text,
    render_vtt,
    write_text_atomic,
)


EXPORT_FORMATS = frozenset({"srt", "vtt", "md", "txt", "jsonl"})
_TIMED_FORMATS = frozenset({"srt", "vtt"})
_VIBEVOICE_MULTI_INPUT_DIARIZATION_REASON = (
    "VibeVoice native speaker labels are local to each independent generation "
    "and cannot be reconciled across multiple input documents"
)
_VIBEVOICE_MULTI_INPUT_DIARIZATION_FIX = (
    "export these VibeVoice documents separately, or rerun the desired ranges "
    "together as one generation"
)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


@dataclass(frozen=True)
class LoadedResult:
    path: Path
    payload: dict[str, Any]
    requested_capabilities: frozenset[str]
    owned_intervals: tuple[tuple[float, float], ...]
    file_identity: ProtectedFileIdentity | None = None


@dataclass(frozen=True)
class MergedTranscript:
    inputs: tuple[Path, ...]
    source: dict[str, Any]
    segments: tuple[dict[str, Any], ...]
    documents: tuple[LoadedResult, ...]
    requested_capabilities: frozenset[str]


@dataclass(frozen=True)
class ExportProduct:
    inputs: tuple[Path, ...]
    output_format: str
    content: str
    segments: tuple[dict[str, Any], ...]
    cues: tuple[Cue, ...]
    speaker_labels_rendered: bool
    warnings: tuple[dict[str, Any], ...]

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def cue_count(self) -> int:
        return len(self.cues)

    def summary(self, output: Path | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": str(self.inputs[0]) if len(self.inputs) == 1 else [
                str(path) for path in self.inputs
            ],
            "output": str(output) if output is not None else None,
            "format": self.output_format,
        }
        if self.output_format in _TIMED_FORMATS:
            payload.update({
                "cues": self.cue_count,
                "source_capability": "word_timestamps",
                "speaker_labels_rendered": self.speaker_labels_rendered,
                "cue_policy": V1_CUE_POLICY.as_dict(),
                "warnings": [dict(item) for item in self.warnings],
            })
        else:
            payload["segments"] = self.segment_count
        return payload


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
        resolved_paths = [path.resolve(strict=False) for path in paths]
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
    if (
        len(set(resolved_paths)) != len(resolved_paths)
        or same_file
        or same_captured_file
    ):
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
            if (
                document.payload["provenance"]["stack"]
                != reference.payload["provenance"]["stack"]
            ):
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


def export_documents(
    inputs: Sequence[Path],
    output_format: str,
    output: Path | None = None,
    force: bool = False,
) -> ExportProduct:
    """Read, merge, render, and optionally atomically publish transcript exports."""
    if output_format not in EXPORT_FORMATS:
        raise ValueError(
            f"output_format must be one of {sorted(EXPORT_FORMATS)}, got {output_format!r}"
        )
    loaded = tuple(load_result_document(Path(path)) for path in inputs)
    merged = merge_documents(loaded)
    source_path: Path | None = None
    source_file_identity: ProtectedFileIdentity | None = None
    if output is not None:
        source_path = Path(merged.source["path"])
        if not source_path.is_absolute():
            raise InvalidResultError(
                merged.inputs[0],
                "source.path must be absolute before writing an export; regenerate "
                "the transcript with the current audio CLI",
            )
        try:
            source_path.resolve(strict=False)
            source_file_identity = capture_file_identity(source_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidResultError(
                merged.inputs[0],
                f"source.path cannot be resolved safely: {exc}",
            ) from exc
    cues: tuple[Cue, ...] = ()
    warnings: tuple[dict[str, Any], ...] = ()
    speaker_labels_rendered = False

    if output_format in _TIMED_FORMATS:
        for document in merged.documents:
            _validate_word_timing_ledger(document)
        _require_word_timing(merged)
        for document in merged.documents:
            try:
                build_cues(
                    document.payload["segments"],
                    duration=float(document.payload["source"]["duration_seconds"]),
                )
            except CueError as exc:
                raise InvalidResultError(document.path, str(exc)) from exc
        try:
            cue_build = build_cues(
                merged.segments,
                duration=float(merged.source["duration_seconds"]),
            )
        except CueError as exc:
            raise IncompatibleResultsError(
                merged.inputs, f"merged cue sequence is invalid: {exc}"
            ) from exc
        cues = cue_build.cues
        warnings = ({
            "code": "cue_timing_unvalidated",
            "blocking": False,
            "detail": (
                "boundary MAE/P95 is unmeasured for the backend that produced this "
                "timing, so cue placement is producible but not claimed "
                "broadcast-acceptable"
            ),
        }, *cue_build.warnings)
        if output_format == "srt":
            content = render_srt(cues)
        else:
            content = render_vtt(cues)
            speaker_labels_rendered = any(
                cue.speaker is not None
                and normalize_voice_annotation(cue.speaker) is not None
                for cue in cues
            )
    elif output_format == "md":
        content = render_markdown(merged.segments)
    elif output_format == "txt":
        content = render_text(merged.segments)
    else:
        content = render_jsonl(merged.segments)

    product = ExportProduct(
        inputs=merged.inputs,
        output_format=output_format,
        content=content,
        segments=merged.segments,
        cues=cues,
        speaker_labels_rendered=speaker_labels_rendered,
        warnings=warnings,
    )
    if output is not None:
        assert source_path is not None
        protected = [*merged.inputs, source_path]
        protected_identities = [
            identity
            for identity in (
                *(document.file_identity for document in merged.documents),
                source_file_identity,
            )
            if identity is not None
        ]
        try:
            write_text_atomic(
                Path(output),
                content,
                force=force,
                protected_paths=protected,
                protected_identities=protected_identities,
            )
        except OSError as exc:
            raise OutputWriteError(Path(output), str(exc)) from exc
    return product


__all__ = [
    "EXPORT_FORMATS",
    "ExportError",
    "ExportProduct",
    "IncompatibleResultsError",
    "InvalidResultError",
    "LoadedResult",
    "MergedTranscript",
    "OutputExistsError",
    "OutputWriteError",
    "TimingRequiredError",
    "UnsafeOutputError",
    "export_documents",
    "load_result_document",
    "merge_documents",
]
