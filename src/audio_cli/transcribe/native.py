"""Orchestration for the two native-structure transcription stacks.

Qwen's processing containers are intentionally kept in :mod:`orchestrator`.  FireRed and
VibeVoice publish real speech bounds, so sharing Qwen's unit loop would risk promoting a
container or a diarizer turn into model output.  This module shares only the preflight,
transport, result, and refusal seams.
"""

from __future__ import annotations

import math
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.environments import packages as package_catalog
from audio_cli.media import temporary_directory
from audio_cli.packages import load_registry

from . import refusals
from .adapters import (
    normalize_aligned_words,
    normalize_firered_result,
    normalize_vibevoice_alignment,
    normalize_vibevoice_result,
    reconcile_turns,
)
from .catalog import InputMetadata, result_source
from .planner import ResolvedRequest, build_plan
from .result import ABSENT, NormalizedResult, ResultError, serialize_result
from .transport import StageFailure, StageOutcome, StageTransport


def _core_helpers():
    # Imported lazily from the dispatching module to avoid maintaining duplicate refusal,
    # metric, and output-target implementations.
    from . import orchestrator

    return orchestrator


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getframerate() != 16_000
            or handle.getnchannels() != 1
            or handle.getsampwidth() != 2
        ):
            raise ValueError("canonical decode is not mono 16 kHz PCM16")
        return round(handle.getnframes() / float(handle.getframerate()), 6)


class _EmptySampleRange(ValueError):
    """A valid second range that owns no sample on the canonical grid."""


def _clip_canonical(
    source: Path, target: Path, *, start: float, end: float
) -> tuple[Path, tuple[float, float]]:
    """Write an internal clip and return its exact source-timeline sample bounds."""
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        # Select samples by their source timestamps: START <= timestamp < END.
        # Using the same ceiling rule on both sides makes adjacent ranged runs
        # meet at one sample index without reprocessing a pre-range sample.
        # These exact bounds drive adapter validation and timeline rebasing.
        frame_count = reader.getnframes()
        first = min(frame_count, max(0, math.ceil(start * rate)))
        last = min(frame_count, max(0, math.ceil(end * rate)))
        if first >= last:
            raise _EmptySampleRange(
                "range selects no complete sample on the canonical 16 kHz timeline"
            )
        reader.setpos(first)
        frames = reader.readframes(last - first)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return target, (first / rate, last / rate)


def _materialized_role_paths(
    entries: Mapping[str, Mapping[str, Any]], identifier: str
) -> dict[str, Path]:
    materialized = entries[identifier].get("materialized", {})
    values = materialized.get("paths") if isinstance(materialized, Mapping) else None
    source = package_catalog()[identifier].source
    repositories = source.get("repos")
    if not isinstance(values, Mapping) or not isinstance(repositories, list):
        raise refusals.package_integrity_failed(({
            "package": identifier,
            "check": "materialized_role_paths",
            "expected": "one path per declared repository role",
            "actual": values,
        },))
    found: dict[str, Path] = {}
    for repository in repositories:
        role = str(repository["role"])
        location = values.get(str(repository["repo"]))
        if not location:
            raise refusals.package_integrity_failed(({
                "package": identifier,
                "check": f"materialized_role_{role}",
                "expected": repository["repo"],
                "actual": None,
            },))
        found[role] = Path(str(location))
    return found


def _checkout(
    entries: Mapping[str, Mapping[str, Any]], identifier: str
) -> Path:
    value = entries[identifier].get("materialized", {}).get("checkout")
    if not value:
        raise refusals.package_integrity_failed(({
            "package": identifier,
            "check": "installed_checkout",
            "expected": "present",
            "actual": value,
        },))
    return Path(str(value))


def _selected_scope(run_range: Any, duration: float) -> tuple[float, float]:
    return (
        float(run_range.start) if run_range is not None else 0.0,
        float(run_range.end) if run_range is not None else duration,
    )


def _published_scope(
    scope: tuple[float, float], duration: float
) -> tuple[float, float]:
    """Express exact processing bounds at the durable schema's precision."""
    return (round(scope[0], 6), min(duration, round(scope[1], 6)))


def _owned(span: Mapping[str, Any], scope: tuple[float, float]) -> bool:
    return scope[0] <= float(span["start"]) < scope[1]


def _intersects(span: Mapping[str, Any], scope: tuple[float, float]) -> bool:
    """Select a whole native unit when any of it intersects the requested range."""
    return float(span["end"]) > scope[0] and float(span["start"]) < scope[1]


def _firered_public_bounds(span: Mapping[str, Any]) -> tuple[float, float]:
    """Project a raw VAD region onto FireRed's published millisecond timeline."""

    start = round(int(float(span["start"]) * 1000) / 1000.0, 6)
    end = round(int(float(span["end"]) * 1000) / 1000.0, 6)
    if end <= start:
        raise ValueError("FireRed VAD region is empty at millisecond precision")
    return start, end


def _firered_intersects(
    span: Mapping[str, Any], scope: tuple[float, float]
) -> bool:
    start, end = _firered_public_bounds(span)
    return end > scope[0] and start < scope[1]


def _intersects_any(
    span: Mapping[str, Any], others: Sequence[Mapping[str, Any]]
) -> bool:
    start, end = float(span["start"]), float(span["end"])
    return any(
        end > float(other["start"]) and start < float(other["end"])
        for other in others
    )


def _expanded_scope(
    regions: Sequence[Mapping[str, Any]], requested: tuple[float, float]
) -> tuple[float, float]:
    if not regions:
        return requested
    return (
        min(float(item["start"]) for item in regions),
        max(float(item["end"]) for item in regions),
    )


def _firered_region_ledger(
    raw_regions: object,
    *,
    source_duration: float,
    requested_scope: tuple[float, float],
) -> list[dict[str, Any]]:
    """Validate the stage's complete processed/unprocessed VAD unit ledger."""

    if not isinstance(raw_regions, list):
        raise TypeError("FireRed stage regions must be an array")
    regions: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    previous_end = 0.0
    saw_unprocessed = False
    for index, item in enumerate(raw_regions):
        field = f"FireRed stage regions[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{field} must be an object")
        if set(item) != {"region_id", "start", "end", "processed"}:
            raise ValueError(f"{field} has an unexpected shape")

        identifier = item["region_id"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
        ):
            raise ValueError("FireRed stage region ids must be unique non-empty strings")
        identifiers.add(identifier)

        bounds: list[float] = []
        for name in ("start", "end"):
            value = item[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field}.{name} must be a number")
            try:
                parsed = float(value)
            except OverflowError as exc:
                raise ValueError(f"{field}.{name} must be finite") from exc
            if not math.isfinite(parsed):
                raise ValueError(f"{field}.{name} must be finite")
            bounds.append(parsed)
        start, end = bounds
        if start < 0 or end > source_duration or end <= start:
            raise ValueError(
                f"{field} must have positive bounds within the source timeline"
            )
        if index and start < previous_end:
            raise ValueError(
                "FireRed stage regions must be chronological and non-overlapping"
            )
        previous_end = end
        if not _firered_intersects(
            {"start": start, "end": end}, requested_scope
        ):
            raise ValueError(
                f"{field} does not intersect the requested processing range"
            )

        processed = item["processed"]
        if not isinstance(processed, bool):
            raise TypeError(f"{field}.processed must be a boolean")
        if saw_unprocessed and processed:
            raise ValueError("FireRed processed regions must form a prefix")
        saw_unprocessed = saw_unprocessed or not processed
        regions.append({
            "region_id": identifier,
            "start": start,
            "end": end,
            "processed": processed,
        })
    return regions


def _firered_published_region_ledger(
    regions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain ledger identity/status while projecting every unit to public bounds."""

    published: list[dict[str, Any]] = []
    for item in regions:
        start, end = _firered_public_bounds(item)
        published.append({
            "region_id": item["region_id"],
            "start": start,
            "end": end,
            "processed": item["processed"],
        })
    return published


def _bind_supplied_firered_region_ledger(
    regions: Sequence[Mapping[str, Any]],
    supplied_vad: Sequence[Mapping[str, Any]],
) -> None:
    """Require the stage ledger to preserve every selected external VAD unit exactly."""

    if len(regions) != len(supplied_vad):
        raise ValueError(
            "FireRed stage regions do not contain every supplied Silero VAD region"
        )
    for index, (actual, expected) in enumerate(zip(regions, supplied_vad, strict=True)):
        if actual["region_id"] != f"vad_{index}":
            raise ValueError(
                "FireRed stage region ids do not preserve supplied Silero VAD order"
            )
        if (
            float(actual["start"]) != float(expected["start"])
            or float(actual["end"]) != float(expected["end"])
        ):
            raise ValueError(
                f"FireRed stage regions[{index}] does not exactly match the supplied "
                "Silero VAD bounds"
            )


def _firered_published_vad_prefix(
    regions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], ...]:
    """Return the already-projected processed region prefix."""

    return tuple({
        "start": float(item["start"]),
        "end": float(item["end"]),
    } for item in regions if item["processed"])


def _speaker_for_span(
    span: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]
) -> str | None:
    """Choose the label owning the most source time, without inventing a bound."""
    start, end = float(span["start"]), float(span["end"])
    scored = []
    for index, turn in enumerate(turns):
        overlap = max(
            0.0,
            min(end, float(turn["end"])) - max(start, float(turn["start"])),
        )
        if overlap > 0:
            scored.append((overlap, -index, str(turn["speaker"])))
    return max(scored)[2] if scored else None


def _diarizer_outputs(
    diarization: Any,
    *,
    scope: tuple[float, float],
    wants: Sequence[str],
) -> tuple[Any, Any, list[dict[str, Any]]]:
    if diarization is None:
        return ABSENT, ABSENT, []
    turns: Any = ABSENT
    if "diarization" in wants:
        turns = [dict(item) for item in diarization.turns if _owned(item, scope)]
    owned_overlaps = [
        dict(item) for item in diarization.overlaps if _owned(item, scope)
    ]
    overlaps: Any = ABSENT
    if "overlapped_speech" in wants:
        overlaps = [
            {"overlap_id": f"overlap_{index}", **item}
            for index, item in enumerate(owned_overlaps)
        ]
    abstentions: list[dict[str, Any]] = []
    for reason, spans in (
        ("raw_fragment", diarization.raw_fragments),
        ("short_turn", diarization.short_turns),
        ("overlap", owned_overlaps),
    ):
        for span in spans:
            if _owned(span, scope):
                abstentions.append({
                    "abstention_id": "",
                    "reason": reason,
                    "start": float(span["start"]),
                    "end": float(span["end"]),
                })
    abstentions.sort(key=lambda item: (item["start"], item["end"], item["reason"]))
    for index, item in enumerate(abstentions):
        item["abstention_id"] = f"ab_{index}"
    return turns, overlaps, abstentions


def _run_diarizer(
    plan: Any,
    entries: Mapping[str, Mapping[str, Any]],
    transport: StageTransport,
    canonical: Path,
    duration: float,
    directory: Path,
    outcomes: list[StageOutcome],
    wants: Sequence[str],
) -> Any:
    if "diarizer" not in plan.roles:
        return None
    entry = entries["fluidaudio"]
    model_entry = entries["speaker-diarization-coreml"]
    source = package_catalog()["fluidaudio"].source
    outcome = transport.diarize(
        checkout=Path(str(entry["materialized"]["path"])),
        product=str(source["product"]),
        product_path=str(entry["materialized"]["product_path"]),
        product_sha256=str(entry["materialized"]["product_sha256"]),
        model=Path(str(model_entry["materialized"]["path"])),
        audio=canonical,
        config=plan.roles["diarizer"]["config"],
        overlap="overlapped_speech" in wants,
        directory=directory,
    )
    outcomes.append(outcome)
    return reconcile_turns(outcome.payload, duration_seconds=duration)


def _executed_plan(
    plan: Any,
    run_range: Any,
    requested_scope: tuple[float, float],
    selected_scope: tuple[float, float],
) -> dict[str, Any]:
    helpers = _core_helpers()
    executed = helpers._core_plan(plan)
    if run_range is not None:
        executed["execution"]["range"] = {
            "requested": [requested_scope[0], requested_scope[1]],
            "selected_unit_scope": [selected_scope[0], selected_scope[1]],
        }
    return executed


def _finish(
    *,
    request: ResolvedRequest,
    metadata: InputMetadata,
    source_identity: Path,
    duration: float,
    plan: Any,
    run_range: Any,
    requested_scope: tuple[float, float],
    selected_scope: tuple[float, float],
    segments: list[dict[str, Any]],
    abstentions: list[dict[str, Any]],
    outcomes: list[StageOutcome],
    turns: Any = ABSENT,
    vad_regions: Any = ABSENT,
    lid_regions: Any = ABSENT,
    overlapped_speech: Any = ABSENT,
    complete: bool = True,
    coverage: Any = ABSENT,
    observed: Mapping[str, Any] | None = None,
    capability_outcomes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    helpers = _core_helpers()
    normalized = NormalizedResult(
        source=result_source(
            metadata, duration, source_identity=source_identity
        ),
        segments=segments,
        abstentions=abstentions,
        provenance={
            "stack": request.stack.id,
            "outcomes": dict(capability_outcomes)
            if capability_outcomes is not None
            else helpers._outcomes(request.wants, segments=segments),
            "observed": {},
            "plan": _executed_plan(
                plan, run_range, requested_scope, selected_scope
            ),
        },
        requested_capabilities=frozenset(request.wants),
        complete=complete,
        coverage=coverage,
        turns=turns,
        vad_regions=vad_regions,
        lid_regions=lid_regions,
        overlapped_speech=overlapped_speech,
    )
    normalized.provenance["observed"].update(helpers._record_metrics(outcomes, normalized))
    if observed:
        normalized.provenance["observed"].update(observed)
    return serialize_result(normalized)


def _write_complete(
    request: ResolvedRequest,
    payload: dict[str, Any],
    output: Path | None,
    output_format: str,
    run_range: Any,
    force: bool,
    protected_source_identity: Any,
) -> None:
    if output is None:
        return
    helpers = _core_helpers()
    helpers._publish_result(
        request,
        payload,
        output,
        output=output,
        output_format=output_format,
        run_range=run_range,
        force=force,
        protected_source_identity=protected_source_identity,
    )


def _run_firered(
    request: ResolvedRequest,
    metadata: InputMetadata,
    *,
    output: Path | None,
    output_format: str,
    run_range: Any,
    registry: Mapping[str, Any] | None,
    transport: StageTransport | None,
    vad_detector: Any | None,
    force: bool,
) -> Any:
    helpers = _core_helpers()
    protected_source_identity = helpers.capture_file_identity(request.input_path)
    helpers.validate_output_targets(
        request, output, output_format=output_format, run_range=run_range, force=force
    )
    source_identity = Path(request.input_path).resolve()
    document = registry if registry is not None else load_registry()
    ready = {
        identifier
        for identifier, entry in document.get("packages", {}).items()
        if isinstance(entry, Mapping) and entry.get("state") == "ready"
    }
    plan = build_plan(request, metadata, provisioned_packages=ready)
    entries = helpers.preflight(plan, document)
    stage_transport = transport or StageTransport()
    stage_outcomes: list[StageOutcome] = []
    active_role, active_backend = "decode", "ffmpeg"

    with temporary_directory(prefix="audio-transcribe-") as directory:
        canonical = directory / "canonical.wav"
        try:
            stage_outcomes.append(stage_transport.decode(source_identity, canonical))
            duration = _duration(canonical)
            run_range = helpers._validate_range(request, run_range, duration)
            requested_scope = _selected_scope(run_range, duration)

            supplied_vad: list[dict[str, Any]] | None = None
            public_vad: Any = ABSENT
            if plan.roles["vad"]["backend"] == "silero-vad":
                active_role, active_backend = "vad", "silero-vad"
                values, wall, peak = helpers._detect_vad(
                    canonical, vad_detector, plan.roles["vad"]["config"]
                )
                supplied_vad = [
                    item
                    for item in values
                    if _firered_intersects(item, requested_scope)
                ]
                stage_outcomes.append(StageOutcome(
                    "vad", "silero-vad", {}, wall, peak_rss_bytes=peak
                ))

            # FluidAudio always observes the canonical whole source.  Ownership is
            # applied only after FireRed reports the actual whole-VAD unit scope, so
            # running this stage now preserves sequential residency without clipping
            # an intersecting native region at a handwritten range boundary.
            active_role, active_backend = "diarizer", "fluidaudio"
            entries = helpers.preflight(plan, document)
            diarization = _run_diarizer(
                plan,
                entries,
                stage_transport,
                canonical,
                duration,
                directory,
                stage_outcomes,
                request.wants,
            )

            active_role, active_backend = "firered_process", "firered-asr2s"
            entries = helpers.preflight(plan, document)
            package_id = "firered-asr2s"
            models = _materialized_role_paths(entries, package_id)
            firered = stage_transport.firered(
                checkout=_checkout(entries, package_id),
                models=models,
                audio=canonical,
                lid_enabled="lid" in request.wants,
                asr_config=dict(plan.roles["asr"]["config"]),
                punctuator_config=dict(plan.roles["punctuator"]["config"]),
                vad_regions=supplied_vad,
                range_start=requested_scope[0],
                range_end=requested_scope[1],
                directory=directory,
            )
            stage_outcomes.append(firered)
            stage_regions = _firered_region_ledger(
                firered.payload.get("regions"),
                source_duration=duration,
                requested_scope=requested_scope,
            )
            if supplied_vad is not None:
                _bind_supplied_firered_region_ledger(stage_regions, supplied_vad)
            published_stage_regions = _firered_published_region_ledger(
                stage_regions
            )
            saw_unprocessed = any(not item["processed"] for item in stage_regions)

            complete_flag = firered.payload.get("complete")
            if not isinstance(complete_flag, bool):
                raise TypeError("FireRed stage complete must be a boolean")
            incomplete = firered.returncode == 4
            processed_count = sum(
                1 for item in stage_regions if item["processed"]
            )
            if incomplete:
                if complete_flag or processed_count == 0 or not saw_unprocessed:
                    raise ValueError(
                        "FireRed exit 4 requires a non-empty processed prefix and suffix"
                    )
            elif firered.returncode != 0 or not complete_flag or saw_unprocessed:
                raise ValueError(
                    "FireRed successful stage status disagrees with its region ledger"
                )

            normalized = normalize_firered_result(
                firered.payload["result"], lid_enabled="lid" in request.wants
            )
            expected_vad = _firered_published_vad_prefix(
                published_stage_regions
            )
            if normalized.vad_regions != expected_vad:
                raise ValueError(
                    "FireRed result VAD regions do not exactly match the processed "
                    "region prefix"
                )

            selected_scope = _expanded_scope(
                published_stage_regions, requested_scope
            )
            auxiliary_scope = (
                min(requested_scope[0], selected_scope[0]),
                max(requested_scope[1], selected_scope[1]),
            )
            coverage: Any = ABSENT
            document_scope = auxiliary_scope
            if incomplete:
                unfinished = [
                    item
                    for item in published_stage_regions
                    if not item["processed"]
                ]
                coverage = helpers._coverage(
                    unfinished,
                    total_units=len(stage_regions),
                    completed_units=processed_count,
                    scope_start=selected_scope[0],
                    scope_end=selected_scope[1],
                )
                document_scope = (
                    auxiliary_scope[0],
                    float(coverage["covered_through_seconds"]),
                )

        except refusals.Refusal:
            raise
        except StageFailure as exc:
            raise refusals.backend_failed(
                exc.role,
                exc.backend,
                exc.detail,
                helpers._backend_fix(exc.role, exc.backend),
            ) from exc
        except (
            EOFError, KeyError, RuntimeError, TypeError, ValueError, wave.Error,
        ) as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                helpers._backend_fix(active_role, active_backend),
            ) from exc

        native_vad = [dict(item) for item in normalized.vad_regions]
        if "vad" in request.wants:
            public_vad = (
                native_vad
                if supplied_vad is None
                else [dict(item) for item in supplied_vad[:processed_count]]
            )

        turns, overlaps, abstentions = _diarizer_outputs(
            diarization, scope=document_scope, wants=request.wants
        )
        ambiguous_overlaps = (
            [
                item for item in diarization.overlaps
                if _intersects(item, document_scope)
            ]
            if diarization is not None and "overlapped_speech" in request.wants
            else []
        )
        public_segments: list[dict[str, Any]] = []
        word_index = 0
        turns_for_labels = list(diarization.turns) if diarization is not None else []
        for index, item in enumerate(normalized.segments):
            segment: dict[str, Any] = {
                "segment_id": f"seg_{index}",
                "text": item["text"],
            }
            if "segment_timestamps" in request.wants:
                segment.update({"start": item["start"], "end": item["end"]})
            if "diarization" in request.wants:
                speaker = _speaker_for_span(item, turns_for_labels)
                if speaker is not None and not _intersects_any(
                    item, ambiguous_overlaps
                ):
                    segment["speaker"] = speaker
            if "word_timestamps" in request.wants:
                words = []
                for word in item["words"]:
                    words.append({"word_id": f"w_{word_index}", **word})
                    word_index += 1
                segment["words"] = words
            public_segments.append(segment)

        lid_regions: Any = ABSENT
        if "lid" in request.wants:
            lid_regions = [dict(item) for item in normalized.lid_regions]
        try:
            payload = _finish(
                request=request,
                metadata=metadata,
                source_identity=source_identity,
                duration=duration,
                plan=plan,
                run_range=run_range,
                requested_scope=requested_scope,
                selected_scope=selected_scope,
                segments=public_segments,
                abstentions=abstentions,
                outcomes=stage_outcomes,
                turns=turns,
                vad_regions=public_vad,
                lid_regions=lid_regions,
                overlapped_speech=overlaps,
                complete=not incomplete,
                coverage=coverage,
                observed={
                    "punctuation_invariant_checked": True,
                    "punctuation_invariant_note": (
                        "each segment's text, stripped of punctuation and whitespace, "
                        "equalled the case-insensitive concatenation of its word texts"
                    ),
                },
            )
        except (ResultError, ValueError) as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                helpers._backend_fix(active_role, active_backend),
            ) from exc

    if incomplete:
        target = helpers._publish_partial(
            request,
            payload,
            output=output,
            output_format=output_format,
            run_range=run_range,
            force=force,
            protected_source_identity=protected_source_identity,
        )
        error = firered.payload.get("error", {})
        detail = error.get("message") if isinstance(error, Mapping) else None
        raise refusals.run_incomplete(
            "firered_process",
            "firered-asr2s",
            str(detail) if detail else (
                f"FireRed stopped after {processed_count} of "
                f"{len(stage_regions)} VAD regions"
            ),
            coverage,
            target,
            helpers._resume_command(request, coverage, target, run_range),
        )

    _write_complete(
        request,
        payload,
        output,
        output_format,
        run_range,
        force,
        protected_source_identity,
    )
    return helpers.RunProduct(payload)


def _native_turns(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose each bounded Vibe speech label without filling gaps between segments."""
    turns: list[dict[str, Any]] = []
    for segment in segments:
        speaker = segment.get("speaker")
        if not segment.get("alignable") or not isinstance(speaker, str):
            continue
        turns.append({
            "turn_id": f"turn_{len(turns)}",
            "speaker": speaker,
            "start": segment["start"],
            "end": segment["end"],
        })
    return turns


def _prefix_coverage(
    *, scope: tuple[float, float], watermark: float
) -> dict[str, Any]:
    if not scope[0] < watermark < scope[1]:
        raise ValueError("VibeVoice truncation must leave a non-empty prefix and suffix")
    covered = [[round(scope[0], 6), round(watermark, 6)]]
    return {
        "scope_intervals": [[round(scope[0], 6), round(scope[1], 6)]],
        "covered_through_seconds": round(watermark, 6),
        "covered_fraction": round(
            (watermark - scope[0]) / (scope[1] - scope[0]), 6
        ),
        "covered_intervals": covered,
        "missing_intervals": [[round(watermark, 6), round(scope[1], 6)]],
        "units_total": 1,
        "units_completed": 0,
    }


def _run_vibevoice(
    request: ResolvedRequest,
    metadata: InputMetadata,
    *,
    output: Path | None,
    output_format: str,
    run_range: Any,
    registry: Mapping[str, Any] | None,
    transport: StageTransport | None,
    vad_detector: Any | None,
    force: bool,
) -> Any:
    helpers = _core_helpers()
    protected_source_identity = helpers.capture_file_identity(request.input_path)
    helpers.validate_output_targets(
        request, output, output_format=output_format, run_range=run_range, force=force
    )
    source_identity = Path(request.input_path).resolve()
    document = registry if registry is not None else load_registry()
    ready = {
        identifier
        for identifier, entry in document.get("packages", {}).items()
        if isinstance(entry, Mapping) and entry.get("state") == "ready"
    }
    plan = build_plan(request, metadata, provisioned_packages=ready)
    entries = helpers.preflight(plan, document)
    stage_transport = transport or StageTransport()
    stage_outcomes: list[StageOutcome] = []
    active_role, active_backend = "decode", "ffmpeg"

    with temporary_directory(prefix="audio-transcribe-") as directory:
        canonical = directory / "canonical.wav"
        try:
            stage_outcomes.append(stage_transport.decode(source_identity, canonical))
            duration = _duration(canonical)
            run_range = helpers._validate_range(request, run_range, duration)
            scope = _selected_scope(run_range, duration)
            selected_audio = canonical
            requested_scope = scope
            if scope != (0.0, duration):
                try:
                    selected_audio, scope = _clip_canonical(
                        canonical,
                        directory / "vibevoice-range.wav",
                        start=scope[0],
                        end=scope[1],
                    )
                except _EmptySampleRange as exc:
                    raise refusals.range_invalid(
                        request.input_path,
                        request.stack.id,
                        request.wants,
                        run_range.provided,
                        str(exc),
                        language=request.language,
                        vad=request.vad,
                        diarizer=request.diarizer,
                    ) from exc

            active_role, active_backend = "diarizer", "fluidaudio"
            entries = helpers.preflight(plan, document)
            diarization = _run_diarizer(
                plan,
                entries,
                stage_transport,
                canonical,
                duration,
                directory,
                stage_outcomes,
                request.wants,
            )

            selected_vad: list[dict[str, Any]] = []
            if "vad" in plan.roles:
                active_role, active_backend = "vad", "silero-vad"
                values, wall, peak = helpers._detect_vad(
                    canonical, vad_detector, plan.roles["vad"]["config"]
                )
                selected_vad = [item for item in values if _owned(item, scope)]
                stage_outcomes.append(StageOutcome(
                    "vad", "silero-vad", {}, wall, peak_rss_bytes=peak
                ))

            active_role, active_backend = "asr", "vibevoice-asr-7b"
            entries = helpers.preflight(plan, document)
            role_paths = _materialized_role_paths(entries, "vibevoice-asr-7b")
            tokenizer_source = plan.roles["asr"].get("tokenizer")
            if not isinstance(tokenizer_source, dict):
                raise ValueError("VibeVoice plan has no tokenizer source")
            tokenizer_role = tokenizer_source.get("materialized_role")
            if not isinstance(tokenizer_role, str):
                raise ValueError("VibeVoice tokenizer has no materialized role")
            vibe = stage_transport.vibevoice(
                checkout=_checkout(entries, "vibevoice-asr-7b"),
                model=role_paths["asr"],
                tokenizer=role_paths[tokenizer_role],
                audio=selected_audio,
                config=dict(plan.roles["asr"]["config"]),
                directory=directory,
            )
            stage_outcomes.append(vibe)
            if vibe.returncode not in {0, 4}:
                raise ValueError(
                    f"VibeVoice stage returned unsupported exit {vibe.returncode}"
                )
            normalized = normalize_vibevoice_result(
                vibe.payload,
                offset_seconds=scope[0],
                clip_duration_seconds=scope[1] - scope[0],
            )
            if vibe.returncode == 4 and not normalized.segments:
                raise StageFailure(
                    "asr",
                    "vibevoice-asr-7b",
                    "generation reached its cap before one complete segment was salvageable",
                )
            if bool(vibe.returncode == 4) != normalized.hit_max_new_tokens:
                raise ValueError(
                    "VibeVoice stage exit status disagrees with hit_max_new_tokens"
                )
            if "diarization" in request.wants and any(
                item["alignable"] and "speaker" not in item
                for item in normalized.segments
            ):
                raise ValueError(
                    "VibeVoice native diarization requires a speaker label on "
                    "every ordinary speech segment"
                )

            alignable = []
            for item in normalized.segments:
                if not item["alignable"]:
                    continue
                alignable.append({
                    "unit_id": f"native_{len(alignable)}",
                    "text": item["text"],
                    "start": item["start"],
                    "end": item["end"],
                })
            aligned: dict[str, list[dict[str, Any]]] = {}
            if "aligner" in plan.roles and alignable:
                active_role, active_backend = "aligner", "qwen3-forcedaligner"
                entries = helpers.preflight(plan, document)
                align = stage_transport.align(
                    model=helpers._materialized_path(entries, "qwen3-forcedaligner"),
                    audio=canonical,
                    segments=alignable,
                    directory=directory,
                )
                stage_outcomes.append(align)
                if align.returncode != 0:
                    raise ValueError(
                        f"aligner stage returned unsupported exit {align.returncode}"
                    )
                aligned = normalize_aligned_words(align.payload, alignable)
                aligned = normalize_vibevoice_alignment(
                    normalized.segments, aligned
                )
        except refusals.Refusal:
            raise
        except StageFailure as exc:
            raise refusals.backend_failed(
                exc.role,
                exc.backend,
                exc.detail,
                helpers._backend_fix(exc.role, exc.backend),
            ) from exc
        except (
            EOFError, KeyError, RuntimeError, TypeError, ValueError, wave.Error,
        ) as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                helpers._backend_fix(active_role, active_backend),
            ) from exc

        incomplete = normalized.hit_max_new_tokens
        coverage: Any = ABSENT
        document_scope = scope
        if incomplete:
            if normalized.covered_through_seconds is None:
                raise refusals.backend_failed(
                    "asr",
                    "vibevoice-asr-7b",
                    "generation reached its cap without a complete segment",
                    helpers._backend_fix("asr", "vibevoice-asr-7b"),
                )
            try:
                watermark = float(normalized.covered_through_seconds)
                coverage = _prefix_coverage(scope=scope, watermark=watermark)
                document_scope = (scope[0], watermark)
            except ValueError as exc:
                raise refusals.backend_failed(
                    "asr",
                    "vibevoice-asr-7b",
                    str(exc),
                    helpers._backend_fix("asr", "vibevoice-asr-7b"),
                ) from exc

        _external_turns, overlaps, abstentions = _diarizer_outputs(
            diarization, scope=document_scope, wants=request.wants
        )
        ambiguous_overlaps = (
            [
                item for item in diarization.overlaps
                if _intersects(item, document_scope)
            ]
            if diarization is not None and "overlapped_speech" in request.wants
            else []
        )
        public_segments: list[dict[str, Any]] = []
        word_index = 0
        alignable_index = 0
        speech_without_words = False
        alignment_abstentions: list[dict[str, Any]] = []
        for index, item in enumerate(normalized.segments):
            segment: dict[str, Any] = {
                "segment_id": f"seg_{index}",
                "text": item["text"],
            }
            if (
                "diarization" in request.wants
                and "speaker" in item
                and not _intersects_any(item, ambiguous_overlaps)
            ):
                segment["speaker"] = item["speaker"]
            if "segment_timestamps" in request.wants:
                segment.update({"start": item["start"], "end": item["end"]})
            if item["alignable"]:
                unit_id = f"native_{alignable_index}"
                alignable_index += 1
                if "word_timestamps" in request.wants:
                    words = aligned.get(unit_id)
                    if words is None:
                        speech_without_words = True
                        alignment_abstentions.append({
                            "abstention_id": "",
                            "reason": "alignment_unavailable",
                            "start": float(item["start"]),
                            "end": float(item["end"]),
                        })
                    else:
                        segment["words"] = []
                        for word in words:
                            segment["words"].append({
                                "word_id": f"w_{word_index}", **word,
                            })
                            word_index += 1
            public_segments.append(segment)

        abstentions.extend(alignment_abstentions)
        abstentions.sort(key=lambda item: (
            float(item["start"]), float(item["end"]), str(item["reason"])
        ))
        for index, item in enumerate(abstentions):
            item["abstention_id"] = f"ab_{index}"
        public_vad: Any = ABSENT
        if "vad" in request.wants:
            public_vad = [
                dict(item) for item in selected_vad
                if _owned(item, document_scope)
            ]
        turns: Any = ABSENT
        if "diarization" in request.wants:
            turns = _native_turns(normalized.segments)
        capability_outcomes = {name: "produced" for name in request.wants}
        if "word_timestamps" in capability_outcomes and speech_without_words:
            capability_outcomes["word_timestamps"] = "abstained"

        try:
            payload = _finish(
                request=request,
                metadata=metadata,
                source_identity=source_identity,
                duration=duration,
                plan=plan,
                run_range=run_range,
                requested_scope=requested_scope,
                # Adapter validation uses the exact sample-aligned scope.  The
                # saved plan uses the same six-decimal source timeline as public
                # timestamps and cannot extend past the published duration at EOF.
                selected_scope=_published_scope(scope, duration),
                segments=public_segments,
                abstentions=abstentions,
                outcomes=stage_outcomes,
                turns=turns,
                vad_regions=public_vad,
                overlapped_speech=overlaps,
                complete=not incomplete,
                coverage=coverage,
                capability_outcomes=capability_outcomes,
            )
        except (ResultError, ValueError) as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                helpers._backend_fix(active_role, active_backend),
            ) from exc

    if incomplete:
        target = helpers._publish_partial(
            request,
            payload,
            output=output,
            output_format=output_format,
            run_range=run_range,
            force=force,
            protected_source_identity=protected_source_identity,
        )
        raise refusals.run_incomplete(
            "asr",
            "vibevoice-asr-7b",
            (
                "generation reached max_new_tokens after salvaging "
                f"{len(public_segments)} complete segment(s)"
            ),
            coverage,
            target,
            helpers._resume_command(request, coverage, target, run_range),
        )

    _write_complete(
        request,
        payload,
        output,
        output_format,
        run_range,
        force,
        protected_source_identity,
    )
    return helpers.RunProduct(payload)


def run_native(
    request: ResolvedRequest,
    metadata: InputMetadata,
    *,
    output: Path | None = None,
    output_format: str = "json",
    run_range: Any = None,
    registry: Mapping[str, Any] | None = None,
    transport: StageTransport | None = None,
    vad_detector: Any | None = None,
    force: bool = False,
) -> Any:
    if request.stack.id == "firered":
        return _run_firered(
            request,
            metadata,
            output=output,
            output_format=output_format,
            run_range=run_range,
            registry=registry,
            transport=transport,
            vad_detector=vad_detector,
            force=force,
        )
    if request.stack.id == "vibevoice":
        return _run_vibevoice(
            request,
            metadata,
            output=output,
            output_format=output_format,
            run_range=run_range,
            registry=registry,
            transport=transport,
            vad_detector=vad_detector,
            force=force,
        )
    raise ValueError(f"{request.stack.id!r} is not a native-structure stack")
