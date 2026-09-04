"""Execute Qwen transcription over fixed units on the canonical source timeline."""

from __future__ import annotations

import math
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.environments import backends
from audio_cli.environments import packages as package_catalog
from audio_cli.media import capture_file_identity, temporary_directory
from audio_cli.packages import load_registry
from audio_cli.vad import VadError

from ..adapters import (
    normalize_aligned_words,
    normalize_qwen_segments,
    reconcile_turns,
    sentence_segments,
)
from ..catalog import InputMetadata, result_source
from ..execution.preflight import preflight
from ..execution.publication import (
    _backend_fix,
    _coverage,
    _has_lexical_text,
    _outcomes,
    _publish_partial,
    _publish_result,
    _record_metrics,
    _resume_command,
    _span_owned,
    validate_output_targets,
)
from ..execution.runtime import (
    RunProduct,
    RunRange,
    _core_plan,
    _materialized_path,
    _validate_range,
)
from ..execution.vad import _detect_vad
from ..planner.build import build_plan
from ..planner.request import ResolvedRequest
from ..refusals import request as refusals
from ..result.serialization import serialize_result
from ..result.types import ABSENT, NormalizedResult, ResultError
from ..transport.service import StageTransport
from ..transport.types import StageFailure, StageOutcome


def _fixed_units(duration: float, request: ResolvedRequest) -> list[dict[str, Any]]:
    rule = request.stack.processing["unit_count_rule"]
    if rule["kind"] != "fixed_seconds":
        raise ValueError(f"{request.stack.id} does not declare fixed-second processing units")
    seconds = float(rule["seconds"])
    return [
        {
            "unit_id": f"unit_{index}",
            "start": round(index * seconds, 6),
            "end": round(min(duration, (index + 1) * seconds), 6),
        }
        for index in range(math.ceil(duration / seconds))
    ]


def _select_range(
    units: Sequence[dict[str, Any]], run_range: RunRange | None, duration: float
) -> tuple[list[dict[str, Any]], float, float]:
    start = run_range.start if run_range else 0.0
    end = min(run_range.end if run_range and run_range.end is not None else duration, duration)
    return [dict(item) for item in units if item["end"] > start and item["start"] < end], start, end


def _run_qwen(
    request: ResolvedRequest,
    metadata: InputMetadata,
    *,
    output: Path | None = None,
    output_format: str = "json",
    run_range: RunRange | None = None,
    registry: Mapping[str, Any] | None = None,
    transport: StageTransport | None = None,
    vad_detector: Any | None = None,
    force: bool = False,
) -> RunProduct:
    """Execute the two Qwen stacks and return their normalized result."""
    protected_source_identity = capture_file_identity(request.input_path)
    validate_output_targets(
        request,
        output,
        output_format=output_format,
        run_range=run_range,
        force=force,
    )
    source_identity = Path(request.input_path).resolve()
    document = registry if registry is not None else load_registry()
    ready = {
        identifier
        for identifier, entry in document.get("packages", {}).items()
        if isinstance(entry, Mapping) and entry.get("state") == "ready"
    }
    plan = build_plan(request, metadata, provisioned_packages=ready)
    entries = preflight(plan, document)
    stage_transport = transport or StageTransport()
    outcomes: list[StageOutcome] = []

    with temporary_directory(prefix="audio-transcribe-") as directory:
        canonical = directory / "canonical.wav"
        active_role = "decode"
        active_backend = "ffmpeg"
        try:
            outcomes.append(stage_transport.decode(source_identity, canonical))
            with wave.open(str(canonical), "rb") as handle:
                if (
                    handle.getframerate() != 16_000
                    or handle.getnchannels() != 1
                    or handle.getsampwidth() != 2
                ):
                    raise ValueError("canonical decode is not mono 16 kHz PCM16")
                canonical_duration = round(handle.getnframes() / float(handle.getframerate()), 6)
            run_range = _validate_range(request, run_range, canonical_duration)
            diarization = None
            if "diarizer" in plan.roles:
                active_role, active_backend = "diarizer", "fluidaudio"
                entries = preflight(plan, document)
                fluid_entry = entries["fluidaudio"]
                fluid_models = entries["speaker-diarization-coreml"]
                source = package_catalog()["fluidaudio"].source
                outcome = stage_transport.diarize(
                    checkout=Path(str(fluid_entry["materialized"]["path"])),
                    product=str(source["product"]),
                    product_path=str(fluid_entry["materialized"]["product_path"]),
                    product_sha256=str(fluid_entry["materialized"]["product_sha256"]),
                    model=Path(str(fluid_models["materialized"]["path"])),
                    audio=canonical,
                    config=plan.roles["diarizer"]["config"],
                    overlap="overlapped_speech" in request.wants,
                    directory=directory,
                )
                outcomes.append(outcome)
                diarization = reconcile_turns(outcome.payload, duration_seconds=canonical_duration)
                all_units = list(diarization.units)
            else:
                all_units = _fixed_units(canonical_duration, request)
            selected_units, requested_start, requested_end = _select_range(
                all_units, run_range, canonical_duration
            )
            if run_range is None:
                scope_start, scope_end = 0.0, canonical_duration
            elif selected_units:
                scope_start = min(float(item["start"]) for item in selected_units)
                scope_end = max(float(item["end"]) for item in selected_units)
            else:
                scope_start, scope_end = requested_start, requested_end

            vad_regions: Any = ABSENT
            if "vad" in plan.roles:
                active_role, active_backend = "vad", "silero-vad"
                config = plan.roles["vad"]["config"]
                vad_regions, vad_wall, vad_peak = _detect_vad(canonical, vad_detector, config)
                outcomes.append(
                    StageOutcome(
                        "vad",
                        "silero-vad",
                        {},
                        vad_wall,
                        peak_rss_bytes=vad_peak,
                    )
                )

            asr_backend = str(plan.roles["asr"]["backend"])
            active_role, active_backend = "asr", asr_backend
            entries = preflight(plan, document)
            package_id = backends()[asr_backend].package
            asr = stage_transport.qwen(
                backend=asr_backend,
                model=_materialized_path(entries, package_id),
                audio=canonical,
                units=selected_units,
                language=request.language,
                max_tokens=int(plan.roles["asr"]["config"]["max_tokens"]),
                batch_size=int(plan.roles["asr"]["config"]["batch_size"]),
                clear_cache_after_every_batch=bool(
                    plan.roles["asr"]["config"]["clear_mlx_cache_after_every_batch"]
                ),
                directory=directory,
            )
            outcomes.append(asr)
            completed, unfinished = normalize_qwen_segments(asr.payload, selected_units)
            if asr.returncode not in {0, 4}:
                raise ValueError(f"Qwen stage returned unsupported exit {asr.returncode}")
            if bool(asr.returncode == 4) != bool(unfinished):
                raise ValueError("Qwen stage exit status disagrees with unfinished unit ledger")
            if unfinished:
                # The backend runs duration-bucketed, so its completed set can have holes in
                # source time. Publish only the chronological prefix; the resume command can
                # then produce a disjoint continuation without asking #24 to guess duplicates.
                watermark = min(float(item["start"]) for item in unfinished)
                completed = [item for item in completed if float(item["end"]) <= watermark]
                unfinished = [
                    dict(item) for item in selected_units if float(item["start"]) >= watermark
                ]

            aligned: dict[str, list[dict[str, Any]]] = {}
            if "aligner" in plan.roles and completed:
                active_role, active_backend = "aligner", "qwen3-forcedaligner"
                entries = preflight(plan, document)
                align = stage_transport.align(
                    model=_materialized_path(entries, "qwen3-forcedaligner"),
                    audio=canonical,
                    segments=completed,
                    directory=directory,
                )
                outcomes.append(align)
                if align.returncode != 0:
                    raise ValueError(f"aligner stage returned unsupported exit {align.returncode}")
                aligned = normalize_aligned_words(align.payload, completed)
        except refusals.Refusal:
            raise
        except StageFailure as exc:
            raise refusals.backend_failed(
                exc.role,
                exc.backend,
                exc.detail,
                _backend_fix(exc.role, exc.backend),
            ) from exc
        except (
            EOFError,
            RuntimeError,
            TypeError,
            ValueError,
            VadError,
            wave.Error,
        ) as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                _backend_fix(active_role, active_backend),
            ) from exc

        completed_ids = {item["unit_id"] for item in completed}
        sentences = sentence_segments(completed, aligned if "aligner" in plan.roles else None)
        alignment_abstentions: list[dict[str, Any]] = []
        if "aligner" in plan.roles:
            for unit in completed:
                unit_sentences = [item for item in sentences if item["unit_id"] == unit["unit_id"]]
                if any(
                    _has_lexical_text(str(item["text"])) and "words" not in item
                    for item in unit_sentences
                ):
                    alignment_abstentions.append(
                        {
                            "abstention_id": "",
                            "reason": "alignment_unavailable",
                            "start": float(unit["start"]),
                            "end": float(unit["end"]),
                        }
                    )

        segments = []
        word_index = 0
        for index, item in enumerate(sentences):
            segment = {"segment_id": f"seg_{index}", "text": item["text"]}
            if "diarization" in request.wants:
                segment["speaker"] = item["speaker"]
            if "word_timestamps" in request.wants and "words" in item:
                words = []
                for word in item["words"]:
                    words.append({"word_id": f"w_{word_index}", **word})
                    word_index += 1
                segment["words"] = words
            segments.append(segment)

        incomplete = bool(unfinished)
        coverage: Any = ABSENT
        if incomplete:
            coverage = _coverage(
                unfinished,
                total_units=len(selected_units),
                completed_units=len(completed),
                scope_start=scope_start,
                scope_end=scope_end,
            )
        if incomplete:
            # Coverage stays at selected unit bounds, but whole-source auxiliary stages can
            # observe evidence between a hand-written range start and the first selected turn.
            # The partial document owns that leading gap; the next resume begins at watermark.
            selected_scope = (
                min(requested_start, scope_start),
                coverage["covered_through_seconds"],
            )
        elif run_range is not None:
            # A requested interval can extend beyond the first/last selected turn, while a
            # fixed processing unit can extend beyond an explicit bound. Own both extents so
            # auxiliary evidence is neither lost at a diarized tail nor clipped from a selected
            # whole unit. Resume watermarks are unit boundaries, preserving disjoint documents.
            selected_scope = (min(requested_start, scope_start), max(requested_end, scope_end))
        else:
            selected_scope = None

        turns: Any = ABSENT
        overlaps: Any = ABSENT
        abstentions = list(alignment_abstentions)
        if diarization is not None:
            if "diarization" in request.wants:
                turns = [item for item in diarization.turns if item["turn_id"] in completed_ids]
            for reason, spans in (
                ("raw_fragment", diarization.raw_fragments),
                ("short_turn", diarization.short_turns),
            ):
                for span in spans:
                    if not _span_owned(span, scope=selected_scope):
                        continue
                    abstentions.append(
                        {
                            "abstention_id": f"ab_{len(abstentions)}",
                            "reason": reason,
                            **span,
                        }
                    )
            owned_overlaps = [
                span for span in diarization.overlaps if _span_owned(span, scope=selected_scope)
            ]
            if "overlapped_speech" in request.wants:
                overlaps = [
                    {"overlap_id": f"overlap_{index}", **span}
                    for index, span in enumerate(owned_overlaps)
                ]
            for span in owned_overlaps:
                abstentions.append(
                    {
                        "abstention_id": f"ab_{len(abstentions)}",
                        "reason": "overlap",
                        **span,
                    }
                )

        if vad_regions is not ABSENT:
            vad_regions = [span for span in vad_regions if _span_owned(span, scope=selected_scope)]
        abstentions.sort(
            key=lambda item: (float(item["start"]), float(item["end"]), str(item["reason"]))
        )
        for index, item in enumerate(abstentions):
            item["abstention_id"] = f"ab_{index}"
        executed_plan = _core_plan(plan)
        if run_range is not None:
            executed_plan["execution"]["range"] = {
                "requested": [requested_start, requested_end],
                "selected_unit_scope": [scope_start, scope_end],
            }
        normalized = NormalizedResult(
            source=result_source(
                metadata,
                canonical_duration,
                source_identity=source_identity,
            ),
            segments=segments,
            abstentions=abstentions,
            provenance={
                "stack": request.stack.id,
                "outcomes": _outcomes(
                    request.wants,
                    segments=segments,
                ),
                "observed": {},
                "plan": executed_plan,
            },
            requested_capabilities=frozenset(request.wants),
            complete=not incomplete,
            coverage=coverage,
            turns=turns,
            vad_regions=vad_regions,
            overlapped_speech=overlaps,
        )
        try:
            normalized.provenance["observed"].update(_record_metrics(outcomes, normalized))
            payload = serialize_result(normalized)
        except (ResultError, ValueError) as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                _backend_fix(active_role, active_backend),
            ) from exc

    if incomplete:
        target = _publish_partial(
            request,
            payload,
            output=output,
            output_format=output_format,
            run_range=run_range,
            force=force,
            protected_source_identity=protected_source_identity,
        )
        stage_error = asr.payload.get("error", {})
        detail = stage_error.get("message") if isinstance(stage_error, Mapping) else None
        raise refusals.run_incomplete(
            "asr",
            asr_backend,
            str(detail)
            if detail
            else f"global generation budget exhausted after {len(completed)} of "
            f"{len(selected_units)} units",
            coverage,
            target,
            _resume_command(request, coverage, target, run_range),
        )
    if output is not None:
        _publish_result(
            request,
            payload,
            output,
            output=output,
            output_format=output_format,
            run_range=run_range,
            force=force,
            protected_source_identity=protected_source_identity,
        )
    return RunProduct(payload)
