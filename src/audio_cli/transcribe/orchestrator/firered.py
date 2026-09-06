"""Execute the FireRed stack over the canonical source timeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from audio_cli.media import canonical_pcm_duration, capture_file_identity, temporary_directory
from audio_cli.packages import load_registry

from ..adapters.firered import normalize_firered_result
from ..adapters.firered_ledger import (
    _bind_supplied_firered_region_ledger,
    _firered_intersects,
    _firered_published_region_ledger,
    _firered_published_vad_prefix,
    _firered_region_ledger,
)
from ..catalog import InputMetadata
from ..execution.materialization import _checkout, _materialized_role_paths
from ..execution.preflight import preflight
from ..execution.publication import (
    PublishedPartial,
    _backend_fix,
    _coverage,
    _publish_partial,
    _resume_command,
    validate_output_targets,
)
from ..execution.runtime import RunProduct, _validate_range
from ..execution.vad import _detect_vad
from ..planner.build import build_plan
from ..planner.request import ResolvedRequest
from ..refusals import request as refusals
from ..result.types import ABSENT, ResultError
from ..transport.service import StageTransport
from ..transport.types import StageFailure, StageOutcome
from .common import (
    _diarizer_outputs,
    _expanded_scope,
    _finish,
    _intersects,
    _intersects_any,
    _run_diarizer,
    _selected_scope,
    _speaker_for_span,
    _write_complete,
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
    protected_source_identity = capture_file_identity(request.input_path)
    validate_output_targets(
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
    entries = preflight(plan, document)
    stage_transport = transport or StageTransport()
    stage_outcomes: list[StageOutcome] = []
    active_role, active_backend = "decode", "ffmpeg"

    with temporary_directory(prefix="audio-transcribe-") as directory:
        canonical = directory / "canonical.wav"
        try:
            stage_outcomes.append(stage_transport.decode(source_identity, canonical))
            duration = canonical_pcm_duration(canonical)
            run_range = _validate_range(request, run_range, duration)
            requested_scope = _selected_scope(run_range, duration)

            supplied_vad: list[dict[str, Any]] | None = None
            public_vad: Any = ABSENT
            if plan.roles["vad"]["backend"] == "silero-vad":
                active_role, active_backend = "vad", "silero-vad"
                values, wall, peak = _detect_vad(
                    canonical, vad_detector, plan.roles["vad"]["config"]
                )
                supplied_vad = [
                    item for item in values if _firered_intersects(item, requested_scope)
                ]
                stage_outcomes.append(
                    StageOutcome("vad", "silero-vad", {}, wall, peak_rss_bytes=peak)
                )

            # FluidAudio always observes the canonical whole source.  Ownership is
            # applied only after FireRed reports the actual whole-VAD unit scope, so
            # running this stage now preserves sequential residency without clipping
            # an intersecting native region at a handwritten range boundary.
            active_role, active_backend = "diarizer", "fluidaudio"
            entries = preflight(plan, document)
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
            entries = preflight(plan, document)
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
            published_stage_regions = _firered_published_region_ledger(stage_regions)
            saw_unprocessed = any(not item["processed"] for item in stage_regions)

            complete_flag = firered.payload.get("complete")
            if not isinstance(complete_flag, bool):
                raise TypeError("FireRed stage complete must be a boolean")
            incomplete = firered.returncode == 4
            processed_count = sum(1 for item in stage_regions if item["processed"])
            if incomplete:
                if complete_flag or processed_count == 0 or not saw_unprocessed:
                    raise ValueError(
                        "FireRed exit 4 requires a non-empty processed prefix and suffix"
                    )
            elif firered.returncode != 0 or not complete_flag or saw_unprocessed:
                raise ValueError("FireRed successful stage status disagrees with its region ledger")

            normalized = normalize_firered_result(
                firered.payload["result"], lid_enabled="lid" in request.wants
            )
            expected_vad = _firered_published_vad_prefix(published_stage_regions)
            if normalized.vad_regions != expected_vad:
                raise ValueError(
                    "FireRed result VAD regions do not exactly match the processed region prefix"
                )

            selected_scope = _expanded_scope(published_stage_regions, requested_scope)
            auxiliary_scope = (
                min(requested_scope[0], selected_scope[0]),
                max(requested_scope[1], selected_scope[1]),
            )
            coverage: Any = ABSENT
            document_scope = auxiliary_scope
            if incomplete:
                unfinished = [item for item in published_stage_regions if not item["processed"]]
                coverage = _coverage(
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
                _backend_fix(exc.role, exc.backend),
            ) from exc
        except (
            EOFError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                _backend_fix(active_role, active_backend),
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
            [item for item in diarization.overlaps if _intersects(item, document_scope)]
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
                if speaker is not None and not _intersects_any(item, ambiguous_overlaps):
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
        error = firered.payload.get("error", {})
        detail = error.get("message") if isinstance(error, Mapping) else None
        raise PublishedPartial(
            refusals.run_incomplete(
                "firered_process",
                "firered-asr2s",
                str(detail)
                if detail
                else (
                    f"FireRed stopped after {processed_count} of {len(stage_regions)} VAD regions"
                ),
                coverage,
                target,
                _resume_command(request, coverage, target, run_range),
            ),
            payload,
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
    return RunProduct(payload)
