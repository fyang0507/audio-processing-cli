"""Execute the native VibeVoice stack over the canonical source timeline."""

from __future__ import annotations

import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.media import capture_file_identity, temporary_directory
from audio_cli.packages import load_registry

from ..adapters import (
    normalize_aligned_words,
    normalize_vibevoice_alignment,
    normalize_vibevoice_result,
)
from ..catalog import InputMetadata
from ..execution.preflight import preflight
from ..execution.publication import (
    _backend_fix,
    _publish_partial,
    _resume_command,
    validate_output_targets,
)
from ..execution.runtime import RunProduct, _materialized_path, _validate_range
from ..execution.vad import _detect_vad
from ..planner.build import build_plan
from ..planner.request import ResolvedRequest
from ..refusals import request as refusals
from ..result.types import ABSENT, ResultError
from ..transport.service import StageTransport
from ..transport.types import StageFailure, StageOutcome
from .common import (
    _diarizer_outputs,
    _finish,
    _run_diarizer,
    _write_complete,
)
from .scope import (
    _checkout,
    _clip_canonical,
    _duration,
    _EmptySampleRange,
    _intersects,
    _intersects_any,
    _materialized_role_paths,
    _owned,
    _published_scope,
    _selected_scope,
)


def _native_turns(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Expose each bounded Vibe speech label without filling gaps between segments."""
    turns: list[dict[str, Any]] = []
    for segment in segments:
        speaker = segment.get("speaker")
        if not segment.get("alignable") or not isinstance(speaker, str):
            continue
        turns.append(
            {
                "turn_id": f"turn_{len(turns)}",
                "speaker": speaker,
                "start": segment["start"],
                "end": segment["end"],
            }
        )
    return turns


def _prefix_coverage(*, scope: tuple[float, float], watermark: float) -> dict[str, Any]:
    if not scope[0] < watermark < scope[1]:
        raise ValueError("VibeVoice truncation must leave a non-empty prefix and suffix")
    covered = [[round(scope[0], 6), round(watermark, 6)]]
    return {
        "scope_intervals": [[round(scope[0], 6), round(scope[1], 6)]],
        "covered_through_seconds": round(watermark, 6),
        "covered_fraction": round((watermark - scope[0]) / (scope[1] - scope[0]), 6),
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
            duration = _duration(canonical)
            run_range = _validate_range(request, run_range, duration)
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

            selected_vad: list[dict[str, Any]] = []
            if "vad" in plan.roles:
                active_role, active_backend = "vad", "silero-vad"
                values, wall, peak = _detect_vad(
                    canonical, vad_detector, plan.roles["vad"]["config"]
                )
                selected_vad = [item for item in values if _owned(item, scope)]
                stage_outcomes.append(
                    StageOutcome("vad", "silero-vad", {}, wall, peak_rss_bytes=peak)
                )

            active_role, active_backend = "asr", "vibevoice-asr-7b"
            entries = preflight(plan, document)
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
                raise ValueError(f"VibeVoice stage returned unsupported exit {vibe.returncode}")
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
                raise ValueError("VibeVoice stage exit status disagrees with hit_max_new_tokens")
            if "diarization" in request.wants and any(
                item["alignable"] and "speaker" not in item for item in normalized.segments
            ):
                raise ValueError(
                    "VibeVoice native diarization requires a speaker label on "
                    "every ordinary speech segment"
                )

            alignable = []
            for item in normalized.segments:
                if not item["alignable"]:
                    continue
                alignable.append(
                    {
                        "unit_id": f"native_{len(alignable)}",
                        "text": item["text"],
                        "start": item["start"],
                        "end": item["end"],
                    }
                )
            aligned: dict[str, list[dict[str, Any]]] = {}
            if "aligner" in plan.roles and alignable:
                active_role, active_backend = "aligner", "qwen3-forcedaligner"
                entries = preflight(plan, document)
                align = stage_transport.align(
                    model=_materialized_path(entries, "qwen3-forcedaligner"),
                    audio=canonical,
                    segments=alignable,
                    directory=directory,
                )
                stage_outcomes.append(align)
                if align.returncode != 0:
                    raise ValueError(f"aligner stage returned unsupported exit {align.returncode}")
                aligned = normalize_aligned_words(align.payload, alignable)
                aligned = normalize_vibevoice_alignment(normalized.segments, aligned)
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
            wave.Error,
        ) as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                _backend_fix(active_role, active_backend),
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
                    _backend_fix("asr", "vibevoice-asr-7b"),
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
                    _backend_fix("asr", "vibevoice-asr-7b"),
                ) from exc

        _external_turns, overlaps, abstentions = _diarizer_outputs(
            diarization, scope=document_scope, wants=request.wants
        )
        ambiguous_overlaps = (
            [item for item in diarization.overlaps if _intersects(item, document_scope)]
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
                        alignment_abstentions.append(
                            {
                                "abstention_id": "",
                                "reason": "alignment_unavailable",
                                "start": float(item["start"]),
                                "end": float(item["end"]),
                            }
                        )
                    else:
                        segment["words"] = []
                        for word in words:
                            segment["words"].append(
                                {
                                    "word_id": f"w_{word_index}",
                                    **word,
                                }
                            )
                            word_index += 1
            public_segments.append(segment)

        abstentions.extend(alignment_abstentions)
        abstentions.sort(
            key=lambda item: (float(item["start"]), float(item["end"]), str(item["reason"]))
        )
        for index, item in enumerate(abstentions):
            item["abstention_id"] = f"ab_{index}"
        public_vad: Any = ABSENT
        if "vad" in request.wants:
            public_vad = [dict(item) for item in selected_vad if _owned(item, document_scope)]
        turns: Any = ABSENT
        if "diarization" in request.wants:
            turns = _native_turns(normalized.segments)
        capability_outcomes = dict.fromkeys(request.wants, "produced")
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
        raise refusals.run_incomplete(
            "asr",
            "vibevoice-asr-7b",
            (
                "generation reached max_new_tokens after salvaging "
                f"{len(public_segments)} complete segment(s)"
            ),
            coverage,
            target,
            _resume_command(request, coverage, target, run_range),
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
