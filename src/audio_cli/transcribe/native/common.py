"""Shared result, diarization, and publication helpers for native stacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_cli.environments import packages as package_catalog

from ..adapters import reconcile_turns
from ..catalog import InputMetadata, result_source
from ..execution.publication import _outcomes, _publish_result, _record_metrics
from ..execution.runtime import _core_plan
from ..planner.request import ResolvedRequest
from ..result.serialization import serialize_result
from ..result.types import ABSENT, NormalizedResult
from ..transport.service import StageTransport
from ..transport.types import StageOutcome
from .scope import _owned


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
    owned_overlaps = [dict(item) for item in diarization.overlaps if _owned(item, scope)]
    overlaps: Any = ABSENT
    if "overlapped_speech" in wants:
        overlaps = [
            {"overlap_id": f"overlap_{index}", **item} for index, item in enumerate(owned_overlaps)
        ]
    abstentions: list[dict[str, Any]] = []
    for reason, spans in (
        ("raw_fragment", diarization.raw_fragments),
        ("short_turn", diarization.short_turns),
        ("overlap", owned_overlaps),
    ):
        for span in spans:
            if _owned(span, scope):
                abstentions.append(
                    {
                        "abstention_id": "",
                        "reason": reason,
                        "start": float(span["start"]),
                        "end": float(span["end"]),
                    }
                )
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
    executed = _core_plan(plan)
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
    normalized = NormalizedResult(
        source=result_source(metadata, duration, source_identity=source_identity),
        segments=segments,
        abstentions=abstentions,
        provenance={
            "stack": request.stack.id,
            "outcomes": dict(capability_outcomes)
            if capability_outcomes is not None
            else _outcomes(request.wants, segments=segments),
            "observed": {},
            "plan": _executed_plan(plan, run_range, requested_scope, selected_scope),
        },
        requested_capabilities=frozenset(request.wants),
        complete=complete,
        coverage=coverage,
        turns=turns,
        vad_regions=vad_regions,
        lid_regions=lid_regions,
        overlapped_speech=overlapped_speech,
    )
    normalized.provenance["observed"].update(_record_metrics(outcomes, normalized))
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
