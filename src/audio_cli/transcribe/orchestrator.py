"""Qwen transcription orchestration over one canonical source timeline."""

from __future__ import annotations

import math
import os
import resource
import shlex
import sys
import time
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_cli import paths
from audio_cli.environments import backends
from audio_cli.environments import environments as environment_catalog
from audio_cli.environments import packages as package_catalog
from audio_cli.media import atomic_write_json, temporary_directory
from audio_cli.packages import load_registry
from audio_cli.vad import SileroOnnxVad, VadError

from . import refusals
from .adapters import (
    normalize_aligned_words,
    normalize_qwen_segments,
    normalize_vad_regions,
    reconcile_turns,
    sentence_segments,
)
from .catalog import InputMetadata
from .plan import Plan, serialize_plan
from .planner import ResolvedRequest, build_plan
from .result import ABSENT, NormalizedResult, ResultError, serialize_result
from .transport import StageFailure, StageOutcome, StageTransport


@dataclass(frozen=True)
class RunRange:
    start: float
    end: float | None
    provided: str


@dataclass(frozen=True)
class RunProduct:
    payload: dict[str, Any]


def _self_peak_rss() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def parse_range(value: str | None) -> RunRange | None:
    if value is None:
        return None
    left, separator, right = value.partition(":")
    if not separator:
        raise ValueError("--range must be START: or START:END")
    try:
        start = float(left)
        end = float(right) if right else None
    except ValueError as exc:
        raise ValueError("--range bounds must be finite seconds") from exc
    if not math.isfinite(start) or start < 0 or (
        end is not None and (not math.isfinite(end) or end <= start)
    ):
        raise ValueError("--range must satisfy 0 <= START < END")
    return RunRange(start, end, value)


def _validate_range(
    request: ResolvedRequest, run_range: RunRange | None, duration: float
) -> RunRange | None:
    if run_range is None:
        return None
    end = min(run_range.end if run_range.end is not None else duration, duration)
    if run_range.start >= duration or end <= run_range.start:
        raise refusals.range_invalid(
            request.input_path,
            request.stack.id,
            request.wants,
            run_range.provided,
            "range does not intersect the source duration",
            language=request.language,
            vad=request.vad,
            diarizer=request.diarizer,
        )
    return RunRange(run_range.start, end, run_range.provided)


def _core_plan(plan: Plan) -> dict[str, Any]:
    payload = serialize_plan(plan)
    payload.pop("sample_output")
    return payload


def _missing_record(record: Mapping[str, Any]) -> dict[str, Any]:
    item = {
        "package": record["package"],
        "kind": record["kind"],
        "bytes": record["bytes"],
    }
    if "requires_tool" in record:
        item["requires_tool"] = list(record["requires_tool"])
    return item


def _paths_exist(materialized: Mapping[str, Any]) -> bool:
    found = []
    if materialized.get("path"):
        found.append(Path(str(materialized["path"])))
    values = materialized.get("paths")
    if isinstance(values, Mapping):
        found.extend(Path(str(value)) for value in values.values())
    return bool(found) and all(path.exists() for path in found)


def preflight(plan: Plan, registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Fail before decode or model load if the selected materialization is not usable."""
    entries = registry.get("packages", {})
    if not isinstance(entries, Mapping):
        entries = {}
    missing = []
    for record in plan.packages:
        if record.get("auto_fetch"):
            continue
        entry = entries.get(record["package"])
        if not isinstance(entry, Mapping) or entry.get("state") != "ready":
            missing.append(_missing_record(record))
    if missing:
        known = sum(int(item["bytes"] or 0) for item in missing)
        unsized = [str(item["package"]) for item in missing if item["bytes"] is None]
        raise refusals.packages_not_provisioned(plan.stack, missing, known, unsized)

    failures = []
    catalog = package_catalog()
    environment_entries = registry.get("environments", {})
    if not isinstance(environment_entries, Mapping):
        environment_entries = {}
    runtime_packages: dict[str, str] = {}
    for record in plan.packages:
        if record.get("auto_fetch"):
            continue
        package = catalog[str(record["package"])]
        if package.environment != "core":
            runtime_packages.setdefault(package.environment, package.id)
    for environment, package_id in runtime_packages.items():
        entry = environment_entries.get(environment, {})
        actual_state = entry.get("state") if isinstance(entry, Mapping) else None
        if actual_state != "ready":
            failures.append({
                "package": package_id,
                "check": f"environment_{environment}_ready",
                "expected": "ready",
                "actual": actual_state or "absent",
            })
            continue
        if environment_catalog()[environment].has_interpreter:
            interpreter = paths.env_python(environment)
            if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                failures.append({
                    "package": package_id,
                    "check": f"environment_{environment}_python_executable",
                    "expected": True,
                    "actual": False,
                })
    selected: dict[str, Mapping[str, Any]] = {}
    for record in plan.packages:
        identifier = str(record["package"])
        if record.get("auto_fetch"):
            continue
        entry = entries[identifier]
        assert isinstance(entry, Mapping)
        selected[identifier] = entry
        materialized = entry.get("materialized", {})
        if not isinstance(materialized, Mapping) or not _paths_exist(materialized):
            failures.append({
                "package": identifier,
                "check": "materialized_paths_exist",
                "expected": True,
                "actual": False,
            })
            continue
        source = catalog[identifier].source
        if source.get("type") in {"huggingface", "git+build"}:
            expected = source.get("revision", source.get("commit"))
            actual = materialized.get("revision")
            if actual != expected:
                failures.append({
                    "package": identifier, "check": "pinned_revision",
                    "expected": expected, "actual": actual,
                })
        elif source.get("type") == "huggingface_multi":
            expected = [item["revision"] for item in source["repos"]]
            actual = materialized.get("revisions")
            if actual != expected:
                failures.append({
                    "package": identifier, "check": "pinned_revisions",
                    "expected": expected, "actual": actual,
                })
        if identifier == "fluidaudio":
            product = str(source["product"])
            if materialized.get("built") is not True:
                failures.append({
                    "package": identifier, "check": "built",
                    "expected": True, "actual": materialized.get("built"),
                })
            elif materialized.get("product_runs") is not True:
                raise refusals.package_build_unusable(identifier, product)
            else:
                checkout = Path(str(materialized["path"]))
                candidates = {
                    path.resolve()
                    for path in checkout.glob(f".build/**/release/{product}")
                    if path.is_file() and os.access(path, os.X_OK)
                }
                if len(candidates) != 1:
                    failures.append({
                        "package": identifier,
                        "check": "built_product_executable",
                        "expected": 1,
                        "actual": len(candidates),
                    })
    if failures:
        raise refusals.package_integrity_failed(failures)
    return selected


def _fixed_units(duration: float, request: ResolvedRequest) -> list[dict[str, Any]]:
    rule = request.stack.processing["unit_count_rule"]
    if rule["kind"] != "fixed_seconds":
        raise ValueError(f"{request.stack.id} does not declare fixed-second processing units")
    seconds = float(rule["seconds"])
    return [{
        "unit_id": f"unit_{index}",
        "start": round(index * seconds, 6),
        "end": round(min(duration, (index + 1) * seconds), 6),
    } for index in range(math.ceil(duration / seconds))]


def _select_range(
    units: Sequence[dict[str, Any]], run_range: RunRange | None, duration: float
) -> tuple[list[dict[str, Any]], float, float]:
    start = run_range.start if run_range else 0.0
    end = min(run_range.end if run_range and run_range.end is not None else duration, duration)
    return [dict(item) for item in units if item["end"] > start and item["start"] < end], start, end


def _materialized_path(entries: Mapping[str, Mapping[str, Any]], identifier: str) -> Path:
    value = entries[identifier].get("materialized", {}).get("path")
    if not value:
        raise refusals.package_integrity_failed(({
            "package": identifier, "check": "materialized_path",
            "expected": "present", "actual": value,
        },))
    return Path(str(value))


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    if rate != 16_000 or channels != 1 or width != 2:
        raise ValueError("canonical decode is not mono 16 kHz PCM16")
    return samples.astype(np.float32) / 32768.0, rate


def _detect_vad(
    path: Path, detector: Any | None, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], float, int]:
    """Run core VAD in a short frame so its PCM and session die before model stages."""
    started = time.perf_counter()
    samples, rate = _read_pcm16(path)
    selected = detector or SileroOnnxVad()
    regions = selected.detect(samples, rate, **config)
    normalized = normalize_vad_regions(regions)
    return normalized, round(time.perf_counter() - started, 6), _self_peak_rss()


def _record_metrics(outcomes: Sequence[StageOutcome], result: NormalizedResult) -> dict[str, Any]:
    walls = {item.role: item.wall_seconds for item in outcomes}
    observed: dict[str, Any] = {
        "stage_wall_seconds": walls,
        "total_wall_seconds": round(sum(walls.values()), 6),
        "segments": len(result.segments),
        "words": sum(len(item.get("words", [])) for item in result.segments),
        "abstentions": len(result.abstentions),
    }
    if result.turns is not ABSENT:
        observed["turns"] = len(result.turns)
    if result.vad_regions is not ABSENT:
        observed["vad_regions"] = len(result.vad_regions)
    if result.overlapped_speech is not ABSENT:
        observed["overlapped_speech"] = len(result.overlapped_speech)
    if "word_timestamps" in result.requested_capabilities:
        observed["segments_without_words"] = sum(
            1 for item in result.segments if "words" not in item
        )
    rss = {item.role: item.peak_rss_bytes for item in outcomes if item.peak_rss_bytes is not None}
    if rss:
        observed["peak_rss_bytes_by_stage"] = rss
        observed["peak_rss_bytes"] = max(rss.values())
    mps = {
        item.role: item.peak_mps_live_bytes
        for item in outcomes if item.peak_mps_live_bytes is not None
    }
    if mps:
        observed["peak_mps_live_bytes_by_stage"] = mps
        observed["peak_mps_live_bytes"] = max(mps.values())
    return observed


def _coverage(
    unfinished: Sequence[Mapping[str, Any]], *, total_units: int,
    completed_units: int, scope_start: float, scope_end: float,
) -> dict[str, Any]:
    watermark = min(float(item["start"]) for item in unfinished)
    missing = [[round(watermark, 6), round(scope_end, 6)]]
    covered = [] if watermark <= scope_start else [[
        round(scope_start, 6), round(watermark, 6)
    ]]
    covered_seconds = sum(end - start for start, end in covered)
    return {
        "scope_intervals": [[round(scope_start, 6), round(scope_end, 6)]],
        "covered_through_seconds": round(watermark, 6),
        "covered_fraction": round(covered_seconds / (scope_end - scope_start), 6),
        "covered_intervals": covered,
        "missing_intervals": missing,
        "units_total": total_units,
        "units_completed": completed_units,
    }


def _span_owned(
    span: Mapping[str, Any], *, scope: tuple[float, float] | None,
) -> bool:
    """Assign each whole-file auxiliary observation to exactly one ranged document."""
    start = float(span["start"])
    return scope is None or scope[0] <= start < scope[1]


def _partial_path(source: Path, output: Path | None) -> Path:
    if output is None:
        return source.with_name(f"{source.stem}.partial.json")
    return output.with_name(f"{output.with_suffix('').name}.partial.json")


def _unused_partial_path(source: Path) -> Path:
    candidate = _partial_path(source, None)
    index = 2
    while candidate.exists():
        candidate = source.with_name(f"{source.stem}.partial.{index}.json")
        index += 1
    return candidate


def validate_output_targets(
    request: ResolvedRequest,
    output: Path | None,
    *,
    output_format: str,
    run_range: RunRange | None,
    force: bool,
) -> None:
    """Refuse every destination collision before decoding or model work begins."""
    source = request.input_path
    if output is not None and source.resolve() == output.resolve():
        raise refusals.output_is_canonical_input(output, output)
    if output is not None and source.resolve() == _partial_path(source, output).resolve():
        raise refusals.output_is_canonical_input(output, _partial_path(source, output))
    targets = (("Output", output),)
    if output is not None:
        targets += (("Partial output", _partial_path(source, output)),)
    for _, target in targets:
        if target is not None and target.exists() and not force:
            raise refusals.output_exists(
                source,
                request.stack.id,
                request.wants,
                output,
                target,
                language=request.language,
                vad=request.vad,
                diarizer=request.diarizer,
                run_range=run_range.provided if run_range is not None else None,
                output_format=output_format,
            )


def _resume_command(
    request: ResolvedRequest,
    coverage: Mapping[str, Any],
    output: Path,
    run_range: RunRange | None,
) -> str:
    if int(coverage["units_completed"]) == 0:
        return (
            "no processing unit completed; --range would repeat the same deterministic "
            "work, so inspect the first unit or backend budget before retrying"
        )
    stem = output.stem
    if stem.endswith(".partial"):
        stem = stem.removesuffix(".partial")
    rest = output.with_name(f"{stem}.rest.json")
    parts = [
        "audio", "transcribe", "run", "--input", str(request.input_path),
        "--stack", request.stack.id,
    ]
    if request.wants:
        parts.extend(("--want", ",".join(request.wants)))
    if request.language:
        parts.extend(("--language", request.language))
    if request.vad:
        parts.extend(("--vad", request.vad))
    if request.diarizer:
        parts.extend(("--diarizer", request.diarizer))
    watermark = coverage["covered_through_seconds"]
    explicit_end = bool(
        run_range is not None and run_range.provided.partition(":")[2]
    )
    range_value = (
        f"{watermark}:{run_range.end}"
        if explicit_end and run_range is not None
        else f"{watermark}:"
    )
    parts.extend(("--range", range_value, "-o", str(rest)))
    return shlex.join(parts)


def render_human(payload: Mapping[str, Any], output_format: str) -> str:
    lines = []
    for item in payload["segments"]:
        prefix = f"[{item['speaker']}] " if "speaker" in item else ""
        lines.append(prefix + item["text"])
    body = "\n".join(lines)
    if output_format == "md":
        return "# Transcript\n\n" + (body + "\n" if body else "")
    return body + ("\n" if body else "")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _backend_fix(role: str, backend: str) -> str:
    return (
        f"inspect the {role} failure from {backend} and correct the reported runtime "
        "condition before retrying"
    )


def _outcomes(
    wants: Sequence[str], *, segments: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    outcomes = {name: "produced" for name in wants}
    if "word_timestamps" in outcomes and any(
        item.get("text") and "words" not in item for item in segments
    ):
        outcomes["word_timestamps"] = "abstained"
    return outcomes


def run(
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
    if request.stack.id not in {"qwen-1.7b", "qwen-0.6b"}:
        issue = 22 if request.stack.id == "firered" else 23
        raise refusals.stack_run_unavailable(request.stack.id, issue)
    validate_output_targets(
        request,
        output,
        output_format=output_format,
        run_range=run_range,
        force=force,
    )
    document = registry if registry is not None else load_registry()
    ready = {
        identifier for identifier, entry in document.get("packages", {}).items()
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
            outcomes.append(stage_transport.decode(request.input_path, canonical))
            with wave.open(str(canonical), "rb") as handle:
                if handle.getframerate() != 16_000 or handle.getnchannels() != 1 \
                        or handle.getsampwidth() != 2:
                    raise ValueError("canonical decode is not mono 16 kHz PCM16")
                canonical_duration = round(
                    handle.getnframes() / float(handle.getframerate()), 6
                )
            run_range = _validate_range(request, run_range, canonical_duration)
            diarization = None
            if "diarizer" in plan.roles:
                active_role, active_backend = "diarizer", "fluidaudio"
                fluid_entry = entries["fluidaudio"]
                source = package_catalog()["fluidaudio"].source
                outcome = stage_transport.diarize(
                    checkout=Path(str(fluid_entry["materialized"]["path"])),
                    product=str(source["product"]),
                    audio=canonical,
                    config=plan.roles["diarizer"]["config"],
                    overlap="overlapped_speech" in request.wants,
                    directory=directory,
                )
                outcomes.append(outcome)
                diarization = reconcile_turns(
                    outcome.payload, duration_seconds=canonical_duration
                )
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
                vad_regions, vad_wall, vad_peak = _detect_vad(
                    canonical, vad_detector, config
                )
                outcomes.append(StageOutcome(
                    "vad", "silero-vad", {}, vad_wall,
                    peak_rss_bytes=vad_peak,
                ))

            asr_backend = str(plan.roles["asr"]["backend"])
            active_role, active_backend = "asr", asr_backend
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
            if unfinished:
                # The backend runs duration-bucketed, so its completed set can have holes in
                # source time. Publish only the chronological prefix; the resume command can
                # then produce a disjoint continuation without asking #24 to guess duplicates.
                watermark = min(float(item["start"]) for item in unfinished)
                completed = [
                    item for item in completed if float(item["end"]) <= watermark
                ]
                unfinished = [
                    dict(item) for item in selected_units
                    if float(item["start"]) >= watermark
                ]

            aligned: dict[str, list[dict[str, Any]]] = {}
            if "aligner" in plan.roles and completed:
                active_role, active_backend = "aligner", "qwen3-forcedaligner"
                try:
                    align = stage_transport.align(
                        model=_materialized_path(entries, "qwen3-forcedaligner"),
                        audio=canonical,
                        segments=completed,
                        directory=directory,
                    )
                    outcomes.append(align)
                    aligned = normalize_aligned_words(align.payload, completed)
                except StageFailure as exc:
                    if exc.role != "aligner":
                        raise
                    aligned = {}
                except (TypeError, ValueError):
                    aligned = {}
        except StageFailure as exc:
            raise refusals.backend_failed(
                exc.role, exc.backend, exc.detail,
                _backend_fix(exc.role, exc.backend),
            ) from exc
        except (TypeError, ValueError, VadError) as exc:
            raise refusals.backend_failed(
                active_role, active_backend, str(exc),
                _backend_fix(active_role, active_backend),
            ) from exc

        completed_ids = {item["unit_id"] for item in completed}
        sentences = sentence_segments(
            completed, aligned if "aligner" in plan.roles else None
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
                unfinished, total_units=len(selected_units), completed_units=len(completed),
                scope_start=scope_start, scope_end=scope_end,
            )
        if incomplete:
            # Coverage stays at selected unit bounds, but whole-source auxiliary stages can
            # observe evidence between a hand-written range start and the first selected turn.
            # The partial document owns that leading gap; the next resume begins at watermark.
            selected_scope = (
                min(requested_start, scope_start), coverage["covered_through_seconds"]
            )
        elif run_range is not None:
            # A requested interval can extend beyond the first/last selected turn, while a
            # fixed processing unit can extend beyond an explicit bound. Own both extents so
            # auxiliary evidence is neither lost at a diarized tail nor clipped from a selected
            # whole unit. Resume watermarks are unit boundaries, preserving disjoint documents.
            selected_scope = (
                min(requested_start, scope_start), max(requested_end, scope_end)
            )
        else:
            selected_scope = None

        turns: Any = ABSENT
        overlaps: Any = ABSENT
        abstentions = []
        if diarization is not None:
            if "diarization" in request.wants:
                turns = [item for item in diarization.turns if item["turn_id"] in completed_ids]
            for reason, spans in (
                ("raw_fragment", diarization.raw_fragments),
                ("short_turn", diarization.short_turns),
            ):
                for span in spans:
                    if not _span_owned(
                        span, scope=selected_scope
                    ):
                        continue
                    abstentions.append({
                        "abstention_id": f"ab_{len(abstentions)}",
                        "reason": reason,
                        **span,
                    })
            owned_overlaps = [
                span for span in diarization.overlaps
                if _span_owned(span, scope=selected_scope)
            ]
            if "overlapped_speech" in request.wants:
                overlaps = [{"overlap_id": f"overlap_{index}", **span}
                            for index, span in enumerate(owned_overlaps)]
            for span in owned_overlaps:
                abstentions.append({
                    "abstention_id": f"ab_{len(abstentions)}", "reason": "overlap", **span,
                })

        if vad_regions is not ABSENT:
            vad_regions = [
                span for span in vad_regions
                if _span_owned(span, scope=selected_scope)
            ]
        abstentions.sort(key=lambda item: (
            float(item["start"]), float(item["end"]), str(item["reason"])
        ))
        for index, item in enumerate(abstentions):
            item["abstention_id"] = f"ab_{index}"
        executed_plan = _core_plan(plan)
        if run_range is not None:
            executed_plan["execution"]["range"] = {
                "requested": [requested_start, requested_end],
                "selected_unit_scope": [scope_start, scope_end],
            }
        source = dict(metadata.source)
        source["duration_seconds"] = canonical_duration
        normalized = NormalizedResult(
            source=source,
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
        normalized.provenance["observed"].update(_record_metrics(outcomes, normalized))
        try:
            payload = serialize_result(normalized)
        except ResultError as exc:
            raise refusals.backend_failed(
                active_role,
                active_backend,
                str(exc),
                _backend_fix(active_role, active_backend),
            ) from exc

    if incomplete:
        target = _partial_path(request.input_path, output)
        if output is None and target.exists() and not force:
            target = _unused_partial_path(request.input_path)
        atomic_write_json(target, payload)
        stage_error = asr.payload.get("error", {})
        detail = stage_error.get("message") if isinstance(stage_error, Mapping) else None
        raise refusals.run_incomplete(
            "asr", asr_backend,
            str(detail) if detail else
            f"global generation budget exhausted after {len(completed)} of "
            f"{len(selected_units)} units",
            coverage,
            target,
            _resume_command(request, coverage, target, run_range),
        )
    if output is not None:
        if output_format == "json":
            atomic_write_json(output, payload)
        else:
            _atomic_write_text(output, render_human(payload, output_format))
    return RunProduct(payload)
