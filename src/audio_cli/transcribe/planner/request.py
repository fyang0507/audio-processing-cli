"""Pure request validation before registry or provisioning checks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

from .. import stacks
from ..refusals import request as refusals

_PIN_VALUES = {"--vad": ("silero-vad",), "--diarizer": ("fluidaudio",)}


@dataclass(frozen=True)
class ResolvedRequest:
    stack: stacks.StackDefinition
    input_path: Path
    wants: tuple[str, ...]
    language: str | None
    vad: str | None
    diarizer: str | None


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def parse_wants(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return _dedupe(value.split(","))
    return _dedupe(value)


def _suggest(provided: str, allowed: Iterable[str], *, cutoff: float = 0.55) -> str | None:
    canonical = {value.casefold(): value for value in allowed}
    prefixes = [
        value for folded, value in canonical.items() if folded.startswith(provided.casefold())
    ]
    if len(prefixes) == 1:
        return prefixes[0]
    matches = get_close_matches(provided.casefold(), canonical, n=1, cutoff=cutoff)
    return canonical[matches[0]] if matches else None


def resolve_request(
    *,
    stack_id: str | None,
    input_path: Path | None,
    wants: str | Iterable[str] | None,
    language: str | None = None,
    vad: str | None = None,
    diarizer: str | None = None,
) -> ResolvedRequest:
    """Reject the complete request before any registry or provisioning check."""
    requested = parse_wants(wants)
    if stack_id is None or stack_id not in stacks.stack_definitions():
        raise refusals.stack_required(input_path, requested)
    definition = stacks.get_stack(stack_id)
    if input_path is None:
        raise refusals.input_required(stack_id, requested)

    for capability in requested:
        if capability not in stacks.capability_order():
            raise refusals.capability_unknown(
                definition,
                input_path,
                capability,
                requested,
                _suggest(capability, stacks.capability_order()),
            )
    for capability in requested:
        resolution = definition.capabilities[capability]["resolution"]
        if resolution == "unsatisfiable_on_stack":
            raise refusals.capability_unsatisfiable_on_stack(
                definition, input_path, capability, requested
            )
        if resolution == "unsupported":
            cell = definition.capabilities[capability]
            raise refusals.capability_unsupported(capability, cell["reason"], cell["refusal_fix"])

    canonical_language = language
    if language is not None:
        if definition.language_vocabulary is None:
            raise refusals.option_unsupported_on_stack(
                definition, input_path, "--language", language, requested
            )
        allowed = stacks.language_vocabulary(definition.language_vocabulary)
        by_fold = {value.casefold(): value for value in allowed}
        canonical_language = by_fold.get(language.casefold())
        if canonical_language is None:
            raise refusals.option_value_unsupported(
                definition,
                input_path,
                "--language",
                language,
                allowed,
                requested,
                _suggest(language, allowed, cutoff=0.4),
            )

    if vad is not None:
        if vad not in _PIN_VALUES["--vad"]:
            fixed_wants = requested if "vad" in requested else (*requested, "vad")
            raise refusals.option_value_unsupported(
                definition,
                input_path,
                "--vad",
                vad,
                _PIN_VALUES["--vad"],
                fixed_wants,
                _suggest(vad, _PIN_VALUES["--vad"]),
            )
        if "vad" not in requested:
            raise refusals.pin_conflicts_with_native_capability(
                definition, input_path, "--vad", vad, "vad", requested
            )

    if diarizer is not None:
        if diarizer not in _PIN_VALUES["--diarizer"]:
            fixed_wants = requested if "diarization" in requested else (*requested, "diarization")
            raise refusals.option_value_unsupported(
                definition,
                input_path,
                "--diarizer",
                diarizer,
                _PIN_VALUES["--diarizer"],
                fixed_wants,
                _suggest(diarizer, _PIN_VALUES["--diarizer"]),
            )
        has_diarizer_role = any(
            definition.capabilities[name]["resolution"] == "add_on"
            and definition.capabilities[name].get("package") == "fluidaudio"
            for name in requested
        )
        if not has_diarizer_role:
            raise refusals.pin_conflicts_with_native_capability(
                definition,
                input_path,
                "--diarizer",
                diarizer,
                "diarization",
                requested,
            )

    return ResolvedRequest(
        stack=definition,
        input_path=input_path,
        wants=requested,
        language=canonical_language,
        vad=vad,
        diarizer=diarizer,
    )
