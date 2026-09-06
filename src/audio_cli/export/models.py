"""Strict loading, merging, rendering, and safe publication for ``audio transcribe export``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_cli.media import ProtectedFileIdentity

from .cues import V1_CUE_POLICY, Cue

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
            "input": str(self.inputs[0])
            if len(self.inputs) == 1
            else [str(path) for path in self.inputs],
            "output": str(output) if output is not None else None,
            "format": self.output_format,
        }
        if self.output_format in _TIMED_FORMATS:
            payload.update(
                {
                    "cues": self.cue_count,
                    "source_capability": "word_timestamps",
                    "speaker_labels_rendered": self.speaker_labels_rendered,
                    "cue_policy": V1_CUE_POLICY.as_dict(),
                    "warnings": [dict(item) for item in self.warnings],
                }
            )
        else:
            payload["segments"] = self.segment_count
        return payload
