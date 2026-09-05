from __future__ import annotations

import json
import os
import pwd
from pathlib import Path

import pytest

from audio_cli import media as media_module
from audio_cli.export import (
    IncompatibleResultsError,
    InvalidResultError,
    OutputExistsError,
    OutputWriteError,
    TimingRequiredError,
    UnsafeOutputError,
    export_documents,
    load_result_document,
    merge_documents,
)
from audio_cli.export.cues import Cue, CueError, build_cues
from audio_cli.export.writers import (
    render_jsonl,
    render_markdown,
    render_srt,
    render_text,
    render_vtt,
    write_text_atomic,
)
from audio_cli.transcribe.catalog import InputMetadata, result_source
from audio_cli.transcribe.result import ABSENT, NormalizedResult, serialize_result


def _payload(
    segments: list[dict],
    *,
    source_path: str = "source.wav",
    duration: float = 2.0,
    outcomes: dict[str, str] | None = None,
    complete: bool = True,
    coverage: dict | object = ABSENT,
    run_range: list[float] | None = None,
    stack: str = "qwen-1.7b",
) -> dict:
    resolved_outcomes = outcomes or {}
    execution: dict = {"partition": "fixture"}
    if run_range is not None:
        execution["range"] = {
            "requested": list(run_range),
            "selected_unit_scope": list(run_range),
        }
    requested = frozenset(resolved_outcomes)
    result = NormalizedResult(
        source={
            "path": source_path,
            "duration_seconds": duration,
            "timebase": "seconds",
        },
        segments=segments,
        abstentions=[],
        provenance={
            "stack": stack,
            "outcomes": resolved_outcomes,
            "observed": {},
            "plan": {"execution": execution},
        },
        requested_capabilities=requested,
        complete=complete,
        coverage=coverage,
        turns=[] if "diarization" in requested else ABSENT,
        vad_regions=[] if "vad" in requested else ABSENT,
        lid_regions=[] if "lid" in requested else ABSENT,
        overlapped_speech=[] if "overlapped_speech" in requested else ABSENT,
    )
    return serialize_result(result)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _timed_segment(
    text: str,
    words: list[tuple[str, float, float]],
    *,
    segment_id: str = "seg_0",
    speaker: str | None = None,
) -> dict:
    segment = {
        "segment_id": segment_id,
        "text": text,
        "words": [
            {"word_id": f"w_{index}", "text": word, "start": start, "end": end}
            for index, (word, start, end) in enumerate(words)
        ],
    }
    if speaker is not None:
        segment["speaker"] = speaker
    return segment


__all__ = [
    "ABSENT",
    "Cue",
    "CueError",
    "IncompatibleResultsError",
    "InputMetadata",
    "InvalidResultError",
    "NormalizedResult",
    "OutputExistsError",
    "OutputWriteError",
    "Path",
    "TimingRequiredError",
    "UnsafeOutputError",
    "_payload",
    "_timed_segment",
    "_write",
    "build_cues",
    "export_documents",
    "json",
    "load_result_document",
    "media_module",
    "merge_documents",
    "os",
    "pwd",
    "pytest",
    "render_jsonl",
    "render_markdown",
    "render_srt",
    "render_text",
    "render_vtt",
    "result_source",
    "serialize_result",
    "write_text_atomic",
]
