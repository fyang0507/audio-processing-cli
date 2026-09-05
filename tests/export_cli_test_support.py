from __future__ import annotations

import json
import os
import shlex
import stat
from pathlib import Path

from audio_cli import cli
from audio_cli.transcribe.result import ABSENT, NormalizedResult, serialize_result


def _write_result(
    path: Path,
    *,
    source: Path,
    segments: list[dict],
    outcomes: dict[str, str],
    abstentions: list[dict] | None = None,
    language: str = "English",
    stack: str = "qwen-0.6b",
    run_range: list[float] | None = None,
) -> Path:
    plan: dict = {"roles": {"asr": {"config": {"language": language}}}}
    if run_range is not None:
        plan["execution"] = {
            "range": {
                "requested": list(run_range),
                "selected_unit_scope": list(run_range),
            }
        }
    payload = serialize_result(
        NormalizedResult(
            source={
                "path": str(source),
                "duration_seconds": 2.0,
                "timebase": "seconds",
            },
            segments=segments,
            abstentions=abstentions or [],
            provenance={
                "stack": stack,
                "outcomes": outcomes,
                "observed": {},
                "plan": plan,
            },
            requested_capabilities=frozenset(outcomes),
            turns=[] if "diarization" in outcomes else ABSENT,
        )
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


__all__ = [
    "ABSENT",
    "NormalizedResult",
    "Path",
    "_write_result",
    "cli",
    "json",
    "os",
    "serialize_result",
    "shlex",
    "stat",
]
