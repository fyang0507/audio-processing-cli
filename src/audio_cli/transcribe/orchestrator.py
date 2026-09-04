"""Compatibility facade for transcription preflight, Qwen execution, and publication."""

from __future__ import annotations

import math
import resource
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_cli.media import (
    atomic_write_json as atomic_write_json,
    atomic_write_text as atomic_write_text,
    capture_file_identity as capture_file_identity,
    hash_file as hash_file,
)
from audio_cli.packages import Toolchain, load_registry as load_registry

from . import refusals
from .catalog import InputMetadata
from .plan import Plan, serialize_plan
from .planner import ResolvedRequest
from .transport import StageFailure as StageFailure
from .transport import StageTransport as StageTransport
from ._orchestrator_output import (
    _backend_fix,
    _coverage,
    _has_lexical_text,
    _outcomes,
    _partial_path,
    _publish_partial,
    _publish_result,
    _record_metrics,
    _resume_command,
    _span_owned,
    _unused_partial_path,
    _write_result_file,
    render_human,
    validate_output_targets,
)
from ._orchestrator_preflight import preflight
from ._orchestrator_qwen import (
    _detect_vad,
    _fixed_units,
    _materialized_path,
    _read_pcm16,
    _run_qwen,
    _select_range,
)

@dataclass(frozen=True)
class RunRange:
    start: float
    end: float | None
    provided: str


@dataclass(frozen=True)
class RunProduct:
    payload: dict[str, Any]


@dataclass(frozen=True)
class _CheckoutState:
    head: str
    modified: tuple[str, ...]
    untracked: tuple[str, ...]


def _inspect_checkout(checkout: Path) -> _CheckoutState:
    """Read live Git state without trusting the provisioning registry's claims."""

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=checkout,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise ValueError(f"could not inspect source checkout: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(
                f"git {' '.join(arguments)} failed for {checkout}: {detail}"
            )
        return completed.stdout

    head = git("rev-parse", "--verify", "HEAD^{commit}").strip()
    modified = tuple(sorted(filter(None, git(
        "diff", "--name-only", "--no-ext-diff", "--no-textconv", "--no-renames",
        "HEAD", "--",
    ).splitlines())))
    ordinary_untracked = filter(None, git(
        "ls-files", "--others", "--exclude-standard",
    ).splitlines())
    ignored_untracked = filter(None, git(
        "ls-files", "--others", "--ignored", "--exclude-standard",
    ).splitlines())
    # Ignored bytecode and extension modules are still importable from a checkout.
    # They are therefore provenance-relevant even though ordinary Git status hides them.
    untracked = tuple(sorted({*ordinary_untracked, *ignored_untracked}))
    return _CheckoutState(head=head, modified=modified, untracked=untracked)


def _checkout_file_digest(path: Path) -> str:
    """Hash live checkout bytes against the immutable manifest expectation."""
    return hash_file(path)


def _built_product_runs(executable: Path) -> bool:
    """Probe an already-built runtime directly, without its provisioning toolchain."""
    try:
        completed = subprocess.run(
            [str(executable), "--help"],
            cwd=executable.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _python_runtime_runs(interpreter: Path) -> bool:
    """Probe a managed interpreter before decode or any model stage begins."""
    try:
        completed = subprocess.run(
            [str(interpreter), "-I", "-c", "pass"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _frozen_packages(interpreter: Path) -> dict[str, str]:
    return Toolchain().frozen_packages(interpreter)


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
    """Execute any shipped transcription stack over the canonical source timeline."""
    if request.stack.id in {"firered", "vibevoice"}:
        # Kept lazy so the Qwen module remains importable while native stage dependencies are
        # absent; model libraries live only in their fresh environment processes.
        from .native import run_native

        return run_native(
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
    return _run_qwen(
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
