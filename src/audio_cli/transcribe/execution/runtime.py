"""Shared execution types, range handling, and preflight probes."""

from __future__ import annotations

import math
import resource
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_cli.media import hash_file
from audio_cli.packages import Toolchain

from ..plan import Plan, serialize_plan
from ..planner.request import ResolvedRequest
from ..refusals import request as refusals


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
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise ValueError(f"could not inspect source checkout: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"git {' '.join(arguments)} failed for {checkout}: {detail}")
        return completed.stdout

    head = git("rev-parse", "--verify", "HEAD^{commit}").strip()
    modified = tuple(
        sorted(
            filter(
                None,
                git(
                    "diff",
                    "--name-only",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--no-renames",
                    "HEAD",
                    "--",
                ).splitlines(),
            )
        )
    )
    ordinary_untracked = filter(
        None, git("ls-files", "--others", "--exclude-standard").splitlines()
    )
    ignored_untracked = filter(
        None,
        git("ls-files", "--others", "--ignored", "--exclude-standard").splitlines(),
    )
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
            capture_output=True,
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
            capture_output=True,
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
    if (
        not math.isfinite(start)
        or start < 0
        or (end is not None and (not math.isfinite(end) or end <= start))
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


def _materialized_path(entries: Mapping[str, Mapping[str, Any]], identifier: str) -> Path:
    value = entries[identifier].get("materialized", {}).get("path")
    if not value:
        raise refusals.package_integrity_failed(
            (
                {
                    "package": identifier,
                    "check": "materialized_path",
                    "expected": "present",
                    "actual": value,
                },
            )
        )
    return Path(str(value))


def _paths_exist(materialized: Mapping[str, Any]) -> bool:
    found = []
    if materialized.get("path"):
        found.append(Path(str(materialized["path"])))
    values = materialized.get("paths")
    if isinstance(values, Mapping):
        found.extend(Path(str(value)) for value in values.values())
    return bool(found) and all(path.exists() for path in found)
