"""Qwen transcription orchestration over one canonical source timeline."""

from __future__ import annotations

import math
import os
import resource
import shlex
import subprocess
import sys
import time
import unicodedata
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_cli import paths
from audio_cli.environments import backends
from audio_cli.environments import environments as environment_catalog
from audio_cli.environments import packages as package_catalog
from audio_cli.media import (
    ProtectedFileIdentity,
    ProtectedOutputError,
    atomic_write_json,
    atomic_write_text,
    capture_file_identity,
    hash_file,
    temporary_directory,
)
from audio_cli.packages import (
    Toolchain,
    _checkout_install_drift,
    _managed_checkout_requirements,
    built_product_candidates,
    checkout_file_matches,
    checkout_patch_expectation,
    hub_materialization_issues,
    load_registry,
    managed_checkout_path,
    managed_environment_path,
    managed_provisioning_root_issue,
    managed_url_artifact_path,
    validated_built_product,
)
from audio_cli.vad import SileroOnnxVad, VadError

from . import refusals
from .adapters import (
    normalize_aligned_words,
    normalize_qwen_segments,
    normalize_vad_regions,
    reconcile_turns,
    sentence_segments,
)
from .catalog import InputMetadata, result_source
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


def preflight(
    plan: Plan,
    registry: Mapping[str, Any],
    *,
    built_product_probe: Callable[[Path], bool] | None = None,
    python_runtime_probe: Callable[[Path], bool] | None = None,
    frozen_packages_probe: Callable[[Path], dict[str, str]] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Fail before decode or model load if the selected materialization is not usable."""
    built_product_probe = built_product_probe or _built_product_runs
    python_runtime_probe = python_runtime_probe or _python_runtime_runs
    frozen_packages_probe = frozen_packages_probe or _frozen_packages
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
    unusable_environments: set[str] = set()
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
            unusable_environments.add(environment)
            failures.append({
                "package": package_id,
                "check": f"environment_{environment}_ready",
                "expected": "ready",
                "actual": actual_state or "absent",
            })
            continue
        _environment_path, environment_path_issue = managed_environment_path(environment)
        if environment_path_issue is not None:
            unusable_environments.add(environment)
            failures.append({
                "package": package_id,
                "check": f"environment_{environment}_managed_root",
                "expected": str(paths.env_dir(environment)),
                "actual": environment_path_issue,
            })
            continue
        if environment_catalog()[environment].has_interpreter:
            interpreter = paths.env_python(environment)
            if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                unusable_environments.add(environment)
                failures.append({
                    "package": package_id,
                    "check": f"environment_{environment}_python_executable",
                    "expected": True,
                    "actual": False,
                })
            elif not python_runtime_probe(interpreter):
                unusable_environments.add(environment)
                failures.append({
                    "package": package_id,
                    "check": f"environment_{environment}_python_runs",
                    "expected": True,
                    "actual": False,
                })
            else:
                selected_ids = {
                    str(record["package"])
                    for record in plan.packages
                    if not record.get("auto_fetch")
                }
                requirements = _managed_checkout_requirements(
                    registry, environment, selected_ids
                )
                if requirements:
                    install_drift = _checkout_install_drift(
                        frozen_packages_probe(interpreter), requirements
                    )
                    if install_drift:
                        unusable_environments.add(environment)
                        failures.append({
                            "package": package_id,
                            "check": f"environment_{environment}_checkout_install",
                            "expected": {
                                name: expected
                                for name, (expected, _actual) in install_drift.items()
                            },
                            "actual": {
                                name: actual
                                for name, (_expected, actual) in install_drift.items()
                            },
                        })
    selected: dict[str, Mapping[str, Any]] = {}
    for record in plan.packages:
        identifier = str(record["package"])
        if record.get("auto_fetch"):
            source = catalog[identifier].source
            if source.get("type") == "url":
                override = os.environ.get("AUDIO_PROCESSING_VAD_MODEL")
                location = (
                    Path(override).expanduser()
                    if override else paths.models_dir() / str(source["filename"])
                )
                location_issue = None
                if not override:
                    root_issue = managed_provisioning_root_issue()
                    if root_issue is not None:
                        location_issue = root_issue
                    elif location.parent.is_symlink():
                        location_issue = (
                            f"managed artifact parent is a symlink: {location.parent}"
                        )
                    elif location.exists() or location.is_symlink():
                        _managed, location_issue = managed_url_artifact_path(
                            catalog[identifier], location
                        )
                digest_changed = False
                if location.exists() and location.is_file():
                    try:
                        digest_changed = hash_file(location) != source["sha256"]
                    except OSError as exc:
                        location_issue = (
                            f"could not hash managed artifact {location}: {exc}"
                        )
                unusable = (
                    (bool(override) and not location.is_file())
                    or location_issue is not None
                    or (
                        location.exists()
                        and (
                            not location.is_file()
                            or digest_changed
                        )
                    )
                )
                if unusable:
                    failures.append({
                        "package": identifier,
                        "check": "url_artifact_sha256",
                        "expected": source["sha256"],
                        "actual": location_issue or (
                            "missing, not a file, or digest changed"
                        ),
                    })
            continue
        entry = entries[identifier]
        assert isinstance(entry, Mapping)
        selected[identifier] = entry
        # An unusable environment is already diagnosed above.  Do not use any path
        # beneath it for Git inspection, hashing, or product launch.
        if catalog[identifier].environment in unusable_environments:
            continue
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
        if source.get("type") in {"huggingface", "huggingface_multi"}:
            hub_issues = hub_materialization_issues(
                catalog[identifier], dict(materialized)
            )
            if hub_issues:
                failures.append({
                    "package": identifier,
                    "check": "hub_snapshot_integrity",
                    "expected": (
                        "snapshot directories, manifest-filtered files, and "
                        "recorded byte size"
                    ),
                    "actual": hub_issues,
                })
        # Mutable receipt revision fields are history, not the pin.  Hub packages are
        # bound above by manifest repository/revision -> live cache-indexed snapshot;
        # source builds are bound below by the live checkout HEAD.
        checkout_spec = catalog[identifier].checkout
        if checkout_spec is not None:
            checkout_value = materialized.get("checkout")
            checkout, location_issue = managed_checkout_path(
                catalog[identifier], checkout_value
            )
            if location_issue is not None or checkout is None or not checkout.is_dir():
                failures.append({
                    "package": identifier,
                    "check": "checkout_directory_exists",
                    "expected": "exact managed non-symlink checkout directory",
                    "actual": location_issue or False,
                })
                # Keep receipt validation below, but never inspect or hash the candidate
                # path after its managed identity or directory kind failed.
                checkout = None
            expected_commit = checkout_spec.get("resolved_commit")
            if not isinstance(expected_commit, str) or len(expected_commit) != 40:
                failures.append({
                    "package": identifier,
                    "check": "checkout_resolved_commit",
                    "expected": "full 40-hex Git SHA",
                    "actual": expected_commit,
                })
                expected_commit = str(expected_commit)
            actual_commit = materialized.get("checkout_commit")
            accepted_commits = (checkout_spec.get("commit"), expected_commit)
            if actual_commit not in accepted_commits:
                failures.append({
                    "package": identifier,
                    "check": "checkout_commit",
                    "expected": list(accepted_commits),
                    "actual": actual_commit,
                })
            checkout_state: _CheckoutState | None = None
            if checkout is not None and checkout.is_dir():
                try:
                    checkout_state = _inspect_checkout(checkout)
                except ValueError as exc:
                    failures.append({
                        "package": identifier,
                        "check": "checkout_git_state",
                        "expected": "inspectable pinned source checkout",
                        "actual": str(exc),
                    })
                else:
                    if checkout_state.head != expected_commit:
                        failures.append({
                            "package": identifier,
                            "check": "checkout_head",
                            "expected": expected_commit,
                            "actual": checkout_state.head,
                        })
            try:
                expected_patches, expected_names, expected_digests = (
                    checkout_patch_expectation(catalog[identifier])
                )
            except (OSError, ValueError) as exc:
                failures.append({
                    "package": identifier,
                    "check": "shipped_checkout_patch",
                    "expected": "readable installed patch",
                    "actual": str(exc),
                })
                expected_patches, expected_names, expected_digests = (), (), {}
            applied = materialized.get("patches_applied", [])
            if not isinstance(applied, list) or applied != list(expected_patches):
                failures.append({
                    "package": identifier,
                    "check": "checkout_patch_applied",
                    "expected": list(expected_patches),
                    "actual": applied,
                })
            digests = materialized.get("patched_file_digests", {})
            if not isinstance(digests, Mapping) or dict(digests) != expected_digests:
                failures.append({
                    "package": identifier,
                    "check": "patched_file_digests",
                    "expected": expected_digests,
                    "actual": dict(digests) if isinstance(digests, Mapping) else digests,
                })
            if checkout is not None and checkout.is_dir():
                changed: list[str] = []
                for name, expected_digest in expected_digests.items():
                    if not checkout_file_matches(
                        checkout, name, expected_digest, _checkout_file_digest
                    ):
                        changed.append(name)
                if changed:
                    failures.append({
                        "package": identifier,
                        "check": "checkout_patch_integrity",
                        "expected": "manifest-pinned post-patch hashes",
                        "actual": changed,
                    })
            if checkout_state is not None:
                if set(checkout_state.modified) != set(expected_names):
                    failures.append({
                        "package": identifier,
                        "check": "checkout_tracked_changes",
                        "expected": sorted(expected_names),
                        "actual": list(checkout_state.modified),
                    })
                if checkout_state.untracked:
                    failures.append({
                        "package": identifier,
                        "check": "checkout_untracked_files",
                        "expected": [],
                        "actual": list(checkout_state.untracked),
                    })
        if identifier == "fluidaudio":
            fluid_failures_before = len(failures)
            product = str(source["product"])
            # `built` and `product_runs` are pull history. The current trust boundary is the
            # live checkout plus the exact executable path and digest recorded after the build.
            package = catalog[identifier]
            checkout, location_issue = managed_checkout_path(
                package, materialized.get("path")
            )
            if location_issue is not None or checkout is None:
                failures.append({
                    "package": identifier,
                    "check": "built_checkout_path",
                    "expected": str(paths.checkout_dir(package.environment, package.id)),
                    "actual": location_issue,
                })
                candidates = []
            else:
                try:
                    state = _inspect_checkout(checkout)
                except ValueError as exc:
                    failures.append({
                        "package": identifier,
                        "check": "built_checkout_git_state",
                        "expected": "inspectable pinned source checkout",
                        "actual": str(exc),
                    })
                else:
                    try:
                        expected_patches, expected_names, expected_digests = (
                            checkout_patch_expectation(package)
                        )
                    except (OSError, ValueError) as exc:
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_patch",
                            "expected": "readable installed patch",
                            "actual": str(exc),
                        })
                        expected_patches, expected_names, expected_digests = (), (), {}
                    if (
                        state.head != source["commit"]
                        or set(state.modified) != set(expected_names)
                    ):
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_git_state",
                            "expected": {
                                "head": source["commit"],
                                "modified": sorted(expected_names),
                            },
                            "actual": {
                                "head": state.head,
                                "modified": list(state.modified),
                            },
                        })
                    if materialized.get("patches_applied", []) != list(
                        expected_patches
                    ):
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_patch_applied",
                            "expected": list(expected_patches),
                            "actual": materialized.get("patches_applied", []),
                        })
                    if materialized.get("patched_file_digests", {}) != expected_digests:
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_patch_digests",
                            "expected": expected_digests,
                            "actual": materialized.get("patched_file_digests", {}),
                        })
                    changed = [
                        name for name, digest in expected_digests.items()
                        if not checkout_file_matches(
                            checkout, name, digest, _checkout_file_digest
                        )
                    ]
                    if changed:
                        failures.append({
                            "package": identifier,
                            "check": "built_checkout_patch_integrity",
                            "expected": "manifest-pinned post-patch hashes",
                            "actual": changed,
                        })
                candidates = built_product_candidates(checkout, product)
            if len(candidates) != 1:
                failures.append({
                    "package": identifier,
                    "check": "built_product_executable",
                    "expected": 1,
                    "actual": len(candidates),
                })
            elif len(failures) == fluid_failures_before:
                executable, product_issue = validated_built_product(
                    checkout, product, dict(materialized)
                )
                if product_issue is not None or executable is None:
                    failures.append({
                        "package": identifier,
                        "check": "built_product_digest",
                        "expected": "pull-recorded path and sha256",
                        "actual": product_issue,
                    })
                elif not built_product_probe(executable):
                    raise refusals.package_build_unusable(identifier, product)
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
    duration = round(len(samples) / float(rate), 6)
    for index, region in enumerate(normalized):
        if region["end"] > duration:
            raise ValueError(
                f"Silero VAD region {index} exceeds canonical source duration"
            )
    return normalized, round(time.perf_counter() - started, 6), _self_peak_rss()


def _record_metrics(outcomes: Sequence[StageOutcome], result: NormalizedResult) -> dict[str, Any]:
    walls: dict[str, float] = {}
    for item in outcomes:
        if item.wall_seconds_by_stage:
            reported = dict(item.wall_seconds_by_stage)
            internal_wall = sum(float(value) for value in reported.values())
            residual = float(item.wall_seconds) - internal_wall
            if residual < -0.005:
                raise ValueError(
                    f"{item.role} internal wall metrics exceed its process wall"
                )
            # A co-resident stage's full process wall includes imports, model loading,
            # audio I/O, and framework overhead outside its explicitly timed phases.
            # Keep that measured residual visible so total_wall_seconds remains the
            # actual sum of non-overlapping stage walls instead of silently dropping it.
            reported[f"{item.role}_overhead"] = max(0.0, residual)
        else:
            reported = {item.role: item.wall_seconds}
        overlap = set(walls) & set(reported)
        if overlap:
            raise ValueError(f"stage wall metrics repeat roles: {sorted(overlap)}")
        walls.update({name: round(float(value), 6) for name, value in reported.items()})
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
    if result.lid_regions is not ABSENT:
        observed["lid_regions"] = len(result.lid_regions)
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
    while os.path.lexists(candidate):
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
    targets = (("Output", output),)
    if output is not None:
        targets += (("Partial output", _partial_path(source, output)),)
    for _, target in targets:
        if (
            target is not None
            and target.is_dir()
            and not target.is_symlink()
        ):
            raise refusals.output_path_invalid(
                output or target,
                target,
                "destination is a directory and cannot be replaced",
            )
        if target is not None and os.path.lexists(target) and not force:
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
    if output is None:
        return
    try:
        source_identity = source.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise refusals.output_path_invalid(output, source, str(exc)) from exc
    for _, target in targets:
        if target is None:
            continue
        try:
            target_identity = target.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise refusals.output_path_invalid(output, target, str(exc)) from exc
        if source_identity == target_identity:
            raise refusals.output_is_canonical_input(output, target)


def _resume_command(
    request: ResolvedRequest,
    coverage: Mapping[str, Any],
    output: Path,
    run_range: RunRange | None,
) -> str:
    prefix_only = request.stack.failure_recovery.get("partial_results") == "prefix_only"
    has_prefix = bool(coverage.get("covered_intervals"))
    if int(coverage["units_completed"]) == 0 and not (prefix_only and has_prefix):
        return (
            "no processing unit completed; --range would repeat the same deterministic "
            "work, so inspect the first unit or backend budget before retrying"
        )
    stem = output.stem
    if stem.endswith(".partial"):
        stem = stem.removesuffix(".partial")
    rest = output.with_name(f"{stem}.rest.json")
    parts = [
        "audio", "transcribe", "run", "--input",
        refusals.command_path_argument(request.input_path),
        "--stack", request.stack.id,
    ]
    if request.wants:
        parts.extend(("--want", ",".join(request.wants)))
    if request.language:
        if request.language.startswith("-"):
            parts.append(f"--language={request.language}")
        else:
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
    parts.extend((
        "--range", range_value, "-o", refusals.command_path_argument(rest),
    ))
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


def _write_result_file(
    path: Path,
    payload: dict[str, Any],
    *,
    output_format: str,
    force: bool,
    protected_path: Path,
    protected_identity: ProtectedFileIdentity | None,
) -> None:
    protected_identities = (
        (protected_identity,) if protected_identity is not None else ()
    )
    if output_format == "json":
        atomic_write_json(
            path,
            payload,
            force=force,
            protected_paths=(protected_path,),
            protected_identities=protected_identities,
        )
    else:
        atomic_write_text(
            path,
            render_human(payload, output_format),
            force=force,
            protected_paths=(protected_path,),
            protected_identities=protected_identities,
        )


def _publish_result(
    request: ResolvedRequest,
    payload: dict[str, Any],
    target: Path,
    *,
    output: Path | None,
    output_format: str,
    run_range: RunRange | None,
    force: bool,
    protected_source_identity: ProtectedFileIdentity | None,
    storage_format: str | None = None,
) -> None:
    try:
        _write_result_file(
            target,
            payload,
            output_format=storage_format or output_format,
            force=force,
            protected_path=request.input_path,
            protected_identity=protected_source_identity,
        )
    except ProtectedOutputError as exc:
        raise refusals.output_is_canonical_input(
            output or target, target
        ) from exc
    except (FileExistsError, IsADirectoryError) as exc:
        if target.is_dir() and not target.is_symlink():
            raise refusals.output_path_invalid(
                output or target,
                target,
                "destination is a directory and cannot be replaced",
            ) from exc
        raise refusals.output_exists(
            request.input_path,
            request.stack.id,
            request.wants,
            output or target,
            target,
            language=request.language,
            vad=request.vad,
            diarizer=request.diarizer,
            run_range=run_range.provided if run_range is not None else None,
            output_format=output_format,
        ) from exc
    except OSError as exc:
        raise refusals.output_path_invalid(
            output or target,
            target,
            str(exc),
        ) from exc


def _publish_partial(
    request: ResolvedRequest,
    payload: dict[str, Any],
    *,
    output: Path | None,
    output_format: str,
    run_range: RunRange | None,
    force: bool,
    protected_source_identity: ProtectedFileIdentity | None,
) -> Path:
    if output is None and not force:
        while True:
            target = _unused_partial_path(request.input_path)
            try:
                atomic_write_json(
                    target,
                    payload,
                    force=False,
                    protected_paths=(request.input_path,),
                    protected_identities=(
                        (protected_source_identity,)
                        if protected_source_identity is not None
                        else ()
                    ),
                )
                return target
            except ProtectedOutputError as exc:
                raise refusals.output_is_canonical_input(target, target) from exc
            except FileExistsError:
                # Another writer claimed the candidate after selection.  Find a
                # new sibling instead of overwriting either result.
                continue
            except OSError as exc:
                raise refusals.output_path_invalid(
                    target,
                    target,
                    str(exc),
                ) from exc
    target = _partial_path(request.input_path, output)
    _publish_result(
        request,
        payload,
        target,
        output=output,
        output_format=output_format,
        run_range=run_range,
        force=force,
        protected_source_identity=protected_source_identity,
        storage_format="json",
    )
    return target


def _backend_fix(role: str, backend: str) -> str:
    return (
        f"inspect the {role} failure from {backend} and correct the reported runtime "
        "condition before retrying"
    )


def _outcomes(
    wants: Sequence[str], *, segments: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    outcomes = {name: "produced" for name in wants}
    wordless_events = {"[Environmental Sounds]", "[Silence]", "[Human Sounds]"}
    if "word_timestamps" in outcomes and any(
        item.get("text")
        and item.get("text") not in wordless_events
        and "words" not in item
        for item in segments
    ):
        outcomes["word_timestamps"] = "abstained"
    return outcomes


def _has_lexical_text(text: str) -> bool:
    return any(
        not character.isspace()
        and not unicodedata.category(character).startswith("P")
        for character in text
    )


def _run_qwen(
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
    protected_source_identity = capture_file_identity(request.input_path)
    validate_output_targets(
        request,
        output,
        output_format=output_format,
        run_range=run_range,
        force=force,
    )
    source_identity = Path(request.input_path).resolve()
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
            outcomes.append(stage_transport.decode(source_identity, canonical))
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
                entries = preflight(plan, document)
                fluid_entry = entries["fluidaudio"]
                fluid_models = entries["speaker-diarization-coreml"]
                source = package_catalog()["fluidaudio"].source
                outcome = stage_transport.diarize(
                    checkout=Path(str(fluid_entry["materialized"]["path"])),
                    product=str(source["product"]),
                    product_path=str(fluid_entry["materialized"]["product_path"]),
                    product_sha256=str(
                        fluid_entry["materialized"]["product_sha256"]
                    ),
                    model=Path(str(fluid_models["materialized"]["path"])),
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
            entries = preflight(plan, document)
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
            if asr.returncode not in {0, 4}:
                raise ValueError(f"Qwen stage returned unsupported exit {asr.returncode}")
            if bool(asr.returncode == 4) != bool(unfinished):
                raise ValueError(
                    "Qwen stage exit status disagrees with unfinished unit ledger"
                )
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
                entries = preflight(plan, document)
                align = stage_transport.align(
                    model=_materialized_path(entries, "qwen3-forcedaligner"),
                    audio=canonical,
                    segments=completed,
                    directory=directory,
                )
                outcomes.append(align)
                if align.returncode != 0:
                    raise ValueError(
                        f"aligner stage returned unsupported exit {align.returncode}"
                    )
                aligned = normalize_aligned_words(align.payload, completed)
        except refusals.Refusal:
            raise
        except StageFailure as exc:
            raise refusals.backend_failed(
                exc.role, exc.backend, exc.detail,
                _backend_fix(exc.role, exc.backend),
            ) from exc
        except (
            EOFError, RuntimeError, TypeError, ValueError, VadError, wave.Error,
        ) as exc:
            raise refusals.backend_failed(
                active_role, active_backend, str(exc),
                _backend_fix(active_role, active_backend),
            ) from exc

        completed_ids = {item["unit_id"] for item in completed}
        sentences = sentence_segments(
            completed, aligned if "aligner" in plan.roles else None
        )
        alignment_abstentions: list[dict[str, Any]] = []
        if "aligner" in plan.roles:
            for unit in completed:
                unit_sentences = [
                    item for item in sentences
                    if item["unit_id"] == unit["unit_id"]
                ]
                if any(
                    _has_lexical_text(str(item["text"])) and "words" not in item
                    for item in unit_sentences
                ):
                    alignment_abstentions.append({
                        "abstention_id": "",
                        "reason": "alignment_unavailable",
                        "start": float(unit["start"]),
                        "end": float(unit["end"]),
                    })

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
        abstentions = list(alignment_abstentions)
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
        normalized = NormalizedResult(
            source=result_source(
                metadata,
                canonical_duration,
                source_identity=source_identity,
            ),
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
        try:
            normalized.provenance["observed"].update(
                _record_metrics(outcomes, normalized)
            )
            payload = serialize_result(normalized)
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
    return RunProduct(payload)


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
