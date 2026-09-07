"""Strictly sequential, fresh-process transport for transcription stages."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from audio_cli import paths
from audio_cli.media import canonical_decode_command, retain_diagnostic_file
from audio_cli.packages import managed_environment_path, validated_built_product

from .process_runner import SubprocessRunner
from .types import ProcessRunner, StageFailure, StageOutcome


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


class StageTransport:
    """Launch each non-core stage exactly once, never keeping a model resident."""

    def __init__(
        self,
        runner: ProcessRunner | None = None,
        *,
        progress=None,
        log_root: Path | None = None,
    ) -> None:
        if runner is not None and log_root is not None:
            raise ValueError("configure log_root on the supplied runner instead")
        self.progress = progress or sys.stderr
        self.runner = runner or SubprocessRunner(self.progress, log_root=log_root)

    def _run(self, command: list[str], stage: str):
        if isinstance(self.runner, SubprocessRunner):
            return self.runner.run(command, stage=stage)
        return self.runner.run(command)

    def _notice(self, message: str) -> None:
        self.progress.write(message.replace("transcribe: ", "transcribe: host: ", 1) + "\n")
        self.progress.flush()

    def _diagnostics(self, completed) -> None:
        if completed.stderr and not (
            getattr(completed, "stderr_preserved", False)
            or getattr(completed, "stderr_streamed", False)
        ):
            self._notice("transcribe: backend/transport stderr follows")
            self.progress.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                self.progress.write("\n")
            self.progress.flush()

    def _retain_response(self, completed, role, backend, result_path, request_path=None):
        directory = getattr(completed, "diagnostics_directory", None)
        if directory is None:
            return
        try:
            if request_path is not None:
                retain_diagnostic_file(request_path, directory / "request.json")
            if result_path.exists() or result_path.is_symlink():
                retain_diagnostic_file(result_path, directory / "result.json")
        except OSError as exc:
            raise StageFailure(
                role, backend, f"cannot retain structured diagnostics: {exc}"
            ) from exc
        self._notice(f"transcribe: {role} structured diagnostics: {directory}")

    def decode(self, source: Path, target: Path) -> StageOutcome:
        self._notice("transcribe: decode started")
        started = time.perf_counter()
        completed = self._run(canonical_decode_command(source, target), "decode")
        wall = time.perf_counter() - started
        if completed.returncode != 0 or not target.is_file():
            detail = completed.stderr.strip() or "ffmpeg did not write the canonical WAV"
            raise StageFailure("decode", "ffmpeg", detail[-4000:])
        self._notice("transcribe: decode finished")
        return StageOutcome(
            "decode",
            "ffmpeg",
            {},
            round(wall, 6),
            peak_rss_bytes=getattr(completed, "peak_rss_bytes", None),
        )

    @staticmethod
    def _stage_script(name: str, role: str, backend: str) -> Path:
        target = Path(__file__).resolve().parents[1] / "stages" / f"{name}.py"
        if not target.is_file():
            raise StageFailure(role, backend, f"installed stage script is missing: {target}")
        return target

    def _json_stage(
        self,
        *,
        role: str,
        backend: str,
        environment: str,
        script: str,
        request: dict[str, Any],
        directory: Path,
    ) -> StageOutcome:
        _managed, environment_issue = managed_environment_path(environment)
        interpreter = paths.env_python(environment)
        if environment_issue is not None:
            raise StageFailure(
                role,
                backend,
                f"managed environment changed after preflight: {environment_issue}",
            )
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise StageFailure(
                role,
                backend,
                f"managed environment interpreter is not executable: {interpreter}",
            )
        request_path = directory / f"{role}.request.json"
        result_path = directory / f"{role}.result.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8")
        command = [
            str(interpreter),
            "-B",
            str(self._stage_script(script, role, backend)),
            str(request_path),
            str(result_path),
        ]
        self._notice(f"transcribe: {role} started ({backend})")
        started = time.perf_counter()
        completed = self._run(command, role)
        transport_wall = time.perf_counter() - started
        self._diagnostics(completed)
        self._retain_response(completed, role, backend, result_path, request_path)
        sampled_peak = getattr(completed, "peak_rss_bytes", None)
        transport_outcome = StageOutcome(
            role,
            backend,
            {},
            round(transport_wall, 6),
            completed.returncode,
            peak_rss_bytes=sampled_peak,
        )
        if not result_path.is_file():
            detail = (
                completed.stderr.strip() or f"stage exited {completed.returncode} without a result"
            )
            raise StageFailure(role, backend, detail[-4000:], outcome=transport_outcome)
        try:
            payload = json.loads(
                result_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (OSError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise StageFailure(
                role,
                backend,
                f"stage result is invalid JSON: {exc}",
                outcome=transport_outcome,
            ) from exc
        if not isinstance(payload, dict):
            raise StageFailure(
                role,
                backend,
                "stage result must be a JSON object",
                outcome=transport_outcome,
            )
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise StageFailure(
                role,
                backend,
                "stage result metrics must be a JSON object",
                outcome=transport_outcome,
            )
        stage_walls = metrics.get("stage_wall_seconds")
        if stage_walls is not None and not isinstance(stage_walls, Mapping):
            raise StageFailure(
                role,
                backend,
                "stage result metrics.stage_wall_seconds must be a JSON object",
                outcome=transport_outcome,
            )
        try:
            raw_wall = metrics.get("wall_seconds", transport_wall)
            if isinstance(raw_wall, bool) or not isinstance(raw_wall, (int, float)):
                raise TypeError("wall_seconds must be a JSON number")
            wall = float(raw_wall)
            if not math.isfinite(wall) or wall < 0:
                raise ValueError("wall_seconds must be finite and non-negative")
            peaks: list[int | None] = []
            for field in ("peak_rss_bytes", "peak_mps_live_bytes"):
                raw_peak = metrics.get(field)
                if raw_peak is None:
                    peaks.append(None)
                    continue
                if isinstance(raw_peak, bool) or not isinstance(raw_peak, int):
                    raise TypeError(f"{field} must be a non-negative integer")
                if raw_peak < 0:
                    raise ValueError(f"{field} must be a non-negative integer")
                peaks.append(raw_peak)
            metric_peak_rss, metric_peak_mps = peaks
            internal_walls: dict[str, float] | None = None
            if stage_walls is not None:
                internal_walls = {}
                for stage, value in stage_walls.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError("stage_wall_seconds values must be JSON numbers")
                    parsed = float(value)
                    if not math.isfinite(parsed) or parsed < 0:
                        raise ValueError(
                            "stage_wall_seconds values must be finite and non-negative"
                        )
                    internal_walls[str(stage)] = round(parsed, 6)
        except (TypeError, ValueError, OverflowError) as exc:
            raise StageFailure(
                role,
                backend,
                f"stage result metrics are invalid: {exc}",
                outcome=transport_outcome,
            ) from exc
        outcome = StageOutcome(
            role,
            backend,
            payload,
            round(wall, 6),
            completed.returncode,
            max(
                value
                for value in (
                    metric_peak_rss,
                    sampled_peak,
                )
                if value is not None
            )
            if metric_peak_rss is not None or sampled_peak is not None
            else None,
            metric_peak_mps,
            internal_walls,
        )
        if completed.returncode not in {0, 4}:
            error = payload.get("error", {})
            detail = (
                (error.get("message") if isinstance(error, Mapping) else None)
                or completed.stderr.strip()
                or f"stage exited {completed.returncode}"
            )
            raise StageFailure(role, backend, str(detail)[-4000:], outcome=outcome)
        self._notice(
            f"transcribe: {role} {'incomplete' if completed.returncode == 4 else 'finished'}"
        )
        return outcome

    def qwen(
        self,
        *,
        backend: str,
        model: Path,
        audio: Path,
        units: list[dict[str, Any]],
        language: str | None,
        max_tokens: int,
        batch_size: int,
        clear_cache_after_every_batch: bool,
        directory: Path,
    ) -> StageOutcome:
        return self._json_stage(
            role="asr",
            backend=backend,
            environment="mlx",
            script="qwen",
            request={
                "model": str(model),
                "audio": str(audio),
                "units": units,
                "language": language,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "clear_cache_after_every_batch": clear_cache_after_every_batch,
            },
            directory=directory,
        )

    def firered(
        self,
        *,
        checkout: Path,
        models: dict[str, Path],
        audio: Path,
        lid_enabled: bool,
        asr_config: dict[str, Any],
        punctuator_config: dict[str, Any],
        vad_regions: list[dict[str, Any]] | None,
        range_start: float,
        range_end: float,
        directory: Path,
    ) -> StageOutcome:
        """Run the complete FireRed system in its one measured co-resident process."""
        return self._json_stage(
            role="firered_process",
            backend="firered-asr2s",
            environment="torch-firered",
            script="firered",
            request={
                "checkout": str(checkout),
                "models": {role: str(path) for role, path in models.items()},
                "audio": str(audio),
                "lid_enabled": lid_enabled,
                "asr_config": asr_config,
                "punctuator_config": punctuator_config,
                # ``None`` means native FireRedVAD.  An empty list is a real, measured
                # external-VAD result and must not silently turn native VAD back on.
                **({"vad_regions": vad_regions} if vad_regions is not None else {}),
                "range_start": range_start,
                "range_end": range_end,
            },
            directory=directory,
        )

    def vibevoice(
        self,
        *,
        checkout: Path,
        model: Path,
        tokenizer: Path,
        audio: Path,
        config: dict[str, Any],
        directory: Path,
    ) -> StageOutcome:
        """Run VibeVoice's single whole-media generate call in a fresh process."""
        return self._json_stage(
            role="asr",
            backend="vibevoice-asr-7b",
            environment="torch-vibevoice",
            script="vibevoice",
            request={
                "checkout": str(checkout),
                "model": str(model),
                "tokenizer": str(tokenizer),
                "audio": str(audio),
                "config": config,
            },
            directory=directory,
        )

    def align(
        self,
        *,
        model: Path,
        audio: Path,
        segments: list[dict[str, Any]],
        directory: Path,
    ) -> StageOutcome:
        return self._json_stage(
            role="aligner",
            backend="qwen3-forcedaligner",
            environment="mlx",
            script="aligner",
            request={"model": str(model), "audio": str(audio), "segments": segments},
            directory=directory,
        )

    @staticmethod
    def _swift_product(
        checkout: Path,
        product: str,
        product_path: str,
        product_sha256: str,
    ) -> Path:
        # Repeat containment, identity, and the pull-recorded byte digest immediately before
        # launch. Decode can be long, so the preflight verdict alone is not a launch boundary.
        executable, issue = validated_built_product(
            checkout,
            product,
            {"product_path": product_path, "product_sha256": product_sha256},
        )
        if issue is not None or executable is None:
            raise StageFailure(
                "diarizer",
                "fluidaudio",
                f"built {product!r} failed its launch-boundary receipt check: {issue}",
            )
        return executable

    @staticmethod
    def _fluid_model_binding(model: Path, directory: Path) -> Path:
        """Expose one preflighted Hub snapshot under FluidAudio's fixed repo-folder name."""
        try:
            if model.is_symlink() or not model.is_dir():
                raise ValueError(f"model snapshot is not a non-symlink directory: {model}")
            resolved = model.resolve(strict=True)
            binding_root = directory / "fluidaudio-models"
            binding_root.mkdir(mode=0o700)
            # Pinned FluidAudio Repo.diarizer.folderName strips the `-coreml` suffix.
            binding = binding_root / "speaker-diarization"
            binding.symlink_to(resolved, target_is_directory=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise StageFailure(
                "diarizer",
                "fluidaudio",
                f"could not bind the pinned model snapshot: {exc}",
            ) from exc
        return binding_root

    def diarize(
        self,
        *,
        checkout: Path,
        product: str,
        product_path: str,
        product_sha256: str,
        model: Path,
        audio: Path,
        config: dict[str, Any],
        overlap: bool,
        directory: Path,
    ) -> StageOutcome:
        raw_path = directory / "diarizer.result.json"
        model_root = self._fluid_model_binding(model, directory)
        command = [
            str(self._swift_product(checkout, product, product_path, product_sha256)),
            "process",
            str(audio),
            "--mode",
            "offline",
            "--model-dir",
            str(model_root),
            "--output",
            str(raw_path),
            "--threshold",
            str(config["threshold"]),
            "--step-ratio",
            str(config["step_ratio"]),
            "--min-segment-duration",
            str(config["min_segment_duration"]),
            "--batch-size",
            str(config["batch_size"]),
        ]
        if overlap:
            command.append("--overlapping-segments")
        self._notice("transcribe: diarizer started (fluidaudio)")
        started = time.perf_counter()
        completed = self._run(command, "diarizer")
        wall = time.perf_counter() - started
        self._diagnostics(completed)
        self._retain_response(completed, "diarizer", "fluidaudio", raw_path)
        if completed.returncode != 0 or not raw_path.is_file():
            detail = completed.stderr.strip() or f"FluidAudio exited {completed.returncode}"
            raise StageFailure("diarizer", "fluidaudio", detail[-4000:])
        try:
            payload = json.loads(
                raw_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except (OSError, json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise StageFailure("diarizer", "fluidaudio", f"invalid result JSON: {exc}") from exc
        self._notice("transcribe: diarizer finished")
        return StageOutcome(
            "diarizer",
            "fluidaudio",
            payload,
            round(wall, 6),
            peak_rss_bytes=getattr(completed, "peak_rss_bytes", None),
        )
