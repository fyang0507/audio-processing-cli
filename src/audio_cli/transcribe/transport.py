"""Strictly sequential, fresh-process transport for transcription stages."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from audio_cli import paths

from .adapters.decode import command as decode_command


@dataclass(frozen=True)
class StageOutcome:
    role: str
    backend: str
    payload: dict[str, Any]
    wall_seconds: float
    returncode: int = 0
    peak_rss_bytes: int | None = None
    peak_mps_live_bytes: int | None = None


class StageFailure(RuntimeError):
    def __init__(self, role: str, backend: str, detail: str) -> None:
        self.role = role
        self.backend = backend
        self.detail = detail
        super().__init__(detail)


class ProcessRunner(Protocol):
    def run(self, command: list[str]) -> subprocess.CompletedProcess[str]: ...


class _MacProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("context_switches", ctypes.c_int32),
        ("thread_count", ctypes.c_int32),
        ("running_threads", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


class _ChildRssReader:
    """Read a child process's current RSS without launching another process."""

    def __init__(self) -> None:
        self._proc_pidinfo = None
        if sys.platform == "darwin":
            try:
                function = ctypes.CDLL(
                    "/usr/lib/libproc.dylib", use_errno=True
                ).proc_pidinfo
                function.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_int
                self._proc_pidinfo = function
            except OSError:
                pass

    def read(self, pid: int) -> int | None:
        if self._proc_pidinfo is not None:
            info = _MacProcTaskInfo()
            returned = self._proc_pidinfo(
                pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info)
            )
            if returned == ctypes.sizeof(info):
                return int(info.resident_size)
        try:
            resident_pages = int(
                Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()[1]
            )
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None


class SubprocessRunner:
    def __init__(self, progress) -> None:
        self.progress = progress

    def run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as exc:
            return subprocess.CompletedProcess(command, 127, "", str(exc))

        stdout: list[str] = []
        stderr: list[str] = []

        def drain(stream, sink: list[str], *, mirror: bool = False) -> None:
            for line in iter(stream.readline, ""):
                sink.append(line)
                if mirror:
                    self.progress.write(line)
                    self.progress.flush()
            stream.close()

        assert process.stdout is not None and process.stderr is not None
        stdout_thread = threading.Thread(target=drain, args=(process.stdout, stdout))
        stderr_thread = threading.Thread(
            target=drain, args=(process.stderr, stderr), kwargs={"mirror": True}
        )
        stdout_thread.start()
        stderr_thread.start()
        reader = _ChildRssReader()
        peak_rss: int | None = None
        while process.poll() is None:
            sample = reader.read(process.pid)
            if sample is not None:
                peak_rss = sample if peak_rss is None else max(peak_rss, sample)
            time.sleep(0.01)
        stdout_thread.join()
        stderr_thread.join()
        completed = subprocess.CompletedProcess(
            command, process.returncode, "".join(stdout), "".join(stderr)
        )
        completed.peak_rss_bytes = peak_rss  # type: ignore[attr-defined]
        completed.stderr_streamed = True  # type: ignore[attr-defined]
        return completed


class StageTransport:
    """Launch each non-core stage exactly once, never keeping a model resident."""

    def __init__(self, runner: ProcessRunner | None = None, *, progress=None) -> None:
        self.progress = progress or sys.stderr
        self.runner = runner or SubprocessRunner(self.progress)

    def _notice(self, message: str) -> None:
        self.progress.write(message + "\n")
        self.progress.flush()

    def decode(self, source: Path, target: Path) -> StageOutcome:
        self._notice("transcribe: decode started")
        started = time.perf_counter()
        completed = self.runner.run(decode_command(source, target))
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
        target = Path(__file__).resolve().parent / "stages" / f"{name}.py"
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
        request_path = directory / f"{role}.request.json"
        result_path = directory / f"{role}.result.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        command = [
            str(paths.env_python(environment)),
            str(self._stage_script(script, role, backend)),
            str(request_path),
            str(result_path),
        ]
        self._notice(f"transcribe: {role} started ({backend})")
        started = time.perf_counter()
        completed = self.runner.run(command)
        transport_wall = time.perf_counter() - started
        if completed.stderr and not getattr(completed, "stderr_streamed", False):
            self.progress.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                self.progress.write("\n")
        if not result_path.is_file():
            detail = completed.stderr.strip() or f"stage exited {completed.returncode} without a result"
            raise StageFailure(role, backend, detail[-4000:])
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StageFailure(role, backend, f"stage result is invalid JSON: {exc}") from exc
        metrics = payload.get("metrics", {})
        wall = float(metrics.get("wall_seconds", transport_wall))
        outcome = StageOutcome(
            role,
            backend,
            payload,
            round(wall, 6),
            completed.returncode,
            max(
                value for value in (
                    int(metrics["peak_rss_bytes"])
                    if metrics.get("peak_rss_bytes") is not None else None,
                    getattr(completed, "peak_rss_bytes", None),
                ) if value is not None
            ) if metrics.get("peak_rss_bytes") is not None
            or getattr(completed, "peak_rss_bytes", None) is not None else None,
            int(metrics["peak_mps_live_bytes"])
            if metrics.get("peak_mps_live_bytes") is not None else None,
        )
        if completed.returncode not in {0, 4}:
            error = payload.get("error", {})
            detail = error.get("message") or completed.stderr.strip() or f"stage exited {completed.returncode}"
            raise StageFailure(role, backend, str(detail)[-4000:])
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
                "model": str(model), "audio": str(audio), "units": units,
                "language": language, "max_tokens": max_tokens,
                "batch_size": batch_size,
                "clear_cache_after_every_batch": clear_cache_after_every_batch,
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
    def _swift_product(checkout: Path, product: str) -> Path:
        candidates = sorted(checkout.glob(f".build/**/release/{product}"))
        candidates.extend(sorted(checkout.glob(f".build/release/{product}")))
        unique = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
        if len(unique) != 1:
            raise StageFailure(
                "diarizer", "fluidaudio",
                f"expected one built {product!r} under {checkout / '.build'}, found {len(unique)}",
            )
        return unique[0]

    def diarize(
        self,
        *,
        checkout: Path,
        product: str,
        audio: Path,
        config: dict[str, Any],
        overlap: bool,
        directory: Path,
    ) -> StageOutcome:
        raw_path = directory / "diarizer.result.json"
        command = [
            str(self._swift_product(checkout, product)),
            "process",
            str(audio),
            "--mode",
            "offline",
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
        completed = self.runner.run(command)
        wall = time.perf_counter() - started
        if completed.stderr and not getattr(completed, "stderr_streamed", False):
            self.progress.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                self.progress.write("\n")
        if completed.returncode != 0 or not raw_path.is_file():
            detail = completed.stderr.strip() or f"FluidAudio exited {completed.returncode}"
            raise StageFailure("diarizer", "fluidaudio", detail[-4000:])
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StageFailure("diarizer", "fluidaudio", f"invalid result JSON: {exc}") from exc
        self._notice("transcribe: diarizer finished")
        return StageOutcome(
            "diarizer",
            "fluidaudio",
            payload,
            round(wall, 6),
            peak_rss_bytes=getattr(completed, "peak_rss_bytes", None),
        )
