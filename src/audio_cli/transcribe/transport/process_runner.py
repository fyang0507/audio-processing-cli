"""Fresh-process execution with retained raw diagnostics and child RSS sampling."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from audio_cli.media import retained_diagnostics_directory, temporary_directory


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
                function = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pidinfo
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
            returned = self._proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
            if returned == ctypes.sizeof(info):
                return int(info.resident_size)
        try:
            resident_pages = int(Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None


class SubprocessRunner:
    def __init__(
        self, progress, *, heartbeat_seconds: float = 10.0, log_root: Path | None = None
    ) -> None:
        self.progress = progress
        self.heartbeat_seconds = heartbeat_seconds
        # Validate requested storage before decode/model execution, with no fallback.
        self.log_directory = (
            retained_diagnostics_directory(log_root, prefix="audio-transcribe-")
            if log_root is not None
            else None
        )
        self._stage_number = 0

    def _notice(self, message: str) -> None:
        self.progress.write(f"transcribe: host: {message}\n")
        self.progress.flush()

    def run(self, command: list[str], *, stage: str = "child") -> subprocess.CompletedProcess[str]:
        if not stage or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in stage):
            raise ValueError("diagnostic stage must contain only lowercase letters, digits, _ or -")
        try:
            return self._run(command, stage)
        except OSError as exc:
            return subprocess.CompletedProcess(
                command, 127, "", f"cannot retain backend diagnostics: {exc}"
            )

    def _run(self, command: list[str], stage: str) -> subprocess.CompletedProcess[str]:
        self._stage_number += 1
        storage = (
            nullcontext(
                retained_diagnostics_directory(
                    self.log_directory, prefix=f"{self._stage_number:03d}-{stage}-"
                )
            )
            if self.log_directory is not None
            else temporary_directory("audio-transcribe-", preserve=True)
        )
        with storage as directory:
            stdout_path = directory / "stdout.log"
            stderr_path = directory / "stderr.log"
            peak_rss: int | None = None
            # Exclusive files inside a media-owned private directory avoid public-path
            # replacement and pipe deadlocks. The child writes bytes directly to disk.
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                self._notice(f"raw backend stdout: {stdout_path}")
                self._notice(f"raw backend stderr: {stderr_path}")
                started = time.monotonic()
                try:
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                except OSError as exc:
                    stderr.write(str(exc).encode("utf-8"))
                    returncode = 127
                else:
                    reader = _ChildRssReader()
                    next_notice = self.heartbeat_seconds
                    try:
                        while process.poll() is None:
                            sample = reader.read(process.pid)
                            if sample is not None:
                                peak_rss = sample if peak_rss is None else max(peak_rss, sample)
                            elapsed = time.monotonic() - started
                            if elapsed >= next_notice:
                                self._notice(f"child running; elapsed {elapsed:.1f}s")
                                next_notice = elapsed + self.heartbeat_seconds
                            time.sleep(0.01)
                        returncode = process.returncode
                    except BaseException:
                        # Reap the direct child before closing the log handles. Even an
                        # interrupted stage leaves its bytes at the announced paths.
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        self._notice("child interrupted; raw diagnostics retained")
                        raise
                stdout.flush()
                stderr.flush()
            elapsed = time.monotonic() - started
            self._notice(f"child exited {returncode}; elapsed {elapsed:.1f}s")
            completed = subprocess.CompletedProcess(
                command,
                returncode,
                stdout_path.read_bytes().decode("utf-8", errors="replace"),
                stderr_path.read_bytes().decode("utf-8", errors="replace"),
            )
            completed.peak_rss_bytes = peak_rss  # type: ignore[attr-defined]
            completed.stderr_preserved = True  # type: ignore[attr-defined]
            completed.diagnostics_directory = directory  # type: ignore[attr-defined]
            return completed
