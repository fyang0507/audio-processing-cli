"""Fresh-process execution with streamed progress and child RSS sampling."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


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
