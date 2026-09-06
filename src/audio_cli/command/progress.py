"""Opt-in stderr lifecycle rendering and elapsed heartbeats for blocking work."""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Literal, TextIO


class ProgressReporter:
    """Context-managed callable sink; nested events follow a synchronous work stack.

    Keep the context around the entire operation, including exceptions. The timer
    only renders elapsed time; it never invokes feature code or predicts completion.
    """

    def __init__(
        self,
        label: str,
        *,
        stream: TextIO | None = None,
        heartbeat_seconds: float = 10.0,
    ) -> None:
        if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be finite and positive")
        self.label = label
        self.stream = sys.stderr if stream is None else stream
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active: list[tuple[str, float]] = []
        self._thread: threading.Thread | None = None
        self._closed = False
        self._stream_failed = False

    def __enter__(self) -> ProgressReporter:
        if self._thread is not None or self._closed:
            raise RuntimeError("progress reporter contexts cannot be reused")
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def _write(self, message: str) -> None:
        if self._stream_failed:
            return
        try:
            self.stream.write(f"{self.label}: {message}\n")
            self.stream.flush()
        except (OSError, ValueError):
            # A closed/broken presentation stream must not change processing results.
            self._stream_failed = True

    def __call__(self, stage: str, status: Literal["started", "finished", "failed"]) -> None:
        with self._lock:
            if self._thread is None or self._closed:
                raise RuntimeError("progress reporter requires an open context")
            if status == "started":
                self._active.append((stage, time.monotonic()))
                self._write(f"{stage} started")
            elif status in {"finished", "failed"}:
                if not self._active or self._active[-1][0] != stage:
                    raise ValueError(f"unmatched progress completion: {stage}")
                _, started = self._active.pop()
                self._write(f"{stage} {status}; elapsed {time.monotonic() - started:.1f}s")
            else:
                raise ValueError(f"unknown progress status: {status}")

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            with self._lock:
                if self._active and not self._closed:
                    stage, started = self._active[-1]
                    self._write(f"{stage} running; elapsed {time.monotonic() - started:.1f}s")

    def close(self) -> None:
        """Stop and join the timer; after return it cannot write further notices."""
        with self._lock:
            self._closed = True
            self._active.clear()
            self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
