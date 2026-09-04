"""Argument, provenance, and telemetry helpers for the FireRed runner."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import re
import resource
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "RssReader",
    "command_version",
    "ffprobe",
    "git_metadata",
    "huggingface_revision",
    "json_default",
    "monotonic",
    "normalized_ru_maxrss",
    "package_versions",
    "parse_args",
    "positive_float",
    "positive_int",
    "sanitize_uttid",
    "sha256",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    default_checkout = repo / "model_tests/firered/FireRedASR2S"
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the CPU FireRedASR2S pipeline in a fresh process and "
            "write a self-contained JSON evidence artifact."
        )
    )
    parser.add_argument(
        "--firered-root",
        default=str(default_checkout),
        help="FireRedASR2S source checkout (default: %(default)s)",
    )
    parser.add_argument("--audio", required=True, help="Source audio or video")
    parser.add_argument("--output", required=True, help="Result JSON path")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--lid",
        choices=("on", "off"),
        required=True,
        help="Load and run FireRedLID for every VAD batch",
    )
    parser.add_argument("--asr-batch-size", type=positive_int, default=1)
    parser.add_argument("--punc-batch-size", type=positive_int, default=1)
    parser.add_argument("--sample-interval", type=positive_float, default=0.2)
    parser.add_argument(
        "--uttid",
        help="Stable utterance ID (default: sanitized source filename stem)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=index,codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def command_version(command: list[str]) -> str | None:
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0] if output else None


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"path": str(path), "commit": None, "tracked_dirty": None}
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        metadata["tracked_dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def huggingface_revision(path: Path) -> dict[str, Any]:
    """Recover the pinned revision written by huggingface-cli download."""
    cache_root: Path | None = None
    for candidate in [path, *list(path.parents)[:3]]:
        possible = candidate / ".cache/huggingface/download"
        if possible.is_dir():
            cache_root = possible
            break
    revisions: set[str] = set()
    if cache_root is not None:
        for metadata_path in cache_root.rglob("*.metadata"):
            try:
                first_line = (
                    metadata_path.read_text(encoding="utf-8", errors="replace")
                    .splitlines()[0]
                    .strip()
                )
            except (OSError, IndexError):
                continue
            if re.fullmatch(r"[0-9a-fA-F]{40}", first_line):
                revisions.add(first_line.lower())
    return {
        "path": str(path),
        "revision_cache_root": str(cache_root.parent.parent) if cache_root is not None else None,
        "revisions": sorted(revisions),
    }


def normalized_ru_maxrss(usage: resource.struct_rusage) -> int:
    value = int(usage.ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


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


class RssReader:
    """Read current process RSS without adding a benchmark dependency."""

    def __init__(self) -> None:
        self.source = "ru_maxrss_high_water"
        self._library: ctypes.CDLL | None = None
        self._proc_pidinfo: Callable[..., int] | None = None
        if sys.platform == "darwin":
            try:
                library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
                function = library.proc_pidinfo
                function.argtypes = [
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_uint64,
                    ctypes.c_void_p,
                    ctypes.c_int,
                ]
                function.restype = ctypes.c_int
                self._library = library
                self._proc_pidinfo = function
                self.source = "macos_proc_pidinfo"
            except OSError:
                pass
        elif Path("/proc/self/statm").is_file():
            self.source = "linux_proc_statm"

    def read(self) -> int:
        if self._proc_pidinfo is not None:
            info = _MacProcTaskInfo()
            returned = self._proc_pidinfo(
                os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info)
            )
            if returned == ctypes.sizeof(info):
                return int(info.resident_size)
        if self.source == "linux_proc_statm":
            try:
                resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
                return resident_pages * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
        return normalized_ru_maxrss(resource.getrusage(resource.RUSAGE_SELF))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def monotonic(items: list[dict[str, Any]], start_key: str, end_key: str) -> bool:
    previous_start = float("-inf")
    for item in items:
        start = float(item[start_key])
        end = float(item[end_key])
        if start < previous_start or end < start:
            return False
        previous_start = start
    return True


def sanitize_uttid(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return sanitized or "benchmark"
