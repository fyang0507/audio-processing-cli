"""Argument, normalization, and telemetry helpers for the MLX-ASR runner."""

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
from pathlib import Path
from typing import Any, Callable


__all__ = [
    "RssReader", "array_sha256", "command_version", "detect_family", "ffprobe",
    "mac_swap_snapshot", "monotonic_segments", "normalize_segments",
    "normalized_ru_maxrss", "package_versions", "parse_args",
    "physical_memory_bytes", "positive_float", "positive_int", "sha256",
    "snapshot_revision", "stable_json_sha256",
]


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one already-cached Qwen3-ASR or Whisper MLX snapshot "
            "and write a scorer-compatible JSON evidence artifact."
        )
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Existing local snapshot directory; Hub IDs are rejected",
    )
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--output", required=True, help="Evidence JSON path")
    parser.add_argument(
        "--family", choices=("auto", "qwen3-asr", "whisper"), default="auto",
    )
    parser.add_argument(
        "--language", required=True,
        help=(
            "Qwen language name (for example Cantonese), 'auto' for Qwen "
            "automatic language identification, or a Whisper code (yue)"
        ),
    )
    parser.add_argument(
        "--qwen-chunk-seconds", type=positive_float, default=180.0,
        help="Bounded long-form chunk size for Qwen3-ASR (default: %(default)s)",
    )
    parser.add_argument(
        "--qwen-batch-size", type=positive_int, default=1,
        help="Maximum parallel Qwen chunks (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens", type=positive_int, default=16384,
        help="Global Qwen generation budget (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper-word-timestamps", action="store_true",
        help="Request costlier Whisper word timestamps in addition to segments",
    )
    parser.add_argument(
        "--sample-interval", type=positive_float, default=0.1,
        help="Resource sample interval in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--target-rtf", type=positive_float, default=1 / 6,
        help="Decision threshold recorded in the artifact (default: 1/6)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    contiguous = value if value.flags.c_contiguous else value.copy(order="C")
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def command_version(command: list[str]) -> str | None:
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0] if output else None


def ffprobe(path: Path) -> dict[str, Any]:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ], text=True))


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
    """Read current-process RSS without adding a runtime dependency."""

    def __init__(self) -> None:
        self.source = "ru_maxrss_high_water"
        self._proc_pidinfo: Callable[..., int] | None = None
        if sys.platform == "darwin":
            try:
                function = ctypes.CDLL(
                    "/usr/lib/libproc.dylib", use_errno=True
                ).proc_pidinfo
                function.argtypes = [
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ctypes.c_void_p, ctypes.c_int,
                ]
                function.restype = ctypes.c_int
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
                pages = int(Path("/proc/self/statm").read_text().split()[1])
                return pages * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
        return normalized_ru_maxrss(resource.getrusage(resource.RUSAGE_SELF))


def mac_swap_snapshot() -> dict[str, int] | None:
    if sys.platform != "darwin":
        return None
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    def parse(label: str) -> int:
        match = re.search(
            rf"\b{label}\s*=\s*([0-9.]+)([KMGT])", raw, re.IGNORECASE
        )
        if not match:
            raise ValueError(f"missing {label} in vm.swapusage")
        scale = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
        return round(float(match.group(1)) * scale[match.group(2).upper()])

    try:
        return {f"{name}_bytes": parse(name) for name in ("total", "used", "free")}
    except ValueError:
        return None


def physical_memory_bytes() -> int | None:
    try:
        return int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def snapshot_revision(model_path: Path) -> str | None:
    if model_path.parent.name == "snapshots" and re.fullmatch(
        r"[0-9a-fA-F]{40}", model_path.name
    ):
        return model_path.name.lower()
    return None


def detect_family(config: dict[str, Any]) -> str:
    model_type = str(config.get("model_type", "")).lower()
    if model_type == "qwen3_asr":
        return "qwen3-asr"
    if model_type == "whisper":
        return "whisper"
    raise ValueError(f"unsupported model_type in config.json: {model_type!r}")


def normalize_segments(
    raw_segments: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized = []
    for segment in raw_segments or []:
        item: dict[str, Any] = {
            "start_s": float(segment.get("start", segment.get("start_s", 0.0))),
            "end_s": float(segment.get("end", segment.get("end_s", 0.0))),
            "text": str(segment.get("text", "")),
            "speaker": None,
        }
        words = segment.get("words")
        if words:
            item["words"] = [{
                "start_s": float(word["start"]),
                "end_s": float(word["end"]),
                "text": str(word.get("word", word.get("text", ""))),
            } for word in words]
        normalized.append(item)
    return normalized


def monotonic_segments(segments: list[dict[str, Any]]) -> bool:
    previous_start = float("-inf")
    for segment in segments:
        start = float(segment["start_s"])
        end = float(segment["end_s"])
        if start < previous_start or end < start:
            return False
        previous_start = start
    return True


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
