"""CLI parsing, hashing, and process telemetry for the turn-attributed MLX benchmark."""

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
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "RssReader",
    "array_sha256",
    "ffprobe",
    "mac_swap_snapshot",
    "nonnegative_float",
    "normalized_ru_maxrss",
    "package_versions",
    "parse_args",
    "positive_float",
    "positive_int",
    "sha256",
    "snapshot_revision",
    "stable_json_sha256",
]


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FluidAudio anonymous speaker turns followed by one "
            "persistent, batched Qwen3-ASR MLX worker"
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio", required=True, help="Canonical 16 kHz mono mix")
    parser.add_argument("--diarization-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="Cantonese")
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--max-tokens", type=positive_int, default=16_384)
    parser.add_argument(
        "--duration-limit",
        type=positive_float,
        help="Optional leading-prefix smoke-test duration in seconds",
    )
    parser.add_argument(
        "--raw-fragment-min-seconds",
        type=nonnegative_float,
        default=0.250,
    )
    parser.add_argument(
        "--merge-silence-max-seconds",
        type=nonnegative_float,
        default=0.300,
    )
    parser.add_argument(
        "--asr-turn-min-seconds",
        type=positive_float,
        default=0.500,
    )
    parser.add_argument("--sample-interval", type=positive_float, default=0.1)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and write the deterministic turn plan without loading MLX",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: Any) -> str:
    contiguous = value if value.flags.c_contiguous else value.copy(order="C")
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def ffprobe(path: Path) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=index,codec_name,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def snapshot_revision(model_path: Path) -> str | None:
    if model_path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", model_path.name):
        return model_path.name.lower()
    return None


def mac_swap_snapshot() -> dict[str, int] | None:
    if sys.platform != "darwin":
        return None
    try:
        raw = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None

    def parse(label: str) -> int:
        match = re.search(rf"\b{label}\s*=\s*([0-9.]+)([KMGT])", raw, re.IGNORECASE)
        if not match:
            raise ValueError(label)
        scale = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
        return round(float(match.group(1)) * scale[match.group(2).upper()])

    try:
        return {f"{name}_bytes": parse(name) for name in ("total", "used", "free")}
    except ValueError:
        return None


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
    def __init__(self) -> None:
        self.source = "ru_maxrss_high_water"
        self._proc_pidinfo: Callable[..., int] | None = None
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
                self.source = "macos_proc_pidinfo"
            except OSError:
                pass

    def read(self) -> int:
        if self._proc_pidinfo is not None:
            info = _MacProcTaskInfo()
            returned = self._proc_pidinfo(
                os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info)
            )
            if returned == ctypes.sizeof(info):
                return int(info.resident_size)
        return normalized_ru_maxrss(resource.getrusage(resource.RUSAGE_SELF))
