"""Fixed probe plan, provenance, and telemetry for Qwen verbatim runs."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "EXPECTED_QWEN3_ASR_SOURCE_SHA256", "FILLER_TOKENS", "FIXTURES",
    "LATIN_REPEAT_RE", "MAX_TOKENS", "MODELS", "REPO_ROOT",
    "REQUIRED_BATCHED_API_PARAMS", "RUN_PLAN", "RssReader", "SAMPLE_RATE",
    "SYSTEM_PROMPTS", "TEMPERATURE", "array_sha256", "build_fixture_info",
    "build_host_info", "count_fillers", "ffprobe", "find_repetitions",
    "hf_snapshot_path", "package_versions", "peak_rss_bytes", "sha256_bytes",
    "sha256_file", "sysctl",
]


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16_000

# Set before any mlx_audio / huggingface_hub / transformers import so cache
# resolution never touches the network, per the task brief.
for _var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_var, "1")

# Recorded by the prior capability probe
# (model_tests/benchmark/results/2026-08-16-qwen-capabilities.json,
# runtime.qwen_adapter_sha256) as the source hash of the installed adapter
# file at mlx-audio==0.4.5. We re-hash our own scratch-env install and
# compare, rather than trusting that record.
EXPECTED_QWEN3_ASR_SOURCE_SHA256 = (
    "c082690575eedcd28fb76207d032cefd7eac2f9ce5d36df5a7a06575bc45d250"
)

REQUIRED_BATCHED_API_PARAMS = {
    "chunks", "max_tokens", "sampler", "language", "system_prompt",
    "batch_size", "verbose",
}


def hf_snapshot_path(org_repo: str, revision: str) -> Path:
    org, name = org_repo.split("/", 1)
    return (
        Path.home() / ".cache" / "huggingface" / "hub"
        / f"models--{org}--{name}" / "snapshots" / revision
    )


MODELS: dict[str, dict[str, Any]] = {
    "qwen3_asr_1.7b_8bit": {
        "repo_id": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55",
    },
    "qwen3_asr_0.6b_8bit": {
        "repo_id": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "revision": "89e96d92ba34aca20b3e29fb10cc284097d1219f",
    },
}
for _spec in MODELS.values():
    _spec["path"] = hf_snapshot_path(_spec["repo_id"], _spec["revision"])

FIXTURES: dict[str, Path] = {
    "multispeaker": REPO_ROOT / "test-sample-multispeaker.m4a",
    "sichuanese_probe": REPO_ROOT / "autio-test-sample.m4a",
}

# Prompt wording is itself a variable under test: a single failed prompt
# does not prove there is no knob, so four distinct strategies are tried
# (direct instruction, editing-workflow framing, Chinese, terse imperative).
SYSTEM_PROMPTS: dict[str, str | None] = {
    "none": None,
    "verbatim_en_direct": (
        "Transcribe this audio verbatim. Include every filler word (um, uh, "
        "like), false start, repetition, and self-correction exactly as "
        "spoken. Do not clean up, paraphrase, or omit any disfluency."
    ),
    "verbatim_en_editing": (
        "This transcript will be used for downstream video editing. "
        "Preserve all hesitations, stutters, and repeated words exactly as "
        "they occur in the recording. Do not smooth, summarize, or correct "
        "the speech in any way."
    ),
    "verbatim_zh": (
        "请逐字转录这段音频，完整保留所有语气词、重复用词和说话中断"
        "（例如“嗯”、“呃”、“那个”、“就是”），"
        "不要删除、合并或改写任何内容。"
    ),
    "verbatim_en_imperative": (
        "VERBATIM MODE: ON. Output every um/uh/repetition/false start "
        "literally. No cleanup, no punctuation normalization, no summary."
    ),
}

# Explicit, stated token list -- counts are auditable per-token, not just a
# total. English tokens are matched case-insensitively on word boundaries;
# Chinese tokens are matched as raw substrings (Chinese text has no
# whitespace word boundaries, so substring counting is the standard cheap
# approach). Simplified/traditional variants are listed and counted
# separately.
FILLER_TOKENS: dict[str, list[str]] = {
    "english": ["um", "uh", "er", "erm", "hmm", "like", "you know"],
    "chinese": ["呃", "嗯", "啊", "那个", "那個", "就是", "这个", "這個"],
}

# Run plan: (model_key, audio_key, prompt_key, label)
RUN_PLAN: list[tuple[str, str, str, str]] = [
    ("qwen3_asr_1.7b_8bit", "multispeaker", "none", "baseline"),
    ("qwen3_asr_1.7b_8bit", "multispeaker", "none", "baseline_repeat"),
    ("qwen3_asr_1.7b_8bit", "multispeaker", "verbatim_en_direct", "prompt_1"),
    ("qwen3_asr_1.7b_8bit", "multispeaker", "verbatim_en_editing", "prompt_2"),
    ("qwen3_asr_1.7b_8bit", "multispeaker", "verbatim_zh", "prompt_3"),
    ("qwen3_asr_1.7b_8bit", "multispeaker", "verbatim_en_imperative", "prompt_4"),
    ("qwen3_asr_1.7b_8bit", "sichuanese_probe", "none", "supplementary_fixture2_baseline"),
    ("qwen3_asr_0.6b_8bit", "multispeaker", "none", "size_comparison_baseline"),
]

MAX_TOKENS = 8192
TEMPERATURE = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def ffprobe(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=index,codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ], text=True))
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"error": f"{type(exc).__name__}: {exc}"}


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
    """Current RSS. On macOS this is a live sample; ru_maxrss (also reported)
    is the true high-water mark and is what "peak RSS" means below."""

    def __init__(self) -> None:
        self.source = "ru_maxrss_high_water"
        self._proc_pidinfo = None
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

    def read(self) -> int:
        if self._proc_pidinfo is not None:
            info = _MacProcTaskInfo()
            returned = self._proc_pidinfo(
                os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info)
            )
            if returned == ctypes.sizeof(info):
                return int(info.resident_size)
        return peak_rss_bytes()


def peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = int(usage.ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def count_fillers(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in FILLER_TOKENS["english"]:
        pattern = r"\b" + re.escape(token) + r"\b"
        counts[token] = len(re.findall(pattern, text, flags=re.IGNORECASE))
    for token in FILLER_TOKENS["chinese"]:
        counts[token] = text.count(token)
    return counts


LATIN_REPEAT_RE = re.compile(
    r"\b([a-zA-Z']+)\b([\s,.’‘-]{1,3})\1\b", flags=re.IGNORECASE
)


def find_repetitions(text: str) -> list[str]:
    """Immediate consecutive repeated Latin word tokens (e.g. 'for for',
    "that's that's"). Deliberately Latin-only: Chinese has no whitespace
    word boundaries, so a naive \\w+ repeat regex over Han text would just
    grab arbitrary character runs, not meaningful repeats. Chinese
    repetition is instead eyeballed qualitatively in the report."""
    return [m.group(0) for m in LATIN_REPEAT_RE.finditer(text)]


def sysctl(name: str) -> str | None:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return None


def build_host_info() -> dict[str, Any]:
    mem = sysctl("hw.memsize")
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version,
        "hw_model": sysctl("hw.model"),
        "physical_memory_bytes": int(mem) if mem else None,
    }


def build_fixture_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "ffprobe": ffprobe(path),
    }
