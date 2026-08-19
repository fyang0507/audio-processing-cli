#!/usr/bin/env python3
"""Ad hoc probe: does Qwen3-ASR retain disfluencies, and is system_prompt a knob?

This is a one-shot experiment script, not part of the production benchmark
suite. It exists to answer two open questions about the `verbatim` capability
declared in the transcription spec ("disfluency- and dialect-form-preserving
text"), which a prior capability probe
(model_tests/benchmark/results/2026-08-16-qwen-capabilities.json) explicitly
left unanswered:

    Q1. Does Qwen3-ASR's default output (system_prompt=None) already retain
        fillers, repetitions, and false starts, without being asked?
    Q2. Does passing a verbatim-requesting `system_prompt` to the private
        batched-inference API measurably change filler retention? If yes,
        `verbatim` is a real configuration switch on this stack. If no, it is
        only a property of whatever the model happens to emit.

Call shape
----------
Model load and the private batched-inference call are reused verbatim in
spirit from model_tests/benchmark/run_turn_attributed_mlx_asr.py (see its
~lines 545-680): `mlx_audio.stt.utils.load_model` followed by
`model._generate_chunks_batched(chunks, max_tokens=..., sampler=make_sampler(
temp=0.0), language=..., system_prompt=..., batch_size=..., verbose=False)`.
That runner verifies the private method's signature before use; this script
does the same, and additionally hashes the installed
`mlx_audio.stt.models.qwen3_asr.qwen3_asr` source file against the sha256
recorded by the prior capability probe.

Unlike run_turn_attributed_mlx_asr.py, there is no FluidAudio diarization
plan feeding this script: each fixture is passed as a single whole-clip
chunk. That matches how the prior capability probe also treated this same
139.284s clip as one chunk (it used chunk_duration=180s, i.e. no split).

Environment
-----------
mlx_audio is intentionally NOT installed in this repo's `.venv`. Run this
script from a throwaway scratch env pinned to mlx-audio==0.4.5 (the pin
matters: the private batched API only matches the runner's assumptions at
that version):

    uv venv /tmp/mlxprobe-venv
    uv pip install --python /tmp/mlxprobe-venv/bin/python "mlx-audio==0.4.5"
    HF_HUB_OFFLINE=1 /tmp/mlxprobe-venv/bin/python \
        model_tests/benchmark/run_qwen_verbatim_probe.py

Output
------
Writes one combined JSON artifact under model_tests/benchmark_runs/
(untracked directory) containing every run's full output text, decode
config, hashes, timing, and memory. Nothing here modifies any tracked file.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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


def main() -> int:
    script_start = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": (
            "Determine whether Qwen3-ASR's default output retains "
            "disfluencies (Q1) and whether mlx-audio's private "
            "system_prompt argument is a usable verbatim control (Q2)."
        ),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "host": build_host_info(),
        "decode_config": {
            "temperature": TEMPERATURE,
            "sampler": "mlx_lm.sample_utils.make_sampler(temp=0.0)",
            "language_argument": None,
            "language_argument_note": (
                "language=None (the no-hint path) is held constant across "
                "every run in this probe so language hinting is not a "
                "confound in the system_prompt comparison."
            ),
            "max_tokens": MAX_TOKENS,
            "batch_size": 1,
            "chunking": (
                "whole-clip single chunk per call; no FluidAudio diarization "
                "plan feeds this script, unlike run_turn_attributed_mlx_asr.py"
            ),
        },
        "system_prompts_tested": SYSTEM_PROMPTS,
        "filler_token_list": FILLER_TOKENS,
        "repetition_method": (
            "regex on cleaned text: \\b([a-zA-Z']+)\\b[\\s,.\\u2018\\u2019-]"
            "{1,3}\\1\\b case-insensitive, immediate consecutive repeats of "
            "a Latin word token only. Chinese repetition is not "
            "regex-counted; see the report's qualitative notes."
        ),
        "runtime_packages": package_versions([
            "mlx", "mlx-metal", "mlx-audio", "mlx-lm", "numpy", "miniaudio",
        ]),
        "offline_environment": {
            name: os.environ.get(name) for name in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            )
        },
        "fixtures": {key: build_fixture_info(path) for key, path in FIXTURES.items()},
        "models": {
            key: {
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "path": str(spec["path"]),
                "snapshot_exists": spec["path"].is_dir(),
            }
            for key, spec in MODELS.items()
        },
        "api_probe": None,
        "runs": [],
        "fatal_error": None,
        "timing": {},
        "memory": {},
    }

    rss_reader = RssReader()

    try:
        import numpy as np
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_audio.stt.utils import load_model, load_audio
        from mlx_lm.sample_utils import make_sampler

        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal device is unavailable")

        # --- Decode each existing fixture once, reused across every run ---
        prepared_audio: dict[str, Any] = {}
        for key, path in FIXTURES.items():
            info = result["fixtures"][key]
            if not info["exists"]:
                info["prepared_audio_error"] = "fixture file not found"
                continue
            t0 = time.perf_counter()
            try:
                audio_mx = load_audio(str(path), sr=SAMPLE_RATE)
                audio_np = np.ascontiguousarray(np.array(audio_mx, dtype=np.float32))
            except Exception as exc:
                info["prepared_audio_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[fixture:{key}] decode FAILED: {exc}", flush=True)
                continue
            prepared_audio[key] = audio_np
            info["prepared_audio_sha256"] = array_sha256(audio_np)
            info["prepared_audio_samples"] = int(len(audio_np))
            info["prepared_audio_duration_s"] = len(audio_np) / SAMPLE_RATE
            info["prepared_audio_decode_wall_s"] = time.perf_counter() - t0
            print(
                f"[fixture:{key}] decoded {len(audio_np)/SAMPLE_RATE:.3f}s "
                f"in {info['prepared_audio_decode_wall_s']:.2f}s "
                f"sha256={info['prepared_audio_sha256'][:12]}...",
                flush=True,
            )

        # --- Run each model in the plan once, executing all its runs ---
        model_keys_in_order = list(dict.fromkeys(item[0] for item in RUN_PLAN))
        for model_key in model_keys_in_order:
            spec = MODELS[model_key]
            model_runs = [item for item in RUN_PLAN if item[0] == model_key]

            if not spec["path"].is_dir():
                for _, audio_key, prompt_key, label in model_runs:
                    result["runs"].append({
                        "model_key": model_key, "audio_key": audio_key,
                        "prompt_key": prompt_key, "label": label,
                        "status": "error",
                        "error": f"local model snapshot not found: {spec['path']}",
                    })
                print(f"[model:{model_key}] SKIPPED: snapshot dir missing", flush=True)
                continue

            print(f"[model:{model_key}] loading from {spec['path']}", flush=True)
            try:
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                model = load_model(str(spec["path"]), lazy=False, strict=False)
                mx.eval(model.parameters())
                mx.synchronize()
                load_s = time.perf_counter() - t0
            except Exception as exc:
                for _, audio_key, prompt_key, label in model_runs:
                    result["runs"].append({
                        "model_key": model_key, "audio_key": audio_key,
                        "prompt_key": prompt_key, "label": label,
                        "status": "error", "phase": "model_load",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    })
                print(f"[model:{model_key}] LOAD FAILED: {exc}", flush=True)
                continue

            if result["api_probe"] is None:
                method = model._generate_chunks_batched
                signature = inspect.signature(method)
                signature_ok = REQUIRED_BATCHED_API_PARAMS.issubset(signature.parameters)
                source_path = Path(inspect.getfile(type(model))).resolve()
                source_hash = sha256_file(source_path)
                result["api_probe"] = {
                    "model_generate_signature": str(inspect.signature(model.generate)),
                    "private_batched_method": "_generate_chunks_batched",
                    "private_batched_signature": str(signature),
                    "signature_matches_runner_contract": signature_ok,
                    "source_path": str(source_path),
                    "source_sha256": source_hash,
                    "expected_source_sha256": EXPECTED_QWEN3_ASR_SOURCE_SHA256,
                    "source_sha256_matches_expected": source_hash == EXPECTED_QWEN3_ASR_SOURCE_SHA256,
                }
                print(
                    f"[api_probe] signature_ok={signature_ok} "
                    f"source_sha256_matches_expected="
                    f"{result['api_probe']['source_sha256_matches_expected']}",
                    flush=True,
                )
                if not signature_ok:
                    raise RuntimeError(
                        "installed mlx-audio private batched API does not "
                        f"match probe assumptions: {signature}"
                    )

            weights_path = spec["path"] / "model.safetensors"
            config_path = spec["path"] / "config.json"
            model_parameter_bytes = sum(
                int(value.nbytes) for _, value in tree_flatten(model.parameters())
            )
            model_info = {
                "load_s": load_s,
                "weight_sha256": sha256_file(weights_path) if weights_path.is_file() else None,
                "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
                "loaded_parameter_bytes": model_parameter_bytes,
                "mlx_peak_active_bytes_after_load": int(mx.get_peak_memory()),
            }
            result["models"][model_key].update(model_info)
            print(
                f"[model:{model_key}] loaded in {load_s:.2f}s, "
                f"{model_parameter_bytes / 1e6:.1f}MB params",
                flush=True,
            )

            for _, audio_key, prompt_key, label in model_runs:
                audio_np = prepared_audio.get(audio_key)
                if audio_np is None:
                    result["runs"].append({
                        "model_key": model_key, "audio_key": audio_key,
                        "prompt_key": prompt_key, "label": label,
                        "status": "error",
                        "error": "prepared audio unavailable for this fixture",
                    })
                    continue

                system_prompt = SYSTEM_PROMPTS[prompt_key]
                run_record: dict[str, Any] = {
                    "model_key": model_key, "audio_key": audio_key,
                    "prompt_key": prompt_key, "label": label,
                    "system_prompt": system_prompt,
                    "language_argument": None,
                    "max_tokens": MAX_TOKENS,
                    "batch_size": 1,
                    "temperature": TEMPERATURE,
                }
                try:
                    mx.reset_peak_memory()
                    t0 = time.perf_counter()
                    chunks = [(audio_np, 0.0)]
                    texts, gen_tokens, prompt_tokens, processed = model._generate_chunks_batched(
                        chunks,
                        max_tokens=MAX_TOKENS,
                        sampler=make_sampler(temp=TEMPERATURE),
                        language=None,
                        system_prompt=system_prompt,
                        batch_size=1,
                        verbose=False,
                    )
                    mx.synchronize()
                    wall_s = time.perf_counter() - t0

                    raw_text = texts[0]
                    was_processed = bool(processed[0])
                    has_language_prefix = (
                        raw_text.startswith("language ") and "<asr_text>" in raw_text
                    )
                    detected_language, clean_text = model.extract_language(raw_text)
                    filler_counts = count_fillers(clean_text)
                    repeats = find_repetitions(clean_text)

                    run_record.update({
                        "status": "ok" if was_processed else "not_processed_budget_exhausted",
                        "processed": was_processed,
                        "prompt_tokens": int(prompt_tokens[0]),
                        "generation_tokens": int(gen_tokens[0]),
                        "wall_s": wall_s,
                        "raw_text": raw_text,
                        "raw_text_had_language_prefix": has_language_prefix,
                        "detected_language": detected_language,
                        "text": clean_text,
                        "text_sha256": sha256_bytes(clean_text.encode("utf-8")),
                        "filler_counts": filler_counts,
                        "filler_total": sum(filler_counts.values()),
                        "repetitions_detected": repeats,
                        "repetition_count": len(repeats),
                        "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                        "mlx_active_bytes_after": int(mx.get_active_memory()),
                        "mlx_cache_bytes_after": int(mx.get_cache_memory()),
                        "rss_high_water_bytes_so_far": peak_rss_bytes(),
                        "rss_source": rss_reader.source,
                    })
                    print(
                        f"[run] {model_key} / {audio_key} / {prompt_key} "
                        f"({label}): {wall_s:.2f}s, {gen_tokens[0]} tokens, "
                        f"fillers={run_record['filler_total']}, "
                        f"lang={detected_language}",
                        flush=True,
                    )
                except Exception as exc:
                    run_record.update({
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    })
                    print(
                        f"[run] {model_key} / {audio_key} / {prompt_key} "
                        f"({label}): FAILED: {exc}",
                        flush=True,
                    )
                mx.clear_cache()
                result["runs"].append(run_record)

            del model
            gc.collect()
            mx.clear_cache()
            mx.synchronize()

    except Exception as exc:
        result["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[FATAL] {exc}", flush=True)

    finally:
        result["timing"]["total_wall_s"] = time.perf_counter() - script_start
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["memory"] = {
            "rss_source": rss_reader.source,
            "ru_maxrss_bytes_final": peak_rss_bytes(),
            "ru_utime_s": usage.ru_utime,
            "ru_stime_s": usage.ru_stime,
        }
        output_path = (
            REPO_ROOT / "model_tests" / "benchmark_runs"
            / "qwen_verbatim_probe_multispeaker_20260817.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {output_path}", flush=True)
        print(f"artifact sha256={sha256_file(output_path)}", flush=True)

    ok_runs = sum(1 for r in result["runs"] if r.get("status") == "ok")
    return 0 if result["fatal_error"] is None and ok_runs > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
