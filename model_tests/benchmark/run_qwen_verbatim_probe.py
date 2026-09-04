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

import gc
import inspect
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from . import _qwen_verbatim_probe_support as _support
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _qwen_verbatim_probe_support as _support


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
            "sha256": _support.sha256_file(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "host": _support.build_host_info(),
        "decode_config": {
            "temperature": _support.TEMPERATURE,
            "sampler": "mlx_lm.sample_utils.make_sampler(temp=0.0)",
            "language_argument": None,
            "language_argument_note": (
                "language=None (the no-hint path) is held constant across "
                "every run in this probe so language hinting is not a "
                "confound in the system_prompt comparison."
            ),
            "max_tokens": _support.MAX_TOKENS,
            "batch_size": 1,
            "chunking": (
                "whole-clip single chunk per call; no FluidAudio diarization "
                "plan feeds this script, unlike run_turn_attributed_mlx_asr.py"
            ),
        },
        "system_prompts_tested": _support.SYSTEM_PROMPTS,
        "filler_token_list": _support.FILLER_TOKENS,
        "repetition_method": (
            "regex on cleaned text: \\b([a-zA-Z']+)\\b[\\s,.\\u2018\\u2019-]"
            "{1,3}\\1\\b case-insensitive, immediate consecutive repeats of "
            "a Latin word token only. Chinese repetition is not "
            "regex-counted; see the report's qualitative notes."
        ),
        "runtime_packages": _support.package_versions([
            "mlx", "mlx-metal", "mlx-audio", "mlx-lm", "numpy", "miniaudio",
        ]),
        "offline_environment": {
            name: os.environ.get(name) for name in (
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            )
        },
        "fixtures": {
            key: _support.build_fixture_info(path)
            for key, path in _support.FIXTURES.items()
        },
        "models": {
            key: {
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "path": str(spec["path"]),
                "snapshot_exists": spec["path"].is_dir(),
            }
            for key, spec in _support.MODELS.items()
        },
        "api_probe": None,
        "runs": [],
        "fatal_error": None,
        "timing": {},
        "memory": {},
    }

    rss_reader = _support.RssReader()

    try:
        import mlx.core as mx
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_audio.stt.utils import load_audio, load_model
        from mlx_lm.sample_utils import make_sampler

        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal device is unavailable")

        # --- Decode each existing fixture once, reused across every run ---
        prepared_audio: dict[str, Any] = {}
        for key, path in _support.FIXTURES.items():
            info = result["fixtures"][key]
            if not info["exists"]:
                info["prepared_audio_error"] = "fixture file not found"
                continue
            t0 = time.perf_counter()
            try:
                audio_mx = load_audio(str(path), sr=_support.SAMPLE_RATE)
                audio_np = np.ascontiguousarray(np.array(audio_mx, dtype=np.float32))
            except Exception as exc:
                info["prepared_audio_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[fixture:{key}] decode FAILED: {exc}", flush=True)
                continue
            prepared_audio[key] = audio_np
            info["prepared_audio_sha256"] = _support.array_sha256(audio_np)
            info["prepared_audio_samples"] = int(len(audio_np))
            info["prepared_audio_duration_s"] = len(audio_np) / _support.SAMPLE_RATE
            info["prepared_audio_decode_wall_s"] = time.perf_counter() - t0
            print(
                f"[fixture:{key}] decoded {len(audio_np)/_support.SAMPLE_RATE:.3f}s "
                f"in {info['prepared_audio_decode_wall_s']:.2f}s "
                f"sha256={info['prepared_audio_sha256'][:12]}...",
                flush=True,
            )

        # --- Run each model in the plan once, executing all its runs ---
        model_keys_in_order = list(
            dict.fromkeys(item[0] for item in _support.RUN_PLAN)
        )
        for model_key in model_keys_in_order:
            spec = _support.MODELS[model_key]
            model_runs = [
                item for item in _support.RUN_PLAN if item[0] == model_key
            ]

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
                signature_ok = _support.REQUIRED_BATCHED_API_PARAMS.issubset(
                    signature.parameters
                )
                source_path = Path(inspect.getfile(type(model))).resolve()
                source_hash = _support.sha256_file(source_path)
                result["api_probe"] = {
                    "model_generate_signature": str(inspect.signature(model.generate)),
                    "private_batched_method": "_generate_chunks_batched",
                    "private_batched_signature": str(signature),
                    "signature_matches_runner_contract": signature_ok,
                    "source_path": str(source_path),
                    "source_sha256": source_hash,
                    "expected_source_sha256": (
                        _support.EXPECTED_QWEN3_ASR_SOURCE_SHA256
                    ),
                    "source_sha256_matches_expected": (
                        source_hash == _support.EXPECTED_QWEN3_ASR_SOURCE_SHA256
                    ),
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
                "weight_sha256": (
                    _support.sha256_file(weights_path)
                    if weights_path.is_file() else None
                ),
                "config_sha256": (
                    _support.sha256_file(config_path)
                    if config_path.is_file() else None
                ),
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

                system_prompt = _support.SYSTEM_PROMPTS[prompt_key]
                run_record: dict[str, Any] = {
                    "model_key": model_key, "audio_key": audio_key,
                    "prompt_key": prompt_key, "label": label,
                    "system_prompt": system_prompt,
                    "language_argument": None,
                    "max_tokens": _support.MAX_TOKENS,
                    "batch_size": 1,
                    "temperature": _support.TEMPERATURE,
                }
                try:
                    mx.reset_peak_memory()
                    t0 = time.perf_counter()
                    chunks = [(audio_np, 0.0)]
                    texts, gen_tokens, prompt_tokens, processed = model._generate_chunks_batched(
                        chunks,
                        max_tokens=_support.MAX_TOKENS,
                        sampler=make_sampler(temp=_support.TEMPERATURE),
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
                    filler_counts = _support.count_fillers(clean_text)
                    repeats = _support.find_repetitions(clean_text)

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
                        "text_sha256": _support.sha256_bytes(
                            clean_text.encode("utf-8")
                        ),
                        "filler_counts": filler_counts,
                        "filler_total": sum(filler_counts.values()),
                        "repetitions_detected": repeats,
                        "repetition_count": len(repeats),
                        "mlx_peak_active_bytes": int(mx.get_peak_memory()),
                        "mlx_active_bytes_after": int(mx.get_active_memory()),
                        "mlx_cache_bytes_after": int(mx.get_cache_memory()),
                        "rss_high_water_bytes_so_far": _support.peak_rss_bytes(),
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
            "ru_maxrss_bytes_final": _support.peak_rss_bytes(),
            "ru_utime_s": usage.ru_utime,
            "ru_stime_s": usage.ru_stime,
        }
        output_path = (
            _support.REPO_ROOT / "model_tests" / "benchmark_runs"
            / "qwen_verbatim_probe_multispeaker_20260817.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {output_path}", flush=True)
        print(f"artifact sha256={_support.sha256_file(output_path)}", flush=True)

    ok_runs = sum(1 for r in result["runs"] if r.get("status") == "ok")
    return 0 if result["fatal_error"] is None and ok_runs > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
