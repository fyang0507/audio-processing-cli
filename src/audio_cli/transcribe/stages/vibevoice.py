"""Execute one whole-media VibeVoice request without importing the core CLI package."""

from __future__ import annotations

import json
import os
import random
import resource
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

_CONFIG = {
    "device": "mps",
    "dtype": "bfloat16",
    "attention": "sdpa",
    "seed": 1234,
    "max_new_tokens": 16384,
}


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"VibeVoice generated JSON repeats key {key!r}")
        result[key] = value
    return result


def _validate_generated_preamble(text: str, json_start: int) -> None:
    if text[:json_start].strip() not in {"", "assistant"}:
        raise ValueError("VibeVoice generated text has an unsupported preamble")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _json_default(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    values = getattr(value, "tolist", None)
    if callable(values):
        return values()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class _MpsHighWater:
    """Sample live MPS allocation without letting telemetry abort model work."""

    def __init__(self, mps: object, *, interval_seconds: float = 0.01) -> None:
        self._mps = mps
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak: int | None = None
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            value = self._mps.current_allocated_memory()
            if isinstance(value, bool):
                return
            measured = int(value)
            if measured < 0:
                return
            with self._lock:
                self._peak = measured if self._peak is None else max(self._peak, measured)
        except Exception:  # noqa: BLE001 - telemetry cannot suppress the result
            return

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def start(self) -> None:
        with suppress(Exception):  # telemetry cannot suppress the result
            self._sample()
        thread: threading.Thread | None = None
        try:
            thread = threading.Thread(
                target=self._run,
                name="vibevoice-mps-high-water",
                daemon=True,
            )
            thread.start()
        except Exception:  # noqa: BLE001 - synchronous sampling remains available
            with suppress(Exception):  # best-effort telemetry cleanup
                self._stop.set()
                if thread is not None and thread.is_alive():
                    thread.join()
            return
        self._thread = thread

    def stop(self) -> int | None:
        thread = self._thread
        with suppress(Exception):  # telemetry cannot suppress the result
            self._sample()
        with suppress(Exception):  # telemetry cannot suppress the result
            self._stop.set()
        if thread is not None:
            with suppress(Exception):  # telemetry cannot suppress the result
                thread.join()
        with suppress(Exception):  # telemetry cannot suppress the result
            self._sample()
        try:
            with self._lock:
                return self._peak
        except Exception:  # noqa: BLE001 - telemetry cannot suppress the result
            return None


def _config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("VibeVoice config must be an object")
    found = dict(value)
    if found != _CONFIG:
        raise ValueError(f"VibeVoice config must equal {_CONFIG!r}")
    return found


def _local_directory(value: object, field: str) -> Path:
    path = Path(str(value))
    if not path.is_dir():
        raise ValueError(f"VibeVoice {field} directory does not exist: {path}")
    return path


def _complete_json_array(text: str) -> list[object]:
    """Require one complete generated JSON array; upstream otherwise collapses errors to []."""
    array_start = text.find("[")
    fence_start = text.find("```json")
    fenced = fence_start >= 0 and (array_start < 0 or fence_start < array_start)
    if fenced:
        _validate_generated_preamble(text, fence_start)
        content_start = fence_start + len("```json")
        array_start = text.find("[", content_start)
        if array_start < 0:
            raise ValueError("VibeVoice generated JSON code block has no array")
        if text[content_start:array_start].strip():
            raise ValueError("VibeVoice generated JSON code block has content before its array")
    else:
        if array_start < 0:
            raise ValueError("VibeVoice generated text has no JSON array")
        _validate_generated_preamble(text, array_start)
    value, end = json.JSONDecoder(object_pairs_hook=_reject_duplicate_json_keys).raw_decode(
        text, array_start
    )
    suffix = text[end:].strip()
    if fenced:
        if not suffix.startswith("```"):
            raise ValueError("VibeVoice generated JSON code block is incomplete")
        if suffix[len("```") :].strip():
            raise ValueError("VibeVoice generated text continues after its JSON code block")
    elif suffix:
        raise ValueError("VibeVoice generated text continues after its JSON array")
    if not isinstance(value, list):
        raise TypeError("VibeVoice generated JSON must be an array")
    return value


def main() -> int:
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    output: dict[str, Any] = {"metrics": {}}
    mps_high_water: _MpsHighWater | None = None
    try:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"

        config = _config(request.get("config"))
        checkout = _local_directory(request.get("checkout"), "checkout")
        model_path = _local_directory(request.get("model"), "model")
        tokenizer_path = _local_directory(request.get("tokenizer"), "tokenizer")
        audio_path = Path(str(request.get("audio")))
        if not audio_path.is_file():
            raise ValueError(f"VibeVoice audio file does not exist: {audio_path}")

        sys.path.insert(0, str(checkout))
        print("vibevoice stage: loading local model", file=sys.stderr, flush=True)
        import numpy as np
        import torch
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        if not torch.backends.mps.is_available():
            raise RuntimeError("VibeVoice v1 requires an available MPS device")
        mps_high_water = _MpsHighWater(torch.mps)
        mps_high_water.start()
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.mps.manual_seed(config["seed"])

        processor = VibeVoiceASRProcessor.from_pretrained(
            str(model_path),
            language_model_pretrained_name=str(tokenizer_path),
            local_files_only=True,
        )
        model = (
            VibeVoiceASRForConditionalGeneration.from_pretrained(
                str(model_path),
                dtype=torch.bfloat16,
                attn_implementation=config["attention"],
                trust_remote_code=True,
                local_files_only=True,
            )
            .to(config["device"])
            .eval()
        )
        torch.mps.synchronize()

        inputs = processor(
            audio=[str(audio_path)],
            sampling_rate=None,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        inputs = {
            key: value.to(config["device"]) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        input_tokens = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=processor.pad_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False,
            )
        torch.mps.synchronize()

        generated = output_ids[0, input_tokens:]
        generated_tokens = int(generated.numel())
        eos = (generated == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        eos_observed = bool(len(eos))
        if eos_observed:
            generated = generated[: int(eos[0]) + 1]
        raw_text = processor.decode(generated, skip_special_tokens=True)
        if not isinstance(raw_text, str):
            raise TypeError("VibeVoice processor decode did not return a string")
        hit_max_new_tokens = generated_tokens == config["max_new_tokens"] and not eos_observed
        segments = processor.post_process_transcription(raw_text)
        if not isinstance(segments, list):
            raise TypeError("VibeVoice post-process result must be an array")
        if not hit_max_new_tokens:
            generated_array = _complete_json_array(raw_text)
            if len(segments) != len(generated_array):
                raise ValueError("VibeVoice post-process dropped generated array entries")
        output.update(
            {
                "raw_text": raw_text,
                "segments": segments,
                "hit_max_new_tokens": hit_max_new_tokens,
                "generated_tokens": generated_tokens,
                "eos_observed": eos_observed,
            }
        )
        code = 4 if hit_max_new_tokens else 0
    except Exception as exc:  # noqa: BLE001 - every stage failure needs a result envelope
        output["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        code = 1

    try:
        peak_mps_live_bytes = mps_high_water.stop() if mps_high_water is not None else None
    except Exception:  # noqa: BLE001 - telemetry cannot suppress the result envelope
        peak_mps_live_bytes = None
    output["metrics"] = {
        "wall_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_bytes": _rss_bytes(),
    }
    if peak_mps_live_bytes is not None:
        output["metrics"]["peak_mps_live_bytes"] = peak_mps_live_bytes
    result_path.write_text(
        json.dumps(output, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
