"""Execute one Qwen ASR request. This file intentionally depends on no core package code."""

from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main() -> int:
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    units = request["units"]
    output = {"units": [], "metrics": {}}
    raw_by_index = {}
    mx = None
    try:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"

        print("qwen stage: loading local model", file=sys.stderr, flush=True)
        import mlx.core as mx
        import numpy as np
        from mlx_audio.audio_io import read as audio_read
        from mlx_audio.stt.utils import load_model
        from mlx_lm.sample_utils import make_sampler

        raw_audio, rate = audio_read(Path(request["audio"]), always_2d=True, dtype="float32")
        if rate != 16_000 or raw_audio.shape[1] != 1:
            raise ValueError(f"canonical input must be mono 16000 Hz, got {rate} Hz")
        audio = np.ascontiguousarray(raw_audio[:, 0], dtype=np.float32)
        model = load_model(Path(request["model"]), lazy=False, strict=False)
        mx.eval(model.parameters())
        mx.synchronize()
        method = model._generate_chunks_batched
        order = sorted(
            range(len(units)),
            key=lambda i: (round((float(units[i]["end"]) - float(units[i]["start"])) * 16_000), i),
        )
        remaining = int(request["max_tokens"])
        batch_size = int(request["batch_size"])
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for offset in range(0, len(order), batch_size):
            if remaining <= 0:
                break
            indices = order[offset : offset + batch_size]
            clips = []
            for index in indices:
                unit = units[index]
                start = max(0, round(float(unit["start"]) * 16_000))
                end = min(len(audio), round(float(unit["end"]) * 16_000))
                clips.append((audio[start:end], 0.0))
            print(
                f"qwen stage: processing batch {offset // batch_size + 1} ({len(indices)} unit(s))",
                file=sys.stderr,
                flush=True,
            )
            texts, generated, prompts, processed = method(
                clips,
                max_tokens=remaining,
                sampler=make_sampler(temp=0.0),
                language=request.get("language"),
                system_prompt=None,
                batch_size=batch_size,
                verbose=False,
            )
            if not all(
                len(values) == len(indices) for values in (texts, generated, prompts, processed)
            ) or not all(isinstance(value, bool) for value in processed):
                raise TypeError("private batched API returned inconsistent per-unit results")
            for local_index, index in enumerate(indices):
                unit = units[index]
                raw_by_index[index] = {
                    "unit_id": unit["unit_id"],
                    "text": texts[local_index],
                    "processed": processed[local_index],
                    "prompt_tokens": int(prompts[local_index]),
                    "generation_tokens": int(generated[local_index]),
                }
            remaining -= sum(int(value) for value in generated)
            if request["clear_cache_after_every_batch"]:
                mx.clear_cache()
                mx.synchronize()
        code = 0
    except Exception as exc:  # noqa: BLE001 - the stage must always write a result envelope
        output["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        code = 4 if any(item.get("processed") for item in raw_by_index.values()) else 1
    for index, unit in enumerate(units):
        output["units"].append(
            raw_by_index.get(
                index,
                {
                    "unit_id": unit["unit_id"],
                    "text": "",
                    "processed": False,
                    "prompt_tokens": 0,
                    "generation_tokens": 0,
                },
            )
        )
    output["metrics"] = {
        "wall_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_bytes": _rss_bytes(),
    }
    if mx is not None:
        with suppress(Exception):  # optional telemetry cannot suppress the result
            output["metrics"]["peak_mps_live_bytes"] = int(mx.get_peak_memory())
    if code == 0 and not all(item["processed"] for item in output["units"]):
        code = 4
    result_path.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
