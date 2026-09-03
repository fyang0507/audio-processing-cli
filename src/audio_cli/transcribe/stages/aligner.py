"""Execute forced alignment per segment without importing the core CLI package."""

from __future__ import annotations

import json
import os
import re
import resource
import sys
import time
import traceback
from pathlib import Path

CJK = re.compile(r"[一-鿿]")


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main() -> int:
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    output = {"segments": [], "metrics": {}}
    try:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"

        print("aligner stage: loading local model", file=sys.stderr, flush=True)
        import mlx.core as mx
        import numpy as np
        from mlx_audio.audio_io import read as audio_read
        from mlx_audio.stt import load as stt_load

        raw_audio, rate = audio_read(Path(request["audio"]), always_2d=True, dtype="float32")
        if rate != 16_000 or raw_audio.shape[1] != 1:
            raise ValueError(f"canonical input must be mono 16000 Hz, got {rate} Hz")
        audio = np.ascontiguousarray(raw_audio[:, 0], dtype=np.float32)
        model = stt_load(Path(request["model"]))
        for index, segment in enumerate(request["segments"]):
            text = segment["text"]
            start_s, end_s = float(segment["start"]), float(segment["end"])
            clip = audio[round(start_s * rate):round(end_s * rate)]
            language = "Chinese" if CJK.search(text) else "English"
            print(
                f"aligner stage: processing segment {index + 1}/"
                f"{len(request['segments'])}",
                file=sys.stderr,
                flush=True,
            )
            try:
                generated = model.generate(audio=clip, text=text, language=language)
                output["segments"].append({
                    "unit_id": segment["unit_id"],
                    "language": language,
                    "words": [{
                        "text": item.text,
                        "start": round(float(item.start_time) + start_s, 3),
                        "end": round(float(item.end_time) + start_s, 3),
                    } for item in generated],
                })
            except Exception as exc:  # noqa: BLE001 - this capability may abstain per segment
                output["segments"].append({
                    "unit_id": segment["unit_id"],
                    "language": language,
                    "words": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                })
        output["metrics"] = {
            "wall_seconds": round(time.perf_counter() - started, 6),
            "peak_rss_bytes": _rss_bytes(),
            "peak_mps_live_bytes": int(mx.get_peak_memory()),
        }
        code = 0
    except Exception as exc:  # noqa: BLE001 - the stage must always write a result envelope
        output["error"] = {
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        output["metrics"] = {
            "wall_seconds": round(time.perf_counter() - started, 6),
            "peak_rss_bytes": _rss_bytes(),
        }
        code = 1
    result_path.write_text(
        json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
