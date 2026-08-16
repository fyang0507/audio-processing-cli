#!/usr/bin/env python3
"""Instrumented, fresh-process FireRedASR2S benchmark runner."""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


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
        "--firered-root", default=str(default_checkout),
        help="FireRedASR2S source checkout (default: %(default)s)",
    )
    parser.add_argument("--audio", required=True, help="Source audio or video")
    parser.add_argument("--output", required=True, help="Result JSON path")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--lid", choices=("on", "off"), required=True,
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
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def command_version(command: list[str]) -> str | None:
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        )
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
    metadata: dict[str, Any] = {"path": str(path), "commit": None,
                                "tracked_dirty": None}
    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain",
             "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL,
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
                first_line = metadata_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[0].strip()
            except (OSError, IndexError):
                continue
            if re.fullmatch(r"[0-9a-fA-F]{40}", first_line):
                revisions.add(first_line.lower())
    return {
        "path": str(path),
        "revision_cache_root": str(cache_root.parent.parent)
        if cache_root is not None else None,
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
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ctypes.c_void_p, ctypes.c_int,
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
                resident_pages = int(
                    Path("/proc/self/statm").read_text().split()[1]
                )
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


def monotonic(items: list[dict[str, Any]], start_key: str,
              end_key: str) -> bool:
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


def main() -> int:
    args = parse_args()
    firered_root = Path(args.firered_root).resolve()
    model_root = firered_root / "pretrained_models"
    model_paths = {
        "vad": model_root / "FireRedVAD/VAD",
        "asr": model_root / "FireRedASR2-AED",
        "lid": model_root / "FireRedLID",
        "punc": model_root / "FireRedPunc",
    }
    audio_path = Path(args.audio).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lid_enabled = args.lid == "on"
    uttid = sanitize_uttid(args.uttid or audio_path.stem)

    status = "error"
    error: dict[str, str] | None = None
    source_probe: dict[str, Any] | None = None
    canonical_probe: dict[str, Any] | None = None
    canonical_sha256: str | None = None
    fire_result: dict[str, Any] = {}
    timing: dict[str, Any] = {"model_load_s": {}}
    stage_time: defaultdict[str, float] = defaultdict(float)
    stage_calls: defaultdict[str, int] = defaultdict(int)
    samples: list[dict[str, Any]] = []
    phase = {"name": "startup"}
    stop = threading.Event()
    rss_reader = RssReader()
    process_start = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)

    def sample_memory() -> None:
        while True:
            samples.append({
                "elapsed_s": time.perf_counter() - process_start,
                "phase": phase["name"],
                "rss_bytes": rss_reader.read(),
            })
            if stop.wait(args.sample_interval):
                break

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    temporary = tempfile.TemporaryDirectory(prefix="firered-benchmark-")
    canonical_audio = Path(temporary.name) / "audio-16k-mono.wav"
    ffmpeg_command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(canonical_audio),
    ]
    recorded_ffmpeg_command = [
        *ffmpeg_command[:-1], "<temporary>/audio-16k-mono.wav"
    ]

    try:
        if not firered_root.is_dir():
            raise FileNotFoundError(f"FireRed checkout not found: {firered_root}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        required_models = ["vad", "asr", "punc"] + (["lid"] if lid_enabled else [])
        for name in required_models:
            if not model_paths[name].is_dir():
                raise FileNotFoundError(
                    f"FireRed {name.upper()} model not found: {model_paths[name]}"
                )

        phase["name"] = "preprocess"
        t0 = time.perf_counter()
        try:
            source_probe = ffprobe(audio_path)
            completed = subprocess.run(
                ffmpeg_command, text=True, capture_output=True, check=False
            )
            if completed.returncode != 0:
                message = completed.stderr.strip() or "ffmpeg conversion failed"
                raise RuntimeError(message)
            canonical_probe = ffprobe(canonical_audio)
        finally:
            timing["preprocess_s"] = time.perf_counter() - t0

        phase["name"] = "import"
        t0 = time.perf_counter()
        try:
            sys.path.insert(0, str(firered_root))
            from fireredasr2s import (  # noqa: PLC0415
                FireRedAsr2System,
                FireRedAsr2SystemConfig,
            )
            from fireredasr2s.fireredasr2 import (  # noqa: PLC0415
                FireRedAsr2,
                FireRedAsr2Config,
            )
            from fireredasr2s.fireredlid import (  # noqa: PLC0415
                FireRedLid,
                FireRedLidConfig,
            )
            from fireredasr2s.fireredpunc import (  # noqa: PLC0415
                FireRedPunc,
                FireRedPuncConfig,
            )
            from fireredasr2s.fireredvad import (  # noqa: PLC0415
                FireRedVad,
                FireRedVadConfig,
            )
        finally:
            timing["import_s"] = time.perf_counter() - t0

        vad_config = FireRedVadConfig(use_gpu=False)
        lid_config = FireRedLidConfig(use_gpu=False, use_half=False)
        asr_config = FireRedAsr2Config(
            use_gpu=False,
            use_half=False,
            beam_size=3,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        )
        punc_config = FireRedPuncConfig(use_gpu=False)
        system_config = FireRedAsr2SystemConfig(
            vad_model_dir=str(model_paths["vad"]),
            lid_model_dir=str(model_paths["lid"]),
            asr_type="aed",
            asr_model_dir=str(model_paths["asr"]),
            punc_model_dir=str(model_paths["punc"]),
            vad_config=vad_config,
            lid_config=lid_config,
            asr_config=asr_config,
            punc_config=punc_config,
            asr_batch_size=args.asr_batch_size,
            punc_batch_size=args.punc_batch_size,
            enable_vad=True,
            enable_lid=lid_enabled,
            enable_punc=True,
        )

        phase["name"] = "load"
        load_start = time.perf_counter()
        try:
            phase["name"] = "load_vad"
            t0 = time.perf_counter()
            try:
                vad_model = FireRedVad.from_pretrained(
                    str(model_paths["vad"]), vad_config
                )
            finally:
                timing["model_load_s"]["vad"] = time.perf_counter() - t0

            lid_model = None
            if lid_enabled:
                phase["name"] = "load_lid"
                t0 = time.perf_counter()
                try:
                    lid_model = FireRedLid.from_pretrained(
                        str(model_paths["lid"]), lid_config
                    )
                finally:
                    timing["model_load_s"]["lid"] = time.perf_counter() - t0
            else:
                timing["model_load_s"]["lid"] = None

            phase["name"] = "load_asr"
            t0 = time.perf_counter()
            try:
                asr_model = FireRedAsr2.from_pretrained(
                    "aed", str(model_paths["asr"]), asr_config
                )
            finally:
                timing["model_load_s"]["asr"] = time.perf_counter() - t0

            phase["name"] = "load_punc"
            t0 = time.perf_counter()
            try:
                punc_model = FireRedPunc.from_pretrained(
                    str(model_paths["punc"]), punc_config
                )
            finally:
                timing["model_load_s"]["punc"] = time.perf_counter() - t0
        finally:
            timing["load_s"] = time.perf_counter() - load_start

        system = FireRedAsr2System.__new__(FireRedAsr2System)
        system.vad = vad_model
        system.lid = lid_model
        system.asr = asr_model
        system.punc = punc_model
        system.config = system_config

        def timed(stage_name: str, function: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(function)
            def wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> Any:
                stage_start = time.perf_counter()
                try:
                    return function(*wrapper_args, **wrapper_kwargs)
                finally:
                    stage_time[stage_name] += time.perf_counter() - stage_start
                    stage_calls[stage_name] += 1
            return wrapper

        system.vad.detect = timed("vad", system.vad.detect)
        system.asr.transcribe = timed("asr", system.asr.transcribe)
        if lid_enabled and system.lid is not None:
            system.lid.process = timed("lid", system.lid.process)
        system.punc.process_with_timestamp = timed(
            "punc", system.punc.process_with_timestamp
        )

        phase["name"] = "inference"
        t0 = time.perf_counter()
        try:
            fire_result = system.process(str(canonical_audio), uttid)
        finally:
            timing["inference_s"] = time.perf_counter() - t0
        # FireRed reports its immediate WAV input, which is a deleted random
        # temporary path in this runner. The stable source and canonical-audio
        # identity are already recorded under ``audio``.
        fire_result.pop("wav_path", None)
        stage_total = sum(stage_time.values())
        timing["inference_framework_overhead_s"] = max(
            0.0, timing["inference_s"] - stage_total
        )
        timing["end_to_end_s"] = time.perf_counter() - process_start
        status = "ok"

        phase["name"] = "evidence"
        canonical_sha256 = sha256(canonical_audio)
    except Exception as exc:  # preserve failure evidence in the result artifact
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": phase["name"],
            "traceback": traceback.format_exc(),
        }
        timing.setdefault("end_to_end_s", time.perf_counter() - process_start)
    finally:
        if canonical_sha256 is None and canonical_audio.is_file():
            try:
                canonical_sha256 = sha256(canonical_audio)
            except OSError:
                pass
        phase["name"] = "complete"
        samples.append({
            "elapsed_s": time.perf_counter() - process_start,
            "phase": phase["name"],
            "rss_bytes": rss_reader.read(),
        })
        stop.set()
        sampler.join()
        temporary.cleanup()

    try:
        source_sha256 = sha256(audio_path) if audio_path.is_file() else None
    except OSError:
        source_sha256 = None
    duration: float | None = None
    if source_probe and source_probe.get("format", {}).get("duration"):
        duration = float(source_probe["format"]["duration"])
    elif canonical_probe and canonical_probe.get("format", {}).get("duration"):
        duration = float(canonical_probe["format"]["duration"])

    normalized_output = fire_result
    normalized_json = json.dumps(
        normalized_output, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=json_default,
    )
    sentences = fire_result.get("sentences", [])
    words = fire_result.get("words", [])
    normalized_segments = [
        {
            "start_s": float(sentence["start_ms"]) / 1000,
            "end_s": float(sentence["end_ms"]) / 1000,
            "text": sentence.get("text", ""),
            "speaker": None,
        }
        for sentence in sentences
    ]
    last_end_s = max(
        (float(sentence["end_ms"]) / 1000 for sentence in sentences),
        default=None,
    )
    peak_by_phase: dict[str, int] = {}
    for sample in samples:
        sample_phase = str(sample["phase"])
        peak_by_phase[sample_phase] = max(
            peak_by_phase.get(sample_phase, 0), int(sample["rss_bytes"])
        )

    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    rtf_inference = None
    rtf_end_to_end = None
    if duration and duration > 0:
        if "inference_s" in timing:
            rtf_inference = timing["inference_s"] / duration
        rtf_end_to_end = timing["end_to_end_s"] / duration
    timing.update({
        "stage_s": dict(stage_time),
        "stage_calls": dict(stage_calls),
        "rtf_inference": rtf_inference,
        "rtf_end_to_end": rtf_end_to_end,
        "cpu_user_s": usage_end.ru_utime - usage_start.ru_utime,
        "cpu_system_s": usage_end.ru_stime - usage_start.ru_stime,
    })

    result = {
        "schema_version": 1,
        "status": status,
        "error": error,
        "runner": {
            "path": str(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "runtime": {
            "packages": package_versions([
                "fireredasr2s", "torch", "transformers", "numpy",
                "soundfile", "torchaudio", "kaldi-native-fbank",
            ]),
            "ffmpeg": command_version(["ffmpeg", "-version"]),
            "ffprobe": command_version(["ffprobe", "-version"]),
            "environment": {
                name: os.environ.get(name) for name in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "PYTORCH_ENABLE_MPS_FALLBACK",
                )
            },
        },
        "configuration": {
            "device": args.device,
            "dtype": "float32",
            "lid": args.lid,
            "asr_batch_size": args.asr_batch_size,
            "punc_batch_size": args.punc_batch_size,
            "sample_interval_s": args.sample_interval,
            "uttid": uttid,
            "canonical_audio": {
                "sample_rate": 16000,
                "channels": 1,
                "codec": "pcm_s16le",
                "ffmpeg_command": recorded_ffmpeg_command,
            },
        },
        "source": {
            "code": git_metadata(firered_root),
            "models": {
                name: huggingface_revision(path)
                for name, path in model_paths.items()
            },
        },
        "audio": {
            "path": str(audio_path),
            "sha256": source_sha256,
            "duration_s": duration,
            "probe": source_probe,
            "canonical_sha256": canonical_sha256,
            "canonical_probe": canonical_probe,
        },
        "timing": timing,
        "memory": {
            "sample_scope": "current_process_only",
            "sample_source": rss_reader.source,
            "peak_sampled_rss_bytes": max(
                (int(sample["rss_bytes"]) for sample in samples), default=0
            ),
            "peak_sampled_rss_by_phase": peak_by_phase,
            "ru_maxrss_bytes": normalized_ru_maxrss(usage_end),
            "ru_children_maxrss_bytes": normalized_ru_maxrss(
                resource.getrusage(resource.RUSAGE_CHILDREN)
            ),
            "samples": samples,
        },
        "stability": {
            "output_parse_valid": status == "ok" and isinstance(fire_result, dict)
            and isinstance(sentences, list) and isinstance(words, list),
            "sentence_timestamps_monotonic": monotonic(
                sentences, "start_ms", "end_ms"
            ) if sentences else None,
            "word_timestamps_monotonic": monotonic(
                words, "start_ms", "end_ms"
            ) if words else None,
            "last_sentence_end_s": last_end_s,
            "last_sentence_end_ratio": last_end_s / duration
            if last_end_s is not None and duration else None,
            "vad_segment_count": len(fire_result.get("vad_segments_ms", [])),
            "sentence_count": len(sentences),
            "word_count": len(words),
        },
        "output": {
            "result": fire_result,
            "segments": normalized_segments,
            "normalized_segments_sha256": hashlib.sha256(json.dumps(
                normalized_segments, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), default=json_default,
            ).encode()).hexdigest(),
            "normalized_result_sha256": hashlib.sha256(
                normalized_json.encode()
            ).hexdigest(),
        },
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": status,
        "output": str(output_path),
        "timing": timing,
        "memory": {
            key: value for key, value in result["memory"].items()
            if key != "samples"
        },
        "normalized_result_sha256": result["output"][
            "normalized_result_sha256"
        ],
        "normalized_segments_sha256": result["output"][
            "normalized_segments_sha256"
        ],
        "error": error,
    }, ensure_ascii=False, default=json_default))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
