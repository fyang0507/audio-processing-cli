from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ENHANCED_MARKER = "audio-processing-cli enhanced"


class MediaError(RuntimeError):
    pass


def _run(
    args: Sequence[str], *, capture_stdout: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(args),
            check=True,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"Required executable not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise MediaError(f"Command failed ({args[0]}): {detail}") from exc


def require_runtime() -> None:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise MediaError(
                f"{executable} is required. Install FFmpeg and ensure both ffmpeg and ffprobe are on PATH."
            )


def ffmpeg_version() -> str:
    result = _run(["ffmpeg", "-version"])
    first = result.stdout.decode("utf-8", errors="replace").splitlines()[0]
    return first.strip()


def probe_media(path: Path) -> dict[str, object]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {path}") from exc
    audio_streams = [
        s for s in payload.get("streams", []) if s.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise MediaError(f"No audio stream found in {path}")
    payload["primary_audio_stream"] = audio_streams[0]
    payload["has_video"] = any(
        s.get("codec_type") == "video" for s in payload.get("streams", [])
    )
    return payload


def media_summary(path: Path, probe: dict[str, object]) -> dict[str, object]:
    stream = probe["primary_audio_stream"]
    assert isinstance(stream, dict)
    fmt = probe.get("format", {})
    assert isinstance(fmt, dict)
    duration_raw = stream.get("duration", fmt.get("duration", 0.0))
    tags = fmt.get("tags", {}) or {}
    return {
        "path": str(path.resolve()),
        "sha256": hash_file(path),
        "duration_seconds": round(float(duration_raw), 6),
        "audio_start_seconds": round(float(stream.get("start_time", 0.0) or 0.0), 6),
        "sample_rate_hz": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
        "channel_layout": stream.get("channel_layout"),
        "audio_codec": stream.get("codec_name"),
        "has_video": bool(probe.get("has_video")),
        "format_name": fmt.get("format_name"),
        "tags": tags,
    }


def is_enhanced_media(probe: dict[str, object]) -> bool:
    fmt = probe.get("format", {})
    if not isinstance(fmt, dict):
        return False
    tags = fmt.get("tags", {}) or {}
    if not isinstance(tags, dict):
        return False
    text = " ".join(str(value) for value in tags.values()).lower()
    return ENHANCED_MARKER in text


def decode_audio(path: Path, *, sample_rate: int = 48_000) -> tuple[np.ndarray, int]:
    probe = probe_media(path)
    stream = probe["primary_audio_stream"]
    assert isinstance(stream, dict)
    source_channels = int(stream.get("channels", 1) or 1)
    if source_channels > 2:
        raise MediaError(
            f"V1 supports mono or stereo sources; {path} has {source_channels} audio channels"
        )
    channels = 1 if source_channels == 1 else 2
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ]
    )
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size == 0 or samples.size % channels:
        raise MediaError(f"Could not decode a complete audio stream from {path}")
    return samples.reshape((-1, channels)), sample_rate


def _wav_duration_ms(path: Path) -> float:
    """Duration of a wav this tool wrote, read through the header rather than the samples."""
    rate, data = wavfile.read(path, mmap=True)
    return len(data) / float(rate) * 1000.0


def write_float_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.asarray(samples, dtype=np.float32))


def _extract_loudnorm_json(stderr: bytes) -> dict[str, float]:
    text = stderr.decode("utf-8", errors="replace")
    matches = re.findall(r"\{\s*\"input_i\"\s*:.*?\}", text, flags=re.DOTALL)
    if not matches:
        raise MediaError("FFmpeg loudnorm did not emit measurement JSON")
    raw = json.loads(matches[-1])
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            result[key] = math.nan
    return result


def measure_loudness(
    path: Path,
    *,
    target_lufs: float = -16.0,
    target_lra: float = 11.0,
    target_true_peak: float = -1.5,
) -> dict[str, float]:
    filter_spec = (
        f"loudnorm=I={target_lufs}:LRA={target_lra}:TP={target_true_peak}:"
        "print_format=json"
    )
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-af",
                filter_spec,
                "-f",
                "null",
                "-",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"").decode("utf-8", errors="replace")[-4000:]
        raise MediaError(f"Could not measure loudness for {path}: {detail}") from exc
    return _extract_loudnorm_json(result.stderr)


def render_loudness_normalized(
    source_wav: Path,
    output_wav: Path,
    *,
    target_lufs: float,
    target_lra: float,
    target_true_peak: float,
    measurement: dict[str, float],
    sample_rate: int,
) -> dict[str, object]:
    required = ("input_i", "input_lra", "input_tp", "input_thresh")
    if any(not math.isfinite(measurement.get(key, math.nan)) for key in required):
        # Two different conditions, and reporting them as one sent anyone with a short clip
        # looking for silence that was not there. A finite true peak is the evidence that the
        # signal exists: integrated loudness is gated in 400 ms blocks (EBU R128), so anything
        # shorter has no measurable value however loud it is, while true silence has no finite
        # peak either.
        peak = measurement.get("input_tp", math.nan)
        if math.isfinite(peak):
            raise MediaError(
                f"Cannot normalize {_wav_duration_ms(source_wav):.0f} ms of audio: integrated "
                f"loudness is measured over 400 ms gating blocks (EBU R128), so a shorter input "
                f"has none however loud it is. This input peaks at {peak:.2f} dBTP, so it is not "
                f"silent -- it is too short to measure. Enhance a longer excerpt, or skip "
                f"program-loudness with --skip program-loudness."
            )
        raise MediaError("Cannot normalize silent audio: it has no measurable level or peak")
    gain_db = target_lufs - measurement["input_i"]
    limiter_amplitude = 10.0 ** (target_true_peak / 20.0)
    iterations: list[dict[str, float]] = []
    measured_output: dict[str, float] = measurement
    for iteration in range(1, 6):
        filter_spec = (
            f"volume={gain_db:.9f}dB,"
            "aresample=192000,"
            f"alimiter=limit={limiter_amplitude:.9f}:attack=5:release=50:"
            "level=false:latency=true,"
            f"aresample={sample_rate}"
        )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(source_wav),
                "-af",
                filter_spec,
                "-c:a",
                "pcm_f32le",
                "-ar",
                str(sample_rate),
                "-y",
                str(output_wav),
            ],
            capture_stdout=False,
        )
        measured_output = measure_loudness(
            output_wav,
            target_lufs=target_lufs,
            target_lra=target_lra,
            target_true_peak=target_true_peak,
        )
        error_lu = target_lufs - measured_output["input_i"]
        iterations.append(
            {
                "iteration": float(iteration),
                "input_gain_db": round(gain_db, 6),
                "measured_lufs": round(measured_output["input_i"], 6),
                "measured_true_peak_dbtp": round(measured_output["input_tp"], 6),
                "loudness_error_lu": round(error_lu, 6),
            }
        )
        if abs(error_lu) <= 0.05:
            break
        gain_db += error_lu
    return {
        "method": "ebu-r128-measured-linear-gain-with-oversampled-true-peak-limiter",
        "resolved_input_gain_db": round(gain_db, 6),
        "limiter_ceiling_dbtp": target_true_peak,
        "limiter_oversample_hz": 192_000,
        "iterations": iterations,
        "measured_output": {
            key: round(value, 6) if math.isfinite(value) else None
            for key, value in sorted(measured_output.items())
        },
    }


def encode_output(
    original: Path,
    enhanced_wav: Path,
    output: Path,
    *,
    original_sha256: str,
    has_video: bool,
) -> None:
    suffix = output.suffix.lower()
    metadata = f"{ENHANCED_MARKER}; original_sha256={original_sha256}"
    if has_video and suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        if suffix == ".webm":
            audio_codec = "libopus"
            audio_args = ["-c:a", audio_codec, "-b:a", "192k"]
        elif suffix == ".mkv":
            audio_args = ["-c:a", "flac"]
        else:
            audio_args = ["-c:a", "aac", "-b:a", "192k"]
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(original),
            "-i",
            str(enhanced_wav),
            "-map",
            "0:v?",
            "-map",
            "1:a:0",
            "-map_metadata",
            "0",
            "-c:v",
            "copy",
            *audio_args,
            "-metadata",
            f"comment={metadata}",
        ]
        if suffix in {".mp4", ".mov"}:
            args.extend(["-movflags", "+faststart"])
        args.extend(["-y", str(output)])
        _run(args, capture_stdout=False)
        return

    codec_args: list[str]
    if suffix == ".wav":
        codec_args = ["-c:a", "pcm_s24le"]
    elif suffix == ".flac":
        codec_args = ["-c:a", "flac"]
    elif suffix == ".mp3":
        codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif suffix in {".m4a", ".aac", ".mp4"}:
        codec_args = ["-c:a", "aac", "-b:a", "192k"]
    elif suffix in {".ogg", ".opus"}:
        codec_args = ["-c:a", "libopus", "-b:a", "160k"]
    else:
        raise MediaError(
            f"Unsupported output extension {suffix!r}; use wav, flac, mp3, m4a, ogg, opus, mp4, mov, mkv, or webm"
        )
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(enhanced_wav),
            *codec_args,
            "-metadata",
            f"comment={metadata}",
            "-y",
            str(output),
        ],
        capture_stdout=False,
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def temporary_directory(prefix: str = "audio-processing-") -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix) as raw:
        yield Path(raw)


@contextmanager
def temporary_output_path(output: Path) -> Iterator[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{output.stem}-",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    temp = Path(raw)
    temp.unlink(missing_ok=True)
    try:
        yield temp
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Written through a sibling and renamed, so a report is never observed half-written.

    Deliberately a plain `open` rather than `mkstemp`. `mkstemp` creates 0600 and `os.replace`
    carries that mode onto the destination, so a report arrived stricter than the render it
    describes -- 0600 beside an 0644 wav, under the same umask, for no reason a reader could act
    on. Everything else this tool publishes honours the umask, including `save_registry`, which
    reached the same conclusion by writing the same way.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
