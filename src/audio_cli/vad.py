from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort

from .paths import models_dir

MODEL_VERSION = "silero-vad-6.2.1"
MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/"
    "src/silero_vad/data/silero_vad.onnx"
)
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
MODEL_FILENAME = f"{MODEL_VERSION}.onnx"


class VadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeechRegion:
    start: float
    end: float
    mean_probability: float
    peak_probability: float

    def as_dict(self) -> dict[str, float]:
        return {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "mean_probability": round(self.mean_probability, 6),
            "peak_probability": round(self.peak_probability, 6),
        }


class VadDetector(Protocol):
    model_version: str

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        threshold: float,
        exit_threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
        speech_pad_ms: int,
    ) -> list[SpeechRegion]: ...


def _cache_root() -> Path:
    """The models directory under the one provisioning root. See paths.py."""
    return models_dir()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_path(explicit: Path | None = None) -> Path:
    env_path = os.environ.get("AUDIO_PROCESSING_VAD_MODEL")
    if explicit is not None or env_path:
        path = explicit or Path(env_path).expanduser()
        assert path is not None
        if not path.is_file():
            raise VadError(f"Silero VAD model not found: {path}")
        return path

    target = _cache_root() / MODEL_FILENAME
    if target.is_file() and _sha256(target) == MODEL_SHA256:
        return target
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.{os.getpid()}.part")
    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "audio-processing-cli/0.1"},
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            partial.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = _sha256(partial)
        if actual != MODEL_SHA256:
            raise VadError(
                f"Downloaded Silero VAD checksum mismatch: expected {MODEL_SHA256}, got {actual}"
            )
        os.replace(partial, target)
    except (OSError, urllib.error.URLError) as exc:
        raise VadError(
            "Could not download the pinned Silero VAD model. Connect once to populate the cache, "
            "or set AUDIO_PROCESSING_VAD_MODEL to a local silero_vad.onnx file."
        ) from exc
    finally:
        partial.unlink(missing_ok=True)
    return target


class SileroOnnxVad:
    model_version = MODEL_VERSION
    frame_samples = 512
    context_samples = 64

    def __init__(self, model_path: Path | None = None) -> None:
        resolved = resolve_model_path(model_path)
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(
                str(resolved),
                providers=["CPUExecutionProvider"],
                sess_options=options,
            )
        except (
            Exception
        ) as exc:  # onnxruntime exposes several provider-specific exceptions
            raise VadError(
                f"Could not load Silero VAD model {resolved}: {exc}"
            ) from exc
        self.model_path = resolved

    def probabilities(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16_000:
            raise VadError(
                f"Silero VAD requires 16000 Hz input, received {sample_rate}"
            )
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, self.context_samples), dtype=np.float32)
        probabilities: list[float] = []
        for offset in range(0, mono.size, self.frame_samples):
            chunk = mono[offset : offset + self.frame_samples]
            if chunk.size < self.frame_samples:
                chunk = np.pad(chunk, (0, self.frame_samples - chunk.size))
            model_input = np.concatenate((context, chunk[None, :]), axis=1).astype(
                np.float32
            )
            output, state = self.session.run(
                None,
                {
                    "input": model_input,
                    "state": state,
                    "sr": np.asarray(sample_rate, dtype=np.int64),
                },
            )
            probabilities.append(float(output[0, 0]))
            context = model_input[:, -self.context_samples :]
        return np.asarray(probabilities, dtype=np.float32)

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        threshold: float,
        exit_threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
        speech_pad_ms: int,
    ) -> list[SpeechRegion]:
        probabilities = self.probabilities(samples, sample_rate)
        frame = self.frame_samples
        min_speech = round(sample_rate * min_speech_ms / 1000)
        min_silence = round(sample_rate * min_silence_ms / 1000)
        pad = round(sample_rate * speech_pad_ms / 1000)
        triggered = False
        start_sample = 0
        possible_end: int | None = None
        raw_regions: list[tuple[int, int]] = []

        for index, probability in enumerate(probabilities):
            position = index * frame
            if probability >= threshold:
                possible_end = None
                if not triggered:
                    triggered = True
                    start_sample = position
                continue
            if triggered and probability < exit_threshold:
                if possible_end is None:
                    possible_end = position
                if position - possible_end >= min_silence:
                    if possible_end - start_sample >= min_speech:
                        raw_regions.append((start_sample, possible_end))
                    triggered = False
                    possible_end = None

        if triggered and samples.size - start_sample >= min_speech:
            raw_regions.append((start_sample, samples.size))

        padded: list[tuple[int, int]] = []
        for index, (start, end) in enumerate(raw_regions):
            left = max(0, start - pad)
            right = min(samples.size, end + pad)
            if index and left < padded[-1][1]:
                midpoint = (padded[-1][1] + left) // 2
                padded[-1] = (padded[-1][0], midpoint)
                left = midpoint
            padded.append((left, right))

        merged: list[tuple[int, int]] = []
        merge_gap = round(0.30 * sample_rate)
        for start, end in padded:
            if merged and start - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        regions: list[SpeechRegion] = []
        for start, end in merged:
            first_frame = max(0, start // frame)
            last_frame = min(probabilities.size, (end + frame - 1) // frame)
            region_probs = probabilities[first_frame:last_frame]
            regions.append(
                SpeechRegion(
                    start=start / sample_rate,
                    end=end / sample_rate,
                    mean_probability=float(np.mean(region_probs))
                    if region_probs.size
                    else 0.0,
                    peak_probability=float(np.max(region_probs))
                    if region_probs.size
                    else 0.0,
                )
            )
        return regions
