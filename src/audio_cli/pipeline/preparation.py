"""Source validation, decoding, VAD, and signal analysis preparation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import signal

from .. import __version__
from ..dsp import analyze_signal, evaluate_profile, regional_measurements
from ..media import (
    decode_audio,
    decoded_audio_timing,
    ffmpeg_version,
    is_enhanced_media,
    measure_loudness,
    media_summary,
    probe_media,
    require_runtime,
)
from ..profiles import PROFILES, Profile
from ..vad import MODEL_SHA256, MODEL_URL, SileroOnnxVad
from ..vad_contract import VadDetector
from .models import PipelineError, PreparedRun
from .outcomes import REGION_BASIS, actual_program
from .progress import ProgressSink, progress_stage
from .reporting import _program_observation, _region_manifest, _round_loudness


def _vad_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono at 16 kHz, never longer than the source timeline it will be timestamped against.

    `resample_poly` returns `ceil(n * up / down)` samples, so 1332160 samples at 48 kHz became
    444054 at 16 kHz where the exact figure is 444053.333. Two thirds of a sample sounds like
    nothing, and it is -- until a speech region runs to the end of the signal and is published
    ending at 27.753375 s in a 27.753333 s file. That is a bound the media does not have, and the
    treatment clamp that pulled it back to the real end then reported a *negative* extension,
    contradicting the guarantee that speech effects are fully engaged at the last detected sample.

    Truncating to the floor keeps the detector's timeline inside the source's. It gives up at most
    one 16 kHz sample of tail, which is 62.5 us and cannot carry speech; inventing a bound past
    the end of the file is the more expensive mistake.
    """
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    if sample_rate == 16_000:
        return mono
    divisor = math.gcd(sample_rate, 16_000)
    resampled = signal.resample_poly(mono, 16_000 // divisor, sample_rate // divisor).astype(
        np.float32
    )
    return resampled[: mono.size * 16_000 // sample_rate]


def _detect_speech(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    detector: VadDetector,
) -> list:
    vad_audio = _vad_audio(audio, sample_rate)
    return detector.detect(
        vad_audio,
        16_000,
        threshold=profile.vad_threshold,
        exit_threshold=profile.vad_exit_threshold,
        min_speech_ms=profile.vad_min_speech_ms,
        min_silence_ms=profile.vad_min_silence_ms,
        speech_pad_ms=profile.vad_speech_pad_ms,
    )


def inspect_source(
    source: Path,
    *,
    profile: Profile | None,
    detector: VadDetector | None = None,
) -> dict[str, object]:
    require_runtime()
    if not source.is_file():
        raise PipelineError(f"Input media does not exist: {source}")
    policy = profile or PROFILES["product-demo"]
    detector = detector or SileroOnnxVad()
    probe = probe_media(source)
    audio, sample_rate = decode_audio(source)
    regions = _detect_speech(audio, sample_rate, policy, detector)
    analysis = analyze_signal(audio, sample_rate, regions, policy)
    program = measure_loudness(
        source,
        target_lufs=policy.target_lufs,
        target_lra=policy.target_lra_lu,
        target_true_peak=policy.target_true_peak_dbtp,
    )
    observations = [*analysis.observations, _program_observation(program)]
    payload: dict[str, object] = {
        "schema_version": "1",
        "kind": "audio_inspection",
        "engine": {
            "name": "audio-processing-cli",
            "version": __version__,
            "ffmpeg": ffmpeg_version(),
            "analysis_policy_version": "1",
            "vad": {
                "model": detector.model_version,
                "model_sha256": MODEL_SHA256,
                "source": MODEL_URL,
            },
        },
        "source": {
            **media_summary(source, probe),
            "decoded_audio": decoded_audio_timing(len(audio), sample_rate),
        },
        "observations": observations,
        "regions": _region_manifest(analysis),
        "region_basis": {**REGION_BASIS, "detection": "inspected_source"},
        "measurements": {
            "program": _round_loudness(program),
            "program_actual": actual_program(program),
        },
    }
    if profile is not None:
        payload["profile"] = profile.as_dict()
        payload["rule_evaluations"] = evaluate_profile(analysis, profile, program)
    return payload


def prepare_run(
    profile: Profile,
    detector: VadDetector | None,
    source: Path,
    *,
    output: Path | None,
    dry_run: bool,
    allow_enhanced_input: bool,
    progress: ProgressSink | None = None,
) -> PreparedRun:
    """Validate and measure the canonical source before any treatment is applied."""
    with progress_stage(progress, "preparation"):
        require_runtime()
        if not source.is_file():
            raise PipelineError(f"Input media does not exist: {source}")
        if not dry_run and output is None:
            raise PipelineError("--output is required unless --dry-run is used")
        if output is not None and source.resolve() == output.resolve():
            raise PipelineError("Output must not overwrite the canonical input")

        probe = probe_media(source)
        if is_enhanced_media(probe) and not allow_enhanced_input:
            raise PipelineError(
                "Input is marked as an enhanced render. Start from the canonical original, "
                "or pass --allow-enhanced-input when this is intentional."
            )
        source_info = media_summary(source, probe)
        resolved_detector = detector or SileroOnnxVad()
    with progress_stage(progress, "decode"):
        audio, sample_rate = decode_audio(source)
        source_info["decoded_audio"] = decoded_audio_timing(len(audio), sample_rate)
    with progress_stage(progress, "inspection"):
        speech_regions = _detect_speech(audio, sample_rate, profile, resolved_detector)
        analysis = analyze_signal(audio, sample_rate, speech_regions, profile)
        before_program = measure_loudness(
            source,
            target_lufs=profile.target_lufs,
            target_lra=profile.target_lra_lu,
            target_true_peak=profile.target_true_peak_dbtp,
        )
        before_regional = regional_measurements(audio, sample_rate, analysis)
    return PreparedRun(
        source=source,
        output=output,
        dry_run=dry_run,
        probe=probe,
        source_info=source_info,
        detector=resolved_detector,
        audio=audio,
        sample_rate=sample_rate,
        analysis=analysis,
        before_program=before_program,
        before_regional=before_regional,
    )
