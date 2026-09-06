"""Explicit model-denoising workflow over the original speech treatment scopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..dsp import (
    SignalAnalysis,
    apply_guided_denoise,
    calibrate_guide_input,
    finish_environment_cleanup,
    prepare_environment_filters,
    resolve_speech_treatment_intervals,
    smooth_time_mask,
)
from ..media import render_rnnoise
from ..profiles import Profile


@dataclass(frozen=True)
class DenoiserModel:
    """A managed model verified by the composition root, never provisioned here."""

    path: Path
    sha256: str
    provenance: dict[str, object]


def apply_model_cleanup(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
    model: DenoiserModel,
) -> tuple[np.ndarray, dict[str, object]]:
    if not analysis.speech_regions:
        return audio, {
            "status": "abstained",
            "reason": "no_speech_detected",
            "operations": [],
            "component_evaluations": [
                {
                    "component": "broadband-denoise",
                    "status": "abstained",
                    "reason": "no_speech_detected",
                    "algorithm": "model-guided-linked-mask-v1",
                }
            ],
        }
    intervals, transition = resolve_speech_treatment_intervals(
        audio, sample_rate, profile, analysis
    )
    mask = smooth_time_mask(
        len(audio),
        intervals,
        sample_rate,
        profile.region_fade_ms,
        transition_placement=profile.speech_transition_placement,
    )
    filtered, operations = prepare_environment_filters(audio, sample_rate, profile, analysis)
    # Keep model state continuous across the recording. Blending afterward protects
    # the samples outside treatment; resetting inference at each boundary would not.
    calibrated, guide_gain, calibration = calibrate_guide_input(
        filtered,
        sample_rate,
        analysis.speech_regions,
        target_rms_dbfs=-24.0,
        peak_limit_dbfs=-3.0,
        maximum_gain_db=40.0,
    )
    calibration["speech_reference"] = "original_detected_regions_after_environment_filters"
    if calibrated is None:
        component = {
            "component": "broadband-denoise",
            "status": "abstained",
            "reason": "no_speech_energy",
        }
        return finish_environment_cleanup(audio, filtered, mask, operations, component, transition)
    denoised = render_rnnoise(calibrated, sample_rate, model.path, model.sha256) / guide_gain
    processed, component, operation = apply_guided_denoise(filtered, denoised, sample_rate, profile)
    evidence = {
        "guide": "ffmpeg-arnndn",
        "guide_calibration": calibration,
        "model": model.provenance,
        "model_file_sha256": model.sha256,
        "sample_rate_hz": sample_rate,
        "compensated_guide_delay_samples": 480,
        "guide_state_scope": "continuous_recording_per_channel",
        "noise_reduction": {"status": "abstained", "reason": "no_clean_reference"},
        "speech_preservation": {"status": "abstained", "reason": "not_measured"},
    }
    component.update(evidence)
    if operation is not None:
        operation.update(evidence)
        operations.append(operation)
    return finish_environment_cleanup(audio, processed, mask, operations, component, transition)
