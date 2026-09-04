"""Signal observations and profile evaluation."""

from __future__ import annotations

import numpy as np

from ..profiles import Profile
from ..vad_contract import SpeechRegion
from .levels import EPSILON, rms_dbfs
from .regions import (
    SignalAnalysis,
    _detect_machine_regions,
    _spectral_environment_metrics,
    _speech_samples,
    _time_scope,
)


def analyze_signal(
    audio: np.ndarray,
    sample_rate: int,
    speech_regions: list[SpeechRegion],
    profile: Profile,
) -> SignalAnalysis:
    duration = audio.shape[0] / sample_rate
    speech = _speech_samples(audio, speech_regions, sample_rate)
    speech_level = rms_dbfs(speech)
    machine_regions, noise_floor, machine_threshold = _detect_machine_regions(
        audio, sample_rate, speech_regions, speech_level
    )
    if audio.shape[1] == 1:
        channel_difference = 0.0
        correlation: float | None = None
        left_rms = right_rms = rms_dbfs(audio[:, 0])
    else:
        left_rms = rms_dbfs(audio[:, 0])
        right_rms = rms_dbfs(audio[:, 1])
        channel_difference = left_rms - right_rms
        if float(np.std(audio[:, 0])) < EPSILON or float(np.std(audio[:, 1])) < EPSILON:
            correlation = None
        else:
            correlation = float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1])
    subbass_ratio, hum_excess, dc_offset = _spectral_environment_metrics(speech, sample_rate)

    observations: list[dict[str, object]] = [
        {
            "observation_id": "channel_001",
            "type": "channel_level_difference",
            "scope": _time_scope(),
            "left_rms_dbfs": round(left_rms, 3),
            "right_rms_dbfs": round(right_rms, 3),
            "difference_db": round(channel_difference, 3),
            "correlation": None if correlation is None else round(correlation, 6),
        },
        {
            "observation_id": "environment_001",
            "type": "speech_environment_spectrum",
            "scope": {"time": "speech_regions", "frequency": "all"},
            "subbass_power_ratio": round(subbass_ratio, 6),
            "maximum_hum_excess_db": round(hum_excess, 3),
            "dc_offset": round(dc_offset, 9),
            "estimated_noise_floor_dbfs": round(noise_floor, 3),
        },
        {
            "observation_id": "speech_001",
            "type": "speech_program_level",
            "region_id": "speech_program",
            "scope": {"time": "speech_regions", "frequency": "all"},
            "measured_rms_dbfs": round(speech_level, 3),
            "non_speech_program_detection_threshold_dbfs": round(machine_threshold, 3),
            "detected_region_count": len(speech_regions),
            "detected_duration_seconds": round(
                sum(region.end - region.start for region in speech_regions), 6
            ),
        },
    ]
    observations.extend(region.as_observation() for region in machine_regions)
    return SignalAnalysis(
        duration_seconds=duration,
        observations=observations,
        speech_regions=speech_regions,
        machine_regions=machine_regions,
        speech_rms_dbfs=speech_level,
        noise_floor_dbfs=noise_floor,
        machine_detection_threshold_dbfs=machine_threshold,
        channel_difference_db=channel_difference,
        channel_correlation=correlation,
        subbass_power_ratio=subbass_ratio,
        hum_excess_db=hum_excess,
        dc_offset=dc_offset,
    )


def evaluate_profile(
    analysis: SignalAnalysis,
    profile: Profile,
    program_loudness: dict[str, float],
) -> list[dict[str, object]]:
    channel_status = (
        "inside_target"
        if abs(analysis.channel_difference_db) <= profile.channel_no_op_db
        else "outside_target"
    )
    evaluations: list[dict[str, object]] = [
        {
            "observation_id": "channel_001",
            "rule": f"{profile.name}.channel-level-difference",
            "target_maximum_absolute_db": profile.channel_no_op_db,
            "observed_db": round(analysis.channel_difference_db, 3),
            "status": channel_status,
        },
        {
            "observation_id": "program_001",
            "rule": f"{profile.name}.program-loudness",
            "target_lufs": profile.target_lufs,
            "observed_lufs": program_loudness.get("input_i"),
            "status": (
                "inside_target"
                if abs(program_loudness.get("input_i", -240.0) - profile.target_lufs) <= 0.5
                else "outside_target"
            ),
        },
    ]
    for region in analysis.machine_regions:
        difference = region.difference_from_speech_db
        evaluations.append(
            {
                "observation_id": f"level_{region.region_id.rsplit('_', 1)[-1]}",
                "rule": f"{profile.name}.machine-audio-relative-level",
                "target_difference_db": {
                    "minimum": profile.machine_relative_minimum_lu,
                    "maximum": profile.machine_relative_maximum_lu,
                },
                "observed_difference_db": round(difference, 3),
                "status": (
                    "inside_target"
                    if profile.machine_relative_minimum_lu
                    <= difference
                    <= profile.machine_relative_maximum_lu
                    else "outside_target"
                ),
            }
        )
    return evaluations
