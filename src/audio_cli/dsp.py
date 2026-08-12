from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from .adjustments import GainAdjustment
from .profiles import Profile
from .vad import SpeechRegion

EPSILON = 1e-12


def amplitude_to_db(value: float) -> float:
    return 20.0 * math.log10(max(abs(value), EPSILON))


def rms_dbfs(samples: np.ndarray) -> float:
    array = np.asarray(samples, dtype=np.float64)
    if array.size == 0:
        return -240.0
    return amplitude_to_db(float(np.sqrt(np.mean(np.square(array)) + EPSILON)))


def peak_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -240.0
    return amplitude_to_db(float(np.max(np.abs(samples))))


def _time_scope(
    start: float | None = None, end: float | None = None
) -> dict[str, object]:
    time: str | dict[str, float]
    if start is None or end is None:
        time = "all"
    else:
        time = {"start": round(start, 6), "end": round(end, 6)}
    return {"time": time, "frequency": "all"}


@dataclass(frozen=True)
class MachineRegion:
    region_id: str
    start: float
    end: float
    measured_rms_dbfs: float
    difference_from_speech_db: float
    overlaps_speech: bool = False

    def as_observation(self) -> dict[str, object]:
        return {
            "observation_id": f"level_{self.region_id.rsplit('_', 1)[-1]}",
            "type": "regional_level_difference",
            "region_id": self.region_id,
            "scope": _time_scope(self.start, self.end),
            "measured_rms_dbfs": round(self.measured_rms_dbfs, 3),
            "reference_region": "speech_program",
            "difference_db": round(self.difference_from_speech_db, 3),
            "overlaps_reference": self.overlaps_speech,
        }


@dataclass
class SignalAnalysis:
    duration_seconds: float
    observations: list[dict[str, object]]
    speech_regions: list[SpeechRegion]
    machine_regions: list[MachineRegion]
    speech_rms_dbfs: float
    noise_floor_dbfs: float
    machine_detection_threshold_dbfs: float
    channel_difference_db: float
    channel_correlation: float | None
    subbass_power_ratio: float
    hum_excess_db: float
    dc_offset: float


def _hard_region_mask(
    length: int, regions: list[SpeechRegion], sample_rate: int
) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for region in regions:
        start = max(0, round(region.start * sample_rate))
        end = min(length, round(region.end * sample_rate))
        mask[start:end] = True
    return mask


def smooth_time_mask(
    length: int,
    intervals: list[tuple[float, float]],
    sample_rate: int,
    fade_ms: int,
    *,
    transition_placement: Literal["inside", "outside"] = "inside",
) -> np.ndarray:
    if transition_placement not in {"inside", "outside"}:
        raise ValueError("transition_placement must be either 'inside' or 'outside'")
    mask = np.zeros(length, dtype=np.float32)
    fade = max(1, round(sample_rate * fade_ms / 1000))
    merged_intervals: list[tuple[float, float]] = []
    for start_seconds, end_seconds in sorted(intervals):
        if merged_intervals and start_seconds <= merged_intervals[-1][1]:
            merged_intervals[-1] = (
                merged_intervals[-1][0],
                max(merged_intervals[-1][1], end_seconds),
            )
        else:
            merged_intervals.append((start_seconds, end_seconds))
    for start_seconds, end_seconds in merged_intervals:
        start = max(0, round(start_seconds * sample_rate))
        end = min(length, round(end_seconds * sample_rate))
        if end <= start:
            continue
        if transition_placement == "inside":
            width = min(fade, max(1, (end - start) // 2))
            local = np.ones(end - start, dtype=np.float32)
            phase = np.linspace(
                0.0, math.pi / 2.0, width, endpoint=False, dtype=np.float32
            )
            ramp = np.square(np.sin(phase))
            local[:width] = ramp
            local[-width:] = ramp[::-1]
            mask[start:end] = np.maximum(mask[start:end], local)
            continue

        # Speech-scoped effects must be fully engaged at the first and last
        # detected samples. Place their equal-power transitions in the
        # surrounding context so opening and closing phonemes are not leveled
        # progressively inside the active region.
        mask[start:end] = 1.0
        left = max(0, start - fade)
        left_width = start - left
        if left_width:
            phase = np.linspace(
                0.0,
                math.pi / 2.0,
                left_width,
                endpoint=False,
                dtype=np.float32,
            )
            mask[left:start] = np.maximum(mask[left:start], np.square(np.sin(phase)))
        right = min(length, end + fade)
        right_width = right - end
        if right_width:
            phase = np.linspace(
                math.pi / 2.0,
                0.0,
                right_width,
                endpoint=False,
                dtype=np.float32,
            )
            mask[end:right] = np.maximum(mask[end:right], np.square(np.sin(phase)))
    return mask


def _speech_samples(
    audio: np.ndarray, regions: list[SpeechRegion], sample_rate: int
) -> np.ndarray:
    mask = _hard_region_mask(audio.shape[0], regions, sample_rate)
    return audio[mask]


def _frame_levels(
    mono: np.ndarray, sample_rate: int, frame_seconds: float = 0.1
) -> tuple[np.ndarray, int]:
    frame = max(1, round(sample_rate * frame_seconds))
    count = math.ceil(mono.size / frame)
    padded = np.pad(mono, (0, count * frame - mono.size))
    framed = padded.reshape(count, frame).astype(np.float64)
    levels = 20.0 * np.log10(np.sqrt(np.mean(np.square(framed), axis=1) + EPSILON))
    return levels, frame


def resolve_speech_treatment_intervals(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[list[tuple[float, float]], dict[str, object]]:
    """Grow VAD seeds to silence-anchored acoustic voice boundaries."""
    frame_seconds = 0.02
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    levels, frame_samples = _frame_levels(mono, sample_rate, frame_seconds)
    activity_threshold = max(
        analysis.noise_floor_dbfs + profile.voice_boundary_noise_margin_db,
        analysis.speech_rms_dbfs - profile.voice_boundary_speech_margin_db,
    )
    active_frames = np.flatnonzero(levels >= activity_threshold)
    maximum_gap_frames = max(
        0,
        round(profile.voice_boundary_bridge_silence_ms / (frame_seconds * 1000.0)),
    )
    clusters: list[tuple[float, float]] = []
    if active_frames.size:
        first = last = int(active_frames[0])
        for raw_index in active_frames[1:]:
            index = int(raw_index)
            if index - last - 1 <= maximum_gap_frames:
                last = index
                continue
            clusters.append(
                (
                    first * frame_samples / sample_rate,
                    min(
                        analysis.duration_seconds,
                        (last + 1) * frame_samples / sample_rate,
                    ),
                )
            )
            first = last = index
        clusters.append(
            (
                first * frame_samples / sample_rate,
                min(
                    analysis.duration_seconds,
                    (last + 1) * frame_samples / sample_rate,
                ),
            )
        )

    fade_seconds = profile.region_fade_ms / 1000.0
    guard_seconds = profile.voice_boundary_guard_ms / 1000.0
    search_seconds = profile.voice_boundary_search_ms / 1000.0
    intervals: list[tuple[float, float]] = []
    resolutions: list[dict[str, object]] = []
    for index, region in enumerate(analysis.speech_regions, 1):
        overlapping = [
            cluster
            for cluster in clusters
            if cluster[0] < region.end and cluster[1] > region.start
        ]
        if overlapping:
            acoustic_start = max(
                region.start - search_seconds,
                min(cluster[0] for cluster in overlapping),
            )
            acoustic_end = min(
                region.end + search_seconds,
                max(cluster[1] for cluster in overlapping),
            )
            start = min(region.start, acoustic_start - guard_seconds)
            end = max(region.end, acoustic_end + guard_seconds)
        else:
            acoustic_start = region.start
            acoustic_end = region.end
            start = region.start
            end = region.end

        preceding_machine_ends = [
            machine.end
            for machine in analysis.machine_regions
            if machine.end <= region.start
        ]
        if preceding_machine_ends:
            start = max(start, max(preceding_machine_ends) + fade_seconds)
        following_machine_starts = [
            machine.start
            for machine in analysis.machine_regions
            if machine.start >= region.end
        ]
        if following_machine_starts:
            end = min(end, min(following_machine_starts) - fade_seconds)

        start = max(0.0, min(start, region.start))
        end = min(analysis.duration_seconds, max(end, region.end))
        intervals.append((start, end))
        resolutions.append(
            {
                "speech_region_id": f"speech_{index:03d}",
                "vad_scope": {
                    "start": round(region.start, 6),
                    "end": round(region.end, 6),
                },
                "acoustic_activity_scope": {
                    "start": round(acoustic_start, 6),
                    "end": round(acoustic_end, 6),
                },
                "treatment_scope": {
                    "start": round(start, 6),
                    "end": round(end, 6),
                },
                "start_extension_ms": round((region.start - start) * 1000.0, 3),
                "end_extension_ms": round((end - region.end) * 1000.0, 3),
            }
        )

    return intervals, {
        "placement": "silence_anchored_outside_voice_activity",
        "fade_in_ms": profile.region_fade_ms,
        "fade_out_ms": profile.region_fade_ms,
        "activity_frame_ms": round(frame_seconds * 1000.0),
        "activity_threshold_dbfs": round(activity_threshold, 3),
        "bridge_silence_ms": profile.voice_boundary_bridge_silence_ms,
        "guard_before_activity_ms": profile.voice_boundary_guard_ms,
        "guard_after_activity_ms": profile.voice_boundary_guard_ms,
        "maximum_search_ms": profile.voice_boundary_search_ms,
        "minimum_mix_inside_treatment_region": 1.0,
        "regions": resolutions,
    }


def _spectral_environment_metrics(
    speech: np.ndarray,
    sample_rate: int,
) -> tuple[float, float, float]:
    if speech.size == 0:
        return 0.0, 0.0, 0.0
    mono = np.mean(speech, axis=1) if speech.ndim == 2 else speech
    dc_offset = float(np.mean(mono))
    if mono.size < 1024:
        return 0.0, 0.0, dc_offset
    frequencies, power = signal.welch(
        mono.astype(np.float64),
        fs=sample_rate,
        nperseg=min(4096, mono.size),
        noverlap=None,
        detrend="constant",
    )
    voice_band = (frequencies >= 20.0) & (
        frequencies <= min(8000.0, sample_rate / 2 - 1)
    )
    subbass = (frequencies >= 20.0) & (frequencies < 70.0)
    total_power = float(np.sum(power[voice_band])) + EPSILON
    subbass_ratio = float(np.sum(power[subbass]) / total_power)
    hum_excesses: list[float] = []
    for center in (60.0, 120.0, 180.0):
        peak_band = np.abs(frequencies - center) <= 1.5
        neighborhood = (np.abs(frequencies - center) >= 4.0) & (
            np.abs(frequencies - center) <= 12.0
        )
        if np.any(peak_band) and np.any(neighborhood):
            peak = float(np.max(power[peak_band])) + EPSILON
            reference = float(np.median(power[neighborhood])) + EPSILON
            hum_excesses.append(10.0 * math.log10(peak / reference))
    return subbass_ratio, max(hum_excesses, default=0.0), dc_offset


def _detect_machine_regions(
    audio: np.ndarray,
    sample_rate: int,
    speech_regions: list[SpeechRegion],
    speech_reference_dbfs: float,
) -> tuple[list[MachineRegion], float, float]:
    mono = np.mean(audio, axis=1)
    levels, frame = _frame_levels(mono, sample_rate)
    midpoint_samples = np.arange(levels.size) * frame + frame // 2
    speech_frames = np.zeros(levels.size, dtype=bool)
    for region in speech_regions:
        speech_frames |= (midpoint_samples >= region.start * sample_rate) & (
            midpoint_samples < region.end * sample_rate
        )
    finite = levels[np.isfinite(levels)]
    noise_floor = float(np.percentile(finite, 20)) if finite.size else -120.0
    # V1 balances only salient non-speech program, not every audible room-noise
    # fluctuation. A region must clear both the estimated floor and untreated
    # speech reference; quiet/ambiguous events remain canonical and unmodified.
    threshold = max(noise_floor + 15.0, speech_reference_dbfs + 3.0, -55.0)
    candidates = (~speech_frames) & (levels >= threshold)
    speech_sample_spans = [
        (
            max(0, round(region.start * sample_rate)),
            min(audio.shape[0], round(region.end * sample_rate)),
        )
        for region in speech_regions
    ]

    # Close gaps up to 200 ms so one sound with internal decay remains one stable region.
    max_gap_frames = 2
    indices = np.flatnonzero(candidates)
    groups: list[tuple[int, int]] = []
    if indices.size:
        start = previous = int(indices[0])
        for current_raw in indices[1:]:
            current = int(current_raw)
            if current - previous > max_gap_frames + 1:
                groups.append((start, previous + 1))
                start = current
            previous = current
        groups.append((start, previous + 1))

    regions: list[MachineRegion] = []
    for start_frame, end_frame in groups:
        start = max(0, start_frame * frame - round(0.04 * sample_rate))
        end = min(audio.shape[0], end_frame * frame + round(0.04 * sample_rate))
        if end - start < round(0.12 * sample_rate):
            continue
        level = rms_dbfs(audio[start:end])
        if level < noise_floor + 8.0:
            continue
        region_id = f"machine_audio_{len(regions) + 1:03d}"
        overlaps_speech = any(
            start < speech_end and end > speech_start
            for speech_start, speech_end in speech_sample_spans
        )
        regions.append(
            MachineRegion(
                region_id=region_id,
                start=start / sample_rate,
                end=end / sample_rate,
                measured_rms_dbfs=level,
                difference_from_speech_db=level - speech_reference_dbfs,
                overlaps_speech=overlaps_speech,
            )
        )
    return regions, noise_floor, threshold


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
    subbass_ratio, hum_excess, dc_offset = _spectral_environment_metrics(
        speech, sample_rate
    )

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
                if abs(program_loudness.get("input_i", -240.0) - profile.target_lufs)
                <= 0.5
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


def apply_channel_balance(
    audio: np.ndarray,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[np.ndarray, dict[str, object]]:
    if audio.shape[1] == 1:
        return audio, {"status": "no_op", "reason": "mono_source", "operations": []}
    difference = analysis.channel_difference_db
    if abs(difference) <= profile.channel_no_op_db:
        return audio, {
            "status": "no_op",
            "reason": "channel_difference_below_threshold",
            "operations": [],
        }
    if (
        analysis.channel_correlation is None
        or analysis.channel_correlation < profile.channel_correlation_minimum
    ):
        return audio, {
            "status": "abstained",
            "reason": "stereo_difference_may_be_intentional",
            "operations": [],
            "observed_correlation": analysis.channel_correlation,
        }
    half = float(
        np.clip(
            difference / 2.0,
            -profile.channel_max_correction_db,
            profile.channel_max_correction_db,
        )
    )
    gains_db = [-half, half]
    gains = np.power(10.0, np.asarray(gains_db, dtype=np.float64) / 20.0)
    output = audio * gains[None, :]
    return output.astype(np.float32), {
        "status": "applied",
        "reason": "correlated_channels_with_level_mismatch",
        "operations": [
            {
                "type": "linked-channel-gain",
                "left_gain_db": round(gains_db[0], 3),
                "right_gain_db": round(gains_db[1], 3),
                "maximum_correction_db": profile.channel_max_correction_db,
            }
        ],
    }


def _blend(original: np.ndarray, processed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    shaped = mask[:, None]
    return (original * (1.0 - shaped) + processed * shaped).astype(np.float32)


def _notch(
    audio: np.ndarray, frequency: float, sample_rate: int, quality: float = 35.0
) -> np.ndarray:
    b, a = signal.iirnotch(frequency, quality, fs=sample_rate)
    return signal.lfilter(b, a, audio, axis=0).astype(np.float32)


def apply_environment_cleanup(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[np.ndarray, dict[str, object]]:
    if not analysis.speech_regions:
        return audio, {
            "status": "abstained",
            "reason": "no_speech_detected",
            "operations": [],
        }
    intervals, resolved_transition = resolve_speech_treatment_intervals(
        audio, sample_rate, profile, analysis
    )
    mask = smooth_time_mask(
        audio.shape[0],
        intervals,
        sample_rate,
        profile.region_fade_ms,
        transition_placement=profile.speech_transition_placement,
    )
    processed = audio.astype(np.float64)
    operations: list[dict[str, object]] = []
    if (
        analysis.subbass_power_ratio >= profile.subbass_ratio_threshold
        or abs(analysis.dc_offset) > 1e-4
    ):
        sos = signal.butter(
            2, profile.highpass_hz, btype="highpass", fs=sample_rate, output="sos"
        )
        processed = signal.sosfilt(sos, processed, axis=0)
        operations.append(
            {
                "type": "minimum-phase-highpass",
                "cutoff_hz": profile.highpass_hz,
                "order": 2,
                "affected_scope": "speech_regions",
            }
        )
    if analysis.hum_excess_db >= profile.hum_excess_db_threshold:
        for frequency in (60.0, 120.0, 180.0):
            processed = _notch(processed, frequency, sample_rate)
        operations.append(
            {
                "type": "minimum-phase-dehum",
                "frequencies_hz": [60.0, 120.0, 180.0],
                "quality_factor": 35.0,
                "affected_scope": "speech_regions",
            }
        )
    broadband: dict[str, object]
    if analysis.speech_rms_dbfs < -45.0:
        broadband = {
            "component": "broadband-denoise",
            "status": "abstained",
            "reason": "input_speech_too_quiet_for_reliable_noise_estimate",
        }
    elif analysis.noise_floor_dbfs <= analysis.speech_rms_dbfs - 20.0:
        broadband = {
            "component": "broadband-denoise",
            "status": "no_op",
            "reason": "stationary_noise_below_threshold",
        }
    else:
        broadband = {
            "component": "broadband-denoise",
            "status": "abstained",
            "reason": "conservative_v1_has_no_reliable_stationary_noise_profile",
        }
    output = (
        _blend(audio, np.asarray(processed, dtype=np.float32), mask)
        if operations
        else audio
    )
    if operations:
        status, reason = "applied", "eligible_environmental_cleanup_resolved"
    elif broadband["status"] == "abstained":
        status, reason = "abstained", str(broadband["reason"])
    else:
        status, reason = "no_op", "environmental_components_below_threshold"
    return output, {
        "status": status,
        "reason": reason,
        "operations": operations,
        "resolved_transition": resolved_transition if operations else None,
        "component_evaluations": [broadband],
    }


def _peaking_coefficients(
    gain_db: float,
    center_hz: float,
    quality: float,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * math.pi * center_hz / sample_rate
    alpha = math.sin(omega) / (2.0 * quality)
    cos_omega = math.cos(omega)
    b = np.asarray(
        [1 + alpha * amplitude, -2 * cos_omega, 1 - alpha * amplitude],
        dtype=np.float64,
    )
    a = np.asarray(
        [1 + alpha / amplitude, -2 * cos_omega, 1 - alpha / amplitude],
        dtype=np.float64,
    )
    return b / a[0], a / a[0]


def apply_frequency_adjustments(
    audio: np.ndarray,
    sample_rate: int,
    adjustments: list[GainAdjustment],
    duration: float,
    fade_ms: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = audio.copy()
    resolved: list[dict[str, object]] = []
    for adjustment in adjustments:
        if adjustment.is_full_band:
            continue
        assert adjustment.frequency_low_hz is not None
        assert adjustment.frequency_high_hz is not None
        center = math.sqrt(adjustment.frequency_low_hz * adjustment.frequency_high_hz)
        bandwidth = adjustment.frequency_high_hz - adjustment.frequency_low_hz
        quality = max(0.25, center / bandwidth)
        b, a = _peaking_coefficients(adjustment.gain_db, center, quality, sample_rate)
        filtered = signal.lfilter(b, a, output, axis=0).astype(np.float32)
        end = adjustment.resolved_end(duration)
        mask = smooth_time_mask(
            output.shape[0], [(adjustment.start, end)], sample_rate, fade_ms
        )
        output = _blend(output, filtered, mask)
        item = adjustment.as_dict(duration)
        item["resolved_filter"] = {
            "kind": "minimum-phase-peaking-biquad",
            "center_hz": round(center, 3),
            "quality_factor": round(quality, 6),
            "boundary_fade_ms": fade_ms,
        }
        resolved.append(item)
    return output, resolved


def _compress(
    audio: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    ratio: float,
) -> tuple[np.ndarray, float]:
    frame = max(1, round(sample_rate * 0.02))
    count = math.ceil(audio.shape[0] / frame)
    pad = count * frame - audio.shape[0]
    padded = np.pad(audio, ((0, pad), (0, 0)))
    levels = 20.0 * np.log10(
        np.sqrt(
            np.mean(
                np.square(padded.reshape(count, frame, audio.shape[1])), axis=(1, 2)
            )
            + EPSILON
        )
    )
    over = np.maximum(levels - threshold_dbfs, 0.0)
    reduction_db = -(over - over / ratio)
    centers = np.minimum(np.arange(count) * frame + frame // 2, audio.shape[0] - 1)
    sample_positions = np.arange(audio.shape[0])
    interpolated = np.interp(
        sample_positions,
        centers,
        reduction_db,
        left=reduction_db[0],
        right=reduction_db[-1],
    )
    gains = np.power(10.0, interpolated / 20.0)
    return (audio * gains[:, None]).astype(np.float32), float(np.min(reduction_db))


def apply_voice_enhancement(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[np.ndarray, dict[str, object]]:
    if not analysis.speech_regions:
        return audio, {
            "status": "abstained",
            "reason": "no_speech_detected",
            "operations": [],
        }
    intervals, resolved_transition = resolve_speech_treatment_intervals(
        audio, sample_rate, profile, analysis
    )
    mask = smooth_time_mask(
        audio.shape[0],
        intervals,
        sample_rate,
        profile.region_fade_ms,
        transition_placement=profile.speech_transition_placement,
    )
    b, a = _peaking_coefficients(
        profile.voice_presence_gain_db, 3000.0, 0.9, sample_rate
    )
    tonal = signal.lfilter(b, a, audio, axis=0).astype(np.float32)
    speech_mask = _hard_region_mask(
        audio.shape[0], analysis.speech_regions, sample_rate
    )
    measured_before = rms_dbfs(tonal[speech_mask])
    desired_gain = profile.voice_target_rms_dbfs - measured_before
    gain_db = float(
        np.clip(
            desired_gain, -profile.voice_max_attenuation_db, profile.voice_max_gain_db
        )
    )
    leveled = tonal * (10.0 ** (gain_db / 20.0))
    compressed, maximum_reduction = _compress(
        leveled,
        sample_rate,
        profile.compressor_threshold_dbfs,
        profile.compressor_ratio,
    )
    output = _blend(audio, compressed, mask)
    measured_after = rms_dbfs(output[speech_mask])
    return output, {
        "status": "applied",
        "reason": "speech_regions_received_bounded_tonal_and_level_correction",
        "operations": [
            {
                "type": "speech-presence-eq",
                "center_hz": 3000.0,
                "gain_db": profile.voice_presence_gain_db,
                "quality_factor": 0.9,
                "filter": "minimum-phase-peaking-biquad",
            },
            {
                "type": "speech-leveling",
                "measured_before_rms_dbfs": round(measured_before, 3),
                "target_rms_dbfs": profile.voice_target_rms_dbfs,
                "resolved_gain_db": round(gain_db, 3),
                "maximum_boost_db": profile.voice_max_gain_db,
                "measured_after_rms_dbfs": round(measured_after, 3),
            },
            {
                "type": "speech-compression",
                "threshold_dbfs": profile.compressor_threshold_dbfs,
                "ratio": profile.compressor_ratio,
                "maximum_gain_reduction_db": round(maximum_reduction, 3),
            },
        ],
        "affected_regions": [
            f"speech_{index:03d}" for index in range(1, len(intervals) + 1)
        ],
        "resolved_transition": resolved_transition,
    }


def apply_source_balance(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    analysis: SignalAnalysis,
) -> tuple[np.ndarray, dict[str, object]]:
    if not analysis.speech_regions:
        return audio, {
            "status": "abstained",
            "reason": "no_speech_reference",
            "operations": [],
        }
    if not analysis.machine_regions:
        return audio, {
            "status": "no_op",
            "reason": "no_non_speech_program_regions",
            "operations": [],
        }
    speech = _speech_samples(audio, analysis.speech_regions, sample_rate)
    speech_reference = rms_dbfs(speech)
    output = audio.copy()
    operations: list[dict[str, object]] = []
    inside_target: list[str] = []
    abstained: list[str] = []
    for region in analysis.machine_regions:
        start = max(0, round(region.start * sample_rate))
        end = min(output.shape[0], round(region.end * sample_rate))
        if region.overlaps_speech:
            abstained.append(region.region_id)
            continue
        measured = rms_dbfs(output[start:end])
        difference = measured - speech_reference
        if (
            profile.machine_relative_minimum_lu
            <= difference
            <= profile.machine_relative_maximum_lu
        ):
            inside_target.append(region.region_id)
            continue
        requested = profile.machine_relative_target_lu - difference
        resolved_gain = float(
            np.clip(
                requested,
                -profile.machine_max_attenuation_db,
                profile.machine_max_boost_db,
            )
        )
        mask = smooth_time_mask(
            output.shape[0],
            [(region.start, region.end)],
            sample_rate,
            profile.region_fade_ms,
        )
        gained = output * (10.0 ** (resolved_gain / 20.0))
        output = _blend(output, gained, mask)
        operations.append(
            {
                "type": "regional-full-band-gain",
                "region_id": region.region_id,
                "scope": _time_scope(region.start, region.end),
                "reference_speech_rms_dbfs": round(speech_reference, 3),
                "measured_before_rms_dbfs": round(measured, 3),
                "observed_difference_db": round(difference, 3),
                "target_difference_db": profile.machine_relative_target_lu,
                "requested_gain_db": round(requested, 3),
                "resolved_gain_db": round(resolved_gain, 3),
                "boundary_fade_ms": profile.region_fade_ms,
            }
        )
    if operations:
        status, reason = (
            "applied",
            "non_overlapping_regions_balanced_to_speech_reference",
        )
    elif abstained:
        status, reason = "abstained", "speech_and_machine_audio_overlap"
    else:
        status, reason = "no_op", "all_machine_regions_inside_target"
    return output, {
        "status": status,
        "reason": reason,
        "operations": operations,
        "inside_target_regions": inside_target,
        "abstained_regions": abstained,
    }


def apply_fullband_adjustments(
    audio: np.ndarray,
    sample_rate: int,
    adjustments: list[GainAdjustment],
    duration: float,
    fade_ms: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = audio.copy()
    resolved: list[dict[str, object]] = []
    for adjustment in adjustments:
        if not adjustment.is_full_band:
            continue
        end = adjustment.resolved_end(duration)
        mask = smooth_time_mask(
            output.shape[0], [(adjustment.start, end)], sample_rate, fade_ms
        )
        gained = output * (10.0 ** (adjustment.gain_db / 20.0))
        output = _blend(output, gained, mask)
        item = adjustment.as_dict(duration)
        item["resolved_transition"] = {"boundary_fade_ms": fade_ms}
        resolved.append(item)
    return output, resolved


def apply_machine_region_corrections(
    audio: np.ndarray,
    sample_rate: int,
    analysis: SignalAnalysis,
    corrections_db: dict[str, float],
    fade_ms: int,
) -> np.ndarray:
    output = audio.copy()
    by_id = {region.region_id: region for region in analysis.machine_regions}
    for region_id, gain_db in corrections_db.items():
        region = by_id[region_id]
        mask = smooth_time_mask(
            output.shape[0],
            [(region.start, region.end)],
            sample_rate,
            fade_ms,
        )
        gained = output * (10.0 ** (gain_db / 20.0))
        output = _blend(output, gained, mask)
    return output


def regional_measurements(
    audio: np.ndarray,
    sample_rate: int,
    analysis: SignalAnalysis,
) -> dict[str, object]:
    speech_level = rms_dbfs(
        _speech_samples(audio, analysis.speech_regions, sample_rate)
    )
    machines: list[dict[str, object]] = []
    for region in analysis.machine_regions:
        start = max(0, round(region.start * sample_rate))
        end = min(audio.shape[0], round(region.end * sample_rate))
        measured = rms_dbfs(audio[start:end])
        machines.append(
            {
                "region_id": region.region_id,
                "measured_rms_dbfs": round(measured, 3),
                "difference_from_speech_db": round(measured - speech_level, 3),
            }
        )
    return {
        "speech_program_rms_dbfs": round(speech_level, 3),
        "machine_regions": machines,
        "sample_peak_dbfs": round(peak_dbfs(audio), 3),
    }
