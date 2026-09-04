"""Speech treatment boundaries and non-speech program region detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from ..profiles import Profile
from ..vad import SpeechRegion
from .levels import EPSILON, rms_dbfs


def _time_scope(start: float | None = None, end: float | None = None) -> dict[str, object]:
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


def _hard_region_mask(length: int, regions: list[SpeechRegion], sample_rate: int) -> np.ndarray:
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
            phase = np.linspace(0.0, math.pi / 2.0, width, endpoint=False, dtype=np.float32)
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


def _speech_samples(audio: np.ndarray, regions: list[SpeechRegion], sample_rate: int) -> np.ndarray:
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
            cluster for cluster in clusters if cluster[0] < region.end and cluster[1] > region.start
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
            machine.end for machine in analysis.machine_regions if machine.end <= region.start
        ]
        if preceding_machine_ends:
            start = max(start, max(preceding_machine_ends) + fade_seconds)
        following_machine_starts = [
            machine.start for machine in analysis.machine_regions if machine.start >= region.end
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
    voice_band = (frequencies >= 20.0) & (frequencies <= min(8000.0, sample_rate / 2 - 1))
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
