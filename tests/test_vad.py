"""Region building in `vad.py`, which had no test file at all.

That absence is why the defect below survived: `vad_min_silence_ms` is a declared profile
parameter, and nothing checked that it decided anything. Speech regions gate `voice-enhance`, so
their boundaries reach the rendered audio -- on a 27.8 s fixture the fix moved a real render from
6 regions to 10.

Every test here stubs `probabilities`, so no ONNX model is loaded and nothing is downloaded. The
detector's arithmetic is the thing under test; the model's judgement is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_cli.profiles import PROFILES
from audio_cli.vad import SileroOnnxVad

SAMPLE_RATE = 16_000
FRAME = SileroOnnxVad.frame_samples
FRAME_MS = FRAME / SAMPLE_RATE * 1000  # 32 ms


class StubVad(SileroOnnxVad):
    """The real region builder over chosen probabilities. No model, no network."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=np.float32)

    def probabilities(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return self._probabilities


def two_bursts(silence_frames: int, *, speech_frames: int = 20):
    """Two confident speech runs separated by `silence_frames` of confident silence."""
    return [0.9] * speech_frames + [0.0] * silence_frames + [0.9] * speech_frames


def detect(probabilities: list[float], profile_name: str = "transcription"):
    profile = PROFILES[profile_name]
    samples = np.zeros(len(probabilities) * FRAME, dtype=np.float32)
    return StubVad(probabilities).detect(
        samples,
        SAMPLE_RATE,
        threshold=profile.vad_threshold,
        exit_threshold=profile.vad_exit_threshold,
        min_speech_ms=profile.vad_min_speech_ms,
        min_silence_ms=profile.vad_min_silence_ms,
        speech_pad_ms=profile.vad_speech_pad_ms,
    )


def test_min_silence_ms_is_what_decides_a_break() -> None:
    """The defect this catches: a declared parameter that decided nothing.

    Regions were merged on the gap left *after* padding, against a hard-coded 300 ms. Padding
    shrinks every gap by `2 * speech_pad_ms`, so with the shipped 120 ms pad a 300 ms silence
    reached the merge as 60 ms and was rejoined: the effective break was 540 ms while the profile
    declared 300, and any value in between changed nothing at all.

    The tolerance is two frames, and it is derived rather than fitted. Silence is only counted on
    frames below `exit_threshold`, so the timer needs `ceil(min_silence / frame)` frames to
    accumulate and then one further silent frame to be evaluated on:

        ceil(4800 / 512) * 512 + 512 - 4800  =  5120 + 512 - 4800  =  832 samples  =  52 ms

    So 352 ms is the grid, and it is bounded by two frames for any `min_silence_ms`. 540 ms was
    not the grid -- it was two thresholds disagreeing, and it grew with `speech_pad_ms` rather
    than with the frame size.
    """
    profile = PROFILES["transcription"]
    declared = profile.vad_min_silence_ms
    crossover_frames = next(
        frames for frames in range(1, 60) if len(detect(two_bursts(frames))) == 2
    )
    crossover_ms = crossover_frames * FRAME_MS

    quantization_ms = (
        -(-profile.vad_min_silence_ms // FRAME_MS) * FRAME_MS  # ceil to the frame grid
        + FRAME_MS
        - profile.vad_min_silence_ms
    )
    assert crossover_ms > declared, "a gap at or under the declared silence must not split"
    assert crossover_ms - declared <= quantization_ms, (
        f"a break needs {crossover_ms:g} ms of silence where the profile declares {declared} ms, "
        f"which overshoots by more than the {quantization_ms:g} ms the frame grid explains; "
        f"something other than vad_min_silence_ms is deciding"
    )
    assert quantization_ms <= 2 * FRAME_MS, "the derivation above no longer holds"


@pytest.mark.parametrize("silence_frames", [1, 4, 8, 11, 16, 32])
def test_regions_stay_ordered_disjoint_and_non_empty(silence_frames: int) -> None:
    """Padding may push two surviving regions together; it must not overlap or invert them."""
    regions = detect(two_bursts(silence_frames))
    assert regions, "confident speech produced no region"
    for region in regions:
        assert region.start < region.end, f"empty or inverted region {region}"
    for earlier, later in zip(regions, regions[1:]):
        assert earlier.end <= later.start, f"{earlier} overlaps {later}"


def test_a_gap_far_below_the_threshold_is_one_region() -> None:
    """The merge still has to do its job: a short pause is not a break."""
    assert len(detect(two_bursts(2))) == 1


def test_regions_never_leave_the_signal() -> None:
    """Padding is clamped, so a region cannot start before zero or end past the last sample."""
    probabilities = [0.9] * 6  # speech from the first frame to the last
    regions = detect(probabilities)
    duration = len(probabilities) * FRAME / SAMPLE_RATE
    assert regions
    assert regions[0].start >= 0.0
    assert regions[-1].end <= duration


def test_probabilities_are_reported_over_the_region_they_describe() -> None:
    """`mean_probability` has to come from the region's own frames, not the whole signal."""
    quiet, loud = 0.0, 0.9
    probabilities = [loud] * 20 + [quiet] * 32 + [loud] * 20
    regions = detect(probabilities)
    assert len(regions) == 2
    for region in regions:
        assert region.peak_probability == pytest.approx(loud, abs=1e-6)
        assert region.mean_probability > 0.5, (
            "a speech region averaging in the silence around it is reporting the wrong span"
        )
