import numpy as np

from audio_cli.dsp import (
    analyze_signal,
    apply_channel_balance,
    apply_source_balance,
    apply_voice_enhancement,
    regional_measurements,
    resolve_speech_treatment_intervals,
    smooth_time_mask,
)
from audio_cli.profiles import PROFILES
from audio_cli.vad import SpeechRegion


def _sine(
    sample_rate: int, duration: float, frequency: float, amplitude: float
) -> np.ndarray:
    time = np.arange(round(sample_rate * duration)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_time_mask_uses_smooth_boundaries_and_unions_overlaps() -> None:
    mask = smooth_time_mask(1000, [(0.1, 0.5), (0.45, 0.8)], 1000, 40)
    assert mask[100] == 0.0
    assert 0.0 < mask[120] < 1.0
    assert mask[300] == 1.0
    assert mask[470] == 1.0
    assert mask[799] == 0.0
    assert np.all((0.0 <= mask) & (mask <= 1.0))


def test_time_mask_can_place_transitions_outside_active_region() -> None:
    mask = smooth_time_mask(
        1000,
        [(0.2, 0.6)],
        1000,
        40,
        transition_placement="outside",
    )
    assert mask[159] == 0.0
    assert 0.0 < mask[180] < 1.0
    assert np.all(mask[200:600] == 1.0)
    assert 0.0 < mask[620] < 1.0
    assert mask[640] == 0.0
    assert np.all((0.0 <= mask) & (mask <= 1.0))


def test_speech_treatment_expands_late_vad_seed_to_preceding_voice_activity() -> None:
    sample_rate = 16_000
    audio = np.zeros((sample_rate * 3, 2), dtype=np.float32)
    first = _sine(sample_rate, 0.16, 180.0, 0.002)
    second = _sine(sample_rate, 0.30, 180.0, 0.002)
    audio[round(0.60 * sample_rate) : round(0.76 * sample_rate)] = first[:, None]
    audio[round(0.90 * sample_rate) : round(1.20 * sample_rate)] = second[:, None]
    speech = [SpeechRegion(1.0, 1.2, 0.9, 1.0)]
    profile = PROFILES["product-demo"]
    analysis = analyze_signal(audio, sample_rate, speech, profile)

    intervals, resolution = resolve_speech_treatment_intervals(
        audio, sample_rate, profile, analysis
    )
    mask = smooth_time_mask(
        audio.shape[0],
        intervals,
        sample_rate,
        profile.region_fade_ms,
        transition_placement="outside",
    )

    assert intervals[0][0] <= 0.52
    assert np.all(mask[round(0.60 * sample_rate) : round(1.20 * sample_rate)] == 1)
    assert resolution["placement"] == "silence_anchored_outside_voice_activity"
    assert resolution["regions"][0]["start_extension_ms"] >= 480


def test_channel_balance_only_corrects_correlated_mismatch() -> None:
    sample_rate = 16_000
    mono = _sine(sample_rate, 2.0, 440.0, 0.05)
    audio = np.column_stack((mono * 2.0, mono))
    speech = [SpeechRegion(0.0, 2.0, 0.9, 1.0)]
    profile = PROFILES["product-demo"]
    analysis = analyze_signal(audio, sample_rate, speech, profile)
    output, stage = apply_channel_balance(audio, profile, analysis)
    corrected = analyze_signal(output, sample_rate, speech, profile)
    assert stage["status"] == "applied"
    assert abs(corrected.channel_difference_db) < 0.05


def test_voice_and_machine_balance_close_the_declared_gap() -> None:
    sample_rate = 16_000
    audio = np.zeros((sample_rate * 6, 2), dtype=np.float32)
    machine = _sine(sample_rate, 1.0, 1000.0, 0.12)
    speech_signal = _sine(sample_rate, 2.4, 220.0, 0.004)
    audio[round(0.5 * sample_rate) : round(1.5 * sample_rate), :] = machine[:, None]
    audio[round(2.0 * sample_rate) : round(4.4 * sample_rate), :] = speech_signal[
        :, None
    ]
    speech = [SpeechRegion(2.0, 4.4, 0.9, 1.0)]
    profile = PROFILES["product-demo"]
    analysis = analyze_signal(audio, sample_rate, speech, profile)
    assert analysis.machine_regions
    initial = regional_measurements(audio, sample_rate, analysis)
    enhanced, voice_stage = apply_voice_enhancement(
        audio, sample_rate, profile, analysis
    )
    balanced, source_stage = apply_source_balance(
        enhanced, sample_rate, profile, analysis
    )
    final = regional_measurements(balanced, sample_rate, analysis)
    assert voice_stage["status"] == "applied"
    assert (
        voice_stage["resolved_transition"]["placement"]
        == "silence_anchored_outside_voice_activity"
    )
    assert voice_stage["resolved_transition"]["fade_in_ms"] == 40
    assert (
        voice_stage["resolved_transition"]["minimum_mix_inside_treatment_region"] == 1
    )
    assert source_stage["status"] == "applied"
    assert initial["machine_regions"][0]["difference_from_speech_db"] > 20
    difference = final["machine_regions"][0]["difference_from_speech_db"]
    assert -4.5 <= difference <= -1.5


def test_machine_region_padding_that_reaches_speech_abstains() -> None:
    sample_rate = 16_000
    audio = np.zeros((sample_rate * 4, 2), dtype=np.float32)
    machine = _sine(sample_rate, 0.5, 1000.0, 0.12)
    speech_signal = _sine(sample_rate, 1.98, 220.0, 0.004)
    audio[round(0.5 * sample_rate) : round(1.0 * sample_rate)] = machine[:, None]
    audio[round(1.02 * sample_rate) : round(3.0 * sample_rate)] = speech_signal[:, None]
    speech = [SpeechRegion(1.02, 3.0, 0.9, 1.0)]
    profile = PROFILES["product-demo"]

    analysis = analyze_signal(audio, sample_rate, speech, profile)

    assert len(analysis.machine_regions) == 1
    region = analysis.machine_regions[0]
    assert region.end > speech[0].start
    assert region.overlaps_speech is True

    balanced, source_stage = apply_source_balance(audio, sample_rate, profile, analysis)

    assert source_stage["status"] == "abstained"
    assert source_stage["reason"] == "speech_and_machine_audio_overlap"
    assert source_stage["abstained_regions"] == [region.region_id]
    assert source_stage["operations"] == []
    np.testing.assert_array_equal(balanced, audio)


SR = 48_000
FADE_MS = 40


def _mask(intervals, length_s=7.0, fade_ms=FADE_MS):
    return smooth_time_mask(
        int(length_s * SR), intervals, SR, fade_ms, transition_placement="outside"
    )


import pytest  # noqa: E402  (kept beside the cases it serves)


@pytest.mark.parametrize(
    "label, intervals",
    [
        ("single", [(1.0, 2.0)]),
        ("far apart", [(1.0, 2.0), (5.0, 6.0)]),
        ("closer than two fades", [(1.0, 2.0), (2.01, 3.0)]),
        ("touching", [(1.0, 2.0), (2.0, 3.0)]),
        ("overlapping", [(1.0, 2.0), (1.5, 3.0)]),
        ("at zero", [(0.0, 1.0)]),
        ("at the end", [(6.9, 7.0)]),
        ("unsorted", [(5.0, 6.0), (1.0, 2.0)]),
        ("shorter than one fade", [(1.0, 1.0005)]),
    ],
)
def test_an_outside_placed_mask_is_fully_engaged_over_every_requested_sample(
    label: str, intervals: list[tuple[float, float]]
) -> None:
    """The promise `outside` placement exists to keep: speech-scoped effects are at full strength
    at the first and last detected sample, with the transition in the surrounding context, so an
    opening or closing phoneme is not levelled progressively."""
    mask = _mask(intervals)
    for start, end in intervals:
        lo, hi = max(0, round(start * SR)), min(mask.size, round(end * SR))
        if hi <= lo:
            continue
        assert np.allclose(mask[lo:hi], 1.0, atol=1e-6), (
            f"{label}: mask dips to {mask[lo:hi].min():.4f} inside {start}-{end}"
        )


@pytest.mark.parametrize(
    "intervals",
    [
        [(1.0, 2.0)],
        [(1.0, 2.0), (2.01, 3.0)],
        [(1.0, 2.0), (2.0, 3.0)],
        [(0.0, 1.0)],
        [(6.9, 7.0)],
        [(1.0, 1.0005)],
    ],
)
def test_a_mask_stays_in_range_and_free_of_steps(intervals) -> None:
    """A discontinuity in the blend weight is a click in the render, so bound the step by the
    fade's own slope rather than merely checking the endpoints."""
    mask = _mask(intervals)
    assert mask.min() >= -1e-6 and mask.max() <= 1.0 + 1e-6, (
        f"mask leaves [0, 1]: [{mask.min():.4f}, {mask.max():.4f}]"
    )
    fade_samples = max(1, round(SR * FADE_MS / 1000))
    largest_step = float(np.max(np.abs(np.diff(mask))))
    assert largest_step <= 4.0 / fade_samples, (
        f"step of {largest_step:.5f} exceeds what a {FADE_MS} ms fade can produce"
    )


def test_a_mask_is_the_length_it_was_asked_for() -> None:
    assert _mask([(1.0, 2.0)], length_s=3.0).size == 3 * SR


def test_nothing_is_engaged_far_from_any_interval() -> None:
    """The scoping guarantee: a speech-scoped effect must reach nowhere near the rest of the file."""
    mask = _mask([(3.0, 4.0)])
    quiet = np.r_[mask[: int(2.9 * SR)], mask[int(4.1 * SR) :]]
    assert float(np.max(quiet)) == 0.0, "the mask is engaged outside its interval plus fade"


def test_an_inside_placed_mask_ramps_within_the_interval() -> None:
    """The other placement still has to behave: `inside` puts the transition in the region, so the
    first sample is not at full strength. Asserted so the two modes cannot converge unnoticed."""
    inside = smooth_time_mask(3 * SR, [(1.0, 2.0)], SR, FADE_MS, transition_placement="inside")
    assert inside[round(1.0 * SR)] < 0.05
    assert inside[round(1.5 * SR)] == pytest.approx(1.0, abs=1e-6)
