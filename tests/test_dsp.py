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
