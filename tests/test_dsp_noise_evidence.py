"""Noise-reference decisions expose actual source scopes and rejection evidence."""

import numpy as np

from audio_cli.dsp import analyze_signal
from audio_cli.dsp.denoise import apply_broadband_denoise
from audio_cli.profiles import PROFILES
from audio_cli.vad_contract import SpeechRegion

RATE = 16_000
PROFILE = PROFILES["product-demo"]


def fixture():
    time = np.arange(4 * RATE) / RATE
    noise = np.random.default_rng(21).normal(0, 0.012, time.size)
    voice = 0.065 * np.sin(2 * np.pi * 220 * time) * ((time >= 1) & (time < 3))
    return (noise + voice).astype(np.float32)[:, None]


def run(audio, exclusions):
    analysis = analyze_signal(audio, RATE, [SpeechRegion(1, 3, 0.9, 1)], PROFILE)
    assert not analysis.machine_regions
    return apply_broadband_denoise(audio, RATE, PROFILE, analysis, exclusions)


def test_reference_scopes_respect_exclusions_and_the_sample_grid():
    audio = fixture()
    output, component, operation = run(audio, [(1, 3)])
    assert component["status"] == "applied"
    scopes = operation["noise_reference_scopes"]
    assert [(s["start"], s["end"]) for s in scopes] == [(0.0, 0.88), (3.12, 4.0)]
    assert all(s["status"] == "eligible" for s in scopes)
    assert all(s["frame_count"] == 107 for s in scopes)
    assert operation["noise_reference_scope_basis"].startswith("source_timeline")
    assert operation["noise_reference_guard_ms"] == 120
    assert not np.array_equal(output, audio)


def test_global_level_rejection_keeps_locally_eligible_scopes_and_measured_spread():
    audio = fixture()
    audio[:RATE] *= 0.1
    snapshot = audio.copy()
    output, component, operation = run(audio, [(1, 3)])
    assert operation is None
    assert component["reason"] == "noise_reference_is_nonstationary"
    assert component["noise_reference_level_spread_db"] > 18
    assert [s["status"] for s in component["noise_reference_scopes"]] == ["eligible", "eligible"]
    np.testing.assert_array_equal(output, snapshot)


def test_local_rejection_identifies_the_changing_reference_instead_of_hiding_it():
    audio = fixture()
    audio[: RATE // 2] *= 0.01
    output, component, operation = run(audio, [(1, 3)])
    left, right = component["noise_reference_scopes"]
    assert left["status"] == "rejected"
    assert left["reason"] == "noise_reference_is_nonstationary"
    assert left["noise_reference_maximum_channel_level_spread_db"] > 20
    assert right["status"] == "eligible"
    assert operation is None
    np.testing.assert_array_equal(output, audio)


def test_too_short_scopes_remain_visible_without_invented_spectral_measurements():
    audio = fixture()
    output, component, operation = run(audio, [(0.3, 3.7)])
    scopes = component["noise_reference_scopes"]
    assert len(scopes) == 2
    assert all(s["reason"] == "noise_reference_too_short" for s in scopes)
    assert all(s["end"] - s["start"] < 0.25 for s in scopes)
    assert all("noise_reference_minimum_block_flatness" not in s for s in scopes)
    assert "noise_reference_level_spread_db" not in component
    assert operation is None
    np.testing.assert_array_equal(output, audio)
