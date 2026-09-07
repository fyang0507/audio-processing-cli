from audio_cli.pipeline import PipelineError, validate_skips
from audio_cli.profiles import PROFILES, STAGE_ORDER


def test_profiles_version_the_complete_fixed_order() -> None:
    assert STAGE_ORDER == (
        "channel-balance",
        "environment-denoise",
        "voice-enhance",
        "source-balance",
        "program-loudness",
    )
    assert PROFILES["product-demo"].source_balance_enabled is True
    assert PROFILES["transcription"].source_balance_enabled is False
    for profile in PROFILES.values():
        assert profile.version == "5"
        assert profile.speech_transition_placement == "outside"
        assert profile.as_dict()["processing_order"] == list(STAGE_ORDER)


def test_skip_parser_disables_exactly_named_stages() -> None:
    assert validate_skips("channel-balance,program-loudness") == {
        "channel-balance",
        "program-loudness",
    }


def test_skip_parser_rejects_unknown_and_duplicate_names() -> None:
    for raw in ("unknown", "voice-enhance,voice-enhance", "voice-enhance,"):
        try:
            validate_skips(raw)
        except PipelineError:
            pass
        else:
            raise AssertionError(f"Expected invalid skip list to fail: {raw}")


def test_bundled_profiles_preserve_version_five_settings() -> None:
    import json
    from pathlib import Path

    baseline = Path(__file__).parents[1] / "fixtures" / "profiles-v5.json"
    assert {name: profile.as_dict() for name, profile in PROFILES.items()} == json.loads(
        baseline.read_text()
    )


def _toml_profile(data: dict) -> str:
    import json

    return (
        "\n".join(
            f"{key} = {repr(value) if type(value) in (int, float) else json.dumps(value)}"
            for key, value in data.items()
        )
        + "\n"
    )


def test_new_bundled_profile_is_discovered_by_cli(tmp_path, monkeypatch) -> None:
    from dataclasses import asdict

    from audio_cli import cli_parser
    from audio_cli.profiles import _load_profiles

    data = asdict(PROFILES["transcription"])
    data["name"] = "meeting"
    (tmp_path / "meeting.toml").write_text(_toml_profile(data))
    loaded = _load_profiles(tmp_path)
    monkeypatch.setattr(cli_parser, "PROFILES", loaded)
    parser = cli_parser.build_parser()
    assert (
        parser.parse_args(["inspect", "recording.wav", "--profile", "meeting"]).profile == "meeting"
    )
    assert (
        parser.parse_args(
            ["enhance", "recording.wav", "--profile", "meeting", "--output", "out.wav"]
        ).profile
        == "meeting"
    )


def test_invalid_bundled_profiles_are_rejected(tmp_path) -> None:
    from dataclasses import asdict

    import pytest

    from audio_cli.profiles import _load_profiles

    original = asdict(PROFILES["transcription"])
    cases = [
        ("unexpected", 1, "unknown fields"),
        ("channel_balance_enabled", 1, "expected bool"),
        ("vad_min_speech_ms", 1.5, "expected int"),
        ("target_lufs", True, "expected float"),
        ("target_lufs", float("nan"), "expected float"),
        ("voice_max_gain_db", -1, "nonnegative"),
        ("vad_threshold", 2, "between 0 and 1"),
        ("highpass_hz", 24000, "Nyquist"),
        ("compressor_ratio", 0.5, "at least 1"),
        ("vad_exit_threshold", 0.9, "must not exceed"),
        ("machine_relative_target_lu", 3, "within"),
        ("target_true_peak_dbtp", 1, "between -9 and 0"),
        ("target_true_peak_dbtp", -10, "between -9 and 0"),
        ("codec_true_peak_headroom_db", 10, "after codec headroom"),
        ("target_lufs", 0, "between -70 and -5"),
        ("target_lra_lu", 0, "between 1 and 50"),
        ("target_lufs", 10**400, "expected float"),
        ("name", "wrong", "match the filename"),
        ("version", "", "positive integer string"),
        ("speech_transition_placement", "inside", "must be 'outside'"),
    ]
    path = tmp_path / "transcription.toml"
    for key, value, message in cases:
        path.write_text(_toml_profile(original | {key: value}))
        with pytest.raises(ValueError, match=message):
            _load_profiles(tmp_path)
    for raw, message in [
        ("", "missing fields"),
        ('name = "a"\nname = "b"', "Cannot overwrite a value"),
        ("[", "Invalid bundled profile transcription.toml"),
        ('name = { value = "transcription" }', "missing fields"),
    ]:
        path.write_text(raw)
        with pytest.raises(ValueError, match=message):
            _load_profiles(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="No bundled profile"):
        _load_profiles(tmp_path)


def test_loudness_validation_accepts_supported_boundaries() -> None:
    from dataclasses import asdict

    from audio_cli.profiles import _validate_profile

    data = asdict(PROFILES["transcription"])
    for lufs, lra, peak in [(-70, 1, -9), (-5, 50, 0)]:
        candidate = data | {
            "target_lufs": lufs,
            "target_lra_lu": lra,
            "target_true_peak_dbtp": peak,
            "codec_true_peak_headroom_db": 0,
        }
        assert _validate_profile(candidate, "transcription").target_lufs == lufs
