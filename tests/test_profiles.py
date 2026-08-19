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
        assert profile.version == "4"
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
