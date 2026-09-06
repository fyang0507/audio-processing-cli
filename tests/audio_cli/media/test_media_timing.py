"""Stream timestamp facts must preserve absence and never stand in for content sync."""

from pathlib import Path

import pytest

from audio_cli.media import media_summary, probed_duration, stream_timing
from audio_cli.pipeline.timing import duration_check
from audio_cli.transcribe.catalog import input_metadata, result_source
from audio_cli.transcribe.result.validation import _validate_source


def probe(audio, *video, duration="8.5"):
    audio = {"codec_type": "audio", "index": 1, "sample_rate": "48000", "channels": 1, **audio}
    return {
        "primary_audio_stream": audio,
        "streams": [audio, *({"codec_type": "video", **s} for s in video)],
        "format": {"duration": duration, "start_time": "-0.5"},
        "has_video": bool(video),
    }


def test_available_stream_origins_and_integer_timebases_are_not_content_offsets(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"identity only")
    info = media_summary(
        source,
        probe(
            {"start_time": "0.044", "duration": "8", "time_base": "1/48000", "start_pts": 2112},
            {"index": 0, "start_time": "-0.5", "duration": "8.5", "time_base": "1/1000"},
            {"index": 2, "start_time": "0.5"},
        ),
    )
    assert info["duration_basis"] == "probed_audio_stream"
    assert info["duration_seconds"] == 8
    assert info["audio_start_seconds"] == 0.044
    timing = info["timing"]
    assert timing["container"] == {"start_seconds": -0.5, "duration_seconds": 8.5}
    assert timing["audio"][0]["start_pts"] == 2112
    assert timing["video"][0]["primary_audio_start_minus_video_start_seconds"] == 0.544
    assert timing["video"][1]["primary_audio_start_minus_video_start_seconds"] == -0.456
    assert "initial_padding" not in timing["audio"][0]


@pytest.mark.parametrize("missing", [None, "N/A", "nan", "inf", "", True])
def test_unavailable_timestamps_and_durations_are_absent(tmp_path, missing):
    source = tmp_path / "unknown.wav"
    source.write_bytes(b"identity")
    raw = probe({"start_time": missing, "duration": missing}, {"index": 0})
    info = media_summary(source, raw)
    assert "audio_start_seconds" not in info
    assert "start_seconds" not in info["timing"]["audio"][0]
    assert "primary_audio_start_minus_video_start_seconds" not in info["timing"]["video"][0]
    assert info["duration_basis"] == "probed_container"
    assert probed_duration(probe({"duration": missing}, duration=missing)) == {}


def test_zero_and_negative_starts_survive_but_invalid_timebase_and_padding_do_not():
    timing = stream_timing(
        probe(
            {
                "start_time": "-0.007",
                "time_base": "0/0",
                "initial_padding": 312,
                "trailing_padding": -1,
                "start_pts": -7,
            },
            {"index": 0, "start_time": "0"},
        )
    )
    assert timing["audio"][0] == {
        "stream_index": 1,
        "start_seconds": -0.007,
        "start_pts": -7,
        "initial_padding": 312,
    }
    assert timing["video"][0]["start_seconds"] == 0


@pytest.mark.parametrize(
    "extra, accepted", [(2399, True), (2400, True), (2401, False), (-2401, False)]
)
def test_duration_gate_compares_unrounded_samples_at_exact_tolerance(extra, accepted):
    passed, delta = duration_check(480000, 48000, 480000 + extra, 48000)
    assert passed is accepted
    assert delta == pytest.approx(extra / 48)


def test_transcription_consumes_media_duration_basis_then_uses_canonical_decode():
    raw = probe({"duration": "N/A", "start_time": "0.044"})
    metadata = input_metadata(Path("input.m4a"), raw)
    assert metadata.source["duration_basis"] == "probed_container"
    assert metadata.catalog_input["timing"] == stream_timing(raw)
    source = result_source(metadata, 8.456)
    assert source["duration_basis"] == "canonical_decoded_pcm"
    assert source["duration_seconds"] == 8.456
    assert _validate_source(source) == 8.456
    # v1 artifacts without this additive field remain readable, without synthesizing it.
    del source["duration_basis"]
    assert _validate_source(source) == 8.456


@pytest.mark.parametrize(
    "basis", [None, "decoded_pcm", "probed_audio_stream", "probed_container", {}]
)
def test_durable_source_cannot_claim_probed_or_unknown_duration_basis(basis):
    from audio_cli.transcribe.result import NormalizedResult, ResultError, serialize_result

    source = {
        "path": "source.wav",
        "duration_seconds": 1,
        "timebase": "seconds",
        "duration_basis": basis,
    }
    result = NormalizedResult(
        source=source,
        segments=[],
        abstentions=[],
        provenance={"stack": "firered", "outcomes": {}, "observed": {}, "plan": {}},
        requested_capabilities=frozenset(),
    )
    with pytest.raises(ResultError, match="duration_basis"):
        serialize_result(result)
