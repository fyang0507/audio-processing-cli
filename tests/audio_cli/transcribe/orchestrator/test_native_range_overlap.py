from __future__ import annotations

from pathlib import Path

from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    InputMetadata,
    StageOutcome,
    _complete_vibe_payload,
    _ready_fluidaudio,
    _ready_multi_package,
    _ready_single_package,
    _runtime,
    orchestrator,
    resolve_request,
    wave,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)


def test_firered_range_masks_cross_boundary_overlap_without_republishing_it(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "firered-cross-boundary.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="vad,segment_timestamps,diarization,overlapped_speech",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 12.0, "wav", 48_000, 1)

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return [{"start": 10.0, "end": 11.0}]

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 192_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == [{"start": 10.0, "end": 11.0}]
            assert (kwargs["range_start"], kwargs["range_end"]) == (10.0, 12.0)
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 10_000,
                                "end_ms": 11_000,
                                "text": "Cross.",
                                "lang": None,
                                "lang_confidence": 0,
                            }
                        ],
                        "words": [
                            {
                                "start_ms": 10_050,
                                "end_ms": 10_900,
                                "text": "cross",
                            }
                        ],
                        "vad_segments_ms": [[10_000, 11_000]],
                    },
                    "regions": [
                        {
                            "region_id": "vad_0",
                            "start": 10.0,
                            "end": 11.0,
                            "processed": True,
                        }
                    ],
                },
                1.0,
            )

        def diarize(self, **_kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {"startTimeSeconds": 9.0, "endTimeSeconds": 11.0, "speakerId": "S0"},
                        {"startTimeSeconds": 9.8, "endTimeSeconds": 10.2, "speakerId": "S1"},
                    ]
                },
                0.2,
            )

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {
                "torch-firered": {"state": "ready"},
                "swift": {"state": "ready"},
            },
            "packages": {
                "firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s"),
                "fluidaudio": _ready_fluidaudio(tmp_path),
                "speaker-diarization-coreml": _ready_single_package(
                    tmp_path, "speaker-diarization-coreml"
                ),
            },
        },
        transport=Transport(),
        vad_detector=Detector(),
        run_range=orchestrator.parse_range("10:12"),
    ).payload

    assert payload["segments"] == [
        {
            "segment_id": "seg_0",
            "text": "Cross.",
            "start": 10.0,
            "end": 11.0,
        }
    ]
    # The second S0 single-speaker span proves a label would otherwise be chosen.
    assert payload["turns"] == [
        {
            "turn_id": "turn_1",
            "speaker": "S0",
            "start": 10.2,
            "end": 11.0,
        }
    ]
    assert payload["overlapped_speech"] == []
    assert payload["abstentions"] == []


def test_vibevoice_range_masks_cross_boundary_overlap_without_republishing_it(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "vibe-cross-boundary.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps,diarization,overlapped_speech",
    )
    metadata = InputMetadata(str(source), 12.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 192_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {
                            "start_time": 0.0,
                            "end_time": 1.0,
                            "speaker_id": 0,
                            "text": "Cross.",
                        }
                    ]
                ),
                1.0,
            )

        def diarize(self, **_kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {"startTimeSeconds": 9.0, "endTimeSeconds": 11.0, "speakerId": "S0"},
                        {"startTimeSeconds": 9.8, "endTimeSeconds": 10.2, "speakerId": "S1"},
                    ]
                },
                0.2,
            )

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {
                "torch-vibevoice": {"state": "ready"},
                "swift": {"state": "ready"},
            },
            "packages": {
                "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
                "fluidaudio": _ready_fluidaudio(tmp_path),
                "speaker-diarization-coreml": _ready_single_package(
                    tmp_path, "speaker-diarization-coreml"
                ),
            },
        },
        transport=Transport(),
        run_range=orchestrator.parse_range("10:12"),
    ).payload

    assert payload["segments"] == [
        {
            "segment_id": "seg_0",
            "text": "Cross.",
            "start": 10.0,
            "end": 11.0,
        }
    ]
    assert payload["turns"] == [
        {
            "turn_id": "turn_0",
            "speaker": "0",
            "start": 10.0,
            "end": 11.0,
        }
    ]
    assert payload["overlapped_speech"] == []
    assert payload["abstentions"] == []
