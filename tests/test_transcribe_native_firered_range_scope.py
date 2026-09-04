from __future__ import annotations

from pathlib import Path

from transcribe_native_test_support import (
    InputMetadata,
    StageOutcome,
    _ready_fluidaudio,
    _ready_multi_package,
    _ready_single_package,
    _runtime,
    orchestrator,
    resolve_request,
    wave,
)
from transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)


def test_firered_silero_range_selects_intersections_and_owns_expanded_auxiliary_scope(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "range.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="vad,segment_timestamps,diarization,overlapped_speech",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return [
                {"start": 0.0, "end": 1.0},
                {"start": 1.4, "end": 2.4},
                {"start": 2.6, "end": 2.9},
            ]

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == [
                {"start": 0.0, "end": 1.0},
                {"start": 1.4, "end": 2.4},
            ]
            assert (kwargs["range_start"], kwargs["range_end"]) == (0.5, 1.5)
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 100,
                                "end_ms": 800,
                                "text": "One.",
                                "lang": None,
                                "lang_confidence": 0,
                            },
                            {
                                "start_ms": 1500,
                                "end_ms": 2200,
                                "text": "Two.",
                                "lang": None,
                                "lang_confidence": 0,
                            },
                        ],
                        "words": [
                            {"start_ms": 100, "end_ms": 800, "text": "one"},
                            {"start_ms": 1500, "end_ms": 2200, "text": "two"},
                        ],
                        "vad_segments_ms": [[0, 1000], [1400, 2400]],
                    },
                    "regions": [
                        {"region_id": "vad_0", "start": 0.0, "end": 1.0, "processed": True},
                        {"region_id": "vad_1", "start": 1.4, "end": 2.4, "processed": True},
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
                        {
                            "startTimeSeconds": 0.1,
                            "endTimeSeconds": 0.9,
                            "speakerId": "S0",
                            "embedding": [0.0] * 256,
                        },
                        {
                            "startTimeSeconds": 0.6,
                            "endTimeSeconds": 0.9,
                            "speakerId": "S2",
                            "embedding": [2.0] * 256,
                        },
                        {
                            "startTimeSeconds": 1.5,
                            "endTimeSeconds": 2.2,
                            "speakerId": "S1",
                            "embedding": [1.0] * 256,
                        },
                    ]
                },
                0.2,
            )

    registry = {
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
    }
    payload = orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=Transport(),
        vad_detector=Detector(),
        run_range=orchestrator.parse_range("0.5:1.5"),
    ).payload
    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.5, 1.5],
        "selected_unit_scope": [0.0, 2.4],
    }
    assert payload["vad_regions"] == [
        {"start": 0.0, "end": 1.0},
        {"start": 1.4, "end": 2.4},
    ]
    assert payload["turns"][0]["start"] == 0.1
    assert "speaker" not in payload["segments"][0]
    assert payload["segments"][1]["speaker"] == "S1"
    assert payload["overlapped_speech"] == [
        {
            "overlap_id": "overlap_0",
            "start": 0.6,
            "end": 0.9,
        }
    ]


def test_firered_native_range_records_expanded_whole_vad_scope(tmp_path: Path, monkeypatch) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "native-range.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,diarization",
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] is None
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 200,
                                "end_ms": 900,
                                "text": "One.",
                                "lang": None,
                                "lang_confidence": 0,
                            },
                            {
                                "start_ms": 1400,
                                "end_ms": 2100,
                                "text": "Two.",
                                "lang": None,
                                "lang_confidence": 0,
                            },
                        ],
                        "words": [
                            {"start_ms": 200, "end_ms": 900, "text": "one"},
                            {"start_ms": 1400, "end_ms": 2100, "text": "two"},
                        ],
                        "vad_segments_ms": [[100, 1000], [1300, 2300]],
                    },
                    "regions": [
                        {"region_id": "vad_0", "start": 0.1, "end": 1.0, "processed": True},
                        {"region_id": "vad_1", "start": 1.3, "end": 2.3, "processed": True},
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
                        {"startTimeSeconds": 0.01, "endTimeSeconds": 0.7, "speakerId": "S0"},
                        {"startTimeSeconds": 2.4, "endTimeSeconds": 2.95, "speakerId": "S1"},
                    ]
                },
                0.2,
            )

    registry = {
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
    }
    payload = orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=Transport(),
        run_range=orchestrator.parse_range("0:3"),
    ).payload
    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.0, 3.0],
        "selected_unit_scope": [0.1, 2.3],
    }
    assert [turn["start"] for turn in payload["turns"]] == [0.01, 2.4]
