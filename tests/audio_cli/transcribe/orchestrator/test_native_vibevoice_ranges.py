from __future__ import annotations

from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    InputMetadata,
    Path,
    StageOutcome,
    _complete_vibe_payload,
    _ready_fluidaudio,
    _ready_multi_package,
    _ready_single_package,
    _runtime,
    export_documents,
    json,
    load_result_document,
    orchestrator,
    pytest,
    resolve_request,
    wave,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)


def test_vibevoice_cap_salvage_has_honest_zero_of_one_units_and_runnable_resume(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "long.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps,vad,overlapped_speech",
    )
    metadata = InputMetadata(str(source), 10.0, "wav", 48_000, 1)

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return [
                {"start": 0.5, "end": 1.0},
                {"start": 4.0, "end": 5.0},
            ]

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 160_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                {
                    "raw_text": (
                        'assistant\n[{"Start":0,"End":2,"Speaker":0,'
                        '"Content":"Complete."},{"Start":2,"End":8,'
                        '"Speaker":1,"Content":"cut'
                    ),
                    "segments": [],
                    "hit_max_new_tokens": True,
                    "generated_tokens": 16384,
                    "eos_observed": False,
                },
                3.0,
                returncode=4,
            )

        def diarize(self, **_kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {"startTimeSeconds": 0.2, "endTimeSeconds": 1.2, "speakerId": "S0"},
                        {"startTimeSeconds": 4.0, "endTimeSeconds": 6.0, "speakerId": "S1"},
                        {"startTimeSeconds": 5.0, "endTimeSeconds": 7.0, "speakerId": "S2"},
                    ]
                },
                0.2,
            )

    registry = {
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
    }
    output = tmp_path / "long.json"
    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=Transport(),
            vad_detector=Detector(),
            output=output,
        )
    refusal = raised.value
    assert refusal.exit_code == 4
    assert refusal.payload["coverage"] == {
        "scope_intervals": [[0.0, 10.0]],
        "covered_through_seconds": 2.0,
        "covered_fraction": 0.2,
        "covered_intervals": [[0.0, 2.0]],
        "missing_intervals": [[2.0, 10.0]],
        "units_total": 1,
        "units_completed": 0,
    }
    assert "--range 2.0:" in refusal.payload["fix"]
    partial = json.loads((tmp_path / "long.partial.json").read_text(encoding="utf-8"))
    assert partial["complete"] is False
    assert partial["segments"] == [
        {
            "segment_id": "seg_0",
            "text": "Complete.",
            "start": 0.0,
            "end": 2.0,
        }
    ]
    assert partial["vad_regions"] == [{"start": 0.5, "end": 1.0}]
    assert partial["overlapped_speech"] == []
    assert all(item["start"] < 2.0 for item in partial["abstentions"])


def test_vibevoice_complete_range_restores_canonical_source_timeline(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "original.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps",
    )
    metadata = InputMetadata(str(source), 10.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 160_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, *, audio, **_kwargs):
            assert audio.name == "vibevoice-range.wav"
            with wave.open(str(audio), "rb") as handle:
                assert handle.getnframes() == 4 * 16_000
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {
                            "start_time": 0.25,
                            "end_time": 1.5,
                            "speaker_id": 0,
                            "text": "Range first.",
                        },
                        {
                            "start_time": 3.0,
                            "end_time": 4.0,
                            "speaker_id": 1,
                            "text": "Range last.",
                        },
                    ],
                    generated_tokens=20,
                    eos_observed=True,
                ),
                1.0,
            )

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {"torch-vibevoice": {"state": "ready"}},
            "packages": {
                "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
            },
        },
        transport=Transport(),
        run_range=orchestrator.parse_range("3:7"),
    ).payload

    assert payload["source"] == {
        "path": str(source.resolve()),
        "duration_seconds": 10.0,
        "timebase": "seconds",
        "duration_basis": "canonical_decoded_pcm",
    }
    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [3.0, 7.0],
        "selected_unit_scope": [3.0, 7.0],
    }
    assert [(segment["start"], segment["end"]) for segment in payload["segments"]] == [
        (3.25, 4.5),
        (6.0, 7.0),
    ]


def test_vibevoice_non_frame_aligned_range_uses_actual_clip_bounds(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "fractional.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps",
    )
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 1_601)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, *, audio, **_kwargs):
            with wave.open(str(audio), "rb") as handle:
                assert handle.getnframes() == 1_600
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {
                            "start_time": 0.0,
                            # The pinned post-processor may be one microsecond over
                            # the exact selected clip. The adapter clamps only that
                            # accepted tolerance before restoring source time.
                            "end_time": (1_600 / 16_000) + 1e-6,
                            "speaker_id": 0,
                            "text": "Sample aligned.",
                        }
                    ],
                    generated_tokens=4,
                    eos_observed=True,
                ),
                1.0,
            )

    saved = tmp_path / "fractional-result.json"
    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {"torch-vibevoice": {"state": "ready"}},
            "packages": {
                "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
            },
        },
        transport=Transport(),
        run_range=orchestrator.parse_range("0.00003:0.10004"),
        output=saved,
    ).payload

    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.00003, 0.10004],
        "selected_unit_scope": [0.000063, 0.100062],
    }
    assert payload["segments"][0]["start"] == 0.000063
    # Both durable fields use the schema's six-decimal source timeline; the
    # adapter was validated against the exact 1601/16000 boundary above.
    assert payload["segments"][0]["end"] == 0.100062
    assert load_result_document(saved).owned_intervals == ((0.000063, 0.100062),)
    assert export_documents([saved], "txt").content == "Sample aligned.\n"


def test_vibevoice_non_frame_partial_and_resume_merge_without_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "resume.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants="segment_timestamps",
    )
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 1)
    registry = {
        "environments": {"torch-vibevoice": {"state": "ready"}},
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
        },
    }

    class BaseTransport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

    class PartialTransport(BaseTransport):
        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                {
                    "raw_text": (
                        'assistant\n[{"Start":0,"End":0.0999775,"Speaker":0,'
                        '"Content":"First."},{"Start":0.2,"End":0.4,'
                        '"Content":"cut'
                    ),
                    "segments": [],
                    "hit_max_new_tokens": True,
                    "generated_tokens": 16_384,
                    "eos_observed": False,
                },
                1.0,
                returncode=4,
            )

    class ResumeTransport(BaseTransport):
        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {
                            "start_time": 0.0,
                            "end_time": 0.1,
                            "speaker_id": 0,
                            "text": "Second.",
                        }
                    ],
                    generated_tokens=4,
                    eos_observed=True,
                ),
                1.0,
            )

    first_output = tmp_path / "first.json"
    with pytest.raises(Exception) as raised:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=PartialTransport(),
            output=first_output,
            run_range=orchestrator.parse_range("0.00003:0.50004"),
        )
    assert raised.value.exit_code == 4
    partial = Path(raised.value.payload["output"])
    assert raised.value.payload["coverage"]["covered_intervals"] == [[0.000063, 0.10004]]

    rest = tmp_path / "rest.json"
    orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=ResumeTransport(),
        output=rest,
        run_range=orchestrator.parse_range("0.10004:"),
    )

    assert load_result_document(partial).owned_intervals == ((0.000063, 0.10004),)
    assert load_result_document(rest).owned_intervals == ((0.100062, 1.0),)
    product = export_documents([partial, rest], "txt")
    assert product.content == "First.\nSecond.\n"
