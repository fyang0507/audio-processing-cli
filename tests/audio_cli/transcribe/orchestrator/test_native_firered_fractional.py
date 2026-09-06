from __future__ import annotations

from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    InputMetadata,
    Path,
    StageOutcome,
    _ready_multi_package,
    _runtime,
    export_documents,
    load_result_document,
    orchestrator,
    pytest,
    refusals,
    resolve_request,
    wave,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)


def test_firered_fractional_public_scope_is_loadable_and_exportable(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "fractional-owned.wav"
    source.write_bytes(b"source")
    selected = [{"start": 0.0009, "end": 1.0009}]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == selected
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 0,
                                "end_ms": 1000,
                                "text": "Whole unit.",
                                "lang": None,
                                "lang_confidence": 0,
                            }
                        ],
                        "words": [
                            {
                                "start_ms": 100,
                                "end_ms": 900,
                                "text": "whole unit",
                            }
                        ],
                        "vad_segments_ms": [[0, 1000]],
                    },
                    "regions": [
                        {
                            "region_id": "vad_0",
                            **selected[0],
                            "processed": True,
                        }
                    ],
                },
                1.0,
            )

    output = tmp_path / "fractional-owned.json"
    payload = orchestrator.run(
        resolve_request(
            stack_id="firered",
            input_path=source,
            wants="vad,segment_timestamps,word_timestamps",
            vad="silero-vad",
        ),
        InputMetadata(str(source), 2.0, "wav", 48_000, 1),
        registry={
            "environments": {"torch-firered": {"state": "ready"}},
            "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
        },
        transport=Transport(),
        vad_detector=Detector(),
        run_range=orchestrator.parse_range("0.0005:0.5"),
        output=output,
    ).payload

    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.0005, 0.5],
        "selected_unit_scope": [0.0, 1.0],
    }
    assert load_result_document(output).owned_intervals == ((0.0, 1.0),)
    assert "Whole unit." in export_documents([output], "srt").content


def test_firered_fractional_partial_resume_uses_public_ownership_once(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "fractional-resume.wav"
    source.write_bytes(b"source")
    selected = [
        {"start": 0.0009, "end": 1.0008},
        {"start": 1.0009, "end": 2.0},
    ]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class BaseTransport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

    class PartialTransport(BaseTransport):
        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == selected
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": False,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 100,
                                "end_ms": 900,
                                "text": "First.",
                                "lang": None,
                                "lang_confidence": 0,
                            }
                        ],
                        "words": [
                            {
                                "start_ms": 100,
                                "end_ms": 900,
                                "text": "first",
                            }
                        ],
                        "vad_segments_ms": [[0, 1000]],
                    },
                    "regions": [
                        {"region_id": "vad_0", **selected[0], "processed": True},
                        {"region_id": "vad_1", **selected[1], "processed": False},
                    ],
                    "error": {"type": "RuntimeError", "message": "batch failed"},
                },
                1.0,
                returncode=4,
            )

    class ResumeTransport(BaseTransport):
        def firered(self, **kwargs):
            assert kwargs["vad_regions"] == [selected[1]]
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 1100,
                                "end_ms": 1900,
                                "text": "Second.",
                                "lang": None,
                                "lang_confidence": 0,
                            }
                        ],
                        "words": [
                            {
                                "start_ms": 1100,
                                "end_ms": 1900,
                                "text": "second",
                            }
                        ],
                        "vad_segments_ms": [[1000, 2000]],
                    },
                    "regions": [
                        {
                            "region_id": "vad_0",
                            **selected[1],
                            "processed": True,
                        }
                    ],
                },
                1.0,
            )

    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="vad,segment_timestamps",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)
    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
    }
    requested = tmp_path / "first.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=PartialTransport(),
            vad_detector=Detector(),
            output=requested,
        )
    partial = Path(caught.value.payload["output"])
    assert caught.value.payload["coverage"]["covered_through_seconds"] == 1.0
    assert "--range 1.0:" in caught.value.payload["fix"]

    rest = tmp_path / "rest.json"
    orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=ResumeTransport(),
        vad_detector=Detector(),
        run_range=orchestrator.parse_range("1.0:"),
        output=rest,
    )

    assert load_result_document(partial).owned_intervals == ((0.0, 1.0),)
    assert load_result_document(rest).owned_intervals == ((1.0, 2.0),)
    assert export_documents([partial, rest], "txt").content == "First.\nSecond.\n"


def test_firered_rejects_same_length_vad_bound_mutation_without_replacing_output(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "ledger-mutation.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="firered", input_path=source, wants="segment_timestamps,vad")
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **_kwargs):
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 300,
                                "end_ms": 900,
                                "text": "Done.",
                                "lang": None,
                                "lang_confidence": 0,
                            }
                        ],
                        "words": [
                            {
                                "start_ms": 300,
                                "end_ms": 900,
                                "text": "done",
                            }
                        ],
                        # Same cardinality as the processed ledger, but a mutated
                        # start bound that the former length-only check accepted.
                        "vad_segments_ms": [[250, 1000]],
                    },
                    "regions": [
                        {
                            "region_id": "vad_0",
                            "start": 0.2,
                            "end": 1.0,
                            "processed": True,
                        }
                    ],
                },
                1.0,
            )

    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
    }
    output = tmp_path / "existing.json"
    original = b"preexisting output must survive"
    output.write_bytes(original)

    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=Transport(),
            output=output,
            force=True,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "firered_process"
    assert caught.value.payload["backend"] == "firered-asr2s"
    assert "exactly match the processed region prefix" in caught.value.payload["detail"]
    assert output.read_bytes() == original
