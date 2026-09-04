from __future__ import annotations

from transcribe_native_test_support import (
    InputMetadata,
    Path,
    StageOutcome,
    _complete_vibe_payload,
    _ready_fluidaudio,
    _ready_multi_package,
    _ready_single_package,
    _runtime,
    json,
    normalize_vibevoice_result,
    orchestrator,
    os,
    pytest,
    refusals,
    resolve_request,
    wave,
)
from transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)

from audio_cli.transcribe.native import vibevoice as vibevoice_execution


def test_vibevoice_native_events_keep_bounds_but_not_speaker_or_words(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "demo.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice",
        input_path=source,
        wants=("verbatim,diarization,segment_timestamps,word_timestamps,overlapped_speech"),
    )
    metadata = InputMetadata(str(source), 10.0, "wav", 48_000, 2)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 160_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **kwargs):
            assert kwargs["model"].name == ("d0c9efdb8d614685062c04425d91e01b6f37d944")
            assert kwargs["tokenizer"].name == ("d149729398750b98c0af14eb82c78cfe92750796")
            assert kwargs["config"] == {
                "device": "mps",
                "dtype": "bfloat16",
                "attention": "sdpa",
                "seed": 1234,
                "max_new_tokens": 16384,
            }
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {"start_time": 0.0, "end_time": 2.0, "speaker_id": 0, "text": "Hello."},
                        {"start_time": 2.0, "end_time": 3.0, "text": "[Environmental Sounds]"},
                        {"start_time": 3.0, "end_time": 5.0, "speaker_id": 0, "text": "Again."},
                    ],
                    generated_tokens=20,
                    eos_observed=True,
                ),
                3.0,
                peak_mps_live_bytes=100,
            )

        def align(self, *, segments, **kwargs):
            assert kwargs["model"].name == ("0e1a68e91d815300c7c9754b2a7639378b23db15")
            assert [item["unit_id"] for item in segments] == ["native_0", "native_1"]
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {
                            "unit_id": "native_0",
                            "words": [
                                {"text": "Hello", "start": 0.2, "end": 1.8},
                            ],
                        },
                        {
                            "unit_id": "native_1",
                            "words": [
                                {"text": "Again", "start": 3.2, "end": 4.8},
                            ],
                        },
                    ]
                },
                1.0,
            )

        def diarize(self, **_kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {"startTimeSeconds": 0.0, "endTimeSeconds": 2.0, "speakerId": "S0"},
                        {"startTimeSeconds": 1.0, "endTimeSeconds": 1.5, "speakerId": "S1"},
                    ]
                },
                0.2,
            )

    registry = {
        "environments": {
            "torch-vibevoice": {"state": "ready"},
            "mlx": {"state": "ready"},
            "swift": {"state": "ready"},
        },
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
            "qwen3-forcedaligner": _ready_single_package(tmp_path, "qwen3-forcedaligner"),
            "fluidaudio": _ready_fluidaudio(tmp_path),
            "speaker-diarization-coreml": _ready_single_package(
                tmp_path, "speaker-diarization-coreml"
            ),
        },
    }
    payload = orchestrator.run(request, metadata, registry=registry, transport=Transport()).payload
    assert payload["segments"][1] == {
        "segment_id": "seg_1",
        "text": "[Environmental Sounds]",
        "start": 2.0,
        "end": 3.0,
    }
    assert "speaker" not in payload["segments"][0]
    assert payload["segments"][2]["speaker"] == "0"
    assert payload["overlapped_speech"] == [
        {
            "overlap_id": "overlap_0",
            "start": 1.0,
            "end": 1.5,
        }
    ]
    assert payload["turns"] == [
        {"turn_id": "turn_0", "speaker": "0", "start": 0.0, "end": 2.0},
        {"turn_id": "turn_1", "speaker": "0", "start": 3.0, "end": 5.0},
    ]
    assert payload["provenance"]["outcomes"]["word_timestamps"] == "produced"
    assert payload["provenance"]["observed"]["segments_without_words"] == 1
    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "overlap",
            "start": 1.0,
            "end": 1.5,
        }
    ]
    assert "N/A" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("segments", "detail"),
    [
        (
            [{"start_time": 0.0, "end_time": 1.0, "text": "Unlabelled speech."}],
            "requires a speaker label",
        ),
        (
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "speaker_id": 0,
                    "text": "[Music]",
                }
            ],
            "must not carry a speaker",
        ),
    ],
)
def test_vibevoice_native_diarization_rejects_uncoupled_segment_labels(
    tmp_path: Path,
    monkeypatch,
    segments: list[dict],
    detail: str,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "invalid-native-diarization.wav"
    source.write_bytes(b"source")
    output = tmp_path / "result.json"

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(segments),
                1.0,
            )

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolve_request(
                stack_id="vibevoice",
                input_path=source,
                wants="diarization,segment_timestamps",
            ),
            InputMetadata(str(source), 1.0, "wav", 48_000, 1),
            registry={
                "environments": {"torch-vibevoice": {"state": "ready"}},
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b")
                },
            },
            transport=Transport(),
            output=output,
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["backend"] == "vibevoice-asr-7b"
    assert detail in raised.value.payload["detail"]
    assert not output.exists()


def test_native_publication_preserves_source_renamed_to_forced_output_after_decode(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    output = tmp_path / "result.json"
    output.write_bytes(b"replaceable")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 16_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 16_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            os.replace(source, output)
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {
                            "start_time": 0.0,
                            "end_time": 1.0,
                            "speaker_id": 0,
                            "text": "Hello.",
                        }
                    ]
                ),
                1.0,
            )

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-vibevoice": {"state": "ready"}},
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b")
                },
            },
            transport=Transport(),
            output=output,
            force=True,
        )

    assert raised.value.payload["code"] == "output_is_canonical_input"
    assert not source.exists()
    assert output.read_bytes() == b"source"


def test_vibevoice_native_turns_preserve_the_recorded_same_speaker_gap() -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[1] / "tests/fixtures/vibevoice_multispeaker_excerpt.json"
        ).read_text(encoding="utf-8")
    )
    normalized = normalize_vibevoice_result(
        {
            "segments": fixture["segments"],
            "raw_text": json.dumps(fixture["segments"]),
            "hit_max_new_tokens": False,
        },
        clip_duration_seconds=60.0,
    )

    turns = vibevoice_execution._native_turns(normalized.segments)
    assert turns[:2] == [
        {"turn_id": "turn_0", "speaker": "1", "start": 16.83, "end": 32.05},
        {"turn_id": "turn_1", "speaker": "1", "start": 33.64, "end": 37.84},
    ]
    assert turns[1]["start"] - turns[0]["end"] == pytest.approx(1.59)
