from __future__ import annotations

from pathlib import Path

from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    InputMetadata,
    StageOutcome,
    _complete_vibe_payload,
    _ready_multi_package,
    _ready_single_package,
    _runtime,
    json,
    orchestrator,
    pytest,
    resolve_request,
    wave,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)


@pytest.mark.parametrize(
    "words, rejection",
    [
        ([], {"code": "text_mismatch"}),
        ([{"text": "Goodbye", "start": 0.2, "end": 1.8}], {"code": "text_mismatch"}),
        (
            [
                {"text": "Hel", "start": 1.0, "end": 1.4},
                {"text": "lo", "start": 0.2, "end": 0.8},
            ],
            {"code": "word_order", "word_index": 1},
        ),
        (
            [
                {"text": "Hel", "start": 0.2, "end": 1.2},
                {"text": "lo", "start": 1.0, "end": 1.8},
            ],
            {"code": "word_order", "word_index": 1},
        ),
        (
            [
                {"text": "", "start": 0.2, "end": 0.3},
                {"text": "Hello", "start": 0.3, "end": 1.8},
            ],
            {"code": "invalid_token", "word_index": 0},
        ),
    ],
)
def test_vibevoice_nonconforming_speech_alignment_is_a_segment_abstention(
    tmp_path: Path,
    monkeypatch,
    words: list[dict],
    rejection: dict,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "bad-alignment.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants="word_timestamps")
    metadata = InputMetadata(str(source), 5.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 80_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                _complete_vibe_payload(
                    [
                        {
                            "start_time": 0.0,
                            "end_time": 2.0,
                            "speaker_id": 0,
                            "text": "Hello.",
                        },
                        {
                            "start_time": 2.0,
                            "end_time": 3.0,
                            "text": "[Environmental Sounds]",
                        },
                        {
                            "start_time": 3.0,
                            "end_time": 5.0,
                            "speaker_id": 0,
                            "text": "Again.",
                        },
                    ],
                    generated_tokens=4,
                    eos_observed=True,
                ),
                1.0,
            )

        def align(self, **_kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {"unit_id": "native_0", "words": words},
                        {"unit_id": "native_1", "words": None},
                    ],
                },
                1.0,
            )

    registry = {
        "environments": {
            "torch-vibevoice": {"state": "ready"},
            "mlx": {"state": "ready"},
        },
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
            "qwen3-forcedaligner": _ready_single_package(tmp_path, "qwen3-forcedaligner"),
        },
    }
    output = tmp_path / "alignment-abstention.json"
    payload = orchestrator.run(
        request,
        metadata,
        registry=registry,
        transport=Transport(),
        output=output,
    ).payload

    assert payload["segments"] == [
        {"segment_id": "seg_0", "text": "Hello."},
        {"segment_id": "seg_1", "text": "[Environmental Sounds]"},
        {"segment_id": "seg_2", "text": "Again."},
    ]
    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "alignment": {"unit_id": "native_0", "segment_ids": ["seg_0"], **rejection},
            "start": 0.0,
            "end": 2.0,
        },
        {
            "abstention_id": "ab_1",
            "reason": "alignment_unavailable",
            "alignment": {
                "unit_id": "native_1",
                "segment_ids": ["seg_2"],
                "code": "provider_unavailable",
            },
            "start": 3.0,
            "end": 5.0,
        },
    ]
    assert payload["provenance"]["outcomes"]["word_timestamps"] == "abstained"
    assert payload["provenance"]["observed"]["segments_without_words"] == 3
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    ("aligner_segments", "expected_outcome", "expected_words"),
    [
        (
            [{"unit_id": "native_0", "words": []}],
            "produced",
            [],
        ),
        ([{"unit_id": "native_0", "words": None}], "abstained", None),
    ],
)
def test_vibevoice_punctuation_only_empty_mapping_differs_from_absence(
    tmp_path: Path,
    monkeypatch,
    aligner_segments: list[dict],
    expected_outcome: str,
    expected_words: list | None,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "punctuation-only.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants="word_timestamps")
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 1)

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
                _complete_vibe_payload(
                    [
                        {
                            "start_time": 0.0,
                            "end_time": 1.0,
                            "speaker_id": 0,
                            "text": "……？！",
                        }
                    ]
                ),
                1.0,
            )

        def align(self, **_kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": aligner_segments,
                },
                1.0,
            )

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {
                "torch-vibevoice": {"state": "ready"},
                "mlx": {"state": "ready"},
            },
            "packages": {
                "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
                "qwen3-forcedaligner": _ready_single_package(tmp_path, "qwen3-forcedaligner"),
            },
        },
        transport=Transport(),
    ).payload

    segment = payload["segments"][0]
    if expected_words is None:
        assert "words" not in segment
        assert payload["provenance"]["observed"]["segments_without_words"] == 1
        assert payload["abstentions"] == [
            {
                "abstention_id": "ab_0",
                "reason": "alignment_unavailable",
                "alignment": {
                    "unit_id": "native_0",
                    "segment_ids": ["seg_0"],
                    "code": "provider_unavailable",
                },
                "start": 0.0,
                "end": 1.0,
            }
        ]
    else:
        assert segment["words"] == expected_words
        assert payload["provenance"]["observed"]["segments_without_words"] == 0
        assert payload["abstentions"] == []
    assert payload["provenance"]["outcomes"]["word_timestamps"] == expected_outcome
