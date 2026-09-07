"""VibeVoice correction evidence follows accepted words and native segment bounds."""

from __future__ import annotations

import json
import wave
from copy import deepcopy

import pytest

from audio_cli.export import export_documents
from audio_cli.transcribe import orchestrator
from audio_cli.transcribe.catalog import InputMetadata
from audio_cli.transcribe.execution.receipt import build_receipt
from audio_cli.transcribe.planner import resolve_request
from audio_cli.transcribe.transport import StageOutcome
from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    _complete_vibe_payload,
    _ready_multi_package,
    _ready_single_package,
    _runtime,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)


class BoundaryTransport:
    def __init__(self, first_word="Hello"):
        self.raw_alignment = {
            "segments": [
                {
                    "unit_id": "native_0",
                    "words": [{"text": first_word, "start": 0.0, "end": 2.014}],
                },
                {
                    "unit_id": "native_1",
                    "words": [{"text": "Again", "start": 2.99, "end": 5.0}],
                },
            ]
        }
        self.original_alignment = deepcopy(self.raw_alignment)

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
                    {"start_time": 0.0, "end_time": 2.0, "speaker_id": 0, "text": "Hello."},
                    {"start_time": 2.0, "end_time": 3.0, "text": "[Environmental Sounds]"},
                    {"start_time": 3.0, "end_time": 5.0, "speaker_id": 0, "text": "Again."},
                ],
                generated_tokens=4,
                eos_observed=True,
            ),
            1.0,
        )

    def align(self, *, segments, **_kwargs):
        assert segments == [
            {"unit_id": "native_0", "text": "Hello.", "start": 0.0, "end": 2.0},
            {"unit_id": "native_1", "text": "Again.", "start": 3.0, "end": 5.0},
        ]
        return StageOutcome("aligner", "qwen3-forcedaligner", self.raw_alignment, 1.0)


@pytest.fixture
def run_boundary(tmp_path, monkeypatch):
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "source.wav"
    source.write_bytes(b"original source")
    registry = {
        "environments": {"torch-vibevoice": {"state": "ready"}, "mlx": {"state": "ready"}},
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(tmp_path, "vibevoice-asr-7b"),
            "qwen3-forcedaligner": _ready_single_package(tmp_path, "qwen3-forcedaligner"),
        },
    }

    def run(name, *, limit=None, first_word="Hello"):
        transport = BoundaryTransport(first_word)
        output = tmp_path / f"{name}.json"
        payload = orchestrator.run(
            resolve_request(
                stack_id="vibevoice",
                input_path=source,
                wants="diarization,segment_timestamps,word_timestamps",
                alignment_max_overrun_ms=limit,
            ),
            InputMetadata(str(source), 5.0, "wav", 48_000, 1),
            registry=registry,
            transport=transport,
            output=output,
        ).payload
        assert json.loads(output.read_text(encoding="utf-8")) == payload
        assert transport.raw_alignment == transport.original_alignment
        assert source.read_bytes() == b"original source"
        return payload, output, build_receipt(payload, output)

    return run


def native_segments():
    return [
        {"segment_id": "seg_0", "text": "Hello.", "speaker": "0", "start": 0.0, "end": 2.0},
        {"segment_id": "seg_1", "text": "[Environmental Sounds]", "start": 2.0, "end": 3.0},
        {"segment_id": "seg_2", "text": "Again.", "speaker": "0", "start": 3.0, "end": 5.0},
    ]


def expected_corrections():
    return [
        {
            "unit_id": "native_0",
            "segment_id": "seg_0",
            "word_id": "w_0",
            "word_index": 0,
            "original_bounds": [0.0, 2.014],
            "unit_bounds": [0.0, 2.0],
            "applied_bounds": [0.0, 2.0],
            "start_overrun_ms": 0.0,
            "end_overrun_ms": 14.0,
            "max_overrun_ms": 20.0,
        },
        {
            "unit_id": "native_1",
            "segment_id": "seg_2",
            "word_id": "w_1",
            "word_index": 0,
            "original_bounds": [2.99, 5.0],
            "unit_bounds": [3.0, 5.0],
            "applied_bounds": [3.0, 5.0],
            "start_overrun_ms": 10.0,
            "end_overrun_ms": 0.0,
            "max_overrun_ms": 20.0,
        },
    ]


def test_configured_clipping_preserves_native_bounds_and_publishes_corrections(run_boundary):
    strict, strict_path, _ = run_boundary("strict")
    strict_bytes = strict_path.read_bytes()
    recovered, recovered_path, receipt = run_boundary("recovered", limit=20)
    expected = native_segments()
    expected[0]["words"] = [{"word_id": "w_0", "text": "Hello", "start": 0.0, "end": 2.0}]
    expected[2]["words"] = [{"word_id": "w_1", "text": "Again", "start": 3.0, "end": 5.0}]
    assert recovered["segments"] == expected
    assert (
        recovered["turns"]
        == strict["turns"]
        == [
            {"turn_id": "turn_0", "speaker": "0", "start": 0.0, "end": 2.0},
            {"turn_id": "turn_1", "speaker": "0", "start": 3.0, "end": 5.0},
        ]
    )
    assert recovered["provenance"]["plan"]["roles"]["aligner"]["config"]["max_overrun_ms"] == 20
    assert recovered["provenance"]["observed"]["alignment_corrections"] == expected_corrections()
    assert receipt["alignment_corrections"] == expected_corrections()
    assert "alignment_rejections" not in receipt
    assert recovered["abstentions"] == []
    assert recovered["provenance"]["outcomes"] == {
        "diarization": "produced",
        "segment_timestamps": "produced",
        "word_timestamps": "produced",
    }
    recovered_bytes = recovered_path.read_bytes()
    for fmt in ("md", "srt", "vtt"):
        product = export_documents(
            [recovered_path], fmt, timestamps=fmt == "md", provenance=fmt == "md"
        )
        assert "Hello" in product.content and "Again" in product.content
        if fmt == "md":
            assert "Alignment boundary corrections" in product.content
    assert strict_path.read_bytes() == strict_bytes
    assert recovered_path.read_bytes() == recovered_bytes


def test_default_refuses_overruns_but_native_timing_remains_exportable(run_boundary):
    result, output, receipt = run_boundary("default")
    assert result["segments"] == native_segments()
    assert result["complete"] is True
    assert result["provenance"]["outcomes"] == {
        "diarization": "produced",
        "segment_timestamps": "produced",
        "word_timestamps": "abstained",
    }
    assert result["provenance"]["plan"]["roles"]["aligner"]["config"]["max_overrun_ms"] == 0.501
    expected = []
    for index, correction in enumerate(expected_corrections()):
        boundary = {
            key: correction[key]
            for key in ("original_bounds", "unit_bounds", "start_overrun_ms", "end_overrun_ms")
        }
        boundary["max_overrun_ms"] = 0.501
        expected.append(
            {
                "abstention_id": f"ab_{index}",
                "reason": "alignment_unavailable",
                "start": correction["unit_bounds"][0],
                "end": correction["unit_bounds"][1],
                "alignment": {
                    "unit_id": correction["unit_id"],
                    "segment_ids": [correction["segment_id"]],
                    "code": "out_of_unit_bounds",
                    "word_index": 0,
                    "boundary": boundary,
                },
            }
        )
    assert receipt["alignment_rejections"] == result["abstentions"] == expected
    assert "alignment_corrections" not in result["provenance"]["observed"]
    assert "alignment_corrections" not in receipt
    product = export_documents([output], "md", timestamps=True)
    assert "Hello" in product.content and "Again" in product.content


def test_text_mismatch_discards_its_correction_and_binds_later_published_word(run_boundary):
    result, _output, receipt = run_boundary("mismatch", limit=20, first_word="Goodbye")
    expected = native_segments()
    expected[2]["words"] = [{"word_id": "w_0", "text": "Again", "start": 3.0, "end": 5.0}]
    assert result["segments"] == expected
    surviving = {**expected_corrections()[1], "word_id": "w_0"}
    assert result["provenance"]["observed"]["alignment_corrections"] == [surviving]
    assert receipt["alignment_corrections"] == [surviving]
    assert (
        receipt["alignment_rejections"]
        == result["abstentions"]
        == [
            {
                "abstention_id": "ab_0",
                "reason": "alignment_unavailable",
                "start": 0.0,
                "end": 2.0,
                "alignment": {
                    "unit_id": "native_0",
                    "segment_ids": ["seg_0"],
                    "code": "text_mismatch",
                },
            }
        ]
    )
    assert result["provenance"]["outcomes"]["word_timestamps"] == "abstained"
