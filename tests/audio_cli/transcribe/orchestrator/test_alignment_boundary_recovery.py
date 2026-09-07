"""Configured clipping must survive real host orchestration and offline exports."""

from __future__ import annotations

import json

import pytest

from audio_cli.export import export_documents
from audio_cli.transcribe.execution.receipt import build_receipt
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    FakeTransport,
    FullFakeTransport,
    InputMetadata,
    StageOutcome,
    full_registry,
    orchestrator,
    resolve_request,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)


class SentenceOverrun(FullFakeTransport):
    def qwen(self, *, units, **kwargs):
        return StageOutcome(
            "asr",
            "qwen3-asr-0.6b-8bit",
            {
                "units": [
                    {"unit_id": unit["unit_id"], "processed": True, "text": "Hello. World!"}
                    for unit in units
                ]
            },
            1,
        )

    def align(self, *, segments, **kwargs):
        return StageOutcome(
            "aligner",
            "qwen3-forcedaligner",
            {
                "segments": [
                    {
                        "unit_id": unit["unit_id"],
                        "words": [
                            {"text": "Hello", "start": 0, "end": 1},
                            {"text": "World", "start": 1, "end": 2.04},
                        ],
                    }
                    for unit in segments
                ]
            },
            1,
        )


@pytest.mark.parametrize("run_range", [None, "0.5:1.8"])
def test_qwen_explicit_recovery_preserves_failed_result_and_records_published_word(
    tmp_path, run_range
):
    run_range = orchestrator.parse_range(run_range) if run_range else None
    source = tmp_path / "source.wav"
    source.write_bytes(b"original source")
    metadata = InputMetadata(str(source), 2, "wav", 48_000, 2)
    registry = full_registry(tmp_path)
    failed_path, recovered_path = tmp_path / "strict.json", tmp_path / "recovered.json"
    original = orchestrator.run(
        resolve_request(stack_id="qwen-0.6b", input_path=source, wants="word_timestamps"),
        metadata,
        registry=registry,
        transport=SentenceOverrun(),
        output=failed_path,
        run_range=run_range,
    ).payload
    failed_bytes = failed_path.read_bytes()
    failed_receipt = build_receipt(original, failed_path)
    rejection = failed_receipt["alignment_rejections"][0]["alignment"]
    assert rejection["boundary"]["end_overrun_ms"] == 40
    assert rejection["boundary"]["max_overrun_ms"] == 0.501
    assert rejection["segment_ids"] == ["seg_0", "seg_1"]
    assert "alignment_corrections" not in failed_receipt
    assert original["provenance"]["outcomes"]["word_timestamps"] == "abstained"

    recovered = orchestrator.run(
        resolve_request(
            stack_id="qwen-0.6b",
            input_path=source,
            wants="word_timestamps",
            alignment_max_overrun_ms=40,
        ),
        metadata,
        registry=registry,
        transport=SentenceOverrun(),
        output=recovered_path,
        run_range=run_range,
    ).payload
    corrections = recovered["provenance"]["observed"]["alignment_corrections"]
    assert corrections == [
        {
            "unit_id": "unit_0",
            "segment_id": "seg_1",
            "word_id": "w_1",
            "word_index": 1,
            "original_bounds": [1.0, 2.04],
            "applied_bounds": [1.0, 2.0],
            "unit_bounds": [0.0, 2.0],
            "start_overrun_ms": 0.0,
            "end_overrun_ms": 40.0,
            "max_overrun_ms": 40.0,
        }
    ]
    assert [s["text"] for s in recovered["segments"]] == [s["text"] for s in original["segments"]]
    assert recovered["segments"][1]["words"][0]["end"] == 2
    assert recovered["provenance"]["outcomes"]["word_timestamps"] == "produced"
    assert recovered["abstentions"] == []
    receipt = build_receipt(recovered, recovered_path)
    assert receipt["alignment_corrections"] == corrections
    assert "alignment_rejections" not in receipt
    recovered_bytes = recovered_path.read_bytes()
    for fmt in ("md", "srt", "vtt"):
        product = export_documents(
            [recovered_path], fmt, timestamps=fmt == "md", provenance=fmt == "md"
        )
        assert "World" in product.content
        if fmt == "md":
            assert "Alignment boundary corrections" in product.content
    assert recovered_path.read_bytes() == recovered_bytes
    assert failed_path.read_bytes() == failed_bytes
    assert source.read_bytes() == b"original source"


def test_qwen_does_not_publish_provisional_corrections_after_text_reconciliation_fails(tmp_path):
    class Mismatched(SentenceOverrun):
        def align(self, **kwargs):
            result = super().align(**kwargs)
            result.payload["segments"][0]["words"][1]["text"] = "Different"
            return result

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    result = orchestrator.run(
        resolve_request(
            stack_id="qwen-0.6b",
            input_path=source,
            wants="word_timestamps",
            alignment_max_overrun_ms=50,
        ),
        InputMetadata(str(source), 2, "wav", 48_000, 2),
        registry=full_registry(tmp_path),
        transport=Mismatched(),
    ).payload
    assert result["abstentions"][0]["alignment"]["code"] == "sentence_reconciliation"
    assert "alignment_corrections" not in result["provenance"]["observed"]
    assert all("words" not in segment for segment in result["segments"])


def test_partial_recovery_correction_stays_within_the_published_prefix(tmp_path):
    class PartialOverrun(FakeTransport):
        def align(self, *, segments, **kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {
                            "unit_id": unit["unit_id"],
                            "words": [
                                {"text": "Hello", "start": unit["start"], "end": unit["end"] + 0.04}
                            ],
                        }
                        for unit in segments
                    ]
                },
                1,
            )

    from audio_cli.transcribe.execution.publication import PublishedPartial

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    with pytest.raises(PublishedPartial) as caught:
        orchestrator.run(
            resolve_request(
                stack_id="qwen-0.6b",
                input_path=source,
                wants="word_timestamps",
                alignment_max_overrun_ms=50,
            ),
            InputMetadata(str(source), 361, "wav", 48_000, 2),
            registry=full_registry(tmp_path),
            transport=PartialOverrun(partial=True),
            output=tmp_path / "requested.json",
        )
    error = caught.value
    saved = error.result
    assert not saved["complete"]
    correction = saved["provenance"]["observed"]["alignment_corrections"][0]
    assert correction["original_bounds"] == [0, 180.04]
    assert correction["applied_bounds"] == [0, 180]
    assert saved["coverage"]["covered_through_seconds"] == 180
    assert "--alignment-max-overrun-ms 50.0" in error.payload["fix"]
    from pathlib import Path

    partial_path = Path(error.payload["output"])
    assert json.loads(partial_path.read_text())["provenance"]["observed"]["alignment_corrections"]
    assert "Hello" in export_documents([partial_path], "srt").content
