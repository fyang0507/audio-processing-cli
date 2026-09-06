from __future__ import annotations

from pathlib import Path

from audio_cli.transcribe.execution import PublishedPartial, build_receipt
from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    InputMetadata,
    StageOutcome,
    _ready_multi_package,
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


def test_firered_partial_writes_prefix_coverage_and_runnable_resume(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "partial.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="firered", input_path=source, wants="segment_timestamps,vad")
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **_kwargs):
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": False,
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
                        "words": [{"start_ms": 300, "end_ms": 900, "text": "done"}],
                        "vad_segments_ms": [[200, 1000]],
                    },
                    "regions": [
                        {"region_id": "vad_0", "start": 0.2, "end": 1.0, "processed": True},
                        {"region_id": "vad_1", "start": 1.5, "end": 2.5, "processed": False},
                    ],
                    "error": {"type": "RuntimeError", "message": "batch failed"},
                },
                1.0,
                returncode=4,
            )

    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
    }
    output = tmp_path / "partial.json"
    with pytest.raises(Exception) as raised:
        orchestrator.run(request, metadata, registry=registry, transport=Transport(), output=output)
    refusal = raised.value
    assert refusal.exit_code == 4
    assert refusal.payload["coverage"] == {
        "scope_intervals": [[0.2, 2.5]],
        "covered_through_seconds": 1.5,
        "covered_fraction": 0.565217,
        "covered_intervals": [[0.2, 1.5]],
        "missing_intervals": [[1.5, 2.5]],
        "units_total": 2,
        "units_completed": 1,
    }
    assert "--range 1.5:" in refusal.payload["fix"]
    partial = json.loads((tmp_path / "partial.partial.json").read_text(encoding="utf-8"))
    assert isinstance(refusal, PublishedPartial)
    assert refusal.result == partial
    receipt = build_receipt(refusal.result, refusal.payload["output"])
    assert receipt["complete"] is False
    assert receipt["coverage"] == refusal.payload["coverage"]
    assert receipt["counts"]["segments"] == 1
    assert partial["complete"] is False
    assert partial["segments"] == [
        {
            "segment_id": "seg_0",
            "text": "Done.",
            "start": 0.3,
            "end": 0.9,
        }
    ]
