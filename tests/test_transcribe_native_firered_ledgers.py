from __future__ import annotations

from pathlib import Path

from transcribe_native_test_support import (
    InputMetadata,
    StageOutcome,
    _ready_multi_package,
    _runtime,
    json,
    orchestrator,
    pytest,
    refusals,
    resolve_request,
    wave,
)
from transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)

from audio_cli.transcribe.adapters import firered_ledger


@pytest.mark.parametrize(
    ("regions", "detail"),
    [
        (
            [
                {"region_id": "same", "start": 0.1, "end": 0.5, "processed": True},
                {"region_id": "same", "start": 0.6, "end": 1.0, "processed": False},
            ],
            "unique non-empty strings",
        ),
        (
            [{"region_id": "vad_0", "start": True, "end": 0.5, "processed": True}],
            "start must be a number",
        ),
        (
            [{"region_id": "vad_0", "start": 10**400, "end": 1.0, "processed": True}],
            "start must be finite",
        ),
        (
            [{"region_id": "vad_0", "start": 0.5, "end": 0.5, "processed": True}],
            "positive bounds within the source timeline",
        ),
        (
            [
                {"region_id": "vad_0", "start": 0.2, "end": 0.8, "processed": True},
                {"region_id": "vad_1", "start": 0.7, "end": 1.0, "processed": False},
            ],
            "chronological and non-overlapping",
        ),
        (
            [{"region_id": "vad_0", "start": 0.2, "end": 2.1, "processed": True}],
            "within the source timeline",
        ),
        (
            [{"region_id": "vad_0", "start": 0.2, "end": 0.8, "processed": True}],
            "does not intersect the requested processing range",
        ),
        (
            [
                {"region_id": "vad_0", "start": 0.2, "end": 0.8, "processed": False},
                {"region_id": "vad_1", "start": 0.9, "end": 1.2, "processed": True},
            ],
            "processed regions must form a prefix",
        ),
    ],
)
def test_firered_region_ledger_rejects_untrusted_stage_mutations(
    regions: list[dict], detail: str
) -> None:
    scope = (1.0, 2.0) if "does not intersect" in detail else (0.0, 2.0)
    with pytest.raises((TypeError, ValueError), match=detail):
        firered_ledger._firered_region_ledger(
            regions,
            source_duration=2.0,
            requested_scope=scope,
        )


def test_firered_ledger_comparison_uses_the_stage_millisecond_truncation() -> None:
    ledger = firered_ledger._firered_published_region_ledger(
        [
            {
                "region_id": "vad_0",
                "start": 0.2009,
                "end": 0.9999,
                "processed": True,
            },
            {
                "region_id": "vad_1",
                "start": 1.1119,
                "end": 1.5559,
                "processed": False,
            },
        ]
    )
    assert firered_ledger._firered_published_vad_prefix(ledger) == ({"start": 0.2, "end": 0.999},)


@pytest.mark.parametrize("mutation", ["dropped", "reordered_ids", "raw_bounds"])
def test_firered_external_silero_stage_ledger_must_preserve_every_selected_unit(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / f"external-ledger-{mutation}.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,vad",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)
    selected = [
        {"start": 0.2009, "end": 0.8009},
        {"start": 1.2009, "end": 1.8009},
    ]

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
            regions = [
                {"region_id": "vad_0", **selected[0], "processed": True},
                {"region_id": "vad_1", **selected[1], "processed": True},
            ]
            if mutation == "dropped":
                regions.pop()
            elif mutation == "reordered_ids":
                regions[0]["region_id"], regions[1]["region_id"] = (
                    regions[1]["region_id"],
                    regions[0]["region_id"],
                )
            else:
                # This remains the same 200 ms public FireRed bound, so only
                # exact binding to the selected Silero ledger can detect it.
                regions[0]["start"] = 0.2008
            count = len(regions)
            sentences = [
                {
                    "start_ms": 300,
                    "end_ms": 700,
                    "text": "First.",
                    "lang": None,
                    "lang_confidence": 0,
                },
                {
                    "start_ms": 1300,
                    "end_ms": 1700,
                    "text": "Second.",
                    "lang": None,
                    "lang_confidence": 0,
                },
            ][:count]
            words = [
                {"start_ms": 300, "end_ms": 700, "text": "first"},
                {"start_ms": 1300, "end_ms": 1700, "text": "second"},
            ][:count]
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": sentences,
                        "words": words,
                        "vad_segments_ms": [
                            [int(item["start"] * 1000), int(item["end"] * 1000)] for item in regions
                        ],
                    },
                    "regions": regions,
                },
                1.0,
            )

    output = tmp_path / "must-not-exist.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-firered": {"state": "ready"}},
                "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
            },
            transport=Transport(),
            vad_detector=Detector(),
            output=output,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "firered_process"
    assert "supplied Silero VAD" in caught.value.payload["detail"]
    assert not output.exists()
    assert not (tmp_path / "must-not-exist.partial.json").exists()


def test_firered_silero_preserves_exact_fractional_vad_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "fractional-silero.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,vad",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)
    selected = [
        {"start": 0.2009, "end": 0.9999},
        {"start": 1.5009, "end": 2.5009},
    ]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
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
                                "start_ms": 300,
                                "end_ms": 900,
                                "text": "First.",
                                "lang": None,
                                "lang_confidence": 0,
                            },
                            {
                                "start_ms": 1600,
                                "end_ms": 2400,
                                "text": "Second.",
                                "lang": None,
                                "lang_confidence": 0,
                            },
                        ],
                        "words": [
                            {"start_ms": 300, "end_ms": 900, "text": "first"},
                            {
                                "start_ms": 1600,
                                "end_ms": 2400,
                                "text": "second",
                            },
                        ],
                        # FireRed's result is validated on its integer-ms grid.
                        "vad_segments_ms": [[200, 999], [1500, 2500]],
                    },
                    # The stage ledger retains the exact selected Silero spans.
                    "regions": [
                        {"region_id": "vad_0", **selected[0], "processed": True},
                        {"region_id": "vad_1", **selected[1], "processed": True},
                    ],
                },
                1.0,
            )

    payload = orchestrator.run(
        request,
        metadata,
        registry={
            "environments": {"torch-firered": {"state": "ready"}},
            "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
        },
        transport=Transport(),
        vad_detector=Detector(),
    ).payload

    assert payload["vad_regions"] == selected


def test_firered_partial_silero_preserves_exact_processed_fractional_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "fractional-silero-partial.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,vad",
        vad="silero-vad",
    )
    metadata = InputMetadata(str(source), 3.0, "wav", 48_000, 1)
    selected = [
        {"start": 0.2009, "end": 0.9999},
        {"start": 1.5009, "end": 2.5009},
    ]

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return selected

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 48_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

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
                                "start_ms": 300,
                                "end_ms": 900,
                                "text": "First.",
                                "lang": None,
                                "lang_confidence": 0,
                            }
                        ],
                        "words": [
                            {
                                "start_ms": 300,
                                "end_ms": 900,
                                "text": "first",
                            }
                        ],
                        "vad_segments_ms": [[200, 999]],
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

    output = tmp_path / "fractional-silero-partial.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-firered": {"state": "ready"}},
                "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
            },
            transport=Transport(),
            vad_detector=Detector(),
            output=output,
        )

    assert caught.value.exit_code == 4
    partial = json.loads(
        (tmp_path / "fractional-silero-partial.partial.json").read_text(encoding="utf-8")
    )
    assert partial["vad_regions"] == [selected[0]]
