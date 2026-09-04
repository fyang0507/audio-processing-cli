from __future__ import annotations

from transcribe_native_test_support import *  # noqa: F403

@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        ("timestamp", "must be a finite non-negative number"),
        ("confidence", "must be between 0 and 1"),
    ],
)
def test_firered_overflowing_stage_number_is_a_typed_backend_failure(
    tmp_path: Path, monkeypatch, mutation: str, detail: str
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / f"firered-{mutation}-overflow.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="segment_timestamps,lid",
    )
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
            raw = {
                "sentences": [{
                    "start_ms": 0,
                    "end_ms": 1000,
                    "text": "Done.",
                    "lang": "en",
                    "lang_confidence": 0.9,
                }],
                "words": [{
                    "start_ms": 100,
                    "end_ms": 900,
                    "text": "done",
                }],
                "vad_segments_ms": [[0, 1000]],
            }
            if mutation == "timestamp":
                raw["words"][0]["start_ms"] = 10**400
            else:
                raw["sentences"][0]["lang_confidence"] = 10**400
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": raw,
                    "regions": [{
                        "region_id": "vad_0",
                        "start": 0.0,
                        "end": 1.0,
                        "processed": True,
                    }],
                },
                1.0,
            )

    output = tmp_path / f"firered-{mutation}-overflow.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-firered": {"state": "ready"}},
                "packages": {
                    "firered-asr2s": _ready_multi_package(
                        tmp_path, "firered-asr2s"
                    ),
                },
            },
            transport=Transport(),
            output=output,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "firered_process"
    assert caught.value.payload["backend"] == "firered-asr2s"
    assert detail in caught.value.payload["detail"]
    assert not output.exists()


@pytest.mark.parametrize(
    "provided",
    ["0.00003:0.000063", "0.0000629:0.000063"],
)
def test_vibevoice_subsample_eof_range_is_typed_before_model_work(
    tmp_path: Path, monkeypatch, provided: str
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    source = tmp_path / "one-frame.wav"
    source.write_bytes(b"source")
    request = resolve_request(stack_id="vibevoice", input_path=source, wants=())
    metadata = InputMetadata(str(source), 0.000063, "wav", 16_000, 1)
    calls: list[str] = []

    class Transport:
        def decode(self, _source, target):
            calls.append("decode")
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0")
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            calls.append("vibevoice")
            raise AssertionError("sub-sample range reached model work")

    output = tmp_path / "one-frame-result.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {"torch-vibevoice": {"state": "ready"}},
                "packages": {
                    "vibevoice-asr-7b": _ready_multi_package(
                        tmp_path, "vibevoice-asr-7b"
                    ),
                },
            },
            transport=Transport(),
            output=output,
            run_range=orchestrator.parse_range(provided),
        )

    assert caught.value.exit_code == 2
    assert caught.value.payload["code"] == "range_invalid"
    assert caught.value.payload["provided"] == provided
    assert "no complete sample" in caught.value.payload["reason"]
    assert calls == ["decode"]
    assert not output.exists()


@pytest.mark.parametrize(
    ("returncode", "transport_failure", "detail"),
    [(1, True, "synthetic failure"), (4, False, "unsupported exit 4")],
)
def test_vibevoice_failed_aligner_is_a_backend_error_and_writes_no_result(
    tmp_path: Path,
    monkeypatch,
    returncode: int,
    transport_failure: bool,
    detail: str,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-vibevoice")
    _runtime(tmp_path, monkeypatch, "mlx")
    source = tmp_path / "aligner-failure.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="vibevoice", input_path=source, wants="word_timestamps"
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def vibevoice(self, **_kwargs):
            return StageOutcome("asr", "vibevoice-asr-7b", _complete_vibe_payload([{
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "speaker_id": 0,
                    "text": "Hello.",
                }]), 1.0)

        def align(self, **_kwargs):
            outcome = StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {"error": {"message": "synthetic failure"}},
                2.5,
                returncode=returncode,
                peak_rss_bytes=250,
            )
            if not transport_failure:
                return outcome
            raise orchestrator.StageFailure(
                "aligner",
                "qwen3-forcedaligner",
                "synthetic failure",
                outcome=outcome,
            )

    registry = {
        "environments": {
            "torch-vibevoice": {"state": "ready"},
            "mlx": {"state": "ready"},
        },
        "packages": {
            "vibevoice-asr-7b": _ready_multi_package(
                tmp_path, "vibevoice-asr-7b"
            ),
            "qwen3-forcedaligner": _ready_single_package(
                tmp_path, "qwen3-forcedaligner"
            ),
        },
    }
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(Exception) as caught:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=Transport(),
            output=output,
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "aligner"
    assert caught.value.payload["backend"] == "qwen3-forcedaligner"
    assert detail in caught.value.payload["detail"]
    assert not output.exists()
