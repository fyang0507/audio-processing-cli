from __future__ import annotations

from pathlib import Path

from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    FakeTransport,
    FullFakeTransport,
    InputMetadata,
    StageOutcome,
    full_registry,
    orchestrator,
    pytest,
    refusals,
    registry,
    request,
    resolve_request,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)


@pytest.mark.parametrize(
    ("partial_payload", "returncode"),
    [(False, 4), (True, 0)],
)
def test_qwen_exit_status_must_match_the_unfinished_unit_ledger(
    tmp_path: Path,
    partial_payload: bool,
    returncode: int,
) -> None:
    class ContradictoryTransport(FakeTransport):
        def __init__(self) -> None:
            super().__init__(partial=partial_payload)

        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            return StageOutcome(
                outcome.role,
                outcome.backend,
                outcome.payload,
                outcome.wall_seconds,
                returncode=returncode,
                peak_rss_bytes=outcome.peak_rss_bytes,
            )

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=ContradictoryTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "asr"
    assert "exit status disagrees with unfinished unit ledger" in (raised.value.payload["detail"])


def test_invalid_internal_stage_walls_are_a_typed_backend_refusal(tmp_path) -> None:
    class InvalidMetricsTransport(FakeTransport):
        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            return StageOutcome(
                outcome.role,
                outcome.backend,
                outcome.payload,
                outcome.wall_seconds,
                returncode=outcome.returncode,
                peak_rss_bytes=outcome.peak_rss_bytes,
                wall_seconds_by_stage={"asr": outcome.wall_seconds + 1.0},
            )

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=InvalidMetricsTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "asr"
    assert "internal wall metrics exceed its process wall" in (raised.value.payload["detail"])


@pytest.mark.parametrize(
    "canonical_bytes",
    [b"", b"not a wave", bytes.fromhex("52494646a8eddd645741564545e52057cfbfd3cf")],
)
def test_malformed_canonical_wav_is_a_typed_decode_failure(
    tmp_path,
    canonical_bytes: bytes,
) -> None:
    class MalformedDecodeTransport(FakeTransport):
        def decode(self, source, target):
            target.write_bytes(canonical_bytes)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=MalformedDecodeTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "decode"
    assert raised.value.payload["backend"] == "ffmpeg"
    assert isinstance(raised.value.payload["detail"], str)


def test_backend_derived_result_error_is_one_json_refusal(tmp_path) -> None:
    class NonLabelTransport(FullFakeTransport):
        def diarize(self, **kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {
                            "startTimeSeconds": 0.0,
                            "endTimeSeconds": 1.0,
                            "speakerId": "N/A",
                        }
                    ]
                },
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=NonLabelTransport(),
        )
    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert "must be absent" in raised.value.payload["detail"]


@pytest.mark.parametrize(
    "invalid_segment",
    [
        {
            "startTimeSeconds": False,
            "endTimeSeconds": 1.0,
            "speakerId": "S1",
        },
        {
            "startTimeSeconds": 0.0,
            "endTimeSeconds": 1.0,
            "speakerId": [],
        },
    ],
)
def test_invalid_diarizer_artifact_is_a_typed_backend_refusal(
    tmp_path,
    invalid_segment: dict[str, object],
) -> None:
    class InvalidDiarizerTransport(FullFakeTransport):
        def diarize(self, **kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {"segments": [invalid_segment]},
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=InvalidDiarizerTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "diarizer"
    assert raised.value.payload["backend"] == "fluidaudio"


def test_invalid_vad_artifact_is_a_typed_backend_refusal(tmp_path) -> None:
    class OverlappingVad:
        def detect(self, samples, rate, **config):
            return [
                {"start": 0.0, "end": 0.8},
                {"start": 0.7, "end": 1.0},
            ]

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("vad",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=FullFakeTransport(),
            vad_detector=OverlappingVad(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "vad"
    assert raised.value.payload["backend"] == "silero-vad"
    assert "overlaps" in raised.value.payload["detail"]


def test_abstentions_are_reidentified_in_source_order(tmp_path) -> None:
    class MixedEvidenceTransport(FullFakeTransport):
        def diarize(self, **kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {"startTimeSeconds": 0.0, "endTimeSeconds": 0.8, "speakerId": "S1"},
                        {"startTimeSeconds": 0.4, "endTimeSeconds": 1.0, "speakerId": "S2"},
                        {"startTimeSeconds": 1.2, "endTimeSeconds": 1.3, "speakerId": "S3"},
                        {"startTimeSeconds": 1.4, "endTimeSeconds": 2.0, "speakerId": "S4"},
                    ]
                },
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization", "overlapped_speech"),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=MixedEvidenceTransport(),
    ).payload
    starts = [item["start"] for item in payload["abstentions"]]
    assert starts == sorted(starts)
    assert [item["abstention_id"] for item in payload["abstentions"]] == [
        f"ab_{index}" for index in range(len(starts))
    ]


def test_empty_transcript_keeps_diarizer_abstention_evidence(tmp_path) -> None:
    class RawOnlyTransport(FullFakeTransport):
        def diarize(self, **kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {
                            "startTimeSeconds": 0.2,
                            "endTimeSeconds": 0.3,
                            "speakerId": "S1",
                        }
                    ]
                },
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=RawOnlyTransport(),
    ).payload
    assert payload["complete"] is True
    assert payload["segments"] == []
    assert payload["turns"] == []
    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "raw_fragment",
            "start": 0.2,
            "end": 0.3,
        }
    ]
    assert payload["provenance"]["outcomes"]["diarization"] == "produced"
