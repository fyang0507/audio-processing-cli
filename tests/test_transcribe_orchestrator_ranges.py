from __future__ import annotations

from transcribe_orchestrator_test_support import (
    FakeTransport,
    FullFakeTransport,
    InputMetadata,
    Path,
    StageOutcome,
    full_registry,
    json,
    orchestrator,
    pytest,
    refusals,
    registry,
    request,
    resolve_request,
    wave,
)
from transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)

from audio_cli.transcribe.orchestrator import qwen as qwen_execution


def test_nonintersecting_range_refuses_after_decode_but_before_model(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)

    class DecodeOnly(FakeTransport):
        decode_calls = 0

        def decode(self, source, target):
            self.decode_calls += 1
            return super().decode(source, target)

        def qwen(self, **kwargs):
            raise AssertionError("range validation reached model transport")

    transport = DecodeOnly()
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            run_range=orchestrator.parse_range("400:"),
        )
    assert transport.decode_calls == 1
    assert raised.value.exit_code == 2
    assert raised.value.payload["code"] == "range_invalid"
    assert raised.value.payload["provided"] == "400:"


def test_open_range_end_resolves_against_canonical_not_shorter_probe(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(stack_id="qwen-0.6b", input_path=source, wants=())
    metadata = InputMetadata(str(source), 1.0, "wav", 48_000, 2)
    transport = FullFakeTransport()
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=registry(tmp_path),
        transport=transport,
        run_range=orchestrator.parse_range("0.5:"),
    ).payload
    assert transport.units == [{"unit_id": "unit_0", "start": 0.0, "end": 2.0}]
    assert payload["provenance"]["plan"]["execution"]["range"]["requested"] == [
        0.5,
        2.0,
    ]


def test_fixed_units_match_the_catalog_rule_even_for_a_subsecond_tail(tmp_path) -> None:
    resolved, _, _ = request(tmp_path)
    assert qwen_execution._fixed_units(360.0000625, resolved) == [
        {"unit_id": "unit_0", "start": 0.0, "end": 180.0},
        {"unit_id": "unit_1", "start": 180.0, "end": 360.0},
        {"unit_id": "unit_2", "start": 360.0, "end": 360.000063},
    ]


class NoncontiguousTransport(FullFakeTransport):
    def __init__(self, *, partial: bool) -> None:
        super().__init__()
        self.partial = partial

    def diarize(self, **kwargs):
        return StageOutcome(
            "diarizer",
            "fluidaudio",
            {
                "segments": [
                    {"startTimeSeconds": 0.0, "endTimeSeconds": 0.6, "speakerId": "S1"},
                    {"startTimeSeconds": 0.6, "endTimeSeconds": 1.3, "speakerId": "S2"},
                    {"startTimeSeconds": 1.35, "endTimeSeconds": 1.45, "speakerId": "S4"},
                    {"startTimeSeconds": 1.5, "endTimeSeconds": 2.0, "speakerId": "S3"},
                ]
            },
            1.0,
        )

    def qwen(self, *, units, **kwargs):
        self.units = list(units)
        return StageOutcome(
            "asr",
            "qwen3-asr-0.6b-8bit",
            {
                "units": [
                    {
                        "unit_id": item["unit_id"],
                        "processed": not self.partial or item["unit_id"] != "turn_1",
                        "text": f"Text {item['unit_id']}.",
                    }
                    for item in units
                ]
            },
            1.0,
            returncode=4 if self.partial else 0,
        )


def test_noncontiguous_partial_resume_reuses_whole_turn_bounds_and_labels(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    first_transport = NoncontiguousTransport(partial=True)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=first_transport,
        )
    partial = json.loads(Path(raised.value.payload["output"]).read_text(encoding="utf-8"))
    assert partial["coverage"]["covered_intervals"] == [[0.0, 0.6]]
    assert partial["coverage"]["missing_intervals"] == [[0.6, 2.0]]
    assert partial["turns"] == [
        {"turn_id": "turn_0", "speaker": "S1", "start": 0.0, "end": 0.6},
    ]
    assert partial["abstentions"] == []

    resume_transport = NoncontiguousTransport(partial=False)
    resumed = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=resume_transport,
        run_range=orchestrator.parse_range("0.6:"),
    ).payload
    assert resumed["turns"] == [
        {"turn_id": "turn_1", "speaker": "S2", "start": 0.6, "end": 1.3},
        {"turn_id": "turn_2", "speaker": "S3", "start": 1.5, "end": 2.0},
    ]
    assert resumed["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "raw_fragment",
            "start": 1.35,
            "end": 1.45,
        }
    ]
    assert resumed["provenance"]["plan"]["execution"]["range"] == {
        "requested": [0.6, 2.0],
        "selected_unit_scope": [0.6, 2.0],
    }


def test_open_resume_owns_auxiliary_evidence_after_the_last_selected_turn(tmp_path) -> None:
    class TailEvidenceTransport(NoncontiguousTransport):
        def decode(self, source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * (3 * 16_000))
            return StageOutcome("decode", "ffmpeg", {}, 1.0)

        def diarize(self, **kwargs):
            base = super().diarize(**kwargs).payload["segments"]
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        *base,
                        {"startTimeSeconds": 2.5, "endTimeSeconds": 2.6, "speakerId": "S5"},
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
    payload = orchestrator.run(
        resolved,
        InputMetadata(str(source), 3.0, "wav", 48_000, 2),
        registry=full_registry(tmp_path),
        transport=TailEvidenceTransport(partial=False),
        run_range=orchestrator.parse_range("0.6:"),
    ).payload
    assert payload["abstentions"][-1] == {
        "abstention_id": "ab_1",
        "reason": "raw_fragment",
        "start": 2.5,
        "end": 2.6,
    }


def test_incomplete_hand_range_owns_evidence_before_its_first_selected_turn(tmp_path) -> None:
    class GapExhaustedTransport(NoncontiguousTransport):
        def diarize(self, **kwargs):
            return StageOutcome(
                "diarizer",
                "fluidaudio",
                {
                    "segments": [
                        {"startTimeSeconds": 0.0, "endTimeSeconds": 1.3, "speakerId": "S1"},
                        {"startTimeSeconds": 1.42, "endTimeSeconds": 1.45, "speakerId": "S4"},
                        {"startTimeSeconds": 1.5, "endTimeSeconds": 2.0, "speakerId": "S2"},
                    ]
                },
                1.0,
            )

        def qwen(self, *, units, **kwargs):
            self.units = list(units)
            return StageOutcome(
                "asr",
                "qwen3-asr-0.6b-8bit",
                {
                    "units": [
                        {
                            "unit_id": item["unit_id"],
                            "processed": False,
                            "text": "",
                        }
                        for item in units
                    ]
                },
                1.0,
                returncode=4,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization",),
    )
    transport = GapExhaustedTransport(partial=True)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            InputMetadata(str(source), 2.0, "wav", 48_000, 2),
            registry=full_registry(tmp_path),
            transport=transport,
            run_range=orchestrator.parse_range("1.4:"),
        )
    partial = json.loads(Path(raised.value.payload["output"]).read_text(encoding="utf-8"))
    assert transport.units == [
        {
            "unit_id": "turn_1",
            "speaker": "S2",
            "start": 1.5,
            "end": 2.0,
        }
    ]
    assert partial["coverage"]["scope_intervals"] == [[1.5, 2.0]]
    assert partial["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "raw_fragment",
            "start": 1.42,
            "end": 1.45,
        }
    ]
