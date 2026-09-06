from __future__ import annotations

from transcribe_orchestrator_test_support import (
    FakeTransport,
    FakeVad,
    FullFakeTransport,
    InputMetadata,
    Path,
    StageOutcome,
    build_plan,
    full_registry,
    json,
    orchestrator,
    pytest,
    refusals,
    registry,
    request,
    resolve_request,
    serialize_plan,
)
from transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)


def test_alignment_abstention_is_observable_and_does_not_invent_words(tmp_path) -> None:
    class AbstainingAligner(FullFakeTransport):
        def align(self, *, segments, **kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {
                            "unit_id": item["unit_id"],
                            "words": None,
                        }
                        for item in segments
                    ]
                },
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("word_timestamps",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=AbstainingAligner(),
    ).payload
    assert all("words" not in item for item in payload["segments"])
    assert payload["provenance"]["outcomes"]["word_timestamps"] == "abstained"
    assert payload["provenance"]["observed"]["segments_without_words"] == 1
    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "start": 0.0,
            "end": 2.0,
        }
    ]


@pytest.mark.parametrize(
    "aligner_payload",
    [
        {},
        {"segments": []},
        {"segments": [{"unit_id": "turn_0"}]},
    ],
)
def test_qwen_rejects_incomplete_aligner_ledgers_without_publication(
    tmp_path: Path,
    aligner_payload: dict,
) -> None:
    class MalformedAligner(FullFakeTransport):
        def align(self, **kwargs):
            return StageOutcome("aligner", "qwen3-forcedaligner", aligner_payload, 1.0)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("word_timestamps",),
    )
    output = tmp_path / "result.json"
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            InputMetadata(str(source), 2.0, "wav", 48_000, 2),
            registry=full_registry(tmp_path),
            transport=MalformedAligner(),
            output=output,
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["backend"] == "qwen3-forcedaligner"
    assert not output.exists()


def test_alignment_text_mismatch_records_the_attempted_unit_bounds(tmp_path) -> None:
    class MismatchedAligner(FullFakeTransport):
        def align(self, *, segments, **kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {
                            "unit_id": item["unit_id"],
                            "words": [{"text": "Goodbye", "start": 0.0, "end": 1.0}],
                        }
                        for item in segments
                    ]
                },
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("word_timestamps",),
    )
    payload = orchestrator.run(
        resolved,
        InputMetadata(str(source), 2.0, "wav", 48_000, 2),
        registry=full_registry(tmp_path),
        transport=MismatchedAligner(),
    ).payload

    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "start": 0.0,
            "end": 2.0,
        }
    ]
    assert all("words" not in item for item in payload["segments"])


def test_partial_qwen_alignment_abstention_covers_only_the_completed_prefix(
    tmp_path,
) -> None:
    class PartialAbstainingAligner(FakeTransport):
        def align(self, *, segments, **kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {
                            "unit_id": item["unit_id"],
                            "words": None,
                        }
                        for item in segments
                    ]
                },
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("word_timestamps",),
    )
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            InputMetadata(str(source), 361.0, "wav", 48_000, 2),
            registry=full_registry(tmp_path),
            transport=PartialAbstainingAligner(partial=True),
        )

    payload = json.loads(Path(raised.value.payload["output"]).read_text(encoding="utf-8"))
    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "start": 0.0,
            "end": 180.0,
        }
    ]


def test_punctuation_only_alignment_with_empty_words_is_produced_not_abstained(
    tmp_path,
) -> None:
    class PunctuationTransport(FakeTransport):
        def qwen(self, *, units, **kwargs):
            return StageOutcome(
                "asr",
                "qwen3-asr-0.6b-8bit",
                {
                    "units": [
                        {
                            "unit_id": item["unit_id"],
                            "processed": True,
                            "text": "……？！",
                        }
                        for item in units
                    ]
                },
                1.0,
            )

        def align(self, *, segments, **kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {
                            "unit_id": item["unit_id"],
                            "words": [],
                        }
                        for item in segments
                    ]
                },
                1.0,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("word_timestamps",),
    )
    payload = orchestrator.run(
        resolved,
        InputMetadata(str(source), 361.0, "wav", 48_000, 2),
        registry=full_registry(tmp_path),
        transport=PunctuationTransport(),
    ).payload
    assert all(item["words"] == [] for item in payload["segments"])
    assert payload["provenance"]["outcomes"]["word_timestamps"] == "produced"
    assert payload["provenance"]["observed"]["segments_without_words"] == 0


@pytest.mark.parametrize(
    ("returncode", "transport_failure", "detail"),
    [(1, True, "out of memory"), (4, False, "unsupported exit 4")],
)
def test_whole_aligner_failure_is_a_backend_error_and_writes_no_result(
    tmp_path, returncode: int, transport_failure: bool, detail: str
) -> None:
    class FailingAligner(FullFakeTransport):
        def align(self, **kwargs):
            from audio_cli.transcribe.transport import StageFailure

            if not transport_failure:
                successful = super().align(**kwargs)
                return StageOutcome(
                    successful.role,
                    successful.backend,
                    successful.payload,
                    successful.wall_seconds,
                    returncode=returncode,
                )
            outcome = StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {"error": {"message": "out of memory"}},
                2.5,
                returncode=returncode,
                peak_rss_bytes=250,
            )
            raise StageFailure(
                "aligner",
                "qwen3-forcedaligner",
                "out of memory",
                outcome=outcome,
            )

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("word_timestamps",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=FailingAligner(),
            output=output,
        )
    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "aligner"
    assert caught.value.payload["backend"] == "qwen3-forcedaligner"
    assert detail in caught.value.payload["detail"]
    assert not output.exists()


def test_canonical_wav_header_owns_duration_and_fixed_unit_bounds(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    metadata = InputMetadata(metadata.source["path"], 999.0, "wav", 48_000, 2)
    transport = FullFakeTransport()
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=registry(tmp_path),
        transport=transport,
    ).payload
    assert payload["source"] == {
        "path": str(resolved.input_path),
        "duration_seconds": 2.0,
        "timebase": "seconds",
        "duration_basis": "canonical_decoded_pcm",
    }
    assert transport.units == [{"unit_id": "unit_0", "start": 0.0, "end": 2.0}]


def test_range_filters_auxiliary_observations_to_selected_processing_scope(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("diarization", "overlapped_speech", "vad"),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=FullFakeTransport(),
        vad_detector=FakeVad(),
        run_range=orchestrator.parse_range("1.2:1.5"),
    ).payload
    assert payload["turns"] == [
        {
            "turn_id": "turn_1",
            "speaker": "S2",
            "start": 1.0,
            "end": 1.8,
        }
    ]
    assert payload["overlapped_speech"] == []
    assert payload["vad_regions"] == []
    assert payload["provenance"]["plan"]["execution"]["range"] == {
        "requested": [1.2, 1.5],
        "selected_unit_scope": [1.0, 1.8],
    }


@pytest.mark.parametrize(
    "wants",
    [
        (),
        ("languages",),
        ("verbatim",),
        ("diarization",),
        ("overlapped_speech",),
        ("vad",),
        ("word_timestamps",),
        ("languages", "verbatim", "diarization", "overlapped_speech", "vad", "word_timestamps"),
    ],
)
def test_real_qwen_result_shape_matches_its_plan_sample_over_derivation_cells(
    tmp_path, wants
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=wants,
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    document = full_registry(tmp_path)
    ready = set(document["packages"])
    sample = serialize_plan(
        build_plan(
            resolved,
            metadata,
            provisioned_packages=ready,
        )
    )["sample_output"]
    actual = orchestrator.run(
        resolved,
        metadata,
        registry=document,
        transport=FullFakeTransport(),
        vad_detector=FakeVad(),
    ).payload

    assert set(actual) == set(sample) - {"sample", "note"}
    assert set(actual["provenance"]) == set(sample["provenance"])
    assert set(actual["segments"][0]) == set(sample["segments"][0])
    assert set(actual["provenance"]["outcomes"]) == set(wants)
    for capability, field in {
        "diarization": "turns",
        "overlapped_speech": "overlapped_speech",
        "vad": "vad_regions",
    }.items():
        assert (field in actual) is (capability in wants)
