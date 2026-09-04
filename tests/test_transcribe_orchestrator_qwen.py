from __future__ import annotations

from transcribe_orchestrator_test_support import *  # noqa: F403



def test_qwen_floor_run_uses_fixed_units_and_never_publishes_container_bounds(tmp_path) -> None:
    resolved, metadata, run_range = request(tmp_path)
    fake = FakeTransport()
    product = orchestrator.run(
        resolved, metadata, registry=registry(tmp_path), transport=fake,
        run_range=run_range,
    )
    assert [(item["start"], item["end"]) for item in fake.units] == [
        (0.0, 180.0), (180.0, 360.0), (360.0, 361.0),
    ]
    assert product.payload["segments"] == [
        {"segment_id": "seg_0", "text": "Hello."},
        {"segment_id": "seg_1", "text": "Hello."},
        {"segment_id": "seg_2", "text": "Hello."},
    ]
    assert product.payload["provenance"]["observed"]["peak_rss_bytes"] == 100
    assert product.payload["provenance"]["observed"]["total_wall_seconds"] == 5.0


def test_run_captures_source_identity_before_decode_and_export_protects_it(
    tmp_path,
) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    link = tmp_path / "input.wav"
    first.write_bytes(b"first-source")
    second.write_bytes(b"second-source")
    link.symlink_to(first)
    request_ = resolve_request(stack_id="qwen-0.6b", input_path=link, wants=())
    metadata = InputMetadata(str(link), 2.0, "wav", 16_000, 1)

    class RetargetingTransport(FullFakeTransport):
        def decode(self, source, target):
            assert Path(source) == first.resolve()
            assert Path(source).read_bytes() == b"first-source"
            link.unlink()
            link.symlink_to(second)
            return super().decode(source, target)

    product = orchestrator.run(
        request_,
        metadata,
        registry=registry(tmp_path),
        transport=RetargetingTransport(),
    )

    assert product.payload["source"]["path"] == str(first.resolve())
    transcript = tmp_path / "result.json"
    transcript.write_text(json.dumps(product.payload), encoding="utf-8")
    with pytest.raises(UnsafeOutputError):
        export_documents([transcript], "txt", output=first, force=True)
    assert first.read_bytes() == b"first-source"


def test_partial_qwen_run_writes_conforming_result_and_exact_resume_ledger(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "requested.md"
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path),
            transport=FakeTransport(partial=True), output=output,
        )
    assert raised.value.exit_code == 4
    assert raised.value.payload["code"] == "run_incomplete"
    partial = tmp_path / "requested.partial.json"
    assert raised.value.payload["output"] == str(partial)
    assert not output.exists()
    payload = json.loads(partial.read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert payload["coverage"] == raised.value.payload["coverage"]
    assert payload["coverage"]["covered_intervals"] == [[0.0, 180.0]]
    assert payload["coverage"]["missing_intervals"] == [[180.0, 361.0]]
    assert payload["coverage"]["scope_intervals"] == [[0.0, 361.0]]
    assert "--range 180.0:" in raised.value.payload["fix"]


@pytest.mark.parametrize("malformation", ["absent_units", "missing_row", "non_bool"])
def test_qwen_malformed_partial_ledger_is_backend_failure_without_publication(
    tmp_path: Path,
    malformation: str,
) -> None:
    class MalformedLedgerTransport(FakeTransport):
        def qwen(self, *, units, **kwargs):
            if malformation == "absent_units":
                payload = {}
            else:
                rows = [{
                    "unit_id": item["unit_id"],
                    "processed": index == 0,
                    "text": "language English<asr_text>Hello.",
                } for index, item in enumerate(units)]
                if malformation == "missing_row":
                    rows.pop()
                else:
                    rows[0]["processed"] = 1
                payload = {"units": rows}
            return StageOutcome(
                "asr",
                "qwen3-asr-0.6b-8bit",
                payload,
                3.0,
                returncode=4,
            )

    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "malformed.json"
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=MalformedLedgerTransport(),
            output=output,
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["backend"] == "qwen3-asr-0.6b-8bit"
    assert not output.exists()
    assert not (tmp_path / "malformed.partial.json").exists()


def test_repeated_incomplete_resumes_never_overwrite_an_earlier_partial(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as first:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
            output=tmp_path / "meeting.timed.json",
        )
    first_partial = Path(first.value.payload["output"])
    first_bytes = first_partial.read_bytes()
    assert first_partial.name == "meeting.timed.partial.json"

    with pytest.raises(refusals.Refusal) as second:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
            output=tmp_path / "meeting.timed.rest.json",
            run_range=orchestrator.parse_range("180:"),
        )
    second_partial = Path(second.value.payload["output"])
    assert second_partial.name == "meeting.timed.rest.partial.json"
    assert first_partial.read_bytes() == first_bytes
    assert first_partial.is_file() and second_partial.is_file()


def test_repeated_implicit_partial_outputs_choose_unused_siblings(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as first:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
        )
    first_partial = Path(first.value.payload["output"])
    first_bytes = first_partial.read_bytes()

    with pytest.raises(refusals.Refusal) as second:
        orchestrator.run(
            resolved, metadata, registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
        )
    second_partial = Path(second.value.payload["output"])
    assert first_partial.name == "source.partial.json"
    assert second_partial.name == "source.partial.2.json"
    assert first_partial.read_bytes() == first_bytes
    assert second_partial.is_file()


def test_implicit_partial_skips_a_broken_symlink_destination(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    first_candidate = tmp_path / "source.partial.json"
    first_candidate.symlink_to("missing-partial.json")

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
        )

    assert raised.value.payload["code"] == "run_incomplete"
    assert Path(raised.value.payload["output"]).name == "source.partial.2.json"
    assert first_candidate.is_symlink()
    assert first_candidate.readlink() == Path("missing-partial.json")


def test_bounded_partial_resume_preserves_the_requested_end(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
            output=tmp_path / "bounded.json",
            run_range=orchestrator.parse_range("100:300"),
        )
    assert "--range 180.0:300.0" in raised.value.payload["fix"]


def test_resume_builder_defensively_quotes_an_option_like_internal_value(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = replace(
        resolve_request(
            stack_id="qwen-0.6b",
            input_path=source,
            wants=(),
            language="English",
        ),
        language="--stack",
    )
    fix = orchestrator._resume_command(
        resolved,
        {
            "units_completed": 1,
            "covered_intervals": [[0.0, 180.0]],
            "covered_through_seconds": 180.0,
        },
        tmp_path / "source.partial.json",
        None,
    )

    arguments = shlex.split(fix)
    parsed = cli._parser().parse_args(arguments[1:])
    assert parsed.language == "--stack"
    assert parsed.stack == "qwen-0.6b"
    assert parsed.run_range == "180.0:"


def test_zero_completed_units_write_a_partial_without_a_replaying_fix(tmp_path) -> None:
    class ExhaustedTransport(FakeTransport):
        def qwen(self, *, units, **kwargs):
            self.units = units
            return StageOutcome("asr", "qwen3-asr-0.6b-8bit", {"units": [{
                "unit_id": item["unit_id"], "processed": False, "text": "",
            } for item in units]}, 3.0, returncode=4)

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=ExhaustedTransport(),
        )
    assert raised.value.exit_code == 4
    assert raised.value.payload["code"] == "run_incomplete"
    coverage = raised.value.payload["coverage"]
    assert coverage["covered_intervals"] == []
    assert coverage["missing_intervals"] == [[0.0, 361.0]]
    assert coverage["covered_fraction"] == 0.0
    assert coverage["units_completed"] == 0
    assert raised.value.payload["fix"] == (
        "no processing unit completed; --range would repeat the same deterministic work, "
        "so inspect the first unit or backend budget before retrying"
    )
    partial = json.loads(
        Path(raised.value.payload["output"]).read_text(encoding="utf-8")
    )
    assert partial["complete"] is False
    assert partial["segments"] == []


def test_zero_prefix_partial_does_not_claim_diarizer_abstained(tmp_path) -> None:
    class ExhaustedDiarizedTransport(FullFakeTransport):
        def qwen(self, *, units, **kwargs):
            return StageOutcome("asr", "qwen3-asr-0.6b-8bit", {"units": [{
                "unit_id": item["unit_id"], "processed": False, "text": "",
            } for item in units]}, 1.0, returncode=4)

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b", input_path=source, wants=("diarization",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=full_registry(tmp_path),
            transport=ExhaustedDiarizedTransport(),
        )
    partial = json.loads(
        Path(raised.value.payload["output"]).read_text(encoding="utf-8")
    )
    assert partial["provenance"]["outcomes"]["diarization"] == "produced"
