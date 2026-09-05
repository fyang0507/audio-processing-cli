from __future__ import annotations

from transcribe_orchestrator_test_support import (
    FakeTransport,
    FullFakeTransport,
    InputMetadata,
    NoncontiguousTransport,
    Path,
    full_registry,
    json,
    orchestrator,
    os,
    pytest,
    refusals,
    registry,
    request,
    resolve_request,
)
from transcribe_orchestrator_test_support import (
    provisioned_runtime_root as provisioned_runtime_root,
)

from audio_cli.transcribe.execution import publication as orchestrator_output


def test_run_refuses_existing_outputs_before_decode_and_force_is_explicit(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("keep", encoding="utf-8")
    transport = FakeTransport()
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
        )
    assert raised.value.payload["code"] == "output_exists"
    assert raised.value.payload["fix"].endswith(f"-o {output} --force")
    assert transport.units == []
    assert output.read_text(encoding="utf-8") == "keep"

    payload = orchestrator.run(
        resolved,
        metadata,
        registry=registry(tmp_path),
        transport=FakeTransport(),
        output=output,
        force=True,
    ).payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_run_treats_broken_symlinks_as_existing_before_decode(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "broken.json"
    output.symlink_to("missing.json")
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
        )

    assert raised.value.payload["code"] == "output_exists"
    assert transport.units == []
    assert output.is_symlink()
    assert output.readlink() == Path("missing.json")


def test_run_treats_broken_derived_partial_as_existing_before_decode(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    partial = tmp_path / "result.partial.json"
    partial.symlink_to("missing-partial.json")
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
        )

    assert raised.value.payload["code"] == "output_exists"
    assert raised.value.payload["existing"] == str(partial)
    assert transport.units == []
    assert partial.is_symlink()


def test_force_refuses_an_output_symlink_loop_as_an_invalid_path(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "loop.json"
    output.symlink_to(output.name)
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
            force=True,
        )

    assert raised.value.payload["code"] == "output_path_invalid"
    assert raised.value.payload["target"] == str(output)
    assert transport.units == []
    assert output.is_symlink()


@pytest.mark.parametrize("partial_target", [False, True])
def test_force_refuses_a_directory_output_before_decode(tmp_path, partial_target: bool) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    target = tmp_path / "result.partial.json" if partial_target else output
    target.mkdir()
    transport = FakeTransport()

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
            force=True,
        )

    assert raised.value.payload["code"] == "output_path_invalid"
    assert raised.value.payload["target"] == str(target)
    assert transport.units == []
    assert target.is_dir()


@pytest.mark.parametrize("partial", [False, True])
def test_late_output_collision_is_never_clobbered(tmp_path, partial: bool) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    collision = tmp_path / "result.partial.json" if partial else output

    class LateCollisionTransport(FakeTransport):
        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            collision.write_text("racer-owned\n", encoding="utf-8")
            return outcome

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=LateCollisionTransport(partial=partial),
            output=output,
        )

    assert raised.value.payload["code"] == "output_exists"
    assert raised.value.payload["existing"] == str(collision)
    assert collision.read_text(encoding="utf-8") == "racer-owned\n"
    if partial:
        assert not output.exists()


def test_late_directory_collision_is_a_typed_refusal(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"

    class LateDirectoryTransport(FakeTransport):
        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            output.mkdir()
            return outcome

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=LateDirectoryTransport(),
            output=output,
        )

    assert raised.value.payload["code"] == "output_path_invalid"
    assert raised.value.payload["target"] == str(output)
    assert output.is_dir()


@pytest.mark.parametrize("partial", [False, True])
def test_late_unwritable_destination_is_a_typed_refusal(
    tmp_path: Path,
    monkeypatch,
    partial: bool,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    transport = FakeTransport(partial=partial)

    def refuse_write(*_args, **_kwargs):
        raise PermissionError("destination is not writable")

    monkeypatch.setattr(orchestrator_output, "atomic_write_json", refuse_write)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=transport,
            output=output,
        )

    target = tmp_path / "result.partial.json" if partial else output
    assert raised.value.exit_code == 2
    assert raised.value.payload["code"] == "output_path_invalid"
    assert raised.value.payload["provided"] == str(output)
    assert raised.value.payload["target"] == str(target)
    assert "not writable" in raised.value.payload["reason"]


def test_run_never_allows_output_to_resolve_to_canonical_input(tmp_path) -> None:
    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=FakeTransport(),
            output=resolved.input_path,
            force=True,
        )
    assert raised.value.payload["code"] == "output_is_canonical_input"
    assert resolved.input_path.read_bytes() == b"source"

    derived_source = tmp_path / "meeting.partial.json"
    derived_source.write_bytes(b"canonical")
    derived_request = resolve_request(
        stack_id="qwen-0.6b",
        input_path=derived_source,
        wants=(),
    )
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            derived_request,
            InputMetadata(str(derived_source), 361.0, "json", 48_000, 2),
            registry=registry(tmp_path),
            transport=FakeTransport(partial=True),
            output=tmp_path / "meeting.json",
            force=True,
        )
    assert raised.value.payload["code"] == "output_is_canonical_input"
    assert raised.value.payload["resolved_target"] == str(derived_source)
    assert derived_source.read_bytes() == b"canonical"


def test_qwen_publication_preserves_source_renamed_to_forced_output_after_decode(
    tmp_path: Path,
) -> None:
    resolved, metadata, _ = request(tmp_path)
    output = tmp_path / "result.json"
    output.write_bytes(b"replaceable")

    class SourceMovingTransport(FakeTransport):
        def qwen(self, **kwargs):
            outcome = super().qwen(**kwargs)
            os.replace(resolved.input_path, output)
            return outcome

    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=SourceMovingTransport(),
            output=output,
            force=True,
        )

    assert raised.value.payload["code"] == "output_is_canonical_input"
    assert not resolved.input_path.exists()
    assert output.read_bytes() == b"source"


def test_diarizer_exclusions_survive_without_requesting_diarization(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    resolved = resolve_request(
        stack_id="qwen-0.6b",
        input_path=source,
        wants=("overlapped_speech",),
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 2)
    payload = orchestrator.run(
        resolved,
        metadata,
        registry=full_registry(tmp_path),
        transport=NoncontiguousTransport(partial=False),
    ).payload
    assert "turns" not in payload
    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "raw_fragment",
            "start": 1.35,
            "end": 1.45,
        }
    ]


def test_overlap_abstention_survives_without_optional_overlap_array(tmp_path) -> None:
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
        transport=FullFakeTransport(),
    ).payload
    assert "overlapped_speech" not in payload
    assert payload["abstentions"] == [
        {
            "abstention_id": "ab_0",
            "reason": "overlap",
            "start": 0.8,
            "end": 1.0,
        }
    ]


def test_backend_failure_fix_does_not_promise_an_unrelated_command(tmp_path) -> None:
    class BrokenTransport(FakeTransport):
        def qwen(self, **kwargs):
            from audio_cli.transcribe.transport import StageFailure

            raise StageFailure("asr", "qwen3-asr-0.6b-8bit", "out of memory")

    resolved, metadata, _ = request(tmp_path)
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            resolved,
            metadata,
            registry=registry(tmp_path),
            transport=BrokenTransport(),
        )
    assert raised.value.exit_code == 1
    assert raised.value.payload["role"] == "asr"
    assert raised.value.payload["backend"] == "qwen3-asr-0.6b-8bit"
    assert not raised.value.payload["fix"].startswith("audio ")
