from __future__ import annotations

import json
import os
import shlex
import stat
from pathlib import Path

from audio_cli import cli
from audio_cli.transcribe.result import ABSENT, NormalizedResult, serialize_result


def _write_result(
    path: Path,
    *,
    source: Path,
    segments: list[dict],
    outcomes: dict[str, str],
    abstentions: list[dict] | None = None,
    language: str = "English",
    stack: str = "qwen-0.6b",
    run_range: list[float] | None = None,
) -> Path:
    plan: dict = {"roles": {"asr": {"config": {"language": language}}}}
    if run_range is not None:
        plan["execution"] = {
            "range": {
                "requested": list(run_range),
                "selected_unit_scope": list(run_range),
            }
        }
    payload = serialize_result(NormalizedResult(
        source={
            "path": str(source), "duration_seconds": 2.0, "timebase": "seconds",
        },
        segments=segments,
        abstentions=abstentions or [],
        provenance={
            "stack": stack,
            "outcomes": outcomes,
            "observed": {},
            "plan": plan,
        },
        requested_capabilities=frozenset(outcomes),
        turns=[] if "diarization" in outcomes else ABSENT,
    ))
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_export_cli_prints_human_formats_without_an_output(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    assert cli.main([
        "export", "--input", str(transcript), "--format", "txt",
    ]) == 0
    assert capsys.readouterr().out == "Hello.\n"


def test_export_cli_refuses_inert_force_without_output(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    assert cli.main([
        "export", "--input", str(transcript), "--format", "txt", "--force",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "code": "output_required_for_force",
        "field": "--force",
        "provided": True,
        "requires": "--output",
        "fix": "remove --force when writing to stdout, or add --output PATH",
    }


def test_export_cli_refuses_directory_output_even_with_force(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    output = tmp_path / "directory-output"
    output.mkdir()
    assert cli.main([
        "export", "--input", str(transcript), "--format", "txt",
        "-o", str(output), "--force",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "output_exists"
    assert "directory cannot be replaced" in error["fix"]
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_export_cli_refuses_replacing_a_fifo_even_with_force(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    output = tmp_path / "out.pipe"
    os.mkfifo(output)

    assert cli.main([
        "export", "--input", str(transcript), "--format", "txt",
        "--output", str(output), "--force",
    ]) == 2
    error = json.loads(capsys.readouterr().err)

    assert error["code"] == "output_exists"
    assert "special file" in error["fix"]
    assert stat.S_ISFIFO(output.lstat().st_mode)


def test_export_cli_refuses_overwriting_an_input_transcript_with_truthful_fix(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )

    assert cli.main([
        "export", "--input", str(transcript), "--format", "txt",
        "-o", str(transcript), "--force",
    ]) == 2
    error = json.loads(capsys.readouterr().err)

    assert error["code"] == "output_is_canonical_input"
    assert error["resolved_target"] == str(transcript)
    assert "input transcript" in error["fix"]
    assert transcript.read_text(encoding="utf-8").startswith("{")


def test_export_cli_rejects_a_self_referential_output_symlink(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    output = tmp_path / "loop"
    output.symlink_to(output.name)
    assert cli.main([
        "export", "--input", str(transcript), "--format", "txt",
        "-o", str(output), "--force",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "code": "output_path_invalid",
        "field": "--output",
        "provided": str(output),
        "target": str(output),
        "reason": f"Symlink loop from '{output}'",
        "fix": (
            "choose an --output whose destination and parent directory can be "
            "resolved and written safely"
        ),
    }
    assert output.is_symlink()


def test_export_cli_types_invalid_and_incompatible_inputs(
    tmp_path: Path, capsys,
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert cli.main([
        "export", "--input", str(invalid), "--format", "txt",
    ]) == 2
    invalid_error = json.loads(capsys.readouterr().err)
    assert invalid_error["code"] == "export_input_invalid"
    assert set(invalid_error) == {"code", "field", "provided", "reason", "fix"}
    assert invalid_error["provided"] == str(invalid)

    first_source = tmp_path / "first.wav"
    second_source = tmp_path / "second.wav"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    first = _write_result(
        tmp_path / "first.json",
        source=first_source,
        segments=[{"segment_id": "seg_0", "text": "First."}],
        outcomes={"verbatim": "produced"},
    )
    second = _write_result(
        tmp_path / "second.json",
        source=second_source,
        segments=[{"segment_id": "seg_0", "text": "Second."}],
        outcomes={"verbatim": "produced"},
    )
    assert cli.main([
        "export", "--input", str(first), "--input", str(second),
        "--format", "txt",
    ]) == 2
    incompatible_error = json.loads(capsys.readouterr().err)
    assert incompatible_error["code"] == "export_inputs_incompatible"
    assert set(incompatible_error) == {"code", "field", "provided", "reason", "fix"}
    assert incompatible_error["provided"] == [str(first), str(second)]


def test_export_cli_refuses_multi_input_vibevoice_native_diarization(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    left = _write_result(
        tmp_path / "left.json",
        source=source,
        segments=[{
            "segment_id": "seg_0",
            "text": "Left.",
            "start": 0.1,
            "end": 0.8,
            "speaker": "Speaker 0",
        }],
        outcomes={"diarization": "produced", "segment_timestamps": "produced"},
        stack="vibevoice",
        run_range=[0.0, 1.0],
    )
    right = _write_result(
        tmp_path / "right.json",
        source=source,
        segments=[{
            "segment_id": "seg_0",
            "text": "Right.",
            "start": 1.1,
            "end": 1.8,
            "speaker": "Speaker 0",
        }],
        outcomes={"diarization": "produced", "segment_timestamps": "produced"},
        stack="vibevoice",
        run_range=[1.0, 2.0],
    )

    assert cli.main([
        "export", "--input", str(left), "--input", str(right), "--format", "txt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "code": "export_inputs_incompatible",
        "field": "--input",
        "provided": [str(left), str(right)],
        "reason": (
            "VibeVoice native speaker labels are local to each independent generation "
            "and cannot be reconciled across multiple input documents"
        ),
        "fix": (
            "export these VibeVoice documents separately, or rerun the desired ranges "
            "together as one generation"
        ),
    }


def test_export_cli_types_an_unresolvable_canonical_source_as_invalid_input(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    source.unlink()
    source.symlink_to(source.name)
    output = tmp_path / "out.txt"

    assert cli.main([
        "export", "--input", str(transcript), "--format", "txt",
        "--output", str(output),
    ]) == 2
    error = json.loads(capsys.readouterr().err)

    assert error["code"] == "export_input_invalid"
    assert error["field"] == "--input"
    assert error["provided"] == str(transcript)
    assert "source.path cannot be resolved safely" in error["reason"]
    assert not output.exists()


def test_export_cli_types_an_embedded_nul_source_path_as_invalid_input(
    tmp_path: Path, capsys,
) -> None:
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=tmp_path / "source.wav",
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    payload["source"]["path"] = "/tmp/audio\0source.wav"
    transcript.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "out.txt"

    assert cli.main([
        "export", "--input", str(transcript), "--format", "txt",
        "--output", str(output),
    ]) == 2
    error = json.loads(capsys.readouterr().err)

    assert error["code"] == "export_input_invalid"
    assert error["provided"] == str(transcript)
    assert "source.path cannot be resolved safely" in error["reason"]
    assert not output.exists()


def test_output_exists_fix_runs_for_option_like_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        Path("-meeting.json"),
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    output = Path("-out.txt")
    output.write_text("old", encoding="utf-8")

    assert cli.main([
        "export", "--input=-meeting.json", "--format", "txt",
        "--output=-out.txt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    arguments = shlex.split(error["fix"])
    assert cli.main(arguments[1:]) == 0
    assert output.read_text(encoding="utf-8") == "Hello.\n"


def test_export_cli_timing_refusal_has_a_runnable_transcribe_fix(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.transcript.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["code"] == "timing_required_for_format"
    assert error["found"] == []
    assert error["fix"].startswith(
        f"audio transcribe run --input {source} --stack qwen-0.6b"
    )
    assert "--want verbatim,word_timestamps" in error["fix"]
    assert "--language English" in error["fix"]
    assert error["fix"].endswith(f"-o {tmp_path / 'meeting.timed.json'}")


def test_export_cli_does_not_repeat_a_word_timing_request_that_abstained(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "vibe.json",
        source=source,
        segments=[{
            "segment_id": "seg_0", "text": "Speech without alignment.",
            "start": 0.0, "end": 1.0,
        }],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
        abstentions=[{
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "start": 0.0,
            "end": 1.0,
        }],
        stack="vibevoice",
    )

    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)

    assert error["code"] == "timing_required_for_format"
    assert not error["fix"].startswith("audio transcribe run")
    assert "already attempted word_timestamps and abstained" in error["fix"]


def test_export_cli_timing_fix_chooses_an_unused_result_path(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.transcript.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    (tmp_path / "meeting.timed.json").write_text("occupied", encoding="utf-8")
    (tmp_path / "meeting.timed.2.partial.json").write_text(
        "occupied", encoding="utf-8"
    )

    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    arguments = shlex.split(error["fix"])
    parsed = cli._parser().parse_args(arguments[1:])
    assert parsed.output == tmp_path / "meeting.timed.3.json"


def test_export_cli_timing_fix_keeps_an_option_like_language_parseable(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.transcript.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
        language="--stack",
    )

    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    arguments = shlex.split(error["fix"])
    parsed = cli._parser().parse_args(arguments[1:])
    assert parsed.language == "--stack"
    assert cli.main(arguments[1:]) == 2
    followup = json.loads(capsys.readouterr().err)
    assert followup["code"] == "option_value_unsupported"


def test_export_cli_timing_fix_keeps_an_option_like_stack_parseable(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.transcript.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
        stack="--input",
    )

    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    arguments = shlex.split(error["fix"])
    parsed = cli._parser().parse_args(arguments[1:])
    assert parsed.stack == "--input"
    assert cli.main(arguments[1:]) == 2
    followup = json.loads(capsys.readouterr().err)
    assert followup["code"] == "stack_required"


def test_export_cli_timing_refusal_handles_a_name_max_input(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / ("a" * 250 + ".json"),
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )

    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "timing_required_for_format"
    parsed = cli._parser().parse_args(shlex.split(error["fix"])[1:])
    assert parsed.output.parent == tmp_path
    assert len(os.fsencode(parsed.output.name)) <= os.pathconf(tmp_path, "PC_NAME_MAX")


def test_export_cli_timing_fix_preserves_recorded_range_and_vad_pin(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "ranged.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
        stack="firered",
    )
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    payload["provenance"]["plan"].update({
        "execution": {
            "range": {
                "requested": [0.5, 1.5],
                "selected_unit_scope": [0.4, 1.6],
            },
        },
        "roles": {
            **payload["provenance"]["plan"]["roles"],
            "vad": {"backend": "silero-vad", "selected_by": "pin:--vad"},
        },
    })
    transcript.write_text(json.dumps(payload), encoding="utf-8")

    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    parsed = cli._parser().parse_args(shlex.split(error["fix"])[1:])
    assert parsed.run_range == "0.5:1.5"
    assert parsed.vad == "silero-vad"


def test_multi_input_timing_fix_does_not_repeat_an_abstained_alignment(
    tmp_path: Path, capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    event = _write_result(
        tmp_path / "event.json",
        source=source,
        segments=[{
            "segment_id": "seg_0",
            "text": "[Music]",
            "start": 0.0,
            "end": 0.8,
        }],
        outcomes={
            "word_timestamps": "produced",
            "segment_timestamps": "produced",
        },
    )
    ordinary = _write_result(
        tmp_path / "ordinary.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Ordinary speech."}],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
        abstentions=[{
            "abstention_id": "ab_0",
            "reason": "alignment_unavailable",
            "start": 1.0,
            "end": 2.0,
        }],
    )
    for path, requested in ((event, [0.0, 1.0]), (ordinary, [1.0, 2.0])):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["provenance"]["plan"]["execution"] = {
            "range": {
                "requested": requested,
                "selected_unit_scope": requested,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    assert cli.main([
        "export",
        "--input", str(event),
        "--input", str(ordinary),
        "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "timing_required_for_format"
    assert not error["fix"].startswith("audio transcribe run")
    assert "already attempted word_timestamps and abstained" in error["fix"]


def test_export_cli_does_not_rerun_a_legacy_relative_source_from_another_cwd(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    original = tmp_path / "original"
    current = tmp_path / "current"
    original.mkdir()
    current.mkdir()
    transcript = _write_result(
        original / "meeting.transcript.json",
        source=Path("source.wav"),
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    (current / "source.wav").write_bytes(b"unrelated")
    monkeypatch.chdir(current)

    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "timing_required_for_format"
    assert not error["fix"].startswith("audio ")
    assert "source.path" in error["fix"]
    assert str(current / "source.wav") not in error["fix"]


def test_export_cli_does_not_prescribe_a_rerun_for_an_unusable_source_identity(
    tmp_path: Path, capsys
) -> None:
    for name in ("missing.wav", "directory", "loop"):
        source = tmp_path / name
        if name == "directory":
            source.mkdir()
        elif name == "loop":
            source.symlink_to(source.name)
        transcript = _write_result(
            tmp_path / f"{name}.transcript.json",
            source=source,
            segments=[{"segment_id": "seg_0", "text": "Hello."}],
            outcomes={"verbatim": "produced"},
        )

        assert cli.main([
            "export", "--input", str(transcript), "--format", "srt",
        ]) == 2
        error = json.loads(capsys.readouterr().err)
        assert error["code"] == "timing_required_for_format"
        assert not error["fix"].startswith("audio ")
        assert "source.path" in error["fix"]


def test_export_cli_event_only_timing_produces_an_empty_subtitle_file(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "events.json",
        source=source,
        segments=[{
            "segment_id": "seg_0", "text": "[Music]", "start": 0.0, "end": 1.0,
        }],
        outcomes={"word_timestamps": "produced", "segment_timestamps": "produced"},
    )
    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_export_cli_does_not_prescribe_an_inert_timing_rerun_for_ordinary_text(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "ordinary.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Ordinary untimed speech."}],
        outcomes={"word_timestamps": "produced"},
    )
    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "export_input_invalid"
    assert "alignment_unavailable" in error["reason"]


def test_export_cli_writes_subtitle_and_prints_a_summary(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "timed.json",
        source=source,
        segments=[{
            "segment_id": "seg_0",
            "text": "Hello.",
            "words": [{
                "word_id": "w_0", "text": "Hello", "start": 0.2, "end": 0.8,
            }],
        }],
        outcomes={"word_timestamps": "produced"},
    )
    output = tmp_path / "meeting.srt"
    assert cli.main([
        "export", "--input", str(transcript), "--format", "srt",
        "-o", str(output),
    ]) == 0
    assert output.read_text(encoding="utf-8") == (
        "1\n00:00:00,200 --> 00:00:00,800\nHello.\n"
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(output)
    assert summary["cues"] == 1
    assert summary["warnings"][0]["code"] == "cue_timing_unvalidated"


def test_export_cli_summary_reports_only_emitted_vtt_voice_tags(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "timed.json",
        source=source,
        segments=[{
            "segment_id": "seg_0",
            "text": "Hello.",
            "speaker": " \n\t ",
            "words": [{
                "word_id": "w_0", "text": "Hello", "start": 0.2, "end": 0.8,
            }],
        }],
        outcomes={"word_timestamps": "produced", "diarization": "produced"},
    )
    output = tmp_path / "meeting.vtt"
    assert cli.main([
        "export", "--input", str(transcript), "--format", "vtt",
        "-o", str(output),
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["speaker_labels_rendered"] is False
    assert "<v " not in output.read_text(encoding="utf-8")
