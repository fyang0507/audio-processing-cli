"""CLI timing refusal and parseable rerun-fix construction."""

from __future__ import annotations

from export_cli_test_support import (
    Path,
    _write_result,
    cli,
    json,
    os,
    shlex,
)


def test_output_exists_fix_runs_for_option_like_paths(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    _write_result(
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
