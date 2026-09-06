"""CLI stdout, output-path safety, and typed input failures."""

from __future__ import annotations

from pathlib import Path

from tests.audio_cli.export.export_cli_test_support import (
    _write_result,
    cli,
    json,
    os,
    stat,
)


def test_export_cli_prints_human_formats_without_an_output(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "txt",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == "Hello.\n"


def test_export_cli_refuses_inert_force_without_output(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "txt",
                "--force",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "code": "output_required_for_force",
        "field": "--force",
        "provided": True,
        "requires": "--output",
        "fix": "remove --force when writing to stdout, or add --output PATH",
    }


def test_export_cli_refuses_directory_output_even_with_force(tmp_path: Path, capsys) -> None:
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
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "txt",
                "-o",
                str(output),
                "--force",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "output_exists"
    assert "directory cannot be replaced" in error["fix"]
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_export_cli_refuses_replacing_a_fifo_even_with_force(
    tmp_path: Path,
    capsys,
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

    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "txt",
                "--output",
                str(output),
                "--force",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)

    assert error["code"] == "output_exists"
    assert "special file" in error["fix"]
    assert stat.S_ISFIFO(output.lstat().st_mode)


def test_export_cli_refuses_overwriting_an_input_transcript_with_truthful_fix(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    transcript = _write_result(
        tmp_path / "meeting.json",
        source=source,
        segments=[{"segment_id": "seg_0", "text": "Hello."}],
        outcomes={"verbatim": "produced"},
    )

    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "txt",
                "-o",
                str(transcript),
                "--force",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)

    assert error["code"] == "output_is_canonical_input"
    assert error["resolved_target"] == str(transcript)
    assert "input transcript" in error["fix"]
    assert transcript.read_text(encoding="utf-8").startswith("{")


def test_export_cli_rejects_a_self_referential_output_symlink(tmp_path: Path, capsys) -> None:
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
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(transcript),
                "--format",
                "txt",
                "-o",
                str(output),
                "--force",
            ]
        )
        == 2
    )
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
    tmp_path: Path,
    capsys,
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(invalid),
                "--format",
                "txt",
            ]
        )
        == 2
    )
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
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(first),
                "--input",
                str(second),
                "--format",
                "txt",
            ]
        )
        == 2
    )
    incompatible_error = json.loads(capsys.readouterr().err)
    assert incompatible_error["code"] == "export_inputs_incompatible"
    assert set(incompatible_error) == {"code", "field", "provided", "reason", "fix"}
    assert incompatible_error["provided"] == [str(first), str(second)]
