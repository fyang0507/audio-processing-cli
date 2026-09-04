"""CLI multi-input compatibility and canonical-source validation."""

from __future__ import annotations

# ruff: noqa: F403, F405
from export_cli_test_support import *

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
