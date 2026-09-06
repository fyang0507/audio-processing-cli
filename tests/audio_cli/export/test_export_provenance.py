"""Provenance is an offline readable header with unchanged text and input evidence."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli.export import core, export_documents
from tests.audio_cli.export.export_test_support import _payload, _timed_segment, _write
from tests.docs.shipped_command_test_support import documented_block


@pytest.mark.parametrize("command", [["transcribe", "export"], ["export"]])
@pytest.mark.parametrize("output_format", ["txt", "md"])
def test_both_commands_export_offline_and_preserve_default_content(
    tmp_path, monkeypatch, capsys, command, output_format
):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline export reached media, stack, packages, or model resolution")

    monkeypatch.setattr(cli.transcribe_planner, "resolve_request", forbidden)
    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    monkeypatch.setattr(cli.transcribe_orchestrator, "run", forbidden)
    payload = _payload([{"segment_id": "seg_0", "text": "嗯，um… keep this!"}])
    payload["source"]["duration_basis"] = "canonical_decoded_pcm"
    path = _write(tmp_path / "original.json", payload)
    original = path.read_bytes()
    args = [*command, "--input", str(path), "--format", output_format]
    assert cli.main(args) == 0
    default = capsys.readouterr()
    assert default.err == ""
    assert (
        default.out
        == ("# Transcript\n\n" if output_format == "md" else "") + "嗯，um… keep this!\n"
    )
    assert cli.main([*args, "--provenance"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith(default.out)
    assert 'Source: "source' in captured.out
    assert 'Stack: "qwen' in captured.out
    assert 'Timebase: "seconds"' in captured.out
    assert 'Duration basis: "canonical' in captured.out
    assert "Complete: true" in captured.out
    assert "Owned intervals (seconds):" in captured.out
    assert path.read_bytes() == original


def test_merged_header_lists_each_saved_input_and_its_actual_coverage(tmp_path):
    source = tmp_path / "original.wav"
    coverage = {
        "scope_intervals": [[0.0, 4.0]],
        "covered_through_seconds": 1.0,
        "covered_fraction": 0.25,
        "covered_intervals": [[0.0, 1.0]],
        "missing_intervals": [[1.0, 4.0]],
        "units_total": 4,
        "units_completed": 1,
    }
    first = _write(
        tmp_path / "first.partial.json",
        _payload(
            [{"segment_id": "seg_0", "text": "First"}],
            source_path=str(source),
            duration=4,
            complete=False,
            coverage=coverage,
        ),
    )
    rest = _write(
        tmp_path / "rest.json",
        _payload(
            [{"segment_id": "seg_0", "text": "Rest"}],
            source_path=str(source),
            duration=4,
            run_range=[2, 4],
        ),
    )
    original = [path.read_bytes() for path in (first, rest)]
    result = export_documents([first, rest], "txt", provenance=True)
    assert result.content == (
        "Provenance\n\n"
        f"Source: {json.dumps(str(source))}\n"
        'Stack: "qwen-1.7b"\nTimebase: "seconds"\n\n'
        f"Input: {json.dumps(str(first))}\nComplete: false\n"
        "Owned intervals (seconds): [[0.0, 1.0]]\n"
        f"Coverage: {json.dumps(coverage)}\n\n"
        f"Input: {json.dumps(str(rest))}\nComplete: true\n"
        "Owned intervals (seconds): [[2.0, 4.0]]\n\nFirst\nRest\n"
    )
    assert "Duration basis" not in result.content
    assert [path.read_bytes() for path in (first, rest)] == original


@pytest.mark.parametrize("output_format", ["srt", "vtt", "jsonl"])
@pytest.mark.parametrize("command", [["transcribe", "export"], ["export"]])
def test_incompatible_format_refuses_before_loading_or_overwriting(
    tmp_path, monkeypatch, capsys, output_format, command
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid provenance option read an input")

    monkeypatch.setattr(core, "load_result_document", forbidden)
    output = tmp_path / "existing"
    output.write_bytes(b"keep")
    assert (
        cli.main(
            [
                *command,
                "--input",
                "missing.json",
                "--format",
                output_format,
                "--provenance",
                "--force",
                "-o",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == "provenance_unsupported_for_format"
    if output_format == "jsonl":
        assert json.loads(captured.err) == documented_block(
            "For example, JSONL with `--provenance` produces",
            document=Path(__file__).resolve().parents[3] / "docs/TRANSCRIBE_CONTRACT.md",
        )
    assert output.read_bytes() == b"keep"


def test_retry_uses_canonical_command_and_preserves_both_readable_options(tmp_path, capsys):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    path = _write(
        tmp_path / "two ' words.json",
        _payload(
            [_timed_segment("Hello!", [("Hello", 0.0, 1.0)])],
            source_path=str(source),
            outcomes={"word_timestamps": "produced"},
        ),
    )
    output = tmp_path / "two ' words.txt"
    output.write_bytes(b"replace")
    assert (
        cli.main(
            [
                "export",
                "--input",
                str(path),
                "--format",
                "txt",
                "--timestamps",
                "--provenance",
                "-o",
                str(output),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    retry = shlex.split(json.loads(captured.err)["fix"])
    assert retry[:3] == ["audio", "transcribe", "export"]
    assert {"--timestamps", "--provenance", "--force"} <= set(retry)
    assert cli.main(retry[1:]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {"input": str(path), "output": str(output), "format": "txt", "segments": 1}
    assert output.read_text().endswith("[00:00:00.000 --> 00:00:01.000] Hello!\n")
    assert output.read_text().startswith("Provenance\n")


@pytest.mark.parametrize("target", ["source", "input"])
def test_provenance_cannot_overwrite_canonical_files_even_with_force(tmp_path, capsys, target):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    path = _write(tmp_path / "result.json", _payload([], source_path=str(source)))
    original = (source.read_bytes(), path.read_bytes())
    assert (
        cli.main(
            [
                "transcribe",
                "export",
                "--input",
                str(path),
                "--format",
                "txt",
                "--provenance",
                "--force",
                "-o",
                str(source if target == "source" else path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["code"] == "output_is_canonical_input"
    assert (source.read_bytes(), path.read_bytes()) == original


def test_markdown_header_quotes_control_characters_and_escapes_markup(tmp_path):
    path = _write(
        tmp_path / "result.json",
        _payload(
            [],
            source_path="x\n<script> [link](target) `ticks` &copy; ~~strike~~ \\ source.wav",
        ),
    )
    content = export_documents([path], "md", provenance=True).content
    assert "\n<script>" not in content
    assert "<script>" not in content
    assert "[link](target)" not in content
    assert "`ticks`" not in content
    assert "\\&copy;" in content
    assert "~~strike~~" not in content
    assert content.endswith("# Transcript\n\n")


@pytest.mark.parametrize("output_format", ["txt", "md", "srt", "vtt", "jsonl"])
def test_alias_and_canonical_exports_are_identical_for_every_format(
    tmp_path, capsys, output_format
):
    path = _write(
        tmp_path / "result.json",
        _payload(
            [_timed_segment("Hello!", [("Hello", 0.0, 1.0)])],
            outcomes={"word_timestamps": "produced"},
        ),
    )
    outputs = []
    for command in (["transcribe", "export"], ["export"]):
        assert cli.main([*command, "--input", str(path), "--format", output_format]) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        outputs.append(captured.out)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "command,option",
    [
        (["transcribe", "plan"], "--compact"),
        (["transcribe", "run"], "--receipt"),
        (["transcribe", "export"], "--provenance"),
        (["export"], "--provenance"),
    ],
)
def test_new_options_are_discoverable_in_help(capsys, command, option):
    with pytest.raises(SystemExit) as raised:
        cli.main([*command, "--help"])
    assert raised.value.code == 0
    assert option in capsys.readouterr().out
