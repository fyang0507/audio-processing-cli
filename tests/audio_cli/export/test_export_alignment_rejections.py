"""Saved alignment evidence remains actionable at the offline export boundary."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli.export import ReadableTimingRequiredError, TimingRequiredError, export_documents
from audio_cli.export.refusals import timing_required_for_format, timing_required_for_timestamps
from tests.audio_cli.export.export_test_support import _payload, _write


def _rejected_payload(source: Path) -> dict:
    payload = _payload(
        [
            {"segment_id": "first", "text": "First."},
            {"segment_id": "second", "text": "Second."},
        ],
        source_path=str(source),
        outcomes={"word_timestamps": "abstained"},
    )
    payload["provenance"]["plan"]["roles"] = {
        "aligner": {"backend": "qwen3-forcedaligner", "config": {"max_overrun_ms": 0.501}}
    }
    payload["abstentions"] = [
        {
            "abstention_id": "first_failure",
            "reason": "alignment_unavailable",
            "start": 0.0,
            "end": 0.8,
            "alignment": {
                "unit_id": "unit_a",
                "segment_ids": ["first"],
                "word_index": 0,
                "code": "out_of_unit_bounds",
                "boundary": {
                    "original_bounds": [0.1, 0.82],
                    "unit_bounds": [0.0, 0.8],
                    "start_overrun_ms": 0.0,
                    "end_overrun_ms": 20.0,
                    "max_overrun_ms": 0.501,
                },
            },
        },
        {
            "abstention_id": "second_failure",
            "reason": "alignment_unavailable",
            "start": 1.0,
            "end": 1.8,
            "alignment": {
                "unit_id": "unit_b",
                "segment_ids": ["second"],
                "code": "provider_unavailable",
            },
        },
    ]
    return payload


@pytest.mark.parametrize("output_format", ["txt", "md", "srt", "vtt"])
def test_export_preserves_scoped_alignment_evidence(tmp_path, capsys, output_format):
    source = tmp_path / "original.wav"
    source.write_bytes(b"original bytes are not processed")
    payload = _rejected_payload(source)
    transcript = _write(tmp_path / "canonical.json", payload)
    before = transcript.read_bytes()
    readable = output_format in {"txt", "md"}
    expected = payload["abstentions"][:1] if readable else payload["abstentions"]
    error_type = ReadableTimingRequiredError if readable else TimingRequiredError
    with pytest.raises(error_type) as raised:
        export_documents([transcript], output_format, timestamps=readable)
    assert raised.value.alignment_rejections == expected

    output = tmp_path / f"export.{output_format}"
    args = [
        "transcribe",
        "export",
        "--input",
        str(transcript),
        "--format",
        output_format,
        "--output",
        str(output),
    ]
    if readable:
        args.append("--timestamps")
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    refusal = json.loads(captured.err)
    assert refusal["alignment_rejections"] == expected
    assert "original_bounds, unit_bounds" in refusal["fix"]
    assert "start_overrun_ms, end_overrun_ms and max_overrun_ms" in refusal["fix"]
    assert "--alignment-max-overrun-ms" in refusal["fix"]
    assert "another supported stack on the original source" in refusal["fix"]
    assert "preserving the required capabilities and scope" in refusal["fix"]
    assert "new canonical output" in refusal["fix"]
    assert "does not switch models automatically" in refusal["fix"]
    assert "neither choice guarantees" in refusal["fix"]
    assert "--alignment-max-overrun-ms 20" not in refusal["fix"]
    assert not refusal["fix"].startswith("audio ")
    assert not output.exists()
    assert transcript.read_bytes() == before
    assert source.read_bytes() == b"original bytes are not processed"


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("readable", [False, True])
def test_older_alignment_evidence_keeps_generic_remedy(tmp_path, capsys, metadata, readable):
    payload = _rejected_payload(tmp_path / "original.wav")
    for entry in payload["abstentions"]:
        if metadata:
            entry["alignment"].pop("boundary", None)
        else:
            del entry["alignment"]
    transcript = _write(tmp_path / "legacy.json", payload)
    args = [
        "transcribe",
        "export",
        "--input",
        str(transcript),
        "--format",
        "md" if readable else "srt",
    ]
    if readable:
        args.append("--timestamps")
    assert cli.main(args) == 2
    refusal = json.loads(capsys.readouterr().err)
    if metadata:
        expected = payload["abstentions"][:1] if readable else payload["abstentions"]
        assert refusal["alignment_rejections"] == expected
    else:
        assert "alignment_rejections" not in refusal
    assert "--alignment-max-overrun-ms" not in refusal["fix"]
    if readable:
        assert "already requested but abstained" in refusal["fix"]
        assert "rerunning the same request is not an established timing repair" in refusal["fix"]
    else:
        assert "this result already attempted word_timestamps and abstained" in refusal["fix"]
        assert "repeating the same command is not a repair" in refusal["fix"]


@pytest.mark.parametrize("entries", [None, []])
def test_refusal_builders_preserve_explicit_list_presence(tmp_path, entries):
    refusals = [
        timing_required_for_timestamps("saved.json", "seg_0", alignment_rejections=entries),
        timing_required_for_format(
            tmp_path / "saved.json",
            "srt",
            (),
            "qwen-1.7b",
            (),
            word_timing_outcome="abstained",
            alignment_rejections=entries,
        ),
    ]
    for refusal in refusals:
        assert ("alignment_rejections" in refusal.payload) is (entries is not None)
        if entries is not None:
            assert refusal.payload["alignment_rejections"] == []


def test_runnable_subtitle_remedy_preserves_saved_aligner_limit(tmp_path, capsys):
    source = tmp_path / "original with spaces.wav"
    source.write_bytes(b"source")
    payload = _payload(
        [{"segment_id": "seg_0", "text": "Hello."}],
        source_path=str(source),
        outcomes={"verbatim": "produced"},
        run_range=[0.0, 2.0],
    )
    payload["provenance"]["plan"]["roles"] = {
        "asr": {"config": {"language": "English"}},
        "aligner": {"backend": "qwen3-forcedaligner", "config": {"max_overrun_ms": 20.0}},
    }
    transcript = _write(tmp_path / "legacy plan.json", payload)
    assert cli.main(["transcribe", "export", "--input", str(transcript), "--format", "srt"]) == 2
    refusal = json.loads(capsys.readouterr().err)
    parsed = cli._parser().parse_args(shlex.split(refusal["fix"])[1:])
    assert parsed.input == source
    assert parsed.stack == "qwen-1.7b"
    assert parsed.language == "English"
    assert parsed.run_range == "0.0:2.0"
    assert float(parsed.alignment_max_overrun_ms) == 20.0
    assert "verbatim,word_timestamps" in shlex.split(refusal["fix"])
    assert parsed.output != transcript
    assert not parsed.output.exists()
    assert "alignment_rejections" not in refusal
