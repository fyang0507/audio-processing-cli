"""Alignment policy examples must match request refusals and valid saved evidence."""

from __future__ import annotations

import json
import shlex

import pytest

from audio_cli import cli
from audio_cli.transcribe.execution.receipt import build_receipt
from audio_cli.transcribe.result import NormalizedResult, serialize_result
from tests.docs.shipped_command_test_support import documented_block
from tests.docs.spec_document_test_support import CONTRACT, json_blocks


@pytest.mark.parametrize(
    "command",
    [
        "audio transcribe plan --input meeting.m4a --stack qwen-1.7b "
        "--want word_timestamps --alignment-max-overrun-ms 0.5",
        "audio transcribe plan --input field.wav --stack firered "
        "--want word_timestamps --alignment-max-overrun-ms 20",
    ],
)
def test_documented_alignment_refusal_is_the_actual_cli_payload(command, capsys):
    assert cli.main(shlex.split(command)[1:]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == documented_block(command)


def test_documented_receipt_fragment_is_valid_canonical_evidence():
    fragment = next(
        doc
        for _, doc in json_blocks(CONTRACT)
        if set(doc) == {"alignment_rejections", "alignment_corrections"}
    )
    result = NormalizedResult(
        source={"path": "example.wav", "duration_seconds": 2.0, "timebase": "seconds"},
        segments=[
            {"segment_id": "seg_0", "text": "Rejected."},
            {
                "segment_id": "seg_1",
                "text": "Accepted.",
                "words": [{"word_id": "w_0", "text": "Accepted", "start": 1.0, "end": 1.5}],
            },
        ],
        abstentions=fragment["alignment_rejections"],
        provenance={
            "stack": "qwen-1.7b",
            "outcomes": {"word_timestamps": "abstained"},
            "observed": {"alignment_corrections": fragment["alignment_corrections"]},
            "plan": {"roles": {"aligner": {"config": {"max_overrun_ms": 20.0}}}},
        },
        requested_capabilities=frozenset({"word_timestamps"}),
    )
    saved = serialize_result(result)
    receipt = build_receipt(saved, "example.json")
    assert {key: receipt[key] for key in fragment} == fragment
