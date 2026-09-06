"""Published-run remedies carry the original explicit host alignment policy."""

import shlex

import pytest

from audio_cli import cli
from audio_cli.command import Refusal
from audio_cli.transcribe.execution import publication
from audio_cli.transcribe.execution.runtime import parse_range
from audio_cli.transcribe.planner import resolve_request


@pytest.mark.parametrize("limit", [None, 0.501, 150.25])
def test_resume_command_keeps_policy_and_source_range(tmp_path, limit):
    request = resolve_request(
        stack_id="qwen-1.7b",
        input_path=tmp_path / "original.wav",
        wants="word_timestamps",
        alignment_max_overrun_ms=limit,
    )
    command = publication._resume_command(
        request,
        {"units_completed": 1, "covered_intervals": [[10, 20]], "covered_through_seconds": 20},
        tmp_path / "canonical.partial.json",
        parse_range("10:90"),
    )
    parsed = cli._parser().parse_args(shlex.split(command)[1:])
    assert parsed.alignment_max_overrun_ms == (str(limit) if limit is not None else None)
    assert parsed.run_range == "20:90.0"
    assert parsed.output == tmp_path / "canonical.rest.json"
    assert parsed.want == "word_timestamps"


def test_publication_race_refusal_preserves_alignment_policy(tmp_path, monkeypatch):
    request = resolve_request(
        stack_id="vibevoice",
        input_path=tmp_path / "original.wav",
        wants="word_timestamps",
        alignment_max_overrun_ms=150.25,
    )
    output = tmp_path / "canonical.json"

    def collision(*args, **kwargs):
        raise FileExistsError(output)

    monkeypatch.setattr(publication, "_write_result_file", collision)
    with pytest.raises(Refusal) as caught:
        publication._publish_result(
            request,
            {},
            output,
            output=output,
            output_format="json",
            run_range=parse_range("10:90"),
            force=False,
            protected_source_identity=None,
        )
    parsed = cli._parser().parse_args(shlex.split(caught.value.payload["fix"])[1:])
    assert float(parsed.alignment_max_overrun_ms) == 150.25
    assert parsed.run_range == "10:90"
    assert parsed.force
