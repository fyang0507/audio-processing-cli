"""Public stack discovery must work before media, models, or request validation."""

import json
from pathlib import Path

import pytest

from audio_cli import cli
from audio_cli.transcribe import stacks


def test_discovery_stdout_matches_complete_documented_output(capsys):
    document = (
        Path(__file__).resolve().parents[1]
        / "docs/transcribe-happy-path/00-overview-and-machine.md"
    ).read_text()
    block = document.split("## Discover declared stacks", 1)[1].split("```json\n", 1)[1]
    expected = json.loads(block.split("```", 1)[0])
    assert cli.main(["transcribe", "stacks"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == expected


def test_stack_discovery_is_offline_declared_and_non_refusing(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("discovery touched request, media, or mutable installed state")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "probe_media", forbidden)
    monkeypatch.setattr(cli, "load_registry", forbidden)
    monkeypatch.setattr(cli.transcribe_planner, "resolve_request", forbidden)
    assert cli.main(["transcribe", "stacks"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    rows = json.loads(captured.out)["stacks"]
    assert [row["id"] for row in rows] == list(stacks.stack_ids())
    for row in rows:
        definition = stacks.get_stack(row["id"])
        assert row == {
            "id": definition.id,
            "characterization": definition.characterization,
            **stacks.availability_groups(definition),
        }
        assert set(row["native"]) | set(row["requires_add_on"]) | set(row["impossible"]) == set(
            stacks.capability_order()
        )


@pytest.mark.parametrize("command", ["capabilities", "plan", "run"])
def test_request_help_routes_to_discovery_without_changing_missing_stack_refusal(command, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["transcribe", command, "--help"])
    assert caught.value.code == 0
    assert "audio transcribe stacks" in " ".join(capsys.readouterr().out.split())
    assert cli.main(["transcribe", command]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == cli.transcribe_refusals.stack_required().payload
