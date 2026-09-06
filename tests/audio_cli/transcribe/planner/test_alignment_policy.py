"""Requested host alignment policy is validated and resolved without model work."""

from dataclasses import replace
from pathlib import Path

import pytest

from audio_cli.command import Refusal
from audio_cli.transcribe import stacks
from audio_cli.transcribe.catalog import InputMetadata
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import ResolvedRequest, build_plan, resolve_request

METADATA = InputMetadata("original.wav", 10, "wav", 16000, 1)


def request(stack="qwen-1.7b", wants="word_timestamps", **options):
    return resolve_request(
        stack_id=stack,
        input_path=Path("original.wav"),
        wants=wants,
        **options,
    )


@pytest.mark.parametrize("stack", ["qwen-0.6b", "qwen-1.7b", "vibevoice"])
def test_forced_aligner_plan_records_effective_default_only_on_aligner(stack):
    resolved = request(stack)
    assert resolved.alignment_max_overrun_ms is None
    plan = serialize_plan(build_plan(resolved, METADATA))
    assert plan["roles"]["aligner"]["config"]["max_overrun_ms"] == 0.501
    assert "max_overrun_ms" not in plan["roles"]["asr"]["config"]
    assert (
        plan["sample_output"]["provenance"]["plan"]["roles"]["aligner"]["config"]
        == (plan["roles"]["aligner"]["config"])
    )


@pytest.mark.parametrize("value", [0.501, "0.501", 1, "150", 987.125, "2e2"])
def test_explicit_valid_limit_reaches_plan_and_preserves_original_request(value):
    resolved = request(alignment_max_overrun_ms=value)
    assert resolved.alignment_max_overrun_ms == float(value)
    plan = serialize_plan(build_plan(resolved, METADATA))
    assert plan["roles"]["aligner"]["config"]["max_overrun_ms"] == float(value)
    assert plan["roles"]["aligner"]["backend"] == "qwen3-forcedaligner"
    assert plan["capabilities"]["word_timestamps"]["backend"] == "qwen3-forcedaligner"
    without = serialize_plan(build_plan(replace(resolved, alignment_max_overrun_ms=None), METADATA))
    assert plan["packages"] == without["packages"]
    assert plan["roles"]["asr"] == without["roles"]["asr"]


@pytest.mark.parametrize(
    "value",
    [0, -1, 0.5, "0.500999", "bad", "", "nan", "inf", "-inf", "1e999", float("nan"), True, False],
)
def test_invalid_limit_is_a_finite_json_request_refusal(value):
    with pytest.raises(Refusal) as caught:
        request(alignment_max_overrun_ms=value)
    assert caught.value.exit_code == 2
    assert caught.value.payload == {
        "code": "alignment_max_overrun_invalid",
        "field": "--alignment-max-overrun-ms",
        "provided": str(value),
        "minimum_ms": 0.501,
        "fix": "set --alignment-max-overrun-ms to a finite number at least 0.501; "
        "repeat the original command, preserving every other argument",
    }


@pytest.mark.parametrize(
    ("stack", "wants"),
    [(stack, None) for stack in stacks.stack_ids()]
    + [
        ("firered", "word_timestamps"),
        ("vibevoice", "segment_timestamps"),
        ("qwen-1.7b", "diarization"),
    ],
)
def test_explicit_option_never_silently_applies_without_forced_aligner(stack, wants):
    with pytest.raises(Refusal) as caught:
        request(stack, wants, alignment_max_overrun_ms=200)
    assert caught.value.exit_code == 2
    assert caught.value.payload == {
        "code": "alignment_option_not_applicable",
        "field": "--alignment-max-overrun-ms",
        "provided": "200",
        "requires_backend": "qwen3-forcedaligner",
        "fix": "remove --alignment-max-overrun-ms and its value; "
        "repeat the original command, preserving every other argument",
    }
    plan = serialize_plan(build_plan(request(stack, wants), METADATA))
    assert "aligner" not in plan["roles"]


def test_legacy_resolved_request_constructor_retains_default_compatibility():
    resolved = ResolvedRequest(
        stacks.get_stack("qwen-1.7b"), Path("original.wav"), ("word_timestamps",), None, None, None
    )
    assert resolved.alignment_max_overrun_ms is None
    assert build_plan(resolved, METADATA).roles["aligner"]["config"]["max_overrun_ms"] == 0.501


def test_one_plan_does_not_mutate_another_request_alignment_policy():
    first = build_plan(request(alignment_max_overrun_ms=100), METADATA)
    second = build_plan(request(), METADATA)
    assert first.roles["aligner"]["config"]["max_overrun_ms"] == 100
    assert second.roles["aligner"]["config"]["max_overrun_ms"] == 0.501
