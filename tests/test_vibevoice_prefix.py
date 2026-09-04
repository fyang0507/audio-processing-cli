from __future__ import annotations

from vibevoice_test_support import (
    _PINNED_POST_PROCESS_TRANSCRIPTION,
    _fixture,
    ast,
    hashlib,
    json,
    normalize_vibevoice_result,
    pytest,
    textwrap,
    types,
    typing,
    vibevoice_stage,
)


@pytest.mark.parametrize(
    "trailing",
    [
        '[{"Start":1,"End":2,"Speaker":1,"Content":"Dropped"}]',
        "additional generated transcript text",
    ],
)
def test_vibevoice_adapter_rejects_content_after_a_closed_capped_array(
    trailing: str,
) -> None:
    raw = (
        'assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Complete"}]\n'
        f"{trailing}"
    )

    with pytest.raises(ValueError, match="continues after its JSON array"):
        normalize_vibevoice_result({
            "segments": [],
            "raw_text": raw,
            "hit_max_new_tokens": True,
        }, clip_duration_seconds=3.0)


def test_vibevoice_parsers_reject_a_transcript_before_the_json_fence() -> None:
    raw = (
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Dropped"}]\n'
        '```json\n'
        '[{"Start":1,"End":2,"Speaker":1,"Content":"Published"}]\n```'
    )

    with pytest.raises(ValueError, match="continues after its JSON array"):
        vibevoice_stage._complete_json_array(raw)
    with pytest.raises(ValueError, match="complete JSON array"):
        normalize_vibevoice_result({
            "segments": [{
                "start_time": 1,
                "end_time": 2,
                "speaker_id": 1,
                "text": "Published",
            }],
            "raw_text": raw,
            "hit_max_new_tokens": False,
        }, clip_duration_seconds=2.0)


def test_vibevoice_adapter_clamps_only_submicrosecond_clip_overrun() -> None:
    segments = [{
        "start_time": 0.0,
        "end_time": 1.000001,
        "speaker_id": 0,
        "text": "Hi.",
    }]
    result = normalize_vibevoice_result(
        {
            "segments": segments,
            "raw_text": json.dumps(segments),
            "hit_max_new_tokens": False,
        },
        clip_duration_seconds=1.0,
    )

    assert result.segments[0]["end"] == 1.0


@pytest.mark.parametrize(
    ("text", "start", "end", "offset"),
    [
        ("Speech.", 1.0, 1.0, 0.0),
        ("[Music]", 1.0, 1.0, 0.0),
        ("Speech.", 0.0, 0.0000004, 7.123456),
        ("[Music]", 0.0, 0.0000004, 7.123456),
    ],
)
def test_vibevoice_adapter_rejects_zero_duration_on_the_public_timeline(
    text: str, start: float, end: float, offset: float
) -> None:
    segments = [{
        "start_time": start,
        "end_time": end,
        "speaker_id": 0,
        "text": text,
    }]
    with pytest.raises(ValueError, match="positive duration"):
        normalize_vibevoice_result({
            "segments": segments,
            "raw_text": json.dumps(segments),
            "hit_max_new_tokens": False,
        }, offset_seconds=offset, clip_duration_seconds=2.0)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_vibevoice_adapter_rejects_empty_or_whitespace_speech(text: str) -> None:
    segments = [{
        "start_time": 0.0,
        "end_time": 1.0,
        "speaker_id": 0,
        "text": text,
    }]
    with pytest.raises(ValueError, match="text must not be empty"):
        normalize_vibevoice_result({
            "segments": segments,
            "raw_text": json.dumps(segments),
            "hit_max_new_tokens": False,
        }, clip_duration_seconds=2.0)


def test_vibevoice_adapter_salvages_every_complete_object_before_arbitrary_cuts() -> None:
    raw = _fixture("vibevoice_30m_raw_prefix_excerpt.json")["raw_text"]
    after_first_comma = raw.index("},{") + 2
    inside_fourth = raw.index('{"Start":28.19') + len('{"Start":28.19')
    inside_first_string = raw.index("[Silence]") + 3
    cases = (
        (raw[:inside_first_string], 0, None),
        (raw[:after_first_comma], 1, 110.91),
        (raw[:inside_fourth], 3, 127.95),
        (raw, 4, 131.72),
    )

    for truncated, count, watermark in cases:
        result = normalize_vibevoice_result({
            "segments": [{"this": "must be ignored on truncation"}],
            "raw_text": truncated,
            "hit_max_new_tokens": True,
        }, offset_seconds=100.0, clip_duration_seconds=40.0)
        assert len(result.segments) == count
        assert result.covered_through_seconds == watermark

    event = normalize_vibevoice_result({
        "segments": [],
        "raw_text": raw[:inside_fourth],
        "hit_max_new_tokens": True,
    }, clip_duration_seconds=40.0).segments[0]
    assert "speaker" not in event
    assert event["alignable"] is False


def test_pinned_upstream_postprocessor_loses_the_recorded_truncated_prefix() -> None:
    parsed = ast.parse(_PINNED_POST_PROCESS_TRANSCRIPTION)
    function = parsed.body[0]
    assert isinstance(function, ast.FunctionDef)
    # The upstream docstring has spaces on otherwise blank lines.  Exclude only that
    # leading string expression so the source-backed digest is semantic and this file
    # remains clean under ``git diff --check``.
    assert isinstance(function.body[0], ast.Expr)
    function.body = function.body[1:]
    digest = hashlib.sha256(
        ast.dump(function, annotate_fields=True, include_attributes=False).encode()
    ).hexdigest()
    assert digest == "1434126abf8ea90bc103972b07f0cdfb9e34a19144d9cc1e3e188f2d9091dd8b"

    logger = types.SimpleNamespace(warning=lambda *_args: None, debug=lambda *_args: None)
    namespace = {
        "Any": typing.Any,
        "Dict": typing.Dict,
        "List": typing.List,
        "json": json,
        "logger": logger,
    }
    exec(
        "class PinnedProcessor:\n"
        + textwrap.indent(_PINNED_POST_PROCESS_TRANSCRIPTION, "    "),
        namespace,
    )
    raw = _fixture("vibevoice_30m_raw_prefix_excerpt.json")["raw_text"]
    assert namespace["PinnedProcessor"]().post_process_transcription(raw) == []


def test_complete_malformed_vibevoice_raw_text_cannot_collapse_to_success() -> None:
    with pytest.raises(ValueError, match="complete JSON array"):
        normalize_vibevoice_result({
            "raw_text": 'assistant\n[{"Start":0,"End":1,"Content":"lost"',
            "segments": [],
            "hit_max_new_tokens": False,
        })


def test_complete_vibevoice_values_must_match_the_generated_json() -> None:
    with pytest.raises(ValueError, match="differs from generated JSON"):
        normalize_vibevoice_result({
            "raw_text": json.dumps([{
                "Start": 0,
                "End": 1,
                "Speaker": 0,
                "Content": "model said raw",
            }]),
            "segments": [{
                "start_time": 0,
                "end_time": 1,
                "speaker_id": 0,
                "text": "postprocessor fabricated",
            }],
            "hit_max_new_tokens": False,
        }, clip_duration_seconds=1.0)


@pytest.mark.parametrize("hit_max_new_tokens", [False, True])
def test_vibevoice_adapter_types_excessively_nested_generated_json(
    hit_max_new_tokens: bool,
) -> None:
    raw = "[" * 10_000 + "0" + "]" * 10_000
    with pytest.raises(ValueError, match="(complete JSON array|nesting is too deep)"):
        normalize_vibevoice_result({
            "segments": [],
            "raw_text": raw,
            "hit_max_new_tokens": hit_max_new_tokens,
        })


def test_vibevoice_prefix_parser_uses_json_rules_inside_quoted_text() -> None:
    raw = (
        'assistant\n[{"Start":0,"End":1,"Speaker":0,'
        '"Content":"literal } and ] remain text"},'
        '{"Start":1,"End":2,"Speaker":1,"Content":"unfinished'
    )
    result = normalize_vibevoice_result({
        "segments": [],
        "raw_text": raw,
        "hit_max_new_tokens": True,
    }, clip_duration_seconds=2.0)
    assert result.segments == ({
        "text": "literal } and ] remain text",
        "start": 0.0,
        "end": 1.0,
        "alignable": True,
        "speaker": "0",
    },)
    assert result.covered_through_seconds == 1.0
