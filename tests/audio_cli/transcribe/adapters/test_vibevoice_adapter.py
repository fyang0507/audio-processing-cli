from __future__ import annotations

from tests.audio_cli.transcribe.vibevoice_test_support import (
    ROOT,
    _fixture,
    hashlib,
    json,
    normalize_vibevoice_result,
    pytest,
    vibevoice_stage,
)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "vibevoice_multispeaker_excerpt.json",
        "vibevoice_30m_raw_prefix_excerpt.json",
    ],
)
def test_vibevoice_excerpts_name_the_exact_local_artifact(fixture_name: str) -> None:
    fixture = _fixture(fixture_name)
    source = ROOT / fixture["source_artifact"]
    if not source.is_file():
        pytest.skip("untracked full VibeVoice artifact is not present")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == fixture["source_sha256"]
    document = json.loads(source.read_text(encoding="utf-8"))
    if fixture_name == "vibevoice_multispeaker_excerpt.json":
        assert fixture["excerpt_rule"] == (
            "segments 7 through 10 inclusive, unchanged and in source order"
        )
        assert document["segments"][7:11] == fixture["segments"]
    else:
        assert fixture["excerpt_rule"] == (
            "output.raw_text from byte zero through the closing brace of array element 3, "
            "unchanged; the outer array close and later elements are omitted"
        )
        source_raw = document["output"]["raw_text"]
        excerpt = fixture["raw_text"]
        assert source_raw.startswith(excerpt)
        assert source_raw[len(excerpt)] == ","


def test_vibevoice_adapter_omits_na_speaker_and_preserves_the_event() -> None:
    fixture = _fixture("vibevoice_multispeaker_excerpt.json")
    result = normalize_vibevoice_result(
        {
            "segments": fixture["segments"],
            "raw_text": json.dumps(fixture["segments"]),
            "hit_max_new_tokens": False,
        },
        clip_duration_seconds=60.0,
    )

    assert result.hit_max_new_tokens is False
    assert result.covered_through_seconds is None
    assert result.segments[0]["speaker"] == "1"
    event = result.segments[2]
    assert event == {
        "text": "[Environmental Sounds]",
        "start": 37.84,
        "end": 40.28,
        "alignable": False,
    }
    assert "N/A" not in json.dumps(result.segments, ensure_ascii=False)


def test_vibevoice_adapter_rejects_speaker_attribution_on_event_tag() -> None:
    segments = [
        {
            "start_time": 0.0,
            "end_time": 1.0,
            "speaker_id": 0,
            "text": "[Music]",
        }
    ]
    with pytest.raises(ValueError, match="event segment 0 must not carry a speaker"):
        normalize_vibevoice_result(
            {
                "segments": segments,
                "raw_text": json.dumps(segments),
                "hit_max_new_tokens": False,
            },
            clip_duration_seconds=1.0,
        )


def test_vibevoice_adapter_accepts_the_stage_fenced_json_envelope() -> None:
    raw = 'assistant\n```json\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]\n```'
    result = normalize_vibevoice_result(
        {
            "segments": [
                {
                    "start_time": 0,
                    "end_time": 1,
                    "speaker_id": 0,
                    "text": "Hi",
                }
            ],
            "raw_text": raw,
            "hit_max_new_tokens": False,
        },
        clip_duration_seconds=1.0,
    )

    assert result.segments == (
        {
            "text": "Hi",
            "start": 0.0,
            "end": 1.0,
            "alignable": True,
            "speaker": "0",
        },
    )


@pytest.mark.parametrize("fenced", [False, True])
def test_vibevoice_stage_preserves_fence_tokens_inside_transcript_content(
    fenced: bool,
) -> None:
    items = [
        {
            "Start": 0,
            "End": 1,
            "Speaker": 0,
            "Content": "say ```json and ``` literally",
        }
    ]
    encoded = json.dumps(items)
    raw = f"assistant\n```json\n{encoded}\n```" if fenced else f"assistant\n{encoded}"

    assert vibevoice_stage._complete_json_array(raw) == items


@pytest.mark.parametrize("fenced", [False, True])
def test_vibevoice_adapter_preserves_fence_tokens_inside_transcript_content(
    fenced: bool,
) -> None:
    text = "say ```json and ``` literally"
    encoded = json.dumps(
        [
            {
                "Start": 0,
                "End": 1,
                "Speaker": 0,
                "Content": text,
            }
        ]
    )
    raw = f"assistant\n```json\n{encoded}\n```" if fenced else f"assistant\n{encoded}"

    result = normalize_vibevoice_result(
        {
            "segments": [
                {
                    "start_time": 0,
                    "end_time": 1,
                    "speaker_id": 0,
                    "text": text,
                }
            ],
            "raw_text": raw,
            "hit_max_new_tokens": False,
        },
        clip_duration_seconds=1.0,
    )

    assert result.segments[0]["text"] == text


@pytest.mark.parametrize(
    "trailing",
    [
        '[{"Start":1,"End":2,"Speaker":1,"Content":"Dropped"}]',
        "additional generated transcript text",
    ],
)
def test_vibevoice_stage_rejects_content_after_complete_fenced_json(
    trailing: str,
) -> None:
    raw = f'assistant\n```json\n[{{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}}]\n```\n{trailing}'

    with pytest.raises(ValueError, match="continues after its JSON code block"):
        vibevoice_stage._complete_json_array(raw)


@pytest.mark.parametrize(
    "trailing",
    [
        '[{"Start":1,"End":2,"Speaker":1,"Content":"Dropped"}]',
        "additional generated transcript text",
    ],
)
def test_vibevoice_adapter_rejects_content_after_complete_fenced_json(
    trailing: str,
) -> None:
    raw = f'assistant\n```json\n[{{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}}]\n```\n{trailing}'

    with pytest.raises(ValueError, match="complete JSON array"):
        normalize_vibevoice_result(
            {
                "segments": [
                    {
                        "start_time": 0,
                        "end_time": 1,
                        "speaker_id": 0,
                        "text": "Hi",
                    }
                ],
                "raw_text": raw,
                "hit_max_new_tokens": False,
            },
            clip_duration_seconds=2.0,
        )


def test_vibevoice_stage_rejects_duplicate_generated_segment_keys() -> None:
    raw = '[{"Start":0,"Start":5,"End":6,"Speaker":0,"Content":"Ambiguous"}]'

    with pytest.raises(ValueError, match="repeats key 'Start'"):
        vibevoice_stage._complete_json_array(raw)


@pytest.mark.parametrize("hit_max_new_tokens", [False, True])
def test_vibevoice_adapter_rejects_duplicate_generated_segment_keys(
    hit_max_new_tokens: bool,
) -> None:
    raw = '[{"Start":0,"Start":5,"End":6,"Speaker":0,"Content":"Ambiguous"}]'
    payload = {
        "segments": [
            {
                "start_time": 5,
                "end_time": 6,
                "speaker_id": 0,
                "text": "Ambiguous",
            }
        ],
        "raw_text": raw,
        "hit_max_new_tokens": hit_max_new_tokens,
    }

    with pytest.raises(ValueError, match="(complete JSON array|repeats key 'Start')"):
        normalize_vibevoice_result(payload, clip_duration_seconds=10.0)


def test_vibevoice_adapter_salvages_a_capped_fenced_prefix() -> None:
    raw = (
        "assistant\n```json\n"
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Complete"},'
        '{"Start":1,"End":2,"Speaker":1,"Content":"cut'
    )

    result = normalize_vibevoice_result(
        {
            "segments": [],
            "raw_text": raw,
            "hit_max_new_tokens": True,
        },
        clip_duration_seconds=3.0,
    )

    assert result.segments == (
        {
            "text": "Complete",
            "start": 0.0,
            "end": 1.0,
            "alignable": True,
            "speaker": "0",
        },
    )
    assert result.covered_through_seconds == 1.0


@pytest.mark.parametrize("partial_close", ["`", "``"])
def test_vibevoice_adapter_salvages_a_cap_inside_the_closing_fence(
    partial_close: str,
) -> None:
    raw = (
        "assistant\n```json\n"
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Complete"}]\n'
        f"{partial_close}"
    )

    result = normalize_vibevoice_result(
        {
            "segments": [],
            "raw_text": raw,
            "hit_max_new_tokens": True,
        },
        clip_duration_seconds=2.0,
    )

    assert result.segments == (
        {
            "text": "Complete",
            "start": 0.0,
            "end": 1.0,
            "alignable": True,
            "speaker": "0",
        },
    )
    assert result.covered_through_seconds == 1.0


@pytest.mark.parametrize("invalid_close", ["`x", "``x", "` ", "`` trailing"])
def test_vibevoice_adapter_rejects_nonprefix_capped_closing_fence_text(
    invalid_close: str,
) -> None:
    raw = (
        "assistant\n```json\n"
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Complete"}]\n'
        f"{invalid_close}"
    )

    with pytest.raises(ValueError, match="continues after its JSON array"):
        normalize_vibevoice_result(
            {
                "segments": [],
                "raw_text": raw,
                "hit_max_new_tokens": True,
            },
            clip_duration_seconds=2.0,
        )
