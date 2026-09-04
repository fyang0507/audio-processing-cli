from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import sys
import textwrap
import threading
import types
import typing
from pathlib import Path

import pytest

from audio_cli.transcribe.adapters import normalize_vibevoice_result
from audio_cli.transcribe.stages import vibevoice as vibevoice_stage

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

# Exact executable method source from
# microsoft/VibeVoice@94da20d98b2fa7688e9cbfaf7692ddb4954f7600,
# vibevoice/processor/vibevoice_asr_processor.py.  The full pinned file's SHA-256 is
# e5551319fac744b5152202a8e41888dad2d94d3fa941aa755d0888caa5a6a4ab; the AST
# digest below makes this executable excerpt a source-backed regression rather than a
# test double that can be configured to return the desired answer.
_PINNED_POST_PROCESS_TRANSCRIPTION = '''\
def post_process_transcription(self, text: str) -> List[Dict[str, Any]]:
        """
        Post-process the generated transcription text to extract structured data.

        Args:
            text: Generated text from the model

        Returns:
            List of dictionaries with transcription segments
        """
        try:
            # Try to parse as JSON
            if "```json" in text:
                # Extract JSON from markdown code block
                json_start = text.find("```json") + 7
                json_end = text.find("```", json_start)
                json_str = text[json_start:json_end].strip()
            else:
                # Try to find JSON array or object
                json_start = text.find("[")
                if json_start == -1:
                    json_start = text.find("{")
                if json_start != -1:
                    # Find matching closing bracket
                    bracket_count = 0
                    json_end = json_start
                    for i in range(json_start, len(text)):
                        if text[i] in "[{":
                            bracket_count += 1
                        elif text[i] in "]}":
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_end = i + 1
                                break
                    json_str = text[json_start:json_end]
                else:
                    json_str = text

            # Parse JSON
            result = json.loads(json_str)

            # Ensure it's a list
            if isinstance(result, dict):
                result = [result]

            # Validate and clean up the result
            cleaned_result = []
            for item in result:
                if isinstance(item, dict):
                    cleaned_item = {}
                    # Map keys to expected format
                    key_mapping = {
                        "Start time": "start_time",
                        "Start": "start_time",
                        "End time": "end_time",
                        "End": "end_time",
                        "Speaker ID": "speaker_id",
                        "Speaker": "speaker_id",
                        "Content": "text",
                    }
                    for key, mapped_key in key_mapping.items():
                        if key in item:
                            cleaned_item[mapped_key] = item[key]

                    if cleaned_item:
                        cleaned_result.append(cleaned_item)

            return cleaned_result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from transcription: {e}")
            logger.debug(f"Raw text: {text}")
            return []
        except Exception as e:
            logger.warning(f"Error post-processing transcription: {e}")
            return []
'''


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


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
    result = normalize_vibevoice_result({
        "segments": fixture["segments"],
        "raw_text": json.dumps(fixture["segments"]),
        "hit_max_new_tokens": False,
    }, clip_duration_seconds=60.0)

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
    segments = [{
        "start_time": 0.0,
        "end_time": 1.0,
        "speaker_id": 0,
        "text": "[Music]",
    }]
    with pytest.raises(ValueError, match="event segment 0 must not carry a speaker"):
        normalize_vibevoice_result({
            "segments": segments,
            "raw_text": json.dumps(segments),
            "hit_max_new_tokens": False,
        }, clip_duration_seconds=1.0)


def test_vibevoice_adapter_accepts_the_stage_fenced_json_envelope() -> None:
    raw = (
        'assistant\n```json\n'
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]\n```'
    )
    result = normalize_vibevoice_result({
        "segments": [{
            "start_time": 0,
            "end_time": 1,
            "speaker_id": 0,
            "text": "Hi",
        }],
        "raw_text": raw,
        "hit_max_new_tokens": False,
    }, clip_duration_seconds=1.0)

    assert result.segments == ({
        "text": "Hi",
        "start": 0.0,
        "end": 1.0,
        "alignable": True,
        "speaker": "0",
    },)


@pytest.mark.parametrize("fenced", [False, True])
def test_vibevoice_stage_preserves_fence_tokens_inside_transcript_content(
    fenced: bool,
) -> None:
    items = [{
        "Start": 0,
        "End": 1,
        "Speaker": 0,
        "Content": "say ```json and ``` literally",
    }]
    encoded = json.dumps(items)
    raw = (
        f"assistant\n```json\n{encoded}\n```"
        if fenced
        else f"assistant\n{encoded}"
    )

    assert vibevoice_stage._complete_json_array(raw) == items


@pytest.mark.parametrize("fenced", [False, True])
def test_vibevoice_adapter_preserves_fence_tokens_inside_transcript_content(
    fenced: bool,
) -> None:
    text = "say ```json and ``` literally"
    encoded = json.dumps([{
        "Start": 0,
        "End": 1,
        "Speaker": 0,
        "Content": text,
    }])
    raw = (
        f"assistant\n```json\n{encoded}\n```"
        if fenced
        else f"assistant\n{encoded}"
    )

    result = normalize_vibevoice_result({
        "segments": [{
            "start_time": 0,
            "end_time": 1,
            "speaker_id": 0,
            "text": text,
        }],
        "raw_text": raw,
        "hit_max_new_tokens": False,
    }, clip_duration_seconds=1.0)

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
    raw = (
        'assistant\n```json\n'
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]\n```\n'
        f"{trailing}"
    )

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
    raw = (
        'assistant\n```json\n'
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]\n```\n'
        f"{trailing}"
    )

    with pytest.raises(ValueError, match="complete JSON array"):
        normalize_vibevoice_result({
            "segments": [{
                "start_time": 0,
                "end_time": 1,
                "speaker_id": 0,
                "text": "Hi",
            }],
            "raw_text": raw,
            "hit_max_new_tokens": False,
        }, clip_duration_seconds=2.0)


def test_vibevoice_stage_rejects_duplicate_generated_segment_keys() -> None:
    raw = (
        '[{"Start":0,"Start":5,"End":6,"Speaker":0,'
        '"Content":"Ambiguous"}]'
    )

    with pytest.raises(ValueError, match="repeats key 'Start'"):
        vibevoice_stage._complete_json_array(raw)


@pytest.mark.parametrize("hit_max_new_tokens", [False, True])
def test_vibevoice_adapter_rejects_duplicate_generated_segment_keys(
    hit_max_new_tokens: bool,
) -> None:
    raw = (
        '[{"Start":0,"Start":5,"End":6,"Speaker":0,'
        '"Content":"Ambiguous"}]'
    )
    payload = {
        "segments": [{
            "start_time": 5,
            "end_time": 6,
            "speaker_id": 0,
            "text": "Ambiguous",
        }],
        "raw_text": raw,
        "hit_max_new_tokens": hit_max_new_tokens,
    }

    with pytest.raises(ValueError, match="(complete JSON array|repeats key 'Start')"):
        normalize_vibevoice_result(payload, clip_duration_seconds=10.0)


def test_vibevoice_adapter_salvages_a_capped_fenced_prefix() -> None:
    raw = (
        'assistant\n```json\n'
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Complete"},'
        '{"Start":1,"End":2,"Speaker":1,"Content":"cut'
    )

    result = normalize_vibevoice_result({
        "segments": [],
        "raw_text": raw,
        "hit_max_new_tokens": True,
    }, clip_duration_seconds=3.0)

    assert result.segments == ({
        "text": "Complete",
        "start": 0.0,
        "end": 1.0,
        "alignable": True,
        "speaker": "0",
    },)
    assert result.covered_through_seconds == 1.0


@pytest.mark.parametrize("partial_close", ["`", "``"])
def test_vibevoice_adapter_salvages_a_cap_inside_the_closing_fence(
    partial_close: str,
) -> None:
    raw = (
        'assistant\n```json\n'
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Complete"}]\n'
        f"{partial_close}"
    )

    result = normalize_vibevoice_result({
        "segments": [],
        "raw_text": raw,
        "hit_max_new_tokens": True,
    }, clip_duration_seconds=2.0)

    assert result.segments == ({
        "text": "Complete",
        "start": 0.0,
        "end": 1.0,
        "alignable": True,
        "speaker": "0",
    },)
    assert result.covered_through_seconds == 1.0


@pytest.mark.parametrize("invalid_close", ["`x", "``x", "` ", "`` trailing"])
def test_vibevoice_adapter_rejects_nonprefix_capped_closing_fence_text(
    invalid_close: str,
) -> None:
    raw = (
        'assistant\n```json\n'
        '[{"Start":0,"End":1,"Speaker":0,"Content":"Complete"}]\n'
        f"{invalid_close}"
    )

    with pytest.raises(ValueError, match="continues after its JSON array"):
        normalize_vibevoice_result({
            "segments": [],
            "raw_text": raw,
            "hit_max_new_tokens": True,
        }, clip_duration_seconds=2.0)


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


def _install_fake_vibevoice(
    monkeypatch,
    *,
    raw_text: str,
    segments: list[dict],
    generated_tokens: int,
    eos_positions: list[int],
    fail_load: bool = False,
    mps_available: bool = True,
    mps_samples: tuple[int, ...] = (10, 30, 20),
) -> dict[str, object]:
    state: dict[str, object] = {
        "generate_calls": 0,
        "torch_seeds": [],
        "mps_seeds": [],
        "numpy_seeds": [],
        "mps_sample_calls": 0,
    }

    class FakeTensor:
        def to(self, device):
            state.setdefault("tensor_devices", []).append(device)
            return self

    class InputTensor(FakeTensor):
        shape = (1, 3)

    class Equality:
        def nonzero(self, *, as_tuple):
            assert as_tuple is True
            return (list(eos_positions),)

    class GeneratedTensor(FakeTensor):
        def numel(self):
            return generated_tokens

        def __eq__(self, _value):
            return Equality()

        def __getitem__(self, _value):
            return self

    class OutputTensor:
        def __getitem__(self, value):
            assert value[0] == 0 and value[1].start == 3
            return GeneratedTensor()

    class Processor:
        pad_id = 0
        tokenizer = types.SimpleNamespace(eos_token_id=99)

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state["processor_load"] = (model, kwargs)
            return cls()

        def __call__(self, **kwargs):
            state["processor_call"] = kwargs
            return {"input_ids": InputTensor(), "attention_mask": InputTensor()}

        def decode(self, _generated, **kwargs):
            state["decode_kwargs"] = kwargs
            return raw_text

        def post_process_transcription(self, text):
            assert text == raw_text
            return segments

    class Model:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state["model_load"] = (model, kwargs)
            if fail_load:
                raise RuntimeError("synthetic model-load OOM")
            return cls()

        def to(self, device):
            state["model_device"] = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            state["generate_calls"] = int(state["generate_calls"]) + 1
            state["generate_kwargs"] = kwargs
            return OutputTensor()

    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.bfloat16 = object()
    torch.manual_seed = lambda seed: state["torch_seeds"].append(seed)
    torch.inference_mode = contextlib.nullcontext
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps_available)
    )

    def current_allocated_memory():
        index = int(state["mps_sample_calls"])
        state["mps_sample_calls"] = index + 1
        return mps_samples[min(index, len(mps_samples) - 1)]

    torch.mps = types.SimpleNamespace(
        manual_seed=lambda seed: state["mps_seeds"].append(seed),
        synchronize=lambda: None,
        current_allocated_memory=current_allocated_memory,
    )
    numpy = types.ModuleType("numpy")
    numpy.random = types.SimpleNamespace(
        seed=lambda seed: state["numpy_seeds"].append(seed)
    )

    modules = {
        "numpy": numpy,
        "torch": torch,
        "vibevoice": types.ModuleType("vibevoice"),
        "vibevoice.modular": types.ModuleType("vibevoice.modular"),
        "vibevoice.modular.modeling_vibevoice_asr": types.ModuleType(
            "vibevoice.modular.modeling_vibevoice_asr"
        ),
        "vibevoice.processor": types.ModuleType("vibevoice.processor"),
        "vibevoice.processor.vibevoice_asr_processor": types.ModuleType(
            "vibevoice.processor.vibevoice_asr_processor"
        ),
    }
    modules[
        "vibevoice.modular.modeling_vibevoice_asr"
    ].VibeVoiceASRForConditionalGeneration = Model
    modules[
        "vibevoice.processor.vibevoice_asr_processor"
    ].VibeVoiceASRProcessor = Processor
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def _stage_request(tmp_path: Path) -> tuple[Path, Path, dict]:
    checkout = tmp_path / "checkout"
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    for directory in (checkout, model, tokenizer):
        directory.mkdir()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wav")
    request = {
        "checkout": str(checkout),
        "model": str(model),
        "tokenizer": str(tokenizer),
        "audio": str(audio),
        "config": {
            "device": "mps",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "seed": 1234,
            "max_new_tokens": 16384,
        },
    }
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path, result_path, request


def test_vibevoice_stage_uses_one_seeded_offline_whole_media_call(
    tmp_path, monkeypatch
) -> None:
    segments = [{"start_time": 0, "end_time": 1, "speaker_id": 0, "text": "Hi"}]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text='assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]',
        segments=segments,
        generated_tokens=3,
        eos_positions=[2],
    )
    request_path, result_path, request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 0
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["segments"] == segments
    assert output["generated_tokens"] == 3
    assert output["eos_observed"] is True
    assert output["hit_max_new_tokens"] is False
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 1
    assert state["torch_seeds"] == [1234]
    assert state["mps_seeds"] == [1234]
    assert state["numpy_seeds"] == [1234]
    assert state["processor_load"] == (
        request["model"],
        {
            "language_model_pretrained_name": request["tokenizer"],
            "local_files_only": True,
        },
    )
    generate = state["generate_kwargs"]
    assert generate["max_new_tokens"] == 16384
    assert generate["do_sample"] is False
    assert generate["input_ids"] is not None
    assert state["processor_call"]["audio"] == [request["audio"]]
    assert state["mps_sample_calls"] >= 3
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        assert __import__("os").environ[name] == "1"


def test_vibevoice_stage_sampler_start_failure_is_nonfatal(
    tmp_path, monkeypatch
) -> None:
    segments = [{"start_time": 0, "end_time": 1, "speaker_id": 0, "text": "Hi"}]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text='assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]',
        segments=segments,
        generated_tokens=3,
        eos_positions=[2],
    )
    state["sampler_join_calls"] = 0

    class UnstartableThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("synthetic sampler start failure")

        def is_alive(self):
            return False

        def join(self):
            state["sampler_join_calls"] = int(state["sampler_join_calls"]) + 1

    monkeypatch.setattr(vibevoice_stage.threading, "Thread", UnstartableThread)
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 0
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["segments"] == segments
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 1
    assert state["sampler_join_calls"] == 0
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )


def test_vibevoice_stage_sampler_join_failure_is_nonfatal(
    tmp_path, monkeypatch
) -> None:
    segments = [{"start_time": 0, "end_time": 1, "speaker_id": 0, "text": "Hi"}]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text='assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]',
        segments=segments,
        generated_tokens=3,
        eos_positions=[2],
    )
    state["sampler_join_calls"] = 0

    class UnjoinableThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def join(self):
            state["sampler_join_calls"] = int(state["sampler_join_calls"]) + 1
            raise RuntimeError("synthetic sampler join failure")

    monkeypatch.setattr(vibevoice_stage.threading, "Thread", UnjoinableThread)
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 0
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["segments"] == segments
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 1
    assert state["sampler_join_calls"] == 1
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )


def test_vibevoice_stage_unexpected_sampler_stop_failure_keeps_result_envelope(
    tmp_path, monkeypatch
) -> None:
    segments = [{"start_time": 0, "end_time": 1, "speaker_id": 0, "text": "Hi"}]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text='assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]',
        segments=segments,
        generated_tokens=3,
        eos_positions=[2],
    )
    monkeypatch.setattr(vibevoice_stage._MpsHighWater, "start", lambda self: None)

    def fail_stop(_self):
        raise RuntimeError("synthetic sampler stop failure")

    monkeypatch.setattr(vibevoice_stage._MpsHighWater, "stop", fail_stop)
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 0
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["segments"] == segments
    assert "peak_mps_live_bytes" not in output["metrics"]
    assert state["generate_calls"] == 1


def test_vibevoice_stage_reports_the_recorded_generation_cap_as_exit_four(
    tmp_path, monkeypatch
) -> None:
    raw = _fixture("vibevoice_30m_raw_prefix_excerpt.json")["raw_text"]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text=raw,
        segments=[],
        generated_tokens=16384,
        eos_positions=[],
    )
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 4
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["hit_max_new_tokens"] is True
    assert output["eos_observed"] is False
    assert output["raw_text"] == raw
    assert output["segments"] == []
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 1


def test_vibevoice_model_load_failure_has_no_partial_result(tmp_path, monkeypatch) -> None:
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text="",
        segments=[],
        generated_tokens=0,
        eos_positions=[],
        fail_load=True,
    )
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 1
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["error"]["type"] == "RuntimeError"
    assert "OOM" in output["error"]["message"]
    assert "raw_text" not in output
    assert "hit_max_new_tokens" not in output
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 0
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )


def test_vibevoice_stage_omits_mps_peak_when_mps_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text="",
        segments=[],
        generated_tokens=0,
        eos_positions=[],
        mps_available=False,
    )
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 1
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["error"]["type"] == "RuntimeError"
    assert "peak_mps_live_bytes" not in output["metrics"]
    assert state["mps_sample_calls"] == 0
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )


def test_vibevoice_stage_is_environment_owned_and_imports_no_core_package() -> None:
    source = Path(vibevoice_stage.__file__).read_text(encoding="utf-8")
    assert "import audio_cli" not in source
    assert "from audio_cli" not in source
