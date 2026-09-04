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


__all__ = [
    "Path",
    "ROOT",
    "_PINNED_POST_PROCESS_TRANSCRIPTION",
    "_fixture",
    "_install_fake_vibevoice",
    "_stage_request",
    "ast",
    "hashlib",
    "json",
    "normalize_vibevoice_result",
    "pytest",
    "sys",
    "textwrap",
    "threading",
    "types",
    "typing",
    "vibevoice_stage",
]
