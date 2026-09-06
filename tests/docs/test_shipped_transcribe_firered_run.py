"""Shipped transcription run shapes against the happy-path specification."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from audio_cli.transcribe.catalog import InputMetadata
from audio_cli.transcribe.orchestrator import run
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.transport import StageOutcome
from tests.docs.shipped_command_test_support import (
    _native_interpreter,
    _native_package_entry,
    assert_documented_shape,
    configure_isolated_root,
    documented_block,
)


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    configure_isolated_root(tmp_path, monkeypatch)


def test_firered_run_emits_the_shape_happy_path_publishes(tmp_path) -> None:
    _native_interpreter(tmp_path, "torch-firered")

    class Transport:
        def decode(self, source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * round(27.8 * 16_000))
            return StageOutcome("decode", "ffmpeg", {}, 0.09)

        def firered(self, **kwargs):
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 380,
                                "end_ms": 1620,
                                "text": "This is a测试。",
                                "lang": "en",
                                "lang_confidence": 0.724,
                            },
                            {
                                "start_ms": 3280,
                                "end_ms": 4490,
                                "text": "我们要来看哈，",
                                "lang": "zh",
                                "lang_confidence": 0.961,
                            },
                        ],
                        "words": [
                            {"start_ms": 410, "end_ms": 620, "text": "this"},
                            {"start_ms": 620, "end_ms": 740, "text": "is"},
                            {"start_ms": 740, "end_ms": 810, "text": "a"},
                            {"start_ms": 1020, "end_ms": 1240, "text": "测"},
                            {"start_ms": 1240, "end_ms": 1480, "text": "试"},
                            {"start_ms": 3310, "end_ms": 3440, "text": "我"},
                            {"start_ms": 3440, "end_ms": 3580, "text": "们"},
                            {"start_ms": 3580, "end_ms": 3720, "text": "要"},
                            {"start_ms": 3720, "end_ms": 3860, "text": "来"},
                            {"start_ms": 3860, "end_ms": 4030, "text": "看"},
                            {"start_ms": 4030, "end_ms": 4290, "text": "哈"},
                        ],
                        "vad_segments_ms": [[380, 1660], [3280, 4520]],
                    },
                    "regions": [
                        {"region_id": "vad_0", "start": 0.38, "end": 1.66, "processed": True},
                        {"region_id": "vad_1", "start": 3.28, "end": 4.52, "processed": True},
                    ],
                },
                20.65,
                peak_rss_bytes=13_169_377_280,
                wall_seconds_by_stage={
                    "vad": 0.61,
                    "lid": 8.83,
                    "asr": 9.14,
                    "punctuator": 1.07,
                },
            )

    entries = {"firered-asr2s": _native_package_entry(tmp_path, "firered-asr2s")}
    request = resolve_request(
        stack_id="firered",
        input_path=Path("field.wav"),
        wants="verbatim,word_timestamps,vad,segment_timestamps,lid",
    )
    metadata = InputMetadata("field.wav", 27.8, "wav", 48_000, 1)
    actual = run(
        request,
        metadata,
        registry={
            "environments": {"torch-firered": {"state": "ready"}},
            "packages": entries,
        },
        transport=Transport(),
    ).payload
    expected_plan = serialize_plan(build_plan(request, metadata, provisioned_packages=set(entries)))
    expected_plan.pop("sample_output")
    documented = documented_block("audio transcribe run --input field.wav --stack firered \\\n")
    documented["provenance"]["plan"] = expected_plan
    assert_documented_shape(actual, documented, "audio transcribe run --stack firered")
