"""Shipped transcription run shapes against the happy-path specification."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
from shipped_command_test_support import (
    _native_interpreter,
    _native_package_entry,
    assert_documented_shape,
    configure_isolated_root,
    documented_block,
)

from audio_cli.transcribe.catalog import InputMetadata
from audio_cli.transcribe.orchestrator import run
from audio_cli.transcribe.plan import serialize_plan
from audio_cli.transcribe.planner import build_plan, resolve_request
from audio_cli.transcribe.transport import StageOutcome


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    configure_isolated_root(tmp_path, monkeypatch)


def test_vibevoice_run_emits_the_shape_happy_path_publishes(tmp_path) -> None:
    _native_interpreter(tmp_path, "torch-vibevoice")
    _native_interpreter(tmp_path, "mlx")

    class Transport:
        def decode(self, source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * round(112.4 * 16_000))
            return StageOutcome("decode", "ffmpeg", {}, 0.44)

        def vibevoice(self, **kwargs):
            segments = [
                {
                    "start_time": 0.0,
                    "end_time": 4.52,
                    "speaker_id": 0,
                    "text": "So, um, this is the new editor.",
                },
                {"start_time": 4.52, "end_time": 6.08, "text": "[Environmental Sounds]"},
                {
                    "start_time": 6.08,
                    "end_time": 9.41,
                    "speaker_id": 1,
                    "text": "And it renders straight away?",
                },
            ]
            return StageOutcome(
                "asr",
                "vibevoice-asr-7b",
                {
                    "raw_text": json.dumps(segments),
                    "segments": segments,
                    "hit_max_new_tokens": False,
                    "generated_tokens": 100,
                    "eos_observed": True,
                },
                53.16,
                peak_mps_live_bytes=19_983_452_160,
            )

        def align(self, *, segments, **kwargs):
            return StageOutcome(
                "aligner",
                "qwen3-forcedaligner",
                {
                    "segments": [
                        {
                            "unit_id": segments[0]["unit_id"],
                            "words": [
                                {"text": "So", "start": 0.31, "end": 0.48},
                                {"text": "um", "start": 0.50, "end": 0.65},
                                {"text": "this", "start": 0.70, "end": 0.90},
                                {"text": "is", "start": 0.95, "end": 1.05},
                                {"text": "the", "start": 1.10, "end": 1.25},
                                {"text": "new", "start": 1.30, "end": 1.55},
                                {"text": "editor", "start": 1.60, "end": 2.00},
                            ],
                        },
                        {
                            "unit_id": segments[1]["unit_id"],
                            "words": [
                                {"text": "And", "start": 6.22, "end": 6.39},
                                {"text": "it", "start": 6.40, "end": 6.50},
                                {"text": "renders", "start": 6.55, "end": 6.90},
                                {"text": "straight", "start": 6.95, "end": 7.30},
                                {"text": "away", "start": 7.35, "end": 7.70},
                            ],
                        },
                    ]
                },
                3.72,
                peak_mps_live_bytes=2_210_398_208,
            )

    entries = {
        "vibevoice-asr-7b": _native_package_entry(tmp_path, "vibevoice-asr-7b"),
        "qwen3-forcedaligner": _native_package_entry(tmp_path, "qwen3-forcedaligner"),
    }
    request = resolve_request(
        stack_id="vibevoice",
        input_path=Path("demo.mp4"),
        wants="verbatim,diarization,segment_timestamps,word_timestamps",
    )
    metadata = InputMetadata("demo.mp4", 112.4, "mp4", 48_000, 2)
    actual = run(
        request,
        metadata,
        registry={
            "environments": {
                "torch-vibevoice": {"state": "ready"},
                "mlx": {"state": "ready"},
            },
            "packages": entries,
        },
        transport=Transport(),
    ).payload
    expected_plan = serialize_plan(build_plan(request, metadata, provisioned_packages=set(entries)))
    expected_plan.pop("sample_output")
    documented = documented_block("audio transcribe run --input demo.mp4 --stack vibevoice \\\n")
    documented["provenance"]["plan"] = expected_plan
    assert_documented_shape(actual, documented, "audio transcribe run --stack vibevoice")
