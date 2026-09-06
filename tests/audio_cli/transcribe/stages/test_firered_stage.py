from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.audio_cli.transcribe.transcribe_firered_test_support import (
    _install_fake_firered,
    _run_stage,
    _stage_request,
    normalize_firered_result,
)


def test_firered_stage_loads_once_with_exact_aed_config_and_native_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {}
    _install_fake_firered(monkeypatch, record)
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=None, range_start=1.5, range_end=2.5),
    )
    assert code == 0
    assert result["complete"] is True
    assert record["load_flags"] == {"vad": True, "lid": False, "punc": True}
    assert record["configs"]["asr"] == {
        "use_gpu": False,
        "use_half": False,
        "beam_size": 3,
        "nbest": 1,
        "decode_max_len": 0,
        "softmax_smoothing": 1.25,
        "aed_length_penalty": 0.6,
        "eos_penalty": 1.0,
        "return_timestamp": True,
    }
    assert record["configs"]["system"]["asr_batch_size"] == 4
    assert record["configs"]["system"]["punc_batch_size"] == 4
    assert record["configs"]["system"]["vad_model_dir"] == "/models/vad/VAD"
    assert record["system_loads"] == 1
    assert record.get("process_calls", 0) == 0
    assert record["asr_batch_sizes"] == [1]
    assert record["punc_batch_sizes"] == [1]
    assert not any(call[0] == "lid" for call in record["calls"])
    assert result["regions"] == [
        {"region_id": "vad_1", "start": 2.0, "end": 3.0, "processed": True}
    ]
    assert result["result"]["vad_segments_ms"] == [[2000, 3000]]
    assert "wav_path" not in result["result"]
    assert set(result["metrics"]["stage_wall_seconds"]) == {"vad", "asr", "punctuator"}


def test_firered_stage_supplied_vad_is_a_real_substitution_and_lid_is_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {}
    _install_fake_firered(monkeypatch, record)
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(
            lid_enabled=True,
            vad_regions=[
                {"region_id": "external_0", "start": 0.0, "end": 1.0},
                {"region_id": "external_1", "start": 2.0, "end": 3.0},
            ],
            range_start=1.5,
            range_end=2.5,
        ),
    )
    assert code == 0
    assert result["complete"] is True
    assert record["load_flags"] == {"vad": False, "lid": True, "punc": True}
    assert any(call[0] == "lid" for call in record["calls"])
    assert result["regions"] == [
        {
            "region_id": "external_1",
            "start": 2.0,
            "end": 3.0,
            "processed": True,
        }
    ]
    assert result["result"]["vad_segments_ms"] == [[2000, 3000]]
    assert set(result["metrics"]["stage_wall_seconds"]) == {"asr", "lid", "punctuator"}


def test_firered_stage_empty_supplied_vad_does_not_load_native_vad_or_run_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {}
    _install_fake_firered(monkeypatch, record)
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=[]),
    )
    assert code == 0
    assert result["complete"] is True
    assert record["load_flags"] == {"vad": False, "lid": False, "punc": True}
    assert record.get("asr_calls", 0) == 0
    assert record.get("punc_calls", 0) == 0
    assert result["regions"] == []
    assert result["result"] == {
        "sentences": [],
        "words": [],
        "vad_segments_ms": [],
    }


def test_firered_stage_never_claims_partial_without_a_completed_region_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {"fail_asr_call": 1}
    _install_fake_firered(monkeypatch, record)
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(
            vad_regions=[
                {
                    "region_id": "external_0",
                    "start": 0.0,
                    "end": 1.0,
                }
            ]
        ),
    )
    assert code == 1
    assert result["complete"] is False
    assert result["result"] == {
        "sentences": [],
        "words": [],
        "vad_segments_ms": [],
    }
    assert result["regions"] == [
        {
            "region_id": "external_0",
            "start": 0.0,
            "end": 1.0,
            "processed": False,
        }
    ]
    assert result["error"]["message"] == "synthetic FireRed failure"


def test_firered_stage_salvages_completed_region_prefix_after_later_batch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {"fail_asr_call": 2}
    _install_fake_firered(monkeypatch, record)
    regions = [
        {"region_id": f"external_{index}", "start": float(index * 2), "end": float(index * 2 + 1)}
        for index in range(5)
    ]
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=regions, lid_enabled=True),
    )
    assert code == 4
    assert result["complete"] is False
    assert record["system_loads"] == 1
    assert record.get("process_calls", 0) == 0
    assert record["asr_batch_sizes"] == [4, 1]
    assert record["lid_batch_sizes"] == [4]
    assert record["punc_batch_sizes"] == [4]
    assert [item["processed"] for item in result["regions"]] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert result["result"] == {
        "sentences": [
            {
                "start_ms": index * 2000,
                "end_ms": index * 2000 + 1000,
                "text": f"region{index * 2000}.",
                "asr_confidence": 0.9,
                "lang": "en",
                "lang_confidence": 0.9,
            }
            for index in range(4)
        ],
        "words": [
            {
                "start_ms": index * 2000 + 1,
                "end_ms": index * 2000 + 999,
                "text": f"region{index * 2000}",
            }
            for index in range(4)
        ],
        "vad_segments_ms": [
            [0, 1000],
            [2000, 3000],
            [4000, 5000],
            [6000, 7000],
        ],
    }
    partial = normalize_firered_result(result["result"], lid_enabled=True)
    assert partial.vad_regions == (
        {"start": 0.0, "end": 1.0},
        {"start": 2.0, "end": 3.0},
        {"start": 4.0, "end": 5.0},
        {"start": 6.0, "end": 7.0},
    )
    assert [segment["text"] for segment in partial.segments] == [
        "region0.",
        "region2000.",
        "region4000.",
        "region6000.",
    ]
    assert partial.lid_regions is not None
    assert [item["language"] for item in partial.lid_regions] == ["en"] * 4
    assert result["error"]["message"] == "synthetic FireRed failure"


def test_firered_stage_treats_later_lid_failure_data_as_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {"malformed_lid_call": 2}
    _install_fake_firered(monkeypatch, record)
    regions = [
        {
            "region_id": f"external_{index}",
            "start": float(index * 2),
            "end": float(index * 2 + 1),
        }
        for index in range(5)
    ]
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=regions, lid_enabled=True),
    )
    assert code == 4
    assert result["complete"] is False
    assert record["asr_batch_sizes"] == [4, 1]
    assert record["lid_batch_sizes"] == [4, 1]
    assert record["punc_batch_sizes"] == [4]
    assert [item["processed"] for item in result["regions"]] == [
        True,
        True,
        True,
        True,
        False,
    ]
    partial = normalize_firered_result(result["result"], lid_enabled=True)
    assert len(partial.segments) == 4
    assert partial.lid_regions is not None
    assert result["error"]["message"] == ("FireRed LID returned an empty region language label")


def test_firered_stage_keeps_one_global_post_filter_punctuation_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The pinned runner filters blank ASR items before punctuation and then batches
    # the complete filtered stream (fireredasr2system.py:84-114).  A blank in the
    # first four-region ASR batch must therefore let region five fill punc batch 1.
    record: dict[str, Any] = {"blank_starts": {2000}}
    _install_fake_firered(monkeypatch, record)
    regions = [
        {
            "region_id": f"external_{index}",
            "start": float(index * 2),
            "end": float(index * 2 + 1),
        }
        for index in range(5)
    ]
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=regions),
    )
    assert code == 0
    assert result["complete"] is True
    assert record["asr_batch_sizes"] == [4, 1]
    assert record["punc_batch_sizes"] == [4]
    assert [sentence["text"] for sentence in result["result"]["sentences"]] == [
        "region0.",
        "region4000.",
        "region6000.",
        "region8000.",
    ]
    assert len(result["result"]["vad_segments_ms"]) == 5
    normalized = normalize_firered_result(result["result"], lid_enabled=False)
    assert len(normalized.segments) == 4
    assert len(normalized.vad_regions) == 5


def test_firered_blank_with_lid_salvages_last_publishable_region_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pinned process() filters the paired LID result but retains the blank VAD
    # region. That cannot become one public LID label per VAD region, so the stage
    # stops at the earlier conforming unit instead of returning raw success that
    # the adapter would reject wholesale.
    record: dict[str, Any] = {"blank_starts": {2000}}
    _install_fake_firered(monkeypatch, record)
    regions = [
        {
            "region_id": f"external_{index}",
            "start": float(index * 2),
            "end": float(index * 2 + 1),
        }
        for index in range(5)
    ]
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=regions, lid_enabled=True),
    )
    assert code == 4
    assert result["complete"] is False
    assert [item["processed"] for item in result["regions"]] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert result["result"]["vad_segments_ms"] == [[0, 1000]]
    normalized = normalize_firered_result(result["result"], lid_enabled=True)
    assert [segment["text"] for segment in normalized.segments] == ["region0."]
    assert normalized.lid_regions == (
        {
            "start": 0.0,
            "end": 1.0,
            "language": "en",
            "confidence": 0.9,
        },
    )
    assert result["error"]["message"] == (
        "FireRed cannot publish region LID for a blank ASR region"
    )
