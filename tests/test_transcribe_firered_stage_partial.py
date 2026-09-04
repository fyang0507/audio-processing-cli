from __future__ import annotations

from pathlib import Path
from typing import Any

from transcribe_firered_test_support import (
    ROOT,
    _install_fake_firered,
    _run_stage,
    _stage_request,
    firered_stage,
    normalize_firered_result,
    os,
    pytest,
)


def test_firered_stage_salvages_only_punctuated_prefix_after_later_punc_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {"fail_punc_call": 2}
    _install_fake_firered(monkeypatch, record)
    regions = [
        {
            "region_id": f"external_{index}",
            "start": float(index * 2),
            "end": float(index * 2 + 1),
        }
        for index in range(6)
    ]
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=regions),
    )
    assert code == 4
    assert result["complete"] is False
    assert record["system_loads"] == 1
    assert record["asr_batch_sizes"] == [4, 2]
    assert record["punc_batch_sizes"] == [4, 2]
    assert [item["processed"] for item in result["regions"]] == [
        True, True, True, True, False, False,
    ]
    assert [sentence["text"] for sentence in result["result"]["sentences"]] == [
        "region0.", "region2000.", "region4000.", "region6000.",
    ]
    assert result["result"]["vad_segments_ms"] == [
        [0, 1000], [2000, 3000], [4000, 5000], [6000, 7000],
    ]
    partial = normalize_firered_result(result["result"], lid_enabled=False)
    assert len(partial.segments) == 4
    assert result["error"]["message"] == "synthetic FireRed failure"


def test_firered_stage_salvages_only_formatted_prefix_after_later_format_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {"malformed_punc_call": 2}
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
    assert code == 4
    assert result["complete"] is False
    assert record["punc_batch_sizes"] == [4, 1]
    assert [item["processed"] for item in result["regions"]] == [
        True, True, True, True, False,
    ]
    partial = normalize_firered_result(result["result"], lid_enabled=False)
    assert len(partial.segments) == 4
    assert result["error"]["message"] == (
        "FireRed punctuation sentence text must be a string"
    )


@pytest.mark.parametrize(
    ("record", "detail"),
    [
        ({"mismatched_punc_starts": {8000}}, "do not reproduce"),
        ({"out_of_region_word_starts": {8000}}, "stay within its VAD region"),
    ],
)
def test_firered_stage_salvages_prefix_before_late_semantic_format_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, Any],
    detail: str,
) -> None:
    # The fifth formatter result is structurally complete but its punctuation text
    # adds a word that does not exist in the ASR timestamp stream.  Detecting this
    # only in the core adapter would discard the four valid earlier regions.
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

    assert code == 4
    assert result["complete"] is False
    assert record["punc_batch_sizes"] == [4, 1]
    assert [item["processed"] for item in result["regions"]] == [
        True, True, True, True, False,
    ]
    assert result["result"]["vad_segments_ms"] == [
        [0, 1000], [2000, 3000], [4000, 5000], [6000, 7000],
    ]
    partial = normalize_firered_result(result["result"], lid_enabled=False)
    assert [segment["text"] for segment in partial.segments] == [
        "region0.", "region2000.", "region4000.", "region6000.",
    ]
    assert detail in result["error"]["message"]


def test_firered_stage_rejects_zero_duration_vad_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {}
    _install_fake_firered(monkeypatch, record)
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=[{
            "region_id": "external_0", "start": 0.5, "end": 0.5,
        }]),
    )

    assert code == 1
    assert result["complete"] is False
    assert result["regions"] == []
    assert record.get("system_loads", 0) == 0
    assert result["error"]["message"] == "vad_regions[0] must satisfy start < end"


def test_firered_stage_semantics_use_published_millisecond_region_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {}
    _install_fake_firered(monkeypatch, record)
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=[{
            "region_id": "external_0", "start": 0.2009, "end": 0.9999,
        }]),
    )

    assert code == 0
    assert result["result"]["vad_segments_ms"] == [[200, 999]]
    normalized = normalize_firered_result(result["result"], lid_enabled=False)
    assert normalized.vad_regions == ({"start": 0.2, "end": 0.999},)


def test_firered_stage_range_selection_uses_published_millisecond_bounds() -> None:
    regions = [
        {"region_id": "prior", "start": 0.0009, "end": 1.0008},
        {"region_id": "next", "start": 1.0009, "end": 2.0},
    ]

    assert firered_stage._region_records(
        regions, range_start=1.0, range_end=None
    ) == [regions[1]]


def test_firered_stage_rejects_region_collapsed_on_public_timeline() -> None:
    with pytest.raises(ValueError, match="empty at FireRed's millisecond precision"):
        firered_stage._region_records(
            [{"region_id": "sub_ms", "start": 0.0001, "end": 0.0009}],
            range_start=0.0,
            range_end=None,
        )


def test_firered_stage_is_offline_and_imports_no_core_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {}
    _install_fake_firered(monkeypatch, record)
    code, _result = _run_stage(tmp_path, monkeypatch, _stage_request())
    assert code == 0
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    source = (ROOT / "src/audio_cli/transcribe/stages/firered.py").read_text(
        encoding="utf-8"
    )
    assert "import audio_cli" not in source
    assert "from audio_cli" not in source
