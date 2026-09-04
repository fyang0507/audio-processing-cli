"""Compatible partial-result merge and captured-input identity."""

from __future__ import annotations

from pathlib import Path

from export_test_support import (
    IncompatibleResultsError,
    _payload,
    _timed_segment,
    _write,
    json,
    load_result_document,
    merge_documents,
    os,
    pytest,
)


def test_merge_partial_and_resume_preserves_source_time_and_reids(tmp_path: Path) -> None:
    partial_coverage = {
        "covered_through_seconds": 1.0,
        "covered_fraction": 0.5,
        "covered_intervals": [[0.0, 1.0]],
        "missing_intervals": [[1.0, 2.0]],
        "scope_intervals": [[0.0, 2.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    partial = _payload(
        [_timed_segment("First.", [("First", 0.1, 0.8)])],
        outcomes={"word_timestamps": "produced"},
        complete=False,
        coverage=partial_coverage,
    )
    resumed = _payload(
        [_timed_segment("Second.", [("Second", 1.1, 1.8)])],
        outcomes={"word_timestamps": "produced"},
        run_range=[1.0, 2.0],
    )
    first_path = _write(tmp_path / "partial.json", partial)
    second_path = _write(tmp_path / "rest.json", resumed)
    merged = merge_documents([load_result_document(first_path), load_result_document(second_path)])

    assert [segment["segment_id"] for segment in merged.segments] == ["seg_0", "seg_1"]
    assert [word["word_id"] for segment in merged.segments for word in segment["words"]] == [
        "w_0",
        "w_1",
    ]
    assert [segment["words"][0]["start"] for segment in merged.segments] == [0.1, 1.1]


def test_merge_refuses_independent_vibevoice_native_speaker_labels(
    tmp_path: Path,
) -> None:
    left = _payload(
        [
            {
                "segment_id": "seg_0",
                "text": "Left.",
                "start": 0.1,
                "end": 0.8,
                "speaker": "Speaker 0",
            }
        ],
        outcomes={"diarization": "produced", "segment_timestamps": "produced"},
        run_range=[0.0, 1.0],
        stack="vibevoice",
    )
    right = _payload(
        [
            {
                "segment_id": "seg_0",
                "text": "Right.",
                "start": 1.1,
                "end": 1.8,
                "speaker": "Speaker 0",
            }
        ],
        outcomes={"diarization": "produced", "segment_timestamps": "produced"},
        run_range=[1.0, 2.0],
        stack="vibevoice",
    )

    with pytest.raises(
        IncompatibleResultsError,
        match="native speaker labels are local to each independent generation",
    ):
        merge_documents(
            [
                load_result_document(_write(tmp_path / "left-vibe.json", left)),
                load_result_document(_write(tmp_path / "right-vibe.json", right)),
            ]
        )


def test_merge_accepts_nondiarized_vibevoice_ranged_documents(tmp_path: Path) -> None:
    left = _payload(
        [{"segment_id": "seg_0", "text": "Left.", "start": 0.1, "end": 0.8}],
        outcomes={"segment_timestamps": "produced"},
        run_range=[0.0, 1.0],
        stack="vibevoice",
    )
    right = _payload(
        [{"segment_id": "seg_0", "text": "Right.", "start": 1.1, "end": 1.8}],
        outcomes={"segment_timestamps": "produced"},
        run_range=[1.0, 2.0],
        stack="vibevoice",
    )

    merged = merge_documents(
        [
            load_result_document(_write(tmp_path / "left-vibe.json", left)),
            load_result_document(_write(tmp_path / "right-vibe.json", right)),
        ]
    )

    assert [segment["text"] for segment in merged.segments] == ["Left.", "Right."]


def test_merge_plan_comparison_distinguishes_json_booleans_from_numbers(
    tmp_path: Path,
) -> None:
    left = _payload([], run_range=[0.0, 1.0])
    right = _payload([], run_range=[1.0, 2.0])
    left["provenance"]["plan"]["flag"] = False
    right["provenance"]["plan"]["flag"] = 0

    with pytest.raises(IncompatibleResultsError, match="executed plans differ"):
        merge_documents(
            [
                load_result_document(_write(tmp_path / "left.json", left)),
                load_result_document(_write(tmp_path / "right.json", right)),
            ]
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_merge_rejects_duplicate_or_reversed_timeline_inputs(tmp_path: Path, reverse: bool) -> None:
    left = _payload(
        [_timed_segment("Left.", [("Left", 0.1, 0.8)])],
        outcomes={"word_timestamps": "produced"},
        run_range=[0.0, 1.2],
    )
    right = _payload(
        [_timed_segment("Right.", [("Right", 1.0, 1.8)])],
        outcomes={"word_timestamps": "produced"},
        run_range=[1.0, 2.0],
    )
    paths = [
        _write(tmp_path / "left.json", left),
        _write(tmp_path / "right.json", right),
    ]
    if reverse:
        paths.reverse()
    with pytest.raises(IncompatibleResultsError, match="overlaps or precedes"):
        merge_documents([load_result_document(path) for path in paths])


def test_merge_rejects_a_captured_input_after_its_paths_are_rebound(
    tmp_path: Path,
) -> None:
    left = _payload(
        [{"segment_id": "seg_0", "text": "Left.", "start": 0.0, "end": 1.0}],
        outcomes={"segment_timestamps": "produced"},
        run_range=[0.0, 1.0],
    )
    right = _payload(
        [{"segment_id": "seg_0", "text": "Right.", "start": 1.0, "end": 2.0}],
        outcomes={"segment_timestamps": "produced"},
        run_range=[1.0, 2.0],
    )
    first_path = _write(tmp_path / "first.json", left)
    second_path = tmp_path / "second.json"
    os.link(first_path, second_path)

    first = load_result_document(first_path)
    second_path.write_text(json.dumps(right), encoding="utf-8")
    second = load_result_document(second_path)
    assert first.file_identity is not None and second.file_identity is not None
    assert (
        first.file_identity.device,
        first.file_identity.inode,
    ) == (
        second.file_identity.device,
        second.file_identity.inode,
    )

    second_path.unlink()
    second_path.write_text(json.dumps(right), encoding="utf-8")
    assert not first_path.samefile(second_path)

    with pytest.raises(IncompatibleResultsError, match="same input document"):
        merge_documents([first, second])


def test_merge_rejects_source_stack_capability_and_plan_drift(tmp_path: Path) -> None:
    base = _payload([{"segment_id": "seg_0", "text": "One."}], run_range=[0.0, 1.0])
    variants = []
    source = _payload(
        [{"segment_id": "seg_0", "text": "Two."}],
        source_path="other.wav",
        run_range=[1.0, 2.0],
    )
    variants.append(source)
    stack = _payload([{"segment_id": "seg_0", "text": "Two."}], run_range=[1.0, 2.0])
    stack["provenance"]["stack"] = "qwen-0.6b"
    variants.append(stack)
    capability = _payload(
        [{"segment_id": "seg_0", "text": "Two."}],
        outcomes={"languages": "produced"},
        run_range=[1.0, 2.0],
    )
    variants.append(capability)
    plan = _payload([{"segment_id": "seg_0", "text": "Two."}], run_range=[1.0, 2.0])
    plan["provenance"]["plan"]["execution"]["partition"] = "different"
    variants.append(plan)

    first = load_result_document(_write(tmp_path / "base.json", base))
    for index, variant in enumerate(variants):
        second = load_result_document(_write(tmp_path / f"variant-{index}.json", variant))
        with pytest.raises(IncompatibleResultsError):
            merge_documents([first, second])
