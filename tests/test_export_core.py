from __future__ import annotations

import json
import os
import pwd
from pathlib import Path

import pytest

from audio_cli import media as media_module
from audio_cli.export import (
    IncompatibleResultsError,
    InvalidResultError,
    OutputExistsError,
    OutputWriteError,
    TimingRequiredError,
    UnsafeOutputError,
    export_documents,
    load_result_document,
    merge_documents,
)
from audio_cli.export.cues import Cue, CueError, build_cues
from audio_cli.export.writers import (
    render_jsonl,
    render_markdown,
    render_srt,
    render_text,
    render_vtt,
    write_text_atomic,
)
from audio_cli.transcribe.catalog import InputMetadata, result_source
from audio_cli.transcribe.result import ABSENT, NormalizedResult, serialize_result


def _payload(
    segments: list[dict],
    *,
    source_path: str = "source.wav",
    duration: float = 2.0,
    outcomes: dict[str, str] | None = None,
    complete: bool = True,
    coverage: dict | object = ABSENT,
    run_range: list[float] | None = None,
    stack: str = "qwen-1.7b",
) -> dict:
    resolved_outcomes = outcomes or {}
    execution: dict = {"partition": "fixture"}
    if run_range is not None:
        execution["range"] = {
            "requested": list(run_range),
            "selected_unit_scope": list(run_range),
        }
    requested = frozenset(resolved_outcomes)
    result = NormalizedResult(
        source={
            "path": source_path,
            "duration_seconds": duration,
            "timebase": "seconds",
        },
        segments=segments,
        abstentions=[],
        provenance={
            "stack": stack,
            "outcomes": resolved_outcomes,
            "observed": {},
            "plan": {"execution": execution},
        },
        requested_capabilities=requested,
        complete=complete,
        coverage=coverage,
        turns=[] if "diarization" in requested else ABSENT,
        vad_regions=[] if "vad" in requested else ABSENT,
        lid_regions=[] if "lid" in requested else ABSENT,
        overlapped_speech=[] if "overlapped_speech" in requested else ABSENT,
    )
    return serialize_result(result)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_export_writer_closes_descriptor_when_temporary_identity_capture_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "transcript.txt"
    captured_descriptor: int | None = None

    def fail_identity(descriptor: int, _path: Path):
        nonlocal captured_descriptor
        captured_descriptor = descriptor
        raise OSError("identity unavailable")

    monkeypatch.setattr(
        "audio_cli.export.writers.file_identity_from_descriptor",
        fail_identity,
    )

    with pytest.raises(OutputWriteError, match="identity unavailable"):
        write_text_atomic(output, "hello\n")

    assert captured_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(captured_descriptor)
    assert not output.exists()


def _timed_segment(
    text: str,
    words: list[tuple[str, float, float]],
    *,
    segment_id: str = "seg_0",
    speaker: str | None = None,
) -> dict:
    segment = {
        "segment_id": segment_id,
        "text": text,
        "words": [
            {"word_id": f"w_{index}", "text": word, "start": start, "end": end}
            for index, (word, start, end) in enumerate(words)
        ],
    }
    if speaker is not None:
        segment["speaker"] = speaker
    return segment


def test_loader_accepts_exact_result_and_rejects_schema_drift(tmp_path: Path) -> None:
    payload = _payload([{"segment_id": "seg_0", "text": "Hello."}])
    valid = _write(tmp_path / "valid.json", payload)
    loaded = load_result_document(valid)
    assert loaded.payload == payload
    assert loaded.owned_intervals == ((0.0, 2.0),)

    payload["invented"] = True
    invalid = _write(tmp_path / "invalid.json", payload)
    with pytest.raises(InvalidResultError, match="exact current normalized result shape"):
        load_result_document(invalid)

    duplicate_keys = tmp_path / "duplicate-keys.json"
    duplicate_keys.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(InvalidResultError, match="duplicate JSON object key"):
        load_result_document(duplicate_keys)

    surrogate = _payload([{"segment_id": "seg_0", "text": "\ud800"}])
    surrogate_path = tmp_path / "surrogate.json"
    surrogate_path.write_text(json.dumps(surrogate), encoding="utf-8")
    with pytest.raises(InvalidResultError, match="surrogates not allowed"):
        load_result_document(surrogate_path)

    huge = _payload([], duration=0.0)
    huge["source"]["duration_seconds"] = 10**400
    with pytest.raises(InvalidResultError, match="finite number"):
        load_result_document(_write(tmp_path / "huge-number.json", huge))

    huge_range = _payload([], run_range=[0.0, 1.0])
    huge_range["provenance"]["plan"]["execution"]["range"]["requested"][1] = 10**400
    with pytest.raises(InvalidResultError, match="too large to convert"):
        load_result_document(_write(tmp_path / "huge-range.json", huge_range))

    deep = _payload([])
    deep["provenance"]["plan"] = {"deep": "__DEEP_PLAN__"}
    encoded = json.dumps(deep).replace(
        '"__DEEP_PLAN__"', "[" * 1_500 + "0" + "]" * 1_500
    )
    deep_path = tmp_path / "deep-plan.json"
    deep_path.write_text(encoded, encoding="utf-8")
    with pytest.raises(InvalidResultError, match="recursion depth"):
        load_result_document(deep_path)

    merge_deep = _payload([])
    merge_deep["provenance"]["plan"] = {"deep": "__MERGE_DEEP_PLAN__"}
    merge_encoded = json.dumps(merge_deep).replace(
        '"__MERGE_DEEP_PLAN__"', "[" * 500 + "0" + "]" * 500
    )
    merge_deep_path = tmp_path / "merge-deep-plan.json"
    merge_deep_path.write_text(merge_encoded, encoding="utf-8")
    with pytest.raises(InvalidResultError, match="recursion depth"):
        export_documents([merge_deep_path], "txt")


def test_loader_refuses_a_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    fifo = tmp_path / "stream.json"
    os.mkfifo(fifo)

    with pytest.raises(InvalidResultError, match="input must be a regular file"):
        load_result_document(fifo)

    empty = _payload([], duration=0.0)
    empty_path = _write(tmp_path / "empty.json", empty)
    assert export_documents([empty_path], "md").content == "# Transcript\n\n"


@pytest.mark.parametrize(
    ("word_start", "word_end"),
    [(0.998, 2.0), (2.0, 3.002)],
)
def test_loader_rejects_words_outside_explicit_segment_bounds(
    tmp_path: Path,
    word_start: float,
    word_end: float,
) -> None:
    payload = _payload(
        [{
            "segment_id": "seg_0",
            "text": "Hello.",
            "start": 1.0,
            "end": 3.0,
            "words": [{
                "word_id": "w_0",
                "text": "Hello",
                "start": 1.5,
                "end": 2.5,
            }],
        }],
        duration=4.0,
        outcomes={
            "segment_timestamps": "produced",
            "word_timestamps": "produced",
        },
    )
    payload["segments"][0]["words"][0].update({
        "start": word_start,
        "end": word_end,
    })
    transcript = _write(tmp_path / "contradictory-bounds.json", payload)

    with pytest.raises(
        InvalidResultError,
        match="words must fall inside the segment bounds",
    ):
        export_documents([transcript], "srt")


def test_export_preserves_the_recorded_one_millisecond_firered_end_seam(
    tmp_path: Path,
) -> None:
    payload = _payload(
        [{
            "segment_id": "seg_0",
            "text": "Hello.",
            "start": 1.0,
            "end": 3.0,
            "words": [{
                "word_id": "w_0",
                "text": "Hello",
                "start": 2.0,
                "end": 3.001,
            }],
        }],
        duration=4.0,
        outcomes={
            "segment_timestamps": "produced",
            "word_timestamps": "produced",
        },
    )
    transcript = _write(tmp_path / "firered-one-ms-seam.json", payload)

    product = export_documents([transcript], "srt")

    assert "00:00:02,000 --> 00:00:03,001" in product.content


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("requested", [False, True]),
        ("selected_unit_scope", ["0", "1"]),
    ],
)
def test_loader_rejects_non_numeric_opaque_plan_range_bounds(
    tmp_path: Path, field: str, values: list[object],
) -> None:
    payload = _payload([], run_range=[0.0, 1.0])
    payload["provenance"]["plan"]["execution"]["range"][field] = values

    with pytest.raises(InvalidResultError, match="must contain numbers"):
        load_result_document(_write(tmp_path / f"bad-{field}.json", payload))


def test_loader_validates_plan_range_even_when_partial_coverage_owns_merge(
    tmp_path: Path,
) -> None:
    coverage = {
        "covered_through_seconds": 1.0,
        "covered_fraction": 0.5,
        "covered_intervals": [[0.0, 1.0]],
        "missing_intervals": [[1.0, 2.0]],
        "scope_intervals": [[0.0, 2.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    payload = _payload(
        [],
        complete=False,
        coverage=coverage,
        run_range=[0.0, 2.0],
    )
    payload["provenance"]["plan"]["execution"]["range"]["requested"] = [
        -1.0,
        1.0,
    ]

    with pytest.raises(InvalidResultError, match="is not a source interval"):
        load_result_document(_write(tmp_path / "partial-bad-range.json", payload))


def test_loader_requires_partial_coverage_scope_to_match_selected_range(
    tmp_path: Path,
) -> None:
    coverage = {
        "covered_through_seconds": 1.5,
        "covered_fraction": 0.5,
        "covered_intervals": [[1.0, 1.5]],
        "missing_intervals": [[1.5, 2.0]],
        "scope_intervals": [[1.0, 2.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    payload = _payload(
        [],
        complete=False,
        coverage=coverage,
        run_range=[0.0, 1.0],
    )

    with pytest.raises(InvalidResultError, match="selected unit scope"):
        load_result_document(_write(tmp_path / "scope-mismatch.json", payload))


@pytest.mark.parametrize("partial", [False, True])
def test_loader_rejects_one_document_with_segments_outside_its_owned_scope(
    tmp_path: Path,
    partial: bool,
) -> None:
    coverage = {
        "covered_through_seconds": 1.0,
        "covered_fraction": 0.5,
        "covered_intervals": [[0.0, 1.0]],
        "missing_intervals": [[1.0, 2.0]],
        "scope_intervals": [[0.0, 2.0]],
        "units_total": 2,
        "units_completed": 1,
    }
    payload = _payload(
        [_timed_segment("Outside.", [("Outside", 1.1, 1.4)])],
        outcomes={"word_timestamps": "produced"},
        complete=not partial,
        coverage=coverage if partial else ABSENT,
        run_range=None if partial else [0.0, 1.0],
    )

    with pytest.raises(InvalidResultError, match="outside.*owned intervals"):
        load_result_document(_write(tmp_path / f"outside-{partial}.json", payload))


def test_loader_rejects_duplicate_ids_and_nonchronological_words(tmp_path: Path) -> None:
    duplicate = _payload([
        {"segment_id": "seg_0", "text": "One."},
        {"segment_id": "seg_0", "text": "Two."},
    ])
    with pytest.raises(InvalidResultError, match="duplicates"):
        load_result_document(_write(tmp_path / "duplicate.json", duplicate))

    overlapping = _payload(
        [_timed_segment("One two.", [("One", 0.2, 0.6), ("two", 0.5, 0.8)])],
        outcomes={"word_timestamps": "produced"},
    )
    with pytest.raises(InvalidResultError, match="overlaps the preceding word"):
        load_result_document(_write(tmp_path / "overlap.json", overlapping))


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
    merged = merge_documents([
        load_result_document(first_path), load_result_document(second_path)
    ])

    assert [segment["segment_id"] for segment in merged.segments] == ["seg_0", "seg_1"]
    assert [
        word["word_id"]
        for segment in merged.segments
        for word in segment["words"]
    ] == ["w_0", "w_1"]
    assert [segment["words"][0]["start"] for segment in merged.segments] == [0.1, 1.1]


def test_merge_refuses_independent_vibevoice_native_speaker_labels(
    tmp_path: Path,
) -> None:
    left = _payload(
        [{
            "segment_id": "seg_0",
            "text": "Left.",
            "start": 0.1,
            "end": 0.8,
            "speaker": "Speaker 0",
        }],
        outcomes={"diarization": "produced", "segment_timestamps": "produced"},
        run_range=[0.0, 1.0],
        stack="vibevoice",
    )
    right = _payload(
        [{
            "segment_id": "seg_0",
            "text": "Right.",
            "start": 1.1,
            "end": 1.8,
            "speaker": "Speaker 0",
        }],
        outcomes={"diarization": "produced", "segment_timestamps": "produced"},
        run_range=[1.0, 2.0],
        stack="vibevoice",
    )

    with pytest.raises(
        IncompatibleResultsError,
        match="native speaker labels are local to each independent generation",
    ):
        merge_documents([
            load_result_document(_write(tmp_path / "left-vibe.json", left)),
            load_result_document(_write(tmp_path / "right-vibe.json", right)),
        ])


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

    merged = merge_documents([
        load_result_document(_write(tmp_path / "left-vibe.json", left)),
        load_result_document(_write(tmp_path / "right-vibe.json", right)),
    ])

    assert [segment["text"] for segment in merged.segments] == ["Left.", "Right."]


def test_merge_plan_comparison_distinguishes_json_booleans_from_numbers(
    tmp_path: Path,
) -> None:
    left = _payload([], run_range=[0.0, 1.0])
    right = _payload([], run_range=[1.0, 2.0])
    left["provenance"]["plan"]["flag"] = False
    right["provenance"]["plan"]["flag"] = 0

    with pytest.raises(IncompatibleResultsError, match="executed plans differ"):
        merge_documents([
            load_result_document(_write(tmp_path / "left.json", left)),
            load_result_document(_write(tmp_path / "right.json", right)),
        ])


@pytest.mark.parametrize("reverse", [False, True])
def test_merge_rejects_duplicate_or_reversed_timeline_inputs(
    tmp_path: Path, reverse: bool
) -> None:
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


def test_timed_export_rejects_mixed_event_only_and_untimed_documents(
    tmp_path: Path,
) -> None:
    event = _payload(
        [{
            "segment_id": "seg_0",
            "text": "[Music]",
            "start": 0.0,
            "end": 0.8,
        }],
        outcomes={
            "word_timestamps": "produced",
            "segment_timestamps": "produced",
        },
        run_range=[0.0, 1.0],
    )
    ordinary = _payload(
        [{
            "segment_id": "seg_0",
            "text": "Ordinary untimed speech.",
            "start": 1.0,
            "end": 1.8,
        }],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
        run_range=[1.0, 2.0],
    )
    ordinary["abstentions"] = [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 1.0,
        "end": 1.8,
    }]
    event_path = _write(tmp_path / "event.json", event)
    ordinary_path = _write(tmp_path / "ordinary.json", ordinary)

    with pytest.raises(TimingRequiredError):
        export_documents([event_path, ordinary_path], "srt")


def test_timed_export_rejects_empty_result_with_only_a_produced_outcome(
    tmp_path: Path,
) -> None:
    payload = _payload([], outcomes={"word_timestamps": "produced"})
    path = _write(tmp_path / "empty-produced.json", payload)

    with pytest.raises(TimingRequiredError):
        export_documents([path], "vtt")


def test_cues_map_canonical_punctuation_and_use_only_word_bounds() -> None:
    segment = _timed_segment(
        "Hello, WORLD! Again.",
        [("hello", 0.1, 0.2), ("world", 0.25, 0.4), ("again", 0.6, 0.8)],
    )
    segment.update({"start": 0.0, "end": 2.0})
    built = build_cues([segment], duration=2.0)
    assert built.cues == (
        Cue(100, 400, "Hello, WORLD!"),
        Cue(600, 800, "Again."),
    )


@pytest.mark.parametrize("output_format", ["srt", "vtt"])
def test_timed_export_rejects_non_whitespace_control_characters(
    tmp_path: Path, output_format: str,
) -> None:
    payload = _payload(
        [_timed_segment("Hel\0lo.", [("Hel\0lo", 0.1, 0.8)])],
        outcomes={"word_timestamps": "produced"},
    )
    path = _write(tmp_path / "control.json", payload)

    with pytest.raises(InvalidResultError, match="control character"):
        export_documents([path], output_format)


def test_timed_export_rejects_words_that_split_one_casefolded_character(
    tmp_path: Path,
) -> None:
    payload = _payload(
        [_timed_segment("ß", [("s", 0.0, 0.1), ("s", 1.0, 1.1)])],
        outcomes={"word_timestamps": "produced"},
    )
    path = _write(tmp_path / "split-casefold.json", payload)

    with pytest.raises(InvalidResultError, match="case-fold expansion"):
        export_documents([path], "srt")


def test_cues_preserve_opening_wrappers_with_the_following_word() -> None:
    segment = _timed_segment(
        "He said, “Hello.” Then left.",
        [
            ("He", 0.0, 0.1),
            ("said", 0.1, 0.2),
            ("Hello", 0.2, 0.4),
            ("Then", 0.5, 0.6),
            ("left", 0.6, 0.8),
        ],
    )
    assert [cue.text for cue in build_cues([segment], duration=1.0).cues] == [
        "He said, “Hello.”",
        "Then left.",
    ]


def test_cues_assign_straight_opening_quote_to_the_following_timed_word() -> None:
    segment = _timed_segment(
        'A sufficiently long phrase, "Hello."',
        [
            ("A", 0.0, 0.1),
            ("sufficiently", 0.1, 0.3),
            ("long", 0.3, 0.4),
            ("phrase", 0.4, 0.6),
            ("Hello", 1.5, 2.0),
        ],
    )

    assert build_cues([segment], duration=2.0).cues == (
        Cue(0, 600, "A sufficiently long phrase,"),
        Cue(1500, 2000, '"Hello."'),
    )


def test_literal_unknown_account_tilde_input_is_not_expanded(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    literal_directory = Path("~codex-no-such-account")
    literal_directory.mkdir()
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    path = _write(
        literal_directory / "result.json",
        _payload(
            [{"segment_id": "seg_0", "text": "Literal path."}],
            source_path=str(source),
        ),
    )

    assert export_documents([path], "txt").content == "Literal path.\n"

    output = tmp_path / "literal-output.txt"
    export_documents([path], "txt", output=output)
    assert output.read_text(encoding="utf-8") == "Literal path.\n"


def test_cues_break_at_ellipsis_before_a_trailing_wrapper() -> None:
    segment = _timed_segment(
        "Wait…” Then.",
        [("Wait", 0.0, 0.4), ("Then", 0.45, 0.8)],
    )

    assert [cue.text for cue in build_cues([segment], duration=1.0).cues] == [
        "Wait…”",
        "Then.",
    ]


def test_cue_wrapping_keeps_boundaries_with_leading_whitespace() -> None:
    first = "A" * 25
    second = "B" * 25
    segment = _timed_segment(
        f"   {first} {second}",
        [(first, 0.1, 0.4), (second, 0.5, 0.8)],
    )
    built = build_cues([segment], duration=1.0)
    assert built.cues == (Cue(100, 800, f"{first}\n{second}"),)


def test_cues_drop_collapsed_ms_bounds_and_never_trim_real_overlap() -> None:
    collapsed = _timed_segment("A.", [("A", 0.0001, 0.0004)])
    built = build_cues([collapsed], duration=1.0)
    assert built.cues == ()
    assert [warning["code"] for warning in built.warnings] == [
        "cue_dropped_after_quantization"
    ]

    overlapping = [
        _timed_segment("First.", [("First", 0.1, 0.6)], segment_id="seg_0"),
        _timed_segment("Second.", [("Second", 0.5, 0.8)], segment_id="seg_1"),
    ]
    with pytest.raises(CueError, match="refusing to trim or nudge"):
        build_cues(overlapping, duration=1.0)

    submillisecond_overlap = [
        _timed_segment(
            "First.", [("First", 0.0994, 0.1004)], segment_id="seg_0"
        ),
        _timed_segment(
            "Second.", [("Second", 0.1003, 0.1014)], segment_id="seg_1"
        ),
    ]
    with pytest.raises(CueError, match="before millisecond quantization"):
        build_cues(submillisecond_overlap, duration=1.0)

    collapsed_then_surviving = [
        _timed_segment("A.", [("A", 0.0001, 0.0004)], segment_id="seg_0"),
        _timed_segment("B.", [("B", 0.0003, 0.0014)], segment_id="seg_1"),
    ]
    with pytest.raises(CueError, match="before millisecond quantization"):
        build_cues(collapsed_then_surviving, duration=1.0)


def test_export_subtitles_omit_event_segments_and_render_real_speakers(
    tmp_path: Path,
) -> None:
    speech = _timed_segment(
        "Hello there.",
        [("Hello", 0.31, 0.8), ("there", 0.9, 1.2)],
        speaker="Speaker 0",
    )
    event = {
        "segment_id": "seg_1",
        "start": 1.2,
        "end": 1.5,
        "text": "[Environmental Sounds]",
    }
    payload = _payload(
        [speech, event],
        outcomes={
            "word_timestamps": "produced",
            "diarization": "produced",
            "segment_timestamps": "produced",
        },
    )
    path = _write(tmp_path / "timed.json", payload)
    product = export_documents([path], "vtt")

    assert product.cue_count == 1
    assert product.speaker_labels_rendered is True
    assert "<v Speaker 0>Hello there." in product.content
    assert "Environmental" not in product.content
    assert product.cues[0].start_ms == 310
    assert product.cues[0].end_ms == 1200


def test_event_only_continuation_does_not_block_real_cues_from_another_input(
    tmp_path: Path,
) -> None:
    event = _payload(
        [{"segment_id": "seg_0", "text": "[Music]", "start": 0.0, "end": 0.8}],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
        run_range=[0.0, 1.0],
    )
    speech_segment = _timed_segment("Hello.", [("Hello", 1.1, 1.8)])
    speech_segment.update({"start": 1.0, "end": 1.9})
    speech = _payload(
        [speech_segment],
        outcomes={
            "word_timestamps": "produced",
            "segment_timestamps": "produced",
        },
        run_range=[1.0, 2.0],
    )
    inputs = [
        _write(tmp_path / "event.json", event),
        _write(tmp_path / "speech.json", speech),
    ]
    product = export_documents(inputs, "srt")
    assert product.cue_count == 1
    assert "Hello." in product.content
    assert "Music" not in product.content


def test_mixed_timed_and_abstained_untimed_text_omits_untimed_segment(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed speech.", [("Timed", 0.1, 0.3), ("speech", 0.4, 0.8)])
    untimed = {
        "segment_id": "seg_1",
        "text": "This speech lost its word stream.",
        "start": 0.9,
        "end": 1.9,
    }
    payload = _payload(
        [timed, untimed],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
    )
    payload["abstentions"] = [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.9,
        "end": 1.9,
    }]
    path = _write(tmp_path / "mixed.json", payload)
    product = export_documents([path], "srt")
    assert product.cue_count == 1
    assert "Timed speech." in product.content
    assert "lost its word stream" not in product.content


def test_mixed_timed_and_unbounded_abstained_text_refuses_unprovable_omission(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed.", [("Timed", 0.1, 0.8)])
    untimed = {"segment_id": "seg_1", "text": "Unbounded alignment failure."}
    payload = _payload(
        [timed, untimed],
        outcomes={"word_timestamps": "abstained"},
    )
    payload["abstentions"] = [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.9,
        "end": 1.9,
    }]

    with pytest.raises(TimingRequiredError):
        export_documents([_write(tmp_path / "unbounded.json", payload)], "srt")


def test_bounded_wordless_speech_rejects_an_unrelated_alignment_abstention(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed.", [("Timed", 0.1, 0.8)])
    missing = {
        "segment_id": "seg_1", "text": "Missing timing.",
        "start": 1.0, "end": 1.8,
    }
    payload = _payload(
        [timed, missing],
        outcomes={
            "word_timestamps": "abstained",
            "segment_timestamps": "produced",
        },
    )
    payload["abstentions"] = [{
        "abstention_id": "ab_0",
        "reason": "alignment_unavailable",
        "start": 0.9,
        "end": 1.9,
    }]

    with pytest.raises(InvalidResultError, match="same-bounds"):
        export_documents([_write(tmp_path / "mismatched.json", payload)], "srt")


def test_mixed_timed_and_wordless_speech_requires_abstention_evidence(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed.", [("Timed", 0.1, 0.8)])
    missing = {"segment_id": "seg_1", "text": "Missing timing."}
    payload = _payload(
        [timed, missing],
        outcomes={"word_timestamps": "produced"},
    )
    path = _write(tmp_path / "contradictory-timing.json", payload)

    with pytest.raises(InvalidResultError, match="alignment_unavailable"):
        export_documents([path], "srt")


def test_explicit_empty_words_remain_valid_for_punctuation_only_text(
    tmp_path: Path,
) -> None:
    timed = _timed_segment("Timed.", [("Timed", 0.1, 0.8)])
    punctuation = {"segment_id": "seg_1", "text": "……？！", "words": []}
    payload = _payload(
        [timed, punctuation],
        outcomes={"word_timestamps": "produced"},
    )

    product = export_documents(
        [_write(tmp_path / "punctuation.json", payload)], "srt"
    )
    assert product.cue_count == 1
    assert "Timed." in product.content


def test_subtitles_fail_closed_without_a_real_word_stream(tmp_path: Path) -> None:
    untimed = _payload([{"segment_id": "seg_0", "text": "Transcript only."}])
    path = _write(tmp_path / "untimed.json", untimed)
    with pytest.raises(TimingRequiredError) as raised:
        export_documents([path], "srt")
    assert raised.value.found == ()
    assert raised.value.source_path == Path("source.wav")
    assert raised.value.stack == "qwen-1.7b"

    empty_claim = _payload(
        [{"segment_id": "seg_0", "text": "[Music]", "start": 0.0, "end": 1.0}],
        outcomes={
            "word_timestamps": "produced",
            "segment_timestamps": "produced",
        },
    )
    empty_path = _write(tmp_path / "empty-claim.json", empty_claim)
    empty_product = export_documents([empty_path], "vtt")
    assert empty_product.cues == ()
    assert empty_product.content == "WEBVTT\n\n"

    false_claim = _payload(
        [{"segment_id": "seg_0", "text": "Ordinary untimed speech."}],
        outcomes={"word_timestamps": "produced"},
    )
    with pytest.raises(InvalidResultError, match="alignment_unavailable"):
        export_documents([_write(tmp_path / "false-claim.json", false_claim)], "vtt")


def test_subtitle_quantization_rejects_finite_but_unrepresentable_bounds(
    tmp_path: Path,
) -> None:
    payload = _payload(
        [_timed_segment("Huge.", [("Huge", 1e307, 1e308)])],
        duration=1e308,
        outcomes={"word_timestamps": "produced"},
    )
    path = _write(tmp_path / "huge-timing.json", payload)
    with pytest.raises(InvalidResultError, match="too large"):
        export_documents([path], "srt")


def test_subtitle_validation_attributes_a_bad_later_document_to_that_input(
    tmp_path: Path,
) -> None:
    first = _payload(
        [_timed_segment("Good.", [("Good", 0.1, 0.8)])],
        outcomes={"word_timestamps": "produced"},
        run_range=[0.0, 1.0],
    )
    second = _payload(
        [_timed_segment("Bad.", [("Mismatch", 1.1, 1.8)])],
        outcomes={"word_timestamps": "produced"},
        run_range=[1.0, 2.0],
    )
    first_path = _write(tmp_path / "first.json", first)
    second_path = _write(tmp_path / "second.json", second)

    with pytest.raises(InvalidResultError) as raised:
        export_documents([first_path, second_path], "srt")

    assert raised.value.input_path == second_path


def test_human_and_jsonl_exports_do_not_require_timing(tmp_path: Path) -> None:
    payload = _payload([
        {"segment_id": "seg_old", "text": "First."},
        {"segment_id": "seg_next", "text": "[Music]"},
    ])
    path = _write(tmp_path / "plain.json", payload)
    text = export_documents([path], "txt")
    markdown = export_documents([path], "md")
    jsonl = export_documents([path], "jsonl")

    assert text.content == "First.\n[Music]\n"
    assert markdown.content == "# Transcript\n\nFirst.\n[Music]\n"
    rows = [json.loads(line) for line in jsonl.content.splitlines()]
    assert rows == [
        {"segment_id": "seg_0", "text": "First."},
        {"segment_id": "seg_1", "text": "[Music]"},
    ]
    assert all("provenance" not in row for row in rows)


def test_writer_shapes_are_real_srt_vtt_markdown_text_and_jsonl() -> None:
    cues = [Cue(2310, 4710, "Hello & <world>.", "Speaker 1")]
    assert render_srt(cues) == (
        "1\n00:00:02,310 --> 00:00:04,710\nHello & <world>.\n"
    )
    assert render_vtt(cues) == (
        "WEBVTT\n\n1\n00:00:02.310 --> 00:00:04.710\n"
        "<v Speaker 1>Hello &amp; &lt;world&gt;.\n"
    )
    assert "<v " not in render_vtt([Cue(0, 1, "No speaker")])
    segments = [{"segment_id": "seg_0", "text": "Hello.", "speaker": "S1"}]
    assert render_text(segments) == "[S1] Hello.\n"
    assert render_markdown(segments) == "# Transcript\n\n[S1] Hello.\n"
    assert json.loads(render_jsonl(segments)) == segments[0]


def test_vtt_voice_annotations_cannot_inject_lines_or_empty_tags() -> None:
    rendered = render_vtt([
        Cue(0, 1000, "Hello.", "Speaker\nInjected"),
        Cue(1000, 2000, "World.", " \t "),
        Cue(2000, 3000, "Safe.", "A > B\x00"),
        Cue(3000, 4000, "Quoted.", "O'Neil \"Two\""),
    ])
    assert "<v Speaker Injected>Hello." in rendered
    assert "<v    >" not in rendered
    assert "\x00" not in rendered
    assert "<v A &gt; B>Safe." in rendered
    assert "<v O'Neil \"Two\">Quoted." in rendered
    assert "&#x27;" not in rendered and "&quot;" not in rendered


def test_atomic_writer_refuses_collisions_and_protected_paths(tmp_path: Path) -> None:
    destination = tmp_path / "out.txt"
    write_text_atomic(destination, "one\n")
    assert destination.read_bytes() == b"one\n"
    with pytest.raises(OutputExistsError):
        write_text_atomic(destination, "two\n")
    assert destination.read_text(encoding="utf-8") == "one\n"
    write_text_atomic(destination, "two\n", force=True)
    assert destination.read_text(encoding="utf-8") == "two\n"

    protected = tmp_path / "source.wav"
    protected.write_bytes(b"source")
    with pytest.raises(UnsafeOutputError):
        write_text_atomic(protected, "destroyed", force=True, protected_paths=[protected])
    assert protected.read_bytes() == b"source"

    directory = tmp_path / "directory-output"
    directory.mkdir()
    with pytest.raises(OutputExistsError) as directory_error:
        write_text_atomic(directory, "destroyed", force=True)
    assert directory_error.value.replaceable is False
    assert list(tmp_path.glob(f".{directory.name}.*.tmp")) == []

    loop = tmp_path / "loop"
    loop.symlink_to(loop.name)
    with pytest.raises(OutputWriteError, match="Symlink loop"):
        write_text_atomic(loop, "destroyed", force=True)
    assert loop.is_symlink()

    symlink_target = tmp_path / "symlink-target.txt"
    symlink_target.write_text("keep\n", encoding="utf-8")
    symlink_output = tmp_path / "symlink-output.txt"
    symlink_output.symlink_to(symlink_target)
    with pytest.raises(OutputExistsError) as symlink_error:
        write_text_atomic(symlink_output, "destroyed", force=True)
    assert symlink_error.value.replaceable is False
    assert symlink_output.is_symlink()
    assert symlink_target.read_text(encoding="utf-8") == "keep\n"

    protected_loop = tmp_path / "protected-loop"
    protected_loop.symlink_to(protected_loop.name)
    separate_output = tmp_path / "separate.txt"
    with pytest.raises(OutputWriteError) as caught:
        write_text_atomic(
            separate_output,
            "destroyed",
            protected_paths=[protected_loop],
        )
    assert caught.value.output == separate_output
    assert f"protected path {protected_loop}" in caught.value.reason
    assert not separate_output.exists()

    alias = tmp_path / "source-alias.wav"
    os.link(protected, alias)
    with pytest.raises(UnsafeOutputError):
        write_text_atomic(alias, "destroyed", force=True, protected_paths=[protected])
    assert protected.read_bytes() == b"source"


def test_atomic_writer_accepts_a_name_max_destination(tmp_path: Path) -> None:
    limit = os.pathconf(tmp_path, "PC_NAME_MAX")
    suffix = ".txt"
    destination = tmp_path / ("a" * (limit - len(suffix)) + suffix)

    write_text_atomic(destination, "published\n")

    assert destination.read_text(encoding="utf-8") == "published\n"


def test_export_writer_cannot_follow_a_parent_swapped_after_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "out.txt"
    external.write_text("KEEP\n", encoding="utf-8")

    def swap_parent():
        safe.rename(tmp_path / "safe-old")
        safe.symlink_to(outside, target_is_directory=True)
        return type("Uuid", (), {"hex": "fixed"})()

    monkeypatch.setattr("audio_cli.export.writers.uuid.uuid4", swap_parent)
    with pytest.raises(OutputWriteError, match="directory identity changed"):
        write_text_atomic(
            safe / "out.txt",
            "DESTROYED\n",
            force=True,
            protected_paths=[external],
        )

    assert external.read_text(encoding="utf-8") == "KEEP\n"
    assert not list(tmp_path.glob(".audio-export-*.tmp"))


def test_export_writer_preserves_a_protected_file_renamed_to_output_mid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"canonical")
    output = tmp_path / "out.txt"
    output.write_bytes(b"replaceable")

    def move_source_to_output():
        os.replace(source, output)
        return type("Uuid", (), {"hex": "fixed"})()

    monkeypatch.setattr(
        "audio_cli.export.writers.uuid.uuid4", move_source_to_output
    )
    with pytest.raises(UnsafeOutputError):
        write_text_atomic(
            output,
            "replacement\n",
            force=True,
            protected_paths=[source],
        )

    assert not source.exists()
    assert output.read_bytes() == b"canonical"
    assert not list(tmp_path.glob(".audio-export-*.tmp"))


def test_export_writer_rejects_a_substituted_private_temporary_at_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out.txt"
    output.write_bytes(b"previous")
    real_exchange = media_module._rename_exchange
    raced = False
    substituted_name: str | None = None

    def substitute_temporary(directory_descriptor, left_name, right_name):
        nonlocal raced, substituted_name
        if not raced:
            raced = True
            substituted_name = left_name
            os.rename(
                left_name,
                "held-legitimate.tmp",
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            descriptor = os.open(
                left_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_descriptor,
            )
            try:
                os.write(descriptor, b"substituted")
            finally:
                os.close(descriptor)
        return real_exchange(directory_descriptor, left_name, right_name)

    monkeypatch.setattr(media_module, "_rename_exchange", substitute_temporary)
    with pytest.raises(OutputWriteError, match="temporary changed identity"):
        write_text_atomic(output, "replacement\n", force=True)

    assert raced is True
    assert output.read_bytes() == b"previous"
    assert (tmp_path / "held-legitimate.tmp").read_bytes() == b"replacement\n"
    assert substituted_name is not None
    assert (tmp_path / substituted_name).read_bytes() == b"substituted"


def test_high_level_write_is_atomic_and_never_overwrites_input_or_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"canonical")
    payload = _payload(
        [{"segment_id": "seg_0", "text": "Hello."}],
        source_path=str(source),
    )
    transcript = _write(tmp_path / "transcript.json", payload)
    output = tmp_path / "transcript.txt"
    product = export_documents([transcript], "txt", output)
    assert output.read_text(encoding="utf-8") == product.content

    with pytest.raises(UnsafeOutputError):
        export_documents([transcript], "txt", transcript, force=True)
    with pytest.raises(UnsafeOutputError):
        export_documents([transcript], "txt", source, force=True)
    assert source.read_bytes() == b"canonical"


def test_producer_and_export_preserve_literal_existing_user_tilde_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    username = pwd.getpwuid(os.getuid()).pw_name
    relative_source = Path(f"~{username}") / "source.wav"
    relative_source.parent.mkdir()
    relative_source.write_bytes(b"canonical")
    metadata = InputMetadata(str(relative_source), 1.0, "wav", 16_000, 1)
    recorded_source = result_source(metadata, 1.0)
    assert recorded_source["path"] == str((tmp_path / relative_source).resolve())

    transcript = _write(
        tmp_path / "transcript.json",
        _payload(
            [{"segment_id": "seg_0", "text": "Hello."}],
            source_path=recorded_source["path"],
        ),
    )
    with pytest.raises(UnsafeOutputError):
        export_documents(
            [transcript], "txt", output=relative_source, force=True,
        )
    assert relative_source.read_bytes() == b"canonical"


def test_high_level_write_refuses_ambiguous_relative_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_directory = tmp_path / "original"
    original_directory.mkdir()
    source = original_directory / "source.wav"
    source.write_bytes(b"canonical")
    transcript = _write(
        original_directory / "transcript.json",
        _payload(
            [{"segment_id": "seg_0", "text": "Hello."}],
            source_path="source.wav",
        ),
    )

    other_directory = tmp_path / "later"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    with pytest.raises(InvalidResultError, match="source.path must be absolute"):
        export_documents([transcript], "txt", source, force=True)
    assert source.read_bytes() == b"canonical"
