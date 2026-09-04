from __future__ import annotations

from transcribe_firered_test_support import (
    ROOT,
    _all_keys,
    _fixture,
    deepcopy,
    hashlib,
    json,
    normalize_firered_result,
    pytest,
)


def test_firered_fixture_provenance_and_strict_normalization() -> None:
    fixture = _fixture()
    provenance = fixture["provenance"]
    assert provenance == {
        "source_artifact": (
            "model_tests/benchmark_runs/firered_lidon_batch4_multispeaker_codeswitch_20260815.json"
        ),
        "source_sha256": ("df32268dad30a0322745c1df85653240fc65ab7c87b428463e8e7abd65308b9d"),
        "excerpt_rule": (
            "output.result.vad_segments_ms[0], every output.result sentence contained by "
            "that region, and the leading output.result words that reproduce those sentences"
        ),
    }
    source = ROOT / provenance["source_artifact"]
    if source.is_file():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance["source_sha256"]

    normalized = normalize_firered_result(fixture["raw"], lid_enabled=True)
    assert normalized.segments == (
        {
            "text": "好，",
            "start": 0.59,
            "end": 0.81,
            "words": [{"text": "好", "start": 0.65, "end": 0.81}],
        },
        {
            "text": "现在开始。",
            "start": 0.81,
            "end": 1.81,
            "words": [
                {"text": "现", "start": 0.81, "end": 0.97},
                {"text": "在", "start": 0.97, "end": 1.09},
                {"text": "开", "start": 1.09, "end": 1.29},
                {"text": "始", "start": 1.29, "end": 1.621},
            ],
        },
    )
    assert normalized.vad_regions == ({"start": 0.59, "end": 1.81},)
    assert normalized.lid_regions == (
        {
            "start": 0.59,
            "end": 1.81,
            "language": "zh xinan",
            "confidence": 0.958,
        },
    )
    assert not {"lang", "lang_confidence", "asr_confidence", "confidence"} & _all_keys(
        normalized.segments
    )
    assert all(
        set(word) == {"text", "start", "end"}
        for segment in normalized.segments
        for word in segment["words"]
    )


def test_firered_word_partition_fails_when_a_recorded_word_is_dropped() -> None:
    raw = deepcopy(_fixture()["raw"])
    del raw["words"][2]
    with pytest.raises(ValueError, match="do not reproduce"):
        normalize_firered_result(raw, lid_enabled=True)


def test_firered_partition_is_casefolded_and_keeps_asr_contractions() -> None:
    normalized = normalize_firered_result(
        {
            "sentences": [
                {
                    "start_ms": 0,
                    "end_ms": 1000,
                    "text": "It's Fine!",
                    "asr_confidence": 0.9,
                    "lang": None,
                    "lang_confidence": 0,
                }
            ],
            "words": [
                {"start_ms": 10, "end_ms": 300, "text": "it's"},
                {"start_ms": 300, "end_ms": 800, "text": "fine"},
            ],
            "vad_segments_ms": [[0, 1000]],
        },
        lid_enabled=False,
    )
    assert [word["text"] for word in normalized.segments[0]["words"]] == ["it's", "fine"]
    assert normalized.lid_regions is None
    assert not {"lang", "lang_confidence", "asr_confidence"} & _all_keys(normalized.segments)


def test_firered_rejects_word_schema_or_timeline_drift() -> None:
    with_confidence = deepcopy(_fixture()["raw"])
    with_confidence["words"][0]["confidence"] = 0.99
    with pytest.raises(ValueError, match="exactly start_ms, end_ms, and text"):
        normalize_firered_result(with_confidence, lid_enabled=True)

    overlapping = deepcopy(_fixture()["raw"])
    overlapping["words"][1]["start_ms"] = 700
    with pytest.raises(ValueError, match="strictly chronological and non-overlapping"):
        normalize_firered_result(overlapping, lid_enabled=True)

    outside_region = deepcopy(_fixture()["raw"])
    outside_region["words"][-1]["end_ms"] = 1900
    with pytest.raises(ValueError, match="stay within its sentence's VAD region"):
        normalize_firered_result(outside_region, lid_enabled=True)


def test_firered_lid_requires_one_verbatim_label_per_vad_region() -> None:
    two_labels = deepcopy(_fixture()["raw"])
    two_labels["sentences"][1]["lang"] = "zh mandarin"
    with pytest.raises(ValueError, match="exactly one LID label"):
        normalize_firered_result(two_labels, lid_enabled=True)

    empty_region = deepcopy(_fixture()["raw"])
    empty_region["vad_segments_ms"].append([2000, 3000])
    with pytest.raises(ValueError, match="exactly one LID label"):
        normalize_firered_result(empty_region, lid_enabled=True)


def test_firered_lid_off_drops_only_the_recorded_backend_defaults() -> None:
    raw = deepcopy(_fixture()["raw"])
    for sentence in raw["sentences"]:
        sentence["lang"] = None
        sentence["lang_confidence"] = 0
    normalized = normalize_firered_result(raw, lid_enabled=False)
    assert normalized.lid_regions is None
    assert not {"lang", "lang_confidence"} & _all_keys(normalized.segments)

    raw["sentences"][0]["lang"] = "zh xinan"
    raw["sentences"][0]["lang_confidence"] = 0.958
    with pytest.raises(ValueError, match="while LID is disabled"):
        normalize_firered_result(raw, lid_enabled=False)


@pytest.mark.parametrize(
    ("filename", "sha256", "sentence_count", "word_count", "lid_enabled"),
    [
        (
            "firered_lidoff_batch4_cantomap150s.json",
            "5eca019507898f0e0e18da45ae4d34944db96efd1c83b82c78445ab581e57fd8",
            59,
            379,
            False,
        ),
        (
            "firered_lidoff_batch4_multispeaker_codeswitch_20260815.json",
            "66cf36d41ec116fb41eb06b45bec727871f6c12dcebfca4ab8288944d6df2564",
            62,
            246,
            False,
        ),
        (
            "firered_lidon_batch4_multispeaker_codeswitch_20260815.json",
            "df32268dad30a0322745c1df85653240fc65ab7c87b428463e8e7abd65308b9d",
            62,
            246,
            True,
        ),
        (
            "firered_lidoff_batch4_spice30m_participant.json",
            "1531ae96a0370873fde5033f32971d9ea2da2f21a290ea74fcf410e49a74436a",
            571,
            3833,
            False,
        ),
        (
            "firered_lidoff_batch4_spice60m_participant_concat.json",
            "94fd21f7a54eb419708841ec0257baee394e102f94ea5ccb9090351c847df06a",
            1142,
            7666,
            False,
        ),
    ],
)
def test_full_firered_artifact_partition_when_untracked_source_is_present(
    filename: str,
    sha256: str,
    sentence_count: int,
    word_count: int,
    lid_enabled: bool,
) -> None:
    path = ROOT / "model_tests/benchmark_runs" / filename
    if not path.is_file():
        pytest.skip("untracked full FireRed artifact is not present")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    raw = json.loads(path.read_text(encoding="utf-8"))["output"]["result"]
    normalized = normalize_firered_result(raw, lid_enabled=lid_enabled)
    assert len(normalized.segments) == sentence_count
    assert sum(len(segment["words"]) for segment in normalized.segments) == word_count
    assert (normalized.lid_regions is not None) is lid_enabled
