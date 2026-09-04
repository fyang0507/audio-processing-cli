from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audio_cli.transcribe.adapters import normalize_firered_result
from audio_cli.transcribe.stages import firered as firered_stage


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/firered_lidon_first_vad_region.json"

# Exact executable method source from
# Xiaohongshu/FireRedASR2S@4e7d9aaf4482a47cec1724807026b9b151926eb5,
# fireredasr2s/fireredasr2system.py:46-200.  The full pinned file's SHA-256 is
# 8bfada7cfc7ff424c73484a94150bc6fa5030a05b7146db983cb7ac374ad1782; the
# body-AST digest below keeps the parity test source-backed when that ignored
# upstream checkout is absent in CI.
_PINNED_FIRERED_PROCESS = r'''
def process(self, wav_path, uttid="tmpid"):
        wav_np, sample_rate = sf.read(wav_path, dtype="int16")
        dur = wav_np.shape[0]/sample_rate

        # 1. VAD
        if self.config.enable_vad:
            vad_result, prob = self.vad.detect(wav_path)
            vad_segments = vad_result["timestamps"]
            logger.info(f"VAD: {vad_result}")
        else:
            vad_segments = [(0, dur)]
            vad_result = {"timestamps" : vad_segments}

        # 2. VAD output to ASR input
        asr_results = []
        lid_results = []
        assert sample_rate == 16000
        batch_asr_uttid = []
        batch_asr_wav = []
        for j, (start_s, end_s) in enumerate(vad_segments):
            wav_segment = wav_np[int(start_s*sample_rate):int(end_s*sample_rate)]
            vad_uttid = f"{uttid}_s{int(start_s*1000)}_e{int(end_s*1000)}"
            batch_asr_uttid.append(vad_uttid)
            batch_asr_wav.append((sample_rate, wav_segment))
            if len(batch_asr_uttid) < self.config.asr_batch_size and j != len(vad_segments) - 1:
                continue

            # 3. ASR
            batch_asr_results = self.asr.transcribe(batch_asr_uttid, batch_asr_wav)
            logger.info(f"ASR: {batch_asr_results}")

            if self.config.enable_lid:
                batch_lid_results = self.lid.process(batch_asr_uttid, batch_asr_wav)
                logger.info(f"LID: {batch_lid_results}")
            else:
                # Note: The original batch size is used here to ensure alignment with the initial number of ASR results
                batch_lid_results = [None] * len(batch_asr_results)

            # Synchronously traverse and filter to ensure that asr_results and lid_results always maintain a one-to-one correspondence
            for a_res, l_res in zip(batch_asr_results, batch_lid_results):
                text = a_res.get("text", "").strip()
                # Filter out <blank>, <sil> and completely empty strings ""
                if not text or re.search(r"(<blank>)|(<sil>)", text):
                    continue
                asr_results.append(a_res)
                lid_results.append(l_res)

            batch_asr_uttid = []
            batch_asr_wav = []

        # 4. ASR output to Postprocess input
        if self.config.enable_punc:
            punc_results = []
            batch_asr_text = []
            batch_asr_uttid = []
            batch_asr_timestamp = []
            for j, asr_result in enumerate(asr_results):
                batch_asr_text.append(asr_result["text"])
                batch_asr_uttid.append(asr_result["uttid"])
                if self.config.asr_config.return_timestamp:
                    batch_asr_timestamp.append(asr_result.get("timestamp", []))
                elif "timestamp" in asr_result:
                    batch_asr_timestamp.append(asr_result["timestamp"])
                if len(batch_asr_text) < self.config.punc_batch_size and j != len(asr_results) - 1:
                    continue

                # 5. Punc
                if self.config.asr_config.return_timestamp:
                    batch_punc_results = self.punc.process_with_timestamp(batch_asr_timestamp, batch_asr_uttid)
                else:
                    batch_punc_results = self.punc.process(batch_asr_text, batch_asr_uttid)
                logger.info(f"Punc: {batch_punc_results}")

                punc_results.extend(batch_punc_results)
                batch_asr_text = []
                batch_asr_uttid = []
                batch_asr_timestamp = []
        else:
            punc_results = asr_results

        # 6. Put all together & Format
        sentences = []
        words = []
        for asr_result, punc_result, lid_result in zip(asr_results, punc_results, lid_results):
            assert asr_result["uttid"] == punc_result["uttid"], f"fix code: {asr_result} | {punc_result}"
            start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
            assert start_ms.startswith("s") and end_ms.startswith("e")
            start_ms, end_ms = int(start_ms[1:]), int(end_ms[1:])
            if self.config.asr_config.return_timestamp:
                sub_sentences = []
                if self.config.enable_punc:
                    for i, punc_sent in enumerate(punc_result["punc_sentences"]):
                        start = start_ms + int(punc_sent["start_s"]*1000)
                        end = start_ms + int(punc_sent["end_s"]*1000)
                        if i == 0:
                            start = start_ms
                        if i == len(punc_result["punc_sentences"]) - 1:
                            end = end_ms
                        sub_sentence = {
                            "start_ms": start,
                            "end_ms": end,
                            "text": punc_sent["punc_text"],
                            "asr_confidence": asr_result["confidence"],
                            "lang": None,
                            "lang_confidence": 0
                        }
                        if lid_result:
                            sub_sentence["lang"] = lid_result["lang"]
                            sub_sentence["lang_confidence"] = lid_result["confidence"]
                        sub_sentences.append(sub_sentence)
                else:
                    sub_sentences = [{
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": asr_result["text"],
                        "asr_confidence": asr_result["confidence"],
                        "lang": None,
                        "lang_confidence": 0
                    }]
                sentences.extend(sub_sentences)
            else:
                text = punc_result["punc_text"] if self.config.enable_punc else asr_result["text"]
                sentence = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0
                }
                if lid_result:
                    sentence["lang"] = lid_result["lang"]
                    sentence["lang_confidence"] = lid_result["confidence"]
                sentences.append(sentence)

            if "timestamp" in asr_result:
                for w, s, e in asr_result["timestamp"]:
                    word = {"start_ms": int(s*1000+start_ms), "end_ms":int(e*1000+start_ms), "text": w}
                    words.append(word)

        vad_segments_ms = [(int(s*1000), int(e*1000)) for s, e in vad_result["timestamps"]]
        text = "".join(s["text"] for s in sentences)
        # Add space after English punctuation when followed by a letter
        text = re.sub(r'([.,!?])\s*([a-zA-Z])', r'\1 \2', text)

        result = {
            "uttid": uttid,
            "text": text,
            "sentences": sentences,
            "vad_segments_ms": vad_segments_ms,
            "dur_s": dur,
            "words": words,
            "wav_path": wav_path
        }
        return result
'''


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, (list, tuple)):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_firered_fixture_provenance_and_strict_normalization() -> None:
    fixture = _fixture()
    provenance = fixture["provenance"]
    assert provenance == {
        "source_artifact": (
            "model_tests/benchmark_runs/"
            "firered_lidon_batch4_multispeaker_codeswitch_20260815.json"
        ),
        "source_sha256": (
            "df32268dad30a0322745c1df85653240fc65ab7c87b428463e8e7abd65308b9d"
        ),
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
    assert normalized.lid_regions == ({
        "start": 0.59,
        "end": 1.81,
        "language": "zh xinan",
        "confidence": 0.958,
    },)
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
    normalized = normalize_firered_result({
        "sentences": [{
            "start_ms": 0,
            "end_ms": 1000,
            "text": "It's Fine!",
            "asr_confidence": 0.9,
            "lang": None,
            "lang_confidence": 0,
        }],
        "words": [
            {"start_ms": 10, "end_ms": 300, "text": "it's"},
            {"start_ms": 300, "end_ms": 800, "text": "fine"},
        ],
        "vad_segments_ms": [[0, 1000]],
    }, lid_enabled=False)
    assert [word["text"] for word in normalized.segments[0]["words"]] == ["it's", "fine"]
    assert normalized.lid_regions is None
    assert not {"lang", "lang_confidence", "asr_confidence"} & _all_keys(
        normalized.segments
    )


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


@pytest.mark.parametrize("lid_enabled", [True, False])
def test_firered_phase_mirror_matches_pinned_upstream_process(
    lid_enabled: bool,
) -> None:
    parsed = ast.parse(_PINNED_FIRERED_PROCESS)
    function = parsed.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = ast.Module(body=function.body, type_ignores=[])
    digest = hashlib.sha256(
        ast.dump(body, annotate_fields=True, include_attributes=False).encode()
    ).hexdigest()
    assert digest == "525cffa63a7f1148d4568f56ef84e5f321ffbdf7890ffd30db6428fc2f3f60db"
    upstream_path = (
        ROOT
        / "model_tests/firered/FireRedASR2S/fireredasr2s/fireredasr2system.py"
    )
    if upstream_path.is_file():
        upstream_source = upstream_path.read_text(encoding="utf-8")
        assert hashlib.sha256(upstream_source.encode()).hexdigest() == (
            "8bfada7cfc7ff424c73484a94150bc6fa5030a05b7146db983cb7ac374ad1782"
        )
        upstream_tree = ast.parse(upstream_source)
        upstream_class = next(
            node
            for node in upstream_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FireRedAsr2System"
        )
        upstream_function = next(
            node
            for node in upstream_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        assert ast.dump(
            ast.Module(body=upstream_function.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        ) == ast.dump(body, annotate_fields=True, include_attributes=False)

    wav = np.zeros(14 * 16_000, dtype=np.int16)
    sample_rate = 16_000
    blank_start = None if lid_enabled else 2000
    native_regions = [
        (float(index * 2), float(index * 2 + 1)) for index in range(7)
    ]
    record: dict[str, Any] = {}

    def bounds(uttid: str) -> tuple[int, int]:
        start_token, end_token = uttid.split("_")[-2:]
        return int(start_token[1:]), int(end_token[1:])

    class Vad:
        def detect(self, audio: str) -> tuple[dict[str, Any], None]:
            record.setdefault("vad_calls", []).append(audio)
            return {"timestamps": native_regions}, None

    class Asr:
        def transcribe(
            self, uttids: list[str], batch_wav: list[tuple[int, Any]]
        ) -> list[dict[str, Any]]:
            assert len(uttids) == len(batch_wav)
            record.setdefault("asr_batch_sizes", []).append(len(uttids))
            results = []
            for uttid in uttids:
                start_ms, end_ms = bounds(uttid)
                text = "" if start_ms == blank_start else f"region{start_ms}"
                record.setdefault("asr_text", {})[uttid] = text
                results.append({
                    "uttid": uttid,
                    "text": text,
                    "confidence": 0.9,
                    "timestamp": (
                        []
                        if not text
                        else [(text, 0.001, (end_ms - start_ms) / 1000 - 0.001)]
                    ),
                })
            return results

    class Lid:
        def process(
            self, uttids: list[str], batch_wav: list[tuple[int, Any]]
        ) -> list[dict[str, Any]]:
            assert len(uttids) == len(batch_wav)
            record.setdefault("lid_batch_sizes", []).append(len(uttids))
            return [
                {"uttid": uttid, "lang": "en", "confidence": 0.9}
                for uttid in uttids
            ]

    class Punc:
        def process_with_timestamp(
            self, timestamps: list[Any], uttids: list[str]
        ) -> list[dict[str, Any]]:
            assert len(timestamps) == len(uttids)
            call = record.get("punc_calls", 0) + 1
            record["punc_calls"] = call
            record.setdefault("punc_batch_sizes", []).append(len(uttids))
            results = []
            for uttid in uttids:
                start_ms, end_ms = bounds(uttid)
                text = record["asr_text"][uttid]
                results.append({
                    "uttid": uttid,
                    "punc_sentences": [{
                        "start_s": 0.0,
                        "end_s": (end_ms - start_ms) / 1000,
                        # Recasing is source-backed punctuator behavior and
                        # survives in raw output while the punctuation floor's
                        # casefolded comparison still reproduces the ASR word.
                        "punc_text": (
                            text.upper() if call == 1 else text.lower()
                        ) + ".",
                    }],
                })
            return results

    config = types.SimpleNamespace(
        enable_vad=True,
        enable_lid=lid_enabled,
        enable_punc=True,
        asr_batch_size=3,
        punc_batch_size=4,
        asr_config=types.SimpleNamespace(return_timestamp=True),
    )
    system = types.SimpleNamespace(
        config=config,
        vad=Vad(),
        asr=Asr(),
        lid=Lid(),
        punc=Punc(),
    )
    namespace = {
        "logger": types.SimpleNamespace(info=lambda *_args: None),
        "re": re,
        "sf": types.SimpleNamespace(
            read=lambda *_args, **_kwargs: (wav, sample_rate)
        ),
    }
    exec(_PINNED_FIRERED_PROCESS, namespace)
    pinned_process = types.MethodType(namespace["process"], system)

    upstream = pinned_process("/canonical.wav", "audio")
    upstream_calls = deepcopy(record)
    record.clear()

    vad_result, _probability = system.vad.detect("/canonical.wav")
    regions = [
        {"region_id": f"vad_{index}", "start": start, "end": end}
        for index, (start, end) in enumerate(vad_result["timestamps"])
    ]
    mirrored, completed, failure = firered_stage._run_pipeline(
        system,
        wav,
        sample_rate,
        regions,
        asr_batch_size=3,
        punc_batch_size=4,
        lid_enabled=lid_enabled,
    )
    mirrored_calls = deepcopy(record)

    assert failure is None
    assert completed == len(regions)
    assert upstream_calls == mirrored_calls
    assert upstream_calls["asr_batch_sizes"] == [3, 3, 1]
    assert upstream_calls["punc_batch_sizes"] == ([4, 3] if lid_enabled else [4, 2])
    if lid_enabled:
        assert upstream_calls["lid_batch_sizes"] == [3, 3, 1]
    else:
        assert "lid_batch_sizes" not in upstream_calls

    def public_subset(result: dict[str, Any]) -> dict[str, Any]:
        # The stage boundary is JSON, which canonicalizes upstream's tuple VAD bounds.
        subset = {
            name: result[name] for name in ("sentences", "words", "vad_segments_ms")
        }
        return json.loads(json.dumps(subset))

    assert public_subset(mirrored) == public_subset(upstream)
    sentence_texts = [item["text"] for item in mirrored["sentences"]]
    if not lid_enabled:
        assert all("region2000" not in text for text in sentence_texts)

    # Mutation witness: resetting punctuation after each ASR batch would group the
    # six post-filter items as 2/3/1 instead of the pinned global 4/2 stream.  The
    # stateful recasing makes that semantic drift observable in raw output without
    # inventing lexical content that the ASR word stream cannot reproduce.
    starts_by_asr_batch = (
        ((0, 2000, 4000), (6000, 8000, 10000), (12000,))
        if lid_enabled
        else ((0, 4000), (6000, 8000, 10000), (12000,))
    )
    reset_batching_texts = [
        (
            f"region{start_ms}".upper()
            if call == 1
            else f"region{start_ms}".lower()
        ) + "."
        for call, starts in enumerate(starts_by_asr_batch, 1)
        for start_ms in starts
    ]
    assert reset_batching_texts != sentence_texts
    first_second_batch = 3 if lid_enabled else 2
    assert reset_batching_texts[first_second_batch].startswith("region")
    assert sentence_texts[first_second_batch].startswith("REGION")


def _install_fake_firered(monkeypatch: pytest.MonkeyPatch, record: dict[str, Any]) -> None:
    def config_type(name: str) -> type:
        class Config:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)
                record.setdefault("configs", {})[name] = dict(kwargs)

        return Config

    class Vad:
        def detect(self, audio: str) -> tuple[dict[str, Any], None]:
            record.setdefault("calls", []).append(("vad", audio))
            return {"timestamps": [(0.0, 1.0), (2.0, 3.0)]}, None

    def bounds(uttid: str) -> tuple[int, int]:
        start_token, end_token = uttid.split("_")[-2:]
        return int(start_token[1:]), int(end_token[1:])

    class Asr:
        def transcribe(
            self, uttids: list[str], batch_wav: list[tuple[int, Any]]
        ) -> list[dict[str, Any]]:
            assert len(uttids) == len(batch_wav)
            call = record.get("asr_calls", 0) + 1
            record["asr_calls"] = call
            record.setdefault("asr_batch_sizes", []).append(len(uttids))
            record.setdefault("calls", []).append(("asr", len(uttids)))
            if record.get("fail_asr_call") == call:
                raise RuntimeError("synthetic FireRed failure")
            results = []
            for uttid in uttids:
                start_ms, end_ms = bounds(uttid)
                text = (
                    ""
                    if start_ms in record.get("blank_starts", set())
                    else f"region{start_ms}"
                )
                record.setdefault("asr_text", {})[uttid] = text
                results.append({
                    "uttid": uttid,
                    "text": text,
                    "confidence": 0.9,
                    "timestamp": (
                        []
                        if not text
                        else [(
                            text,
                            0.001,
                            (
                                (end_ms - start_ms) / 1000 + 0.5
                                if start_ms in record.get(
                                    "out_of_region_word_starts", set()
                                )
                                else (end_ms - start_ms) / 1000 - 0.001
                            ),
                        )]
                    ),
                })
            return results

    class Lid:
        def process(
            self, uttids: list[str], batch_wav: list[tuple[int, Any]]
        ) -> list[dict[str, Any]]:
            assert len(uttids) == len(batch_wav)
            call = record.get("lid_calls", 0) + 1
            record["lid_calls"] = call
            record.setdefault("lid_batch_sizes", []).append(len(uttids))
            record.setdefault("calls", []).append(("lid", len(uttids)))
            if record.get("fail_lid_call") == call:
                raise RuntimeError("synthetic FireRed failure")
            if record.get("malformed_lid_call") == call:
                # Pinned FireRedLID uses this data shape for a caught feature
                # extraction failure (fireredlid/lid.py:62-66).
                return [{"uttid": uttid, "lang": ""} for uttid in uttids]
            return [
                {"uttid": uttid, "lang": "en", "confidence": 0.9}
                for uttid in uttids
            ]

    class Punc:
        def process_with_timestamp(
            self, timestamps: list[Any], uttids: list[str]
        ) -> list[dict[str, Any]]:
            assert len(timestamps) == len(uttids)
            call = record.get("punc_calls", 0) + 1
            record["punc_calls"] = call
            record.setdefault("punc_batch_sizes", []).append(len(uttids))
            record.setdefault("calls", []).append(("punctuator", len(uttids)))
            if record.get("fail_punc_call") == call:
                raise RuntimeError("synthetic FireRed failure")
            results = []
            for uttid in uttids:
                start_ms, end_ms = bounds(uttid)
                text = record["asr_text"][uttid]
                punctuated = (
                    text + " missing."
                    if start_ms in record.get("mismatched_punc_starts", set())
                    else text + "."
                )
                results.append({
                    "uttid": uttid,
                    "punc_sentences": [{
                        "start_s": 0.0,
                        "end_s": (end_ms - start_ms) / 1000,
                        "punc_text": punctuated,
                    }],
                })
            if record.get("malformed_punc_call") == call:
                del results[0]["punc_sentences"][0]["punc_text"]
            return results

    class System:
        def __init__(self, config: Any) -> None:
            self.config = config
            record["system_loads"] = record.get("system_loads", 0) + 1
            record["load_flags"] = {
                "vad": config.enable_vad,
                "lid": config.enable_lid,
                "punc": config.enable_punc,
            }
            self.vad = Vad() if config.enable_vad else None
            self.lid = Lid() if config.enable_lid else None
            self.asr = Asr()
            self.punc = Punc()

        def process(self, _audio: str, _uttid: str) -> dict[str, Any]:
            record["process_calls"] = record.get("process_calls", 0) + 1
            raise AssertionError(
                "the stage must mirror upstream phases without resetting batches"
            )

    root = types.ModuleType("fireredasr2s")
    root.__path__ = []
    root.FireRedAsr2System = System
    root.FireRedAsr2SystemConfig = config_type("system")
    modules = {
        "fireredasr2s": root,
        "fireredasr2s.fireredasr2": types.ModuleType("fireredasr2s.fireredasr2"),
        "fireredasr2s.fireredlid": types.ModuleType("fireredasr2s.fireredlid"),
        "fireredasr2s.fireredpunc": types.ModuleType("fireredasr2s.fireredpunc"),
        "fireredasr2s.fireredvad": types.ModuleType("fireredasr2s.fireredvad"),
    }
    modules["fireredasr2s.fireredasr2"].FireRedAsr2Config = config_type("asr")
    modules["fireredasr2s.fireredlid"].FireRedLidConfig = config_type("lid")
    modules["fireredasr2s.fireredpunc"].FireRedPuncConfig = config_type("punctuator")
    modules["fireredasr2s.fireredvad"].FireRedVadConfig = config_type("vad")
    soundfile = types.ModuleType("soundfile")
    soundfile.read = lambda *_args, **_kwargs: (
        np.zeros(160_000, dtype=np.int16),
        16_000,
    )
    modules["soundfile"] = soundfile
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _stage_request(**updates: Any) -> dict[str, Any]:
    request = {
        "audio": "/canonical.wav",
        "checkout": "/firered/source",
        "models": {
            "vad": "/models/vad",
            "lid": "/models/lid",
            "asr": "/models/asr",
            "punctuator": "/models/punc",
        },
        "lid_enabled": False,
        "asr_config": {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 4,
            "return_timestamp": True,
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6,
            "eos_penalty": 1.0,
        },
        "punctuator_config": {"batch_size": 4},
    }
    request.update(updates)
    return request


def _run_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    request_path = tmp_path / "firered.request.json"
    result_path = tmp_path / "firered.result.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["firered.py", str(request_path), str(result_path)]
    )
    code = firered_stage.main()
    return code, json.loads(result_path.read_text(encoding="utf-8"))


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
    assert set(result["metrics"]["stage_wall_seconds"]) == {
        "vad", "asr", "punctuator"
    }


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
    assert result["regions"] == [{
        "region_id": "external_1",
        "start": 2.0,
        "end": 3.0,
        "processed": True,
    }]
    assert result["result"]["vad_segments_ms"] == [[2000, 3000]]
    assert set(result["metrics"]["stage_wall_seconds"]) == {
        "asr", "lid", "punctuator"
    }


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
        "sentences": [], "words": [], "vad_segments_ms": [],
    }


def test_firered_stage_never_claims_partial_without_a_completed_region_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {"fail_asr_call": 1}
    _install_fake_firered(monkeypatch, record)
    code, result = _run_stage(
        tmp_path,
        monkeypatch,
        _stage_request(vad_regions=[{
            "region_id": "external_0", "start": 0.0, "end": 1.0,
        }]),
    )
    assert code == 1
    assert result["complete"] is False
    assert result["result"] == {
        "sentences": [], "words": [], "vad_segments_ms": [],
    }
    assert result["regions"] == [{
        "region_id": "external_0",
        "start": 0.0,
        "end": 1.0,
        "processed": False,
    }]
    assert result["error"]["message"] == "synthetic FireRed failure"


def test_firered_stage_salvages_completed_region_prefix_after_later_batch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, Any] = {"fail_asr_call": 2}
    _install_fake_firered(monkeypatch, record)
    regions = [
        {"region_id": f"external_{index}", "start": float(index * 2),
         "end": float(index * 2 + 1)}
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
        True, True, True, True, False,
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
            [0, 1000], [2000, 3000], [4000, 5000], [6000, 7000],
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
        "region0.", "region2000.", "region4000.", "region6000.",
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
        True, True, True, True, False,
    ]
    partial = normalize_firered_result(result["result"], lid_enabled=True)
    assert len(partial.segments) == 4
    assert partial.lid_regions is not None
    assert result["error"]["message"] == (
        "FireRed LID returned an empty region language label"
    )


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
        True, False, False, False, False,
    ]
    assert result["result"]["vad_segments_ms"] == [[0, 1000]]
    normalized = normalize_firered_result(result["result"], lid_enabled=True)
    assert [segment["text"] for segment in normalized.segments] == ["region0."]
    assert normalized.lid_regions == ({
        "start": 0.0,
        "end": 1.0,
        "language": "en",
        "confidence": 0.9,
    },)
    assert result["error"]["message"] == (
        "FireRed cannot publish region LID for a blank ASR region"
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
