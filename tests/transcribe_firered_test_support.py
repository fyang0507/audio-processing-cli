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



__all__ = [name for name in globals() if not name.startswith("__")]
