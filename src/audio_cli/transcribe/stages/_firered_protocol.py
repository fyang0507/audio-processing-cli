"""Validate and normalize FireRed stage protocol values.

This module is deliberately stdlib-only: the stage imports it both as a package module
in core tests and as a sibling module in the isolated model environment.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from typing import Any


def _number(value: object, field: str) -> float:
    # FireRed can retain NumPy/Torch scalar timestamps in-process; the recorded
    # runner likewise serializes scalar ``item()`` values in
    # model_tests/benchmark/_firered_benchmark_support.py:243-249.
    if not isinstance(value, (bool, int, float)):
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def _probability(value: object, field: str) -> float:
    parsed = _number(value, field)
    if parsed > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return parsed


def _plain(text: str) -> str:
    return "".join(
        character.casefold()
        for character in text
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _range_bounds(request: Mapping[str, Any]) -> tuple[float, float | None]:
    start = _number(request.get("range_start", 0.0), "range_start")
    raw_end = request.get("range_end")
    end = None if raw_end is None else _number(raw_end, "range_end")
    if end is not None and end < start:
        raise ValueError("range_end must be greater than or equal to range_start")
    return start, end


def _public_region_bounds(start: float, end: float) -> tuple[float, float]:
    """Project raw VAD seconds onto FireRed's durable integer-ms timeline."""

    published_start = round(int(start * 1000) / 1000.0, 6)
    published_end = round(int(end * 1000) / 1000.0, 6)
    if published_end <= published_start:
        raise ValueError("VAD region is empty at FireRed's millisecond precision")
    return published_start, published_end


def _region_records(
    values: object, *, range_start: float, range_end: float | None
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise TypeError("VAD regions must be an array")
    result: list[dict[str, Any]] = []
    previous_end = -1.0
    seen: set[str] = set()
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            start = _number(item.get("start"), f"vad_regions[{index}].start")
            end = _number(item.get("end"), f"vad_regions[{index}].end")
            identifier = item.get("region_id", item.get("unit_id", f"vad_{index}"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start = _number(item[0], f"vad_regions[{index}][0]")
            end = _number(item[1], f"vad_regions[{index}][1]")
            identifier = f"vad_{index}"
        else:
            raise TypeError(f"vad_regions[{index}] must be an object or [start, end]")
        if end <= start:
            raise ValueError(f"vad_regions[{index}] must satisfy start < end")
        if start < previous_end:
            raise ValueError("VAD regions must be chronological and non-overlapping")
        previous_end = end
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("VAD region ids must be unique non-empty strings")
        seen.add(identifier)
        # FireRed publishes integer-millisecond bounds. Select on that same
        # timeline so a continuation at the next published start cannot
        # reselect a predecessor whose raw end falls inside the same millisecond.
        published_start, published_end = _public_region_bounds(start, end)
        if published_end > range_start and (range_end is None or published_start < range_end):
            result.append({"region_id": identifier, "start": start, "end": end})
    return result


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _result_array(value: object, field: str, expected: int) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != expected:
        raise TypeError(f"{field} must contain one object per requested region")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field}[{index}] must be an object")
        result.append(item)
    return result


def _format_region(
    entry: Mapping[str, Any], punc_result: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mirror the pinned formatter at ``fireredasr2system.py:126-184``."""

    asr_result = _mapping(entry["asr"], "FireRed ASR result")
    lid_result = entry.get("lid")
    region = _mapping(entry["region"], "FireRed region")
    uttid = asr_result.get("uttid")
    if not isinstance(uttid, str) or punc_result.get("uttid") != uttid:
        raise ValueError("FireRed ASR and punctuation utterance ids must match")
    start_ms = int(_number(region.get("start"), "FireRed region start") * 1000)
    end_ms = int(_number(region.get("end"), "FireRed region end") * 1000)
    raw_sentences = punc_result.get("punc_sentences")
    if not isinstance(raw_sentences, list) or not raw_sentences:
        raise ValueError("FireRed punctuation must return at least one sentence")
    confidence = _probability(asr_result.get("confidence"), "FireRed ASR sentence confidence")

    sentences: list[dict[str, Any]] = []
    for index, raw_sentence in enumerate(raw_sentences):
        sentence = _mapping(raw_sentence, f"FireRed punctuation sentence {index}")
        text = sentence.get("punc_text")
        if not isinstance(text, str):
            raise TypeError("FireRed punctuation sentence text must be a string")
        start_s = _number(sentence.get("start_s"), "punctuation sentence start_s")
        end_s = _number(sentence.get("end_s"), "punctuation sentence end_s")
        sentence_start = start_ms + int(start_s * 1000)
        sentence_end = start_ms + int(end_s * 1000)
        if index == 0:
            sentence_start = start_ms
        if index == len(raw_sentences) - 1:
            sentence_end = end_ms
        normalized = {
            "start_ms": sentence_start,
            "end_ms": sentence_end,
            "text": text,
            "asr_confidence": confidence,
            "lang": None,
            "lang_confidence": 0,
        }
        if lid_result is not None:
            lid = _mapping(lid_result, "FireRed LID result")
            normalized["lang"] = lid.get("lang")
            normalized["lang_confidence"] = _probability(
                lid.get("confidence"), "FireRed LID confidence"
            )
        sentences.append(normalized)

    timestamps = asr_result.get("timestamp")
    if not isinstance(timestamps, list):
        raise TypeError("FireRed ASR timestamps must be an array")
    words: list[dict[str, Any]] = []
    for index, item in enumerate(timestamps):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise TypeError(f"FireRed ASR timestamp {index} must be [text, start, end]")
        text, start, end = item
        if not isinstance(text, str):
            raise TypeError(f"FireRed ASR timestamp {index} text must be a string")
        start_s = _number(start, f"FireRed ASR timestamp {index} start")
        end_s = _number(end, f"FireRed ASR timestamp {index} end")
        words.append(
            {
                "start_ms": int(start_s * 1000 + start_ms),
                "end_ms": int(end_s * 1000 + start_ms),
                "text": text,
            }
        )
    return sentences, words


def _validate_region_semantics(
    sentences: list[dict[str, Any]],
    words: list[dict[str, Any]],
    *,
    region: Mapping[str, Any],
    previous_sentence_end: float,
    previous_word_start: float,
    previous_word_end: float,
) -> tuple[float, float, float]:
    """Validate one formatted unit before it can enter the salvageable prefix.

    The pinned formatter only zips stage arrays and copies punctuation text.  The
    core adapter later enforces the semantic contract, but waiting until then would
    make one malformed late region discard every earlier valid region.  Keep this
    stage-local mirror narrow: sentence/VAD membership and chronology, native-word
    chronology, and exact reproduction after punctuation removal.
    """

    # The pinned formatter publishes integer-millisecond truncation.  Validate
    # against those exact durable bounds rather than the higher-precision input
    # seconds, or a legitimate 0.2009 s region would reject its 200 ms sentence.
    region_start = int(_number(region.get("start"), "FireRed region start") * 1000)
    region_end = int(_number(region.get("end"), "FireRed region end") * 1000)
    word_cursor = 0
    for sentence_index, sentence in enumerate(sentences):
        field = f"FireRed punctuation sentence {sentence_index}"
        start = _number(sentence.get("start_ms"), f"{field} start_ms")
        end = _number(sentence.get("end_ms"), f"{field} end_ms")
        if end <= start:
            raise ValueError("FireRed sentence bounds must have positive duration")
        if start < previous_sentence_end:
            raise ValueError("FireRed sentences must be chronological and non-overlapping")
        if start < region_start or end > region_end:
            raise ValueError("FireRed sentence must belong to its VAD region")
        previous_sentence_end = end

        text = sentence.get("text")
        if not isinstance(text, str):
            raise TypeError(f"{field} text must be a string")
        target = _plain(text)
        if not target:
            raise ValueError(f"{field} text must contain a non-punctuation character")
        joined = ""
        while word_cursor < len(words) and len(joined) < len(target):
            word = words[word_cursor]
            word_field = f"FireRed ASR word {word_cursor}"
            word_text = word.get("text")
            if not isinstance(word_text, str):
                raise TypeError(f"{word_field} text must be a string")
            plain_word = _plain(word_text)
            if not plain_word:
                raise ValueError(f"{word_field} text must contain a non-punctuation character")
            word_start = _number(word.get("start_ms"), f"{word_field} start_ms")
            word_end = _number(word.get("end_ms"), f"{word_field} end_ms")
            if word_end <= word_start:
                raise ValueError("FireRed word bounds must have positive duration")
            if word_start < region_start or word_end > region_end:
                raise ValueError("FireRed word must stay within its VAD region")
            if word_start <= previous_word_start or word_start < previous_word_end:
                raise ValueError("FireRed words must be strictly chronological and non-overlapping")
            candidate = joined + plain_word
            if not target.startswith(candidate):
                raise ValueError(
                    f"FireRed words do not reproduce {field} text after punctuation removal"
                )
            joined = candidate
            previous_word_start = word_start
            previous_word_end = word_end
            word_cursor += 1
        if joined != target:
            raise ValueError(
                f"FireRed words do not reproduce {field} text after punctuation removal"
            )

    if word_cursor != len(words):
        raise ValueError("FireRed words remain after the final punctuation sentence")
    return previous_sentence_end, previous_word_start, previous_word_end
