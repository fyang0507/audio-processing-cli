"""Normalize the pinned FireRedASR2S pipeline result at the core boundary.

The raw shape comes from ``fireredasr2system.py`` at the pinned source checkout.  In
particular, its word objects are exactly ``start_ms``, ``end_ms``, and ``text``
(``fireredasr2system.py:181-184``); the confidence beside a sentence is deliberately
not promoted to a word or public capability.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FireRedResult:
    """Backend-free values produced by one FireRed environment process."""

    segments: tuple[dict[str, Any], ...]
    vad_regions: tuple[dict[str, float], ...]
    lid_regions: tuple[dict[str, Any], ...] | None


def _array(raw: Mapping[str, Any], field: str) -> list[Any]:
    value = raw.get(field)
    if not isinstance(value, list):
        raise TypeError(f"FireRed result {field} must be an array")
    return value


def _milliseconds(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number of milliseconds")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return parsed


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be between 0 and 1") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return parsed


def _bounds(item: Mapping[str, Any], field: str) -> tuple[float, float]:
    start = _milliseconds(item.get("start_ms"), f"{field}.start_ms")
    end = _milliseconds(item.get("end_ms"), f"{field}.end_ms")
    if end < start:
        raise ValueError(f"{field} must satisfy start_ms <= end_ms")
    return start, end


def _seconds(milliseconds: float) -> float:
    return round(milliseconds / 1000.0, 6)


def _plain(text: str) -> str:
    """Return the comparison form required by the punctuation floor.

    FireRedPunc recases its sentence output, so this must use ``casefold`` rather
    than compare the recorded ASR word spellings byte-for-byte.
    """

    return "".join(
        character.casefold()
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _words(
    raw: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    result: list[dict[str, Any]] = []
    raw_bounds: list[tuple[float, float]] = []
    previous_start = -1.0
    previous_end = -1.0
    for index, item in enumerate(_array(raw, "words")):
        if not isinstance(item, Mapping):
            raise TypeError(f"FireRed result words[{index}] must be an object")
        # This exact key set is runner-source and artifact backed.  Rejecting drift
        # here prevents a newly appearing backend default from becoming public by
        # accident, especially the retired per-word confidence claim.
        if set(item) != {"start_ms", "end_ms", "text"}:
            raise ValueError(
                "FireRed result words["
                f"{index}] must contain exactly start_ms, end_ms, and text"
            )
        if not isinstance(item["text"], str):
            raise TypeError(f"FireRed result words[{index}].text must be a string")
        if not _plain(item["text"]):
            raise ValueError(
                f"FireRed result words[{index}].text must contain a non-punctuation character"
            )
        start, end = _bounds(item, f"FireRed result words[{index}]")
        if end == start:
            raise ValueError("FireRed result word bounds must have positive duration")
        if start <= previous_start or start < previous_end:
            raise ValueError(
                "FireRed result words must be strictly chronological and non-overlapping"
            )
        previous_start = start
        previous_end = end
        raw_bounds.append((start, end))
        result.append({
            "text": item["text"],
            "start": _seconds(start),
            "end": _seconds(end),
        })
    return result, raw_bounds


def _vad_regions(raw: Mapping[str, Any]) -> tuple[
    list[tuple[float, float]], tuple[dict[str, float], ...]
]:
    raw_bounds: list[tuple[float, float]] = []
    normalized: list[dict[str, float]] = []
    previous_end = -1.0
    for index, item in enumerate(_array(raw, "vad_segments_ms")):
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError(
                f"FireRed result vad_segments_ms[{index}] must be [start_ms, end_ms]"
            )
        start = _milliseconds(item[0], f"vad_segments_ms[{index}][0]")
        end = _milliseconds(item[1], f"vad_segments_ms[{index}][1]")
        if end <= start:
            raise ValueError(f"vad_segments_ms[{index}] must satisfy start < end")
        if start < previous_end:
            raise ValueError("FireRed VAD regions must be chronological and non-overlapping")
        previous_end = end
        raw_bounds.append((start, end))
        normalized.append({"start": _seconds(start), "end": _seconds(end)})
    return raw_bounds, tuple(normalized)


def _sentence_language(
    sentence: Mapping[str, Any], field: str
) -> tuple[str, float]:
    language = sentence.get("lang")
    confidence = sentence.get("lang_confidence")
    if not isinstance(language, str) or not language:
        raise ValueError(f"{field}.lang must be a non-empty backend label when LID ran")
    return language, _probability(
        confidence, f"{field}.lang_confidence"
    )


def normalize_firered_result(
    raw: Mapping[str, Any], *, lid_enabled: bool
) -> FireRedResult:
    """Normalize sentences, native words, VAD regions, and optional region LID.

    Word partitioning is intentionally strict.  The raw FireRed word stream is
    pre-punctuation and flat; greedily consuming it until each punctuation sentence
    matches is the invariant needed by subtitle splitting.  A dropped, extra, or
    reordered word is therefore a backend failure, never an invitation to omit words.
    """

    if not isinstance(raw, Mapping):
        raise TypeError("FireRed result must be an object")
    raw_vad, vad_regions = _vad_regions(raw)
    words, raw_word_bounds = _words(raw)
    sentence_values = _array(raw, "sentences")

    segments: list[dict[str, Any]] = []
    sentence_inputs: list[tuple[Mapping[str, Any], float, float]] = []
    sentence_word_ranges: list[tuple[int, int]] = []
    word_cursor = 0
    previous_sentence_end = -1.0
    for index, sentence in enumerate(sentence_values):
        field = f"FireRed result sentences[{index}]"
        if not isinstance(sentence, Mapping):
            raise TypeError(f"{field} must be an object")
        text = sentence.get("text")
        if not isinstance(text, str):
            raise TypeError(f"{field}.text must be a string")
        start_ms, end_ms = _bounds(sentence, field)
        if end_ms == start_ms:
            raise ValueError("FireRed sentence bounds must have positive duration")
        if start_ms < previous_sentence_end:
            raise ValueError("FireRed sentences must be chronological and non-overlapping")
        previous_sentence_end = end_ms
        target = _plain(text)
        if not target:
            raise ValueError(f"{field}.text must contain a non-punctuation character")
        joined = ""
        partition: list[dict[str, Any]] = []
        first_word = word_cursor
        while word_cursor < len(words) and len(joined) < len(target):
            word = words[word_cursor]
            candidate = joined + _plain(str(word["text"]))
            if not target.startswith(candidate):
                raise ValueError(
                    f"FireRed words do not reproduce {field}.text after punctuation removal"
                )
            partition.append(word)
            joined = candidate
            word_cursor += 1
        if joined != target:
            raise ValueError(
                f"FireRed words do not reproduce {field}.text after punctuation removal"
            )
        segments.append({
            "text": text,
            "start": _seconds(start_ms),
            "end": _seconds(end_ms),
            "words": partition,
        })
        sentence_inputs.append((sentence, start_ms, end_ms))
        sentence_word_ranges.append((first_word, word_cursor))

    if word_cursor != len(words):
        raise ValueError("FireRed words remain after the final punctuation sentence")

    memberships: list[int] = []
    for sentence_index, (_sentence, start, end) in enumerate(sentence_inputs):
        containing = [
            region_index
            for region_index, (region_start, region_end) in enumerate(raw_vad)
            if start >= region_start and end <= region_end
        ]
        if len(containing) != 1:
            raise ValueError(
                f"FireRed sentence {sentence_index} must belong to exactly one VAD region"
            )
        memberships.append(containing[0])

    for sentence_index, (first_word, last_word) in enumerate(sentence_word_ranges):
        region_start, region_end = raw_vad[memberships[sentence_index]]
        for word_index in range(first_word, last_word):
            word_start, word_end = raw_word_bounds[word_index]
            if word_start < region_start or word_end > region_end:
                raise ValueError(
                    f"FireRed word {word_index} must stay within its sentence's VAD region"
                )

    if not lid_enabled:
        for sentence_index, (sentence, _start, _end) in enumerate(sentence_inputs):
            # The pinned backend fills these defaults even when LID is disabled
            # (fireredasr2system.py:149-150).  A non-default value means the stage
            # unexpectedly ran LID and must not be silently hidden by gating.
            confidence = sentence.get("lang_confidence", 0)
            try:
                invalid_confidence = _probability(
                    confidence,
                    f"FireRed result sentences[{sentence_index}].lang_confidence",
                ) != 0.0
            except (TypeError, ValueError):
                invalid_confidence = True
            if sentence.get("lang") is not None or invalid_confidence:
                raise ValueError(
                    f"FireRed sentence {sentence_index} carries LID output while LID is disabled"
                )

    lid_regions: tuple[dict[str, Any], ...] | None = None
    if lid_enabled:
        grouped: list[dict[str, Any]] = []
        for region_index, ((_region_start, _region_end), region) in enumerate(
            zip(raw_vad, vad_regions, strict=True)
        ):
            labels: set[tuple[str, float]] = set()
            for sentence_index, (sentence, start, end) in enumerate(sentence_inputs):
                if memberships[sentence_index] == region_index:
                    labels.add(_sentence_language(
                        sentence, f"FireRed result sentences[{sentence_index}]"
                    ))
            if len(labels) != 1:
                raise ValueError(
                    f"FireRed VAD region {region_index} must carry exactly one LID label"
                )
            language, confidence = labels.pop()
            grouped.append({
                **region,
                "language": language,
                "confidence": confidence,
            })
        lid_regions = tuple(grouped)

    return FireRedResult(tuple(segments), vad_regions, lid_regions)
