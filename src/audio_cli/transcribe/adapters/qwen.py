"""Normalize the raw result of the pinned Qwen private batched API."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

# Observed by model_tests/benchmark/run_qwen_verbatim_probe.py and returned by the private
# _generate_chunks_batched path used in run_turn_attributed_mlx_asr.py.  The public path strips
# this wrapper; keeping it would put backend protocol text into the transcript.
_SCAFFOLD = re.compile(r"^language ([^<]+)<asr_text>")
_SENTENCE_END = frozenset(".!?。！？")
_CLOSERS = frozenset("\"'”’」』】）)]")


def strip_qwen_scaffold(text: str) -> str:
    return _SCAFFOLD.sub("", text, count=1)


def split_sentences(text: str) -> list[str]:
    """Split punctuated model text without inventing or removing punctuation."""
    result: list[str] = []
    pending = ""
    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        boundary = character in _SENTENCE_END or character == "\n"
        if (
            character == "."
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            boundary = False
        if not boundary:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in _SENTENCE_END:
            end += 1
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        sentence = text[start:end].strip()
        if _plain(sentence):
            result.append(pending + sentence)
            pending = ""
        elif sentence:
            if result:
                result[-1] += sentence
            else:
                pending += sentence
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
        index = start
    remainder = text[start:].strip()
    if remainder:
        result.append(pending + remainder)
    elif pending and result:
        result[-1] += pending
    elif pending:
        result.append(pending)
    return result


def _plain(text: str) -> str:
    return "".join(
        character.casefold()
        for character in text
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def sentence_segments(
    completed: Sequence[Mapping[str, Any]],
    aligned: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    rejections: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Turn processing-unit text into the sentence artifacts required by the floor.

    When alignment ran, its word stream is partitioned by the punctuation invariant instead
    of treating processing-unit bounds as sentence timing.
    """
    result: list[dict[str, Any]] = []
    aligned = aligned or {}
    for unit in completed:
        identifier = str(unit["unit_id"])
        sentences = split_sentences(str(unit["text"]))
        words = list(aligned.get(identifier, ()))
        unit_segments = []
        try:
            word_cursor = 0
            for text in sentences:
                segment = {"unit_id": identifier, "text": text}
                if "speaker" in unit:
                    segment["speaker"] = unit["speaker"]
                if identifier in aligned:
                    target = _plain(text)
                    collected = []
                    joined = ""
                    while word_cursor < len(words) and len(joined) < len(target):
                        word = dict(words[word_cursor])
                        candidate = joined + _plain(str(word["text"]))
                        if not target.startswith(candidate):
                            raise ValueError(
                                f"aligned words do not reproduce sentence text for {identifier!r}"
                            )
                        collected.append(word)
                        joined = candidate
                        word_cursor += 1
                    if joined != target:
                        raise ValueError(
                            f"aligned words do not reproduce sentence text for {identifier!r}"
                        )
                    segment["words"] = collected
                unit_segments.append(segment)
            if identifier in aligned and word_cursor != len(words):
                raise ValueError(f"aligned words remain after sentence split for {identifier!r}")
        except ValueError:
            if rejections is not None:
                rejections[identifier] = {"code": "sentence_reconciliation"}
            unit_segments = []
            for text in sentences:
                segment = {"unit_id": identifier, "text": text}
                if "speaker" in unit:
                    segment["speaker"] = unit["speaker"]
                unit_segments.append(segment)
        result.extend(unit_segments)
    return result


def normalize_qwen_segments(
    raw: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return chronological completed segments and unfinished units.

    Unit bounds and speakers are deterministic orchestration inputs. Qwen supplies text only;
    its language scaffold is transport metadata and is intentionally not promoted to a result.
    """
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in units:
        identifier = str(item["unit_id"])
        if identifier in by_id:
            raise ValueError(f"Qwen received duplicate requested unit {identifier!r}")
        by_id[identifier] = item
    completed: list[dict[str, Any]] = []
    unfinished: list[dict[str, Any]] = []
    seen: set[str] = set()
    if "units" not in raw:
        raise ValueError("Qwen stage result is missing units")
    values = raw["units"]
    if not isinstance(values, list):
        raise TypeError("Qwen stage result units must be an array")
    for item in values:
        if not isinstance(item, Mapping):
            raise TypeError("Qwen stage result unit must be an object")
        identifier = str(item.get("unit_id", ""))
        if identifier not in by_id or identifier in seen:
            raise ValueError(f"Qwen stage returned unknown or duplicate unit {identifier!r}")
        seen.add(identifier)
        unit = by_id[identifier]
        processed = item.get("processed")
        if not isinstance(processed, bool):
            raise TypeError(f"Qwen stage unit {identifier!r} processed must be a boolean")
        if not processed:
            unfinished.append(dict(unit))
            continue
        text = item.get("text")
        if not isinstance(text, str):
            raise TypeError(f"Qwen stage unit {identifier!r} has no string text")
        segment = {
            "unit_id": identifier,
            "start": float(unit["start"]),
            "end": float(unit["end"]),
            "text": strip_qwen_scaffold(text),
        }
        if "speaker" in unit:
            segment["speaker"] = str(unit["speaker"])
        completed.append(segment)
    missing = set(by_id) - seen
    if missing:
        rendered = ", ".join(repr(identifier) for identifier in sorted(missing))
        raise ValueError(f"Qwen stage result is missing requested units: {rendered}")
    completed.sort(key=lambda item: (item["start"], item["end"], item["unit_id"]))
    unfinished.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return completed, unfinished
