"""Normalize VibeVoice's structured transcript and salvage a truncated prefix.

The pinned processor returns an empty array when generation stops before the outer JSON
array closes.  Complete objects before that cut are still valid backend evidence, so the
adapter decodes one array element at a time instead of asking the upstream all-or-nothing
post-processor to parse the document again.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_EVENT_TAG = re.compile(r"^\[[^\[\]\r\n]+\]$")
_MISSING = object()
_RAW_KEYS = {
    "Start",
    "Start time",
    "End",
    "End time",
    "Speaker",
    "Speaker ID",
    "Content",
}
_NORMALIZED_KEYS = {"start_time", "end_time", "speaker_id", "text"}


@dataclass(frozen=True)
class VibeVoiceResult:
    """Core-owned values from one whole-media VibeVoice generation."""

    segments: tuple[dict[str, Any], ...]
    hit_max_new_tokens: bool
    covered_through_seconds: float | None


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"VibeVoice generated JSON repeats key {key!r}")
        result[key] = value
    return result


def _raw_decoder() -> json.JSONDecoder:
    return json.JSONDecoder(object_pairs_hook=_reject_duplicate_json_keys)


def _validate_generated_preamble(text: str, json_start: int) -> None:
    if text[:json_start].strip() not in {"", "assistant"}:
        raise ValueError("VibeVoice generated text has an unsupported preamble")


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def _plain(text: str) -> str:
    return "".join(
        character.casefold()
        for character in text
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def normalize_vibevoice_alignment(
    segments: Sequence[Mapping[str, Any]],
    aligned: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    rejections: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Keep only speech-unit word streams that satisfy the punctuation floor.

    The adapter's narrow event-tag classification is the intentional no-word case:
    those segments are not sent to the aligner and therefore have no ``native_*`` id.
    An invalid or missing speech-unit stream instead becomes a per-segment aligner
    abstention; it does not erase otherwise valid VibeVoice text.
    """

    valid: dict[str, list[dict[str, Any]]] = {}
    alignable_index = 0
    for segment in segments:
        if not segment.get("alignable"):
            continue
        identifier = f"native_{alignable_index}"
        alignable_index += 1
        if identifier not in aligned:
            continue
        words = aligned[identifier]
        target = _plain(str(segment["text"]))
        plain_words = [_plain(str(word["text"])) for word in words]
        if any(not word for word in plain_words):
            continue
        joined = "".join(plain_words)
        if joined == target:
            valid[identifier] = [dict(word) for word in words]
        elif rejections is not None:
            rejections[identifier] = {"code": "text_mismatch"}
    return valid


def _one_alias(item: Mapping[str, Any], aliases: Sequence[str], field: str) -> object:
    present = [name for name in aliases if name in item]
    if len(present) != 1:
        raise ValueError(f"VibeVoice segment must carry exactly one {field} field")
    return item[present[0]]


def _speaker(value: object) -> str | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, str):
        label = value.strip()
        if label == "N/A":
            return None
        if not label:
            raise ValueError("VibeVoice speaker label must not be empty")
        return label
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("VibeVoice speaker label must be a string or integer")
    return str(value)


def _raw_array_prefix(text: str) -> list[Mapping[str, Any]]:
    """Decode complete objects from the first JSON array, stopping at the first cut.

    ``JSONDecoder.raw_decode`` understands quoted delimiters and escapes.  Counting braces
    does not: recorded event text itself contains square brackets, and ordinary speech can
    contain any of the same characters as the surrounding JSON syntax.
    """

    start = text.find("[")
    fence_start = text.find("```json")
    fenced = fence_start >= 0 and (start < 0 or fence_start < start)
    if fenced:
        _validate_generated_preamble(text, fence_start)
        content_start = fence_start + len("```json")
        start = text.find("[", content_start)
        if start < 0:
            if text[content_start:].strip():
                raise ValueError("VibeVoice generated JSON code block has no array")
            return []
        if text[content_start:start].strip():
            raise ValueError("VibeVoice generated JSON code block has content before its array")
    if start < 0:
        return []
    if not fenced:
        _validate_generated_preamble(text, start)
    decoder = _raw_decoder()
    cursor = start + 1
    found: list[Mapping[str, Any]] = []

    def finish_closed_array() -> list[Mapping[str, Any]]:
        suffix = text[cursor + 1 :].lstrip()
        if fenced and suffix in {"`", "``"}:
            # A capped generation may end inside the closing fence after the
            # transcript array itself has closed.  Only exact byte prefixes of
            # that wrapper are salvageable; they cannot contain transcript data.
            return found
        if fenced and suffix.startswith("```"):
            suffix = suffix[len("```") :]
        suffix = suffix.strip()
        if suffix:
            raise ValueError("VibeVoice generated text continues after its JSON array")
        return found

    while True:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            return found
        if text[cursor] == "]":
            return finish_closed_array()
        try:
            value, cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            return found
        except RecursionError as exc:
            raise ValueError("VibeVoice generated JSON nesting is too deep") from exc
        if not isinstance(value, Mapping):
            raise TypeError("VibeVoice transcript array elements must be objects")
        found.append(value)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            return found
        if text[cursor] == "]":
            return finish_closed_array()
        if text[cursor] != ",":
            raise ValueError("VibeVoice generated JSON is malformed after its complete prefix")
        cursor += 1


def _complete_raw_array(text: str) -> list[object]:
    """Parse the same complete fenced or unfenced array accepted by the stage."""

    array_start = text.find("[")
    fence_start = text.find("```json")
    fenced = fence_start >= 0 and (array_start < 0 or fence_start < array_start)
    if fenced:
        _validate_generated_preamble(text, fence_start)
        content_start = fence_start + len("```json")
        array_start = text.find("[", content_start)
        if array_start < 0:
            raise ValueError("VibeVoice generated JSON code block has no array")
        if text[content_start:array_start].strip():
            raise ValueError("VibeVoice generated JSON code block has content before its array")
    else:
        if array_start < 0:
            raise ValueError("VibeVoice generated text has no JSON array")
        _validate_generated_preamble(text, array_start)
    value, end = _raw_decoder().raw_decode(text, array_start)
    suffix = text[end:].strip()
    if fenced:
        if not suffix.startswith("```"):
            raise ValueError("VibeVoice generated JSON code block is incomplete")
        if suffix[len("```") :].strip():
            raise ValueError("VibeVoice generated text continues after its JSON code block")
    elif suffix:
        raise ValueError("VibeVoice generated text continues after its JSON array")
    if not isinstance(value, list):
        raise TypeError("VibeVoice generated JSON must be an array")
    return value


def _normalize_segment(
    item: Mapping[str, Any],
    *,
    index: int,
    offset_seconds: float,
    clip_duration_seconds: float | None,
) -> dict[str, Any]:
    keys = set(item)
    if keys <= _NORMALIZED_KEYS and {"start_time", "end_time", "text"} <= keys:
        start_value = item["start_time"]
        end_value = item["end_time"]
        text = item["text"]
        speaker = item.get("speaker_id", _MISSING)
    elif keys <= _RAW_KEYS and "Content" in keys:
        start_value = _one_alias(item, ("Start", "Start time"), "start")
        end_value = _one_alias(item, ("End", "End time"), "end")
        text = item["Content"]
        speaker = (
            _one_alias(item, ("Speaker", "Speaker ID"), "speaker")
            if {"Speaker", "Speaker ID"} & keys
            else _MISSING
        )
    else:
        raise ValueError(f"VibeVoice segment {index} has unsupported keys {sorted(keys)}")

    start = _number(start_value, f"VibeVoice segment {index} start")
    end = _number(end_value, f"VibeVoice segment {index} end")
    if end <= start:
        raise ValueError(f"VibeVoice segment {index} must have positive duration")
    if clip_duration_seconds is not None:
        if end > clip_duration_seconds + 1e-6:
            raise ValueError(f"VibeVoice segment {index} exceeds the selected audio clip")
        # Preserve the existing backend-rounding tolerance without publishing a
        # timestamp outside the exact sample-aligned clip that this document owns.
        end = min(end, clip_duration_seconds)
    if not isinstance(text, str):
        raise TypeError(f"VibeVoice segment {index} text must be a string")
    if not text.strip():
        raise ValueError(f"VibeVoice segment {index} text must not be empty")

    published_start = round(start + offset_seconds, 6)
    published_end = round(end + offset_seconds, 6)
    if published_end <= published_start:
        raise ValueError(
            f"VibeVoice segment {index} must have positive duration on the source timeline"
        )
    alignable = _EVENT_TAG.fullmatch(text.strip()) is None
    normalized: dict[str, Any] = {
        "text": text,
        "start": published_start,
        "end": published_end,
        "alignable": alignable,
    }
    label = _speaker(speaker)
    if not alignable and label is not None:
        raise ValueError(f"VibeVoice non-speech event segment {index} must not carry a speaker")
    if label is not None:
        normalized["speaker"] = label
    return normalized


def _normalize_segments(
    values: Sequence[object],
    *,
    offset_seconds: float,
    clip_duration_seconds: float | None,
) -> tuple[dict[str, Any], ...]:
    """Normalize one chronological segment stream at the public boundary."""

    segments: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise TypeError(f"VibeVoice segment {index} must be an object")
        segment = _normalize_segment(
            item,
            index=index,
            offset_seconds=offset_seconds,
            clip_duration_seconds=clip_duration_seconds,
        )
        if float(segment["start"]) < previous_end:
            raise ValueError("VibeVoice segments must be chronological and non-overlapping")
        previous_end = float(segment["end"])
        segments.append(segment)
    return tuple(segments)


def normalize_vibevoice_result(
    payload: Mapping[str, Any],
    *,
    offset_seconds: float = 0.0,
    clip_duration_seconds: float | None = None,
) -> VibeVoiceResult:
    """Normalize a complete result or salvage the complete prefix of a capped decode.

    Bounds in a ranged VibeVoice process are relative to its internal clip.  The offset is
    applied here so no backend-relative timestamp can cross the adapter boundary.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("VibeVoice stage result must be an object")
    offset = _number(offset_seconds, "VibeVoice timeline offset")
    clip_duration = (
        None
        if clip_duration_seconds is None
        else _number(clip_duration_seconds, "VibeVoice clip duration")
    )
    hit_max_new_tokens = payload.get("hit_max_new_tokens")
    if not isinstance(hit_max_new_tokens, bool):
        raise TypeError("VibeVoice hit_max_new_tokens must be a boolean")

    if hit_max_new_tokens:
        raw_text = payload.get("raw_text")
        if not isinstance(raw_text, str):
            raise TypeError("VibeVoice truncated result raw_text must be a string")
        values: Sequence[object] = _raw_array_prefix(raw_text)
        segments = _normalize_segments(
            values,
            offset_seconds=offset,
            clip_duration_seconds=clip_duration,
        )
    else:
        raw_text = payload.get("raw_text")
        if not isinstance(raw_text, str):
            raise TypeError("VibeVoice complete result raw_text must be a string")
        try:
            generated_values = _complete_raw_array(raw_text)
        except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("VibeVoice complete raw_text is not a complete JSON array") from exc
        cleaned_values = payload.get("segments")
        if not isinstance(cleaned_values, list):
            raise TypeError("VibeVoice stage result segments must be an array")
        # The pinned post-processor only maps generated aliases to normalized
        # names.  Bind every public value to the generated JSON itself, and use
        # the independently returned cleaned stream only as a drift check.
        segments = _normalize_segments(
            generated_values,
            offset_seconds=offset,
            clip_duration_seconds=clip_duration,
        )
        cleaned_segments = _normalize_segments(
            cleaned_values,
            offset_seconds=offset,
            clip_duration_seconds=clip_duration,
        )
        if cleaned_segments != segments:
            raise ValueError("VibeVoice post-process result differs from generated JSON")

    watermark = float(segments[-1]["end"]) if hit_max_new_tokens and segments else None
    return VibeVoiceResult(segments, hit_max_new_tokens, watermark)
