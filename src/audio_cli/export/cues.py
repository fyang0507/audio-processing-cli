"""Deterministic subtitle cue construction from normalized per-word timing.

Cue times are a representation of word bounds, not a new timing estimate: the first and
last word in a cue own its start and end.  This module therefore never reads a segment
extent, extends a cue for readability, trims an overlap, or nudges a start after rounding.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

_SENTENCE_FINAL = frozenset(".!?…。！？；‼⁇⁈⁉")
_CLAUSE_FINAL = frozenset(",;:，；：、")
_TRAILING_WRAPPERS = frozenset("'\"’”»》」』】〕〗〙〛）)]}")
_LEADING_WRAPPERS = frozenset("‘“«《「『【〔〖〘〚（([{")
_AMBIGUOUS_ASCII_QUOTES = frozenset("'\"")


class CueError(ValueError):
    """Timed words cannot be rendered without violating cue invariants."""


@dataclass(frozen=True)
class CuePolicy:
    """The fixed v1 cue policy consumed by export.

    These values are deliberately not command-line tuning options.  Changing them is a
    versioned policy decision, not per-export nondeterminism.
    """

    max_duration_s: float = 7.0
    max_lines: int = 2
    max_chars_per_line_latin: int = 42
    max_chars_per_line_cjk: int = 16
    max_words_latin: int = 24
    max_gap_s: float = 0.8
    min_clause_chars: int = 20
    quantization_ms: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Return the public policy summary, kept intentionally smaller than machinery."""
        return {
            "max_duration_s": self.max_duration_s,
            "max_lines": self.max_lines,
            "max_chars_per_line_cjk": self.max_chars_per_line_cjk,
            "break_priority": [
                "sentence_end",
                "clause_punctuation",
                "word_gap",
            ],
            "never_spans_speaker_change": True,
            "quantization_ms": self.quantization_ms,
        }


V1_CUE_POLICY = CuePolicy()


@dataclass(frozen=True)
class Cue:
    """One fully quantized subtitle cue."""

    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class CueBuild:
    cues: tuple[Cue, ...]
    warnings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _MappedWord:
    start: float
    end: float
    char_start: int
    char_end: int


def _plain(value: str) -> str:
    return "".join(
        character
        for character in value.casefold()
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _plain_positions(value: str) -> tuple[str, list[int]]:
    plain: list[str] = []
    positions: list[int] = []
    for position, character in enumerate(value):
        if character.isspace() or unicodedata.category(character).startswith("P"):
            continue
        for folded in character.casefold():
            if folded.isspace() or unicodedata.category(folded).startswith("P"):
                continue
            plain.append(folded)
            positions.append(position)
    return "".join(plain), positions


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CueError(f"{field} must be a finite number")
    found = float(value)
    if not math.isfinite(found):
        raise CueError(f"{field} must be a finite number")
    return found


def _reject_non_whitespace_controls(value: str, field: str) -> None:
    for character in value:
        if unicodedata.category(character) == "Cc" and not character.isspace():
            raise CueError(
                f"{field} contains a non-whitespace control character that cannot "
                "be represented safely in a subtitle cue"
            )


def _map_words(
    segment: Mapping[str, Any],
    *,
    segment_index: int,
    duration: float,
) -> list[_MappedWord]:
    text = segment["text"]
    words = segment["words"]
    _reject_non_whitespace_controls(text, f"segments[{segment_index}].text")
    plain_text, positions = _plain_positions(text)
    normalized_words: list[str] = []
    for word_index, word in enumerate(words):
        _reject_non_whitespace_controls(
            word["text"], f"segments[{segment_index}].words[{word_index}].text"
        )
        normalized = _plain(word["text"])
        if not normalized:
            raise CueError(
                f"segments[{segment_index}].words[{word_index}].text has no lexical content"
            )
        normalized_words.append(normalized)
    if "".join(normalized_words) != plain_text:
        raise CueError(
            f"segments[{segment_index}] word text does not map to segment text under "
            "the punctuation invariant"
        )

    lexical_starts: list[int] = []
    lexical_ends: list[int] = []
    cursor = 0
    for normalized in normalized_words:
        lexical_starts.append(positions[cursor])
        cursor += len(normalized)
        lexical_ends.append(positions[cursor - 1] + 1)
    for word_index in range(1, len(lexical_starts)):
        if lexical_starts[word_index] < lexical_ends[word_index - 1]:
            raise CueError(
                f"segments[{segment_index}] word partition splits one source "
                "character's case-fold expansion across timed words"
            )

    word_starts = [0]
    for word_index in range(1, len(normalized_words)):
        lexical_start = lexical_starts[word_index]
        separator_start = lexical_ends[word_index - 1]
        separator = text[separator_start:lexical_start]
        cursor = len(separator)
        while cursor and separator[cursor - 1].isspace():
            cursor -= 1
        wrapper_end = cursor
        while cursor:
            wrapper = separator[cursor - 1]
            if wrapper in _LEADING_WRAPPERS:
                cursor -= 1
                continue
            if wrapper in _AMBIGUOUS_ASCII_QUOTES:
                before = separator[: cursor - 1]
                after = separator[cursor:]
                # Straight quotes are symmetric.  Treat one as opening only when
                # its local typography says it introduces the next lexical token:
                # either it follows separating whitespace, or it directly hugs
                # that token.  A quote followed by whitespace after sentence
                # punctuation (``Hello.\" Then``) remains with the prior word.
                if (before and before[-1].isspace()) or not after:
                    cursor -= 1
                    continue
            break
        word_starts.append(separator_start + cursor if cursor < wrapper_end else lexical_start)

    mapped: list[_MappedWord] = []
    previous_end = -1.0
    for word_index, word in enumerate(words):
        field = f"segments[{segment_index}].words[{word_index}]"
        start = _finite_number(word["start"], f"{field}.start")
        end = _finite_number(word["end"], f"{field}.end")
        if start < 0 or end < start or end > duration:
            raise CueError(f"{field} bounds must satisfy 0 <= start <= end <= source duration")
        if start < previous_end:
            raise CueError(f"{field} overlaps the preceding word")
        previous_end = end
        mapped.append(
            _MappedWord(
                start=start,
                end=end,
                char_start=word_starts[word_index],
                char_end=(
                    word_starts[word_index + 1] if word_index + 1 < len(words) else len(text)
                ),
            )
        )
    return mapped


def _ends_with(text: str, marks: frozenset[str]) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in _TRAILING_WRAPPERS:
        stripped = stripped[:-1].rstrip()
    return bool(stripped) and stripped[-1] in marks


def _contains_cjk(text: str) -> bool:
    for character in text:
        point = ord(character)
        if (
            0x3400 <= point <= 0x4DBF
            or 0x4E00 <= point <= 0x9FFF
            or 0x3040 <= point <= 0x30FF
            or 0xAC00 <= point <= 0xD7AF
            or 0xF900 <= point <= 0xFAFF
        ):
            return True
    return False


def _text_for(text: str, words: Sequence[_MappedWord]) -> str:
    # Keep the raw slice coordinates aligned with ``_MappedWord.char_start``.
    # Rendering normalizes surrounding whitespace in ``_one_line``; stripping here
    # would shift every later word boundary and could split a word during wrapping.
    return text[words[0].char_start : words[-1].char_end]


def _visible_length(text: str) -> int:
    return len(" ".join(text.split()))


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _wrap(
    text: str,
    words: Sequence[_MappedWord],
    *,
    policy: CuePolicy,
) -> tuple[str, bool]:
    """Balance at a real word boundary and report an unavoidable overlong line."""
    width = (
        policy.max_chars_per_line_cjk if _contains_cjk(text) else policy.max_chars_per_line_latin
    )
    one_line = _one_line(text)
    if _visible_length(one_line) <= width:
        return one_line, False
    if len(words) < 2 or policy.max_lines < 2:
        return one_line, True

    origin = words[0].char_start
    candidates: list[tuple[tuple[int, int], str, str]] = []
    for index in range(1, len(words)):
        boundary = words[index].char_start - origin
        head = _one_line(text[:boundary])
        tail = _one_line(text[boundary:])
        if not head or not tail:
            continue
        head_length = _visible_length(head)
        tail_length = _visible_length(tail)
        candidates.append(
            (
                (max(head_length, tail_length), abs(head_length - tail_length)),
                head,
                tail,
            )
        )
    if not candidates:
        return one_line, True
    within = [
        candidate
        for candidate in candidates
        if _visible_length(candidate[1]) <= width and _visible_length(candidate[2]) <= width
    ]
    _, head, tail = min(within or candidates, key=lambda item: item[0])
    rendered = f"{head}\n{tail}"
    overlong = _visible_length(head) > width or _visible_length(tail) > width
    return rendered, overlong


def _to_ms(seconds: float, quantum_ms: int) -> int:
    try:
        units = round(seconds * 1000 / quantum_ms)
    except (OverflowError, ValueError) as exc:
        raise CueError(
            "word timestamp is finite but too large for subtitle millisecond quantization"
        ) from exc
    return units * quantum_ms


def build_cues(
    segments: Sequence[Mapping[str, Any]],
    *,
    duration: float,
    policy: CuePolicy = V1_CUE_POLICY,
) -> CueBuild:
    """Build source-timeline cues from real word arrays.

    Segments without a word stream are deliberately omitted.  Every segment is processed
    independently, so a cue can never cross a segment or speaker change.
    """
    cues: list[Cue] = []
    exact_cue_bounds: list[tuple[float, float]] = []
    warnings: list[dict[str, Any]] = []

    for segment_index, segment in enumerate(segments):
        words = segment.get("words")
        if not isinstance(words, (list, tuple)) or not words:
            continue
        mapped = _map_words(segment, segment_index=segment_index, duration=duration)
        text = segment["text"]
        speaker = segment.get("speaker")
        current: list[_MappedWord] = []

        def flush(*, segment_text: str = text, segment_speaker: object = speaker) -> None:
            nonlocal current
            if not current:
                return
            cue_text = _text_for(segment_text, current)
            start_ms = _to_ms(current[0].start, policy.quantization_ms)
            end_ms = _to_ms(current[-1].end, policy.quantization_ms)
            if cue_text:
                # Exact chronology is an input invariant, including candidates that
                # later collapse on the subtitle millisecond grid.
                exact_cue_bounds.append((current[0].start, current[-1].end))
            if cue_text and end_ms > start_ms:
                wrapped, overlong = _wrap(cue_text, current, policy=policy)
                cues.append(
                    Cue(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=wrapped,
                        speaker=(segment_speaker if isinstance(segment_speaker, str) else None),
                    )
                )
                if overlong:
                    warnings.append(
                        {
                            "code": "cue_line_overlong",
                            "blocking": False,
                            "detail": (
                                "a cue cannot meet the line-width policy without splitting "
                                "inside a timed word"
                            ),
                        }
                    )
                if current[-1].end - current[0].start > policy.max_duration_s:
                    warnings.append(
                        {
                            "code": "cue_duration_overlong",
                            "blocking": False,
                            "detail": (
                                "a single timed word exceeds the cue-duration policy and "
                                "cannot be split without fabricating a word boundary"
                            ),
                        }
                    )
            elif cue_text:
                warnings.append(
                    {
                        "code": "cue_dropped_after_quantization",
                        "blocking": False,
                        "detail": (
                            "a cue whose exact word bounds collapse on the millisecond grid "
                            "was omitted rather than assigned fabricated timing"
                        ),
                    }
                )
            current = []

        for word in mapped:
            if current:
                prospective = [*current, word]
                prospective_text = _text_for(text, prospective)
                cjk = _contains_cjk(prospective_text)
                line_width = (
                    policy.max_chars_per_line_cjk if cjk else policy.max_chars_per_line_latin
                )
                gap = word.start - current[-1].end
                too_long = word.end - current[0].start > policy.max_duration_s
                too_wide = _visible_length(prospective_text) > (line_width * policy.max_lines)
                too_many = (
                    not cjk
                    and policy.max_words_latin > 0
                    and len(current) >= policy.max_words_latin
                )
                if gap >= policy.max_gap_s or too_long or too_wide or too_many:
                    flush()
            current.append(word)
            current_text = _text_for(text, current)
            word_text = text[word.char_start : word.char_end]
            if _ends_with(word_text, _SENTENCE_FINAL) or (
                _ends_with(word_text, _CLAUSE_FINAL)
                and _visible_length(current_text) >= policy.min_clause_chars
            ):
                flush()
        flush()

    for previous, current in pairwise(exact_cue_bounds):
        if current[0] < previous[1]:
            raise CueError(
                "word-derived cues overlap before millisecond quantization; refusing to "
                "trim or nudge a real word bound"
            )
    for previous, current in pairwise(cues):
        if current.start_ms < previous.end_ms:
            raise CueError(
                "word-derived cues overlap after millisecond quantization; refusing to "
                "trim or nudge a real word bound"
            )
    unique_warnings: list[dict[str, Any]] = []
    seen_warnings: set[tuple[str, str]] = set()
    for warning in warnings:
        identity = (str(warning["code"]), str(warning["detail"]))
        if identity not in seen_warnings:
            seen_warnings.add(identity)
            unique_warnings.append(warning)
    return CueBuild(tuple(cues), tuple(unique_warnings))
