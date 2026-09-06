"""Shared punctuation invariant for associating timed words with canonical text."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _plain(value: str) -> str:
    return "".join(
        character
        for character in value.casefold()
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def plain_positions(value: str) -> tuple[str, list[int]]:
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


def normalized_word_texts(
    text: str,
    words: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[str]:
    """Require full lexical coverage, tolerating punctuation, whitespace and casing.

    Empty lexical words cannot supply timing. This checks text coverage only;
    subtitle splitting, line limits, and character-boundary mapping remain cue policy.
    """
    normalized = []
    for index, word in enumerate(words):
        value = _plain(word["text"])
        if not value:
            raise ValueError(f"{field}.words[{index}].text has no lexical content")
        normalized.append(value)
    if "".join(normalized) != _plain(text):
        raise ValueError(
            f"{field} word text does not map to segment text under the punctuation invariant"
        )
    return normalized
