"""Join accepted alignment corrections to the words each workflow publishes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class CorrectionLedger:
    """Record only corrections that survive sentence/text reconciliation."""

    def __init__(self, corrections: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._by_unit = {
            unit_id: {row["word_index"]: row for row in rows}
            for unit_id, rows in corrections.items()
        }
        self._offsets: dict[str, int] = {}
        self.records: list[dict[str, Any]] = []

    def add_words(self, unit_id: str, segment_id: str, words: Sequence[Mapping[str, Any]]) -> None:
        offset = self._offsets.get(unit_id, 0)
        rows = self._by_unit.get(unit_id, {})
        for index, word in enumerate(words, start=offset):
            row = rows.get(index)
            if row is None:
                continue
            if list(row["applied_bounds"]) != [word["start"], word["end"]]:
                raise ValueError("alignment correction differs from the word being published")
            self.records.append(
                {"unit_id": unit_id, "segment_id": segment_id, "word_id": word["word_id"], **row}
            )
        self._offsets[unit_id] = offset + len(words)

    def observed(self) -> dict[str, Any]:
        return {"alignment_corrections": self.records} if self.records else {}
