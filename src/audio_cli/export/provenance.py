"""Readable provenance from the exact saved inputs to an offline export."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import MergedTranscript


def render_provenance(merged: MergedTranscript, output_format: str) -> str:
    markdown = output_format == "md"
    lines = ["# Provenance" if markdown else "Provenance", ""]

    def field(label: str, value: Any) -> None:
        # JSON quoting keeps embedded newlines and control characters literal. Markdown
        # metacharacters in saved paths/text must remain data rather than links or markup.
        rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if markdown:
            rendered = re.sub(r"([\\`*_{}\[\]<>()#+.!|\-&~])", r"\\\1", rendered)
        lines.append(f"{'- ' if markdown else ''}{label}: {rendered}")

    field("Source", merged.source["path"])
    field("Stack", merged.documents[0].payload["provenance"]["stack"])
    field("Timebase", merged.source["timebase"])
    if "duration_basis" in merged.source:
        field("Duration basis", merged.source["duration_basis"])
    for document in merged.documents:
        lines.append("")
        field("Input", str(document.path))
        field("Complete", document.payload["complete"])
        field("Owned intervals (seconds)", document.owned_intervals)
        if "coverage" in document.payload:
            field("Coverage", document.payload["coverage"])
        corrections = document.payload["provenance"]["observed"].get("alignment_corrections")
        if corrections:
            field("Alignment boundary corrections", corrections)
    return "\n".join(lines) + "\n\n"
