"""Human-readable and JSONL renderer shapes."""

from __future__ import annotations

from pathlib import Path

import pytest
from export_test_support import (
    Cue,
    _payload,
    _write,
    export_documents,
    json,
    render_jsonl,
    render_markdown,
    render_srt,
    render_text,
    render_vtt,
)


@pytest.mark.parametrize("first_text", ["First.", "First.\nStill first.\n\nFinal line.\n"])
def test_human_and_jsonl_exports_do_not_require_timing(tmp_path: Path, first_text: str) -> None:
    payload = _payload(
        [
            {"segment_id": "seg_old", "text": first_text},
            {"segment_id": "seg_next", "text": "[Music]"},
        ]
    )
    path = _write(tmp_path / "plain.json", payload)
    original_bytes = path.read_bytes()
    text = export_documents([path], "txt")
    markdown = export_documents([path], "md")
    jsonl = export_documents([path], "jsonl")

    assert text.content == first_text + "\n[Music]\n"
    assert markdown.content == "# Transcript\n\n" + first_text + "\n\n[Music]\n"
    rows = [json.loads(line) for line in jsonl.content.splitlines()]
    assert rows == [
        {"segment_id": "seg_0", "text": first_text},
        {"segment_id": "seg_1", "text": "[Music]"},
    ]
    assert all("provenance" not in row for row in rows)
    assert path.read_bytes() == original_bytes


def test_writer_shapes_are_real_srt_vtt_markdown_text_and_jsonl() -> None:
    cues = [Cue(2310, 4710, "Hello & <world>.", "Speaker 1")]
    assert render_srt(cues) == ("1\n00:00:02,310 --> 00:00:04,710\nHello & <world>.\n")
    assert render_vtt(cues) == (
        "WEBVTT\n\n1\n00:00:02.310 --> 00:00:04.710\n<v Speaker 1>Hello &amp; &lt;world&gt;.\n"
    )
    assert "<v " not in render_vtt([Cue(0, 1, "No speaker")])
    segments = [{"segment_id": "seg_0", "text": "Hello.", "speaker": "S1"}]
    assert render_text(segments) == "[S1] Hello.\n"
    assert render_markdown(segments) == "# Transcript\n\n[S1] Hello.\n"
    assert json.loads(render_jsonl(segments)) == segments[0]


def test_vtt_voice_annotations_cannot_inject_lines_or_empty_tags() -> None:
    rendered = render_vtt(
        [
            Cue(0, 1000, "Hello.", "Speaker\nInjected"),
            Cue(1000, 2000, "World.", " \t "),
            Cue(2000, 3000, "Safe.", "A > B\x00"),
            Cue(3000, 4000, "Quoted.", 'O\'Neil "Two"'),
        ]
    )
    assert "<v Speaker Injected>Hello." in rendered
    assert "<v    >" not in rendered
    assert "\x00" not in rendered
    assert "<v A &gt; B>Safe." in rendered
    assert '<v O\'Neil "Two">Quoted.' in rendered
    assert "&#x27;" not in rendered and "&quot;" not in rendered
