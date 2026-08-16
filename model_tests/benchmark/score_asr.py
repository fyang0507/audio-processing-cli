#!/usr/bin/env python3
"""Score frozen mixed Chinese/Latin references without claiming semantic quality."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


LATIN_WORD = re.compile(
    r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*"
)


def is_han(character: str) -> bool:
    code = ord(character)
    return any(start <= code <= end for start, end in (
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
        (0x20000, 0x2FA1F),
    ))


def mixed_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    # SpiCE appends source-language tags such as @m to a spoken token and uses
    # xxx for audio that was deliberately silenced. Neither is audible content.
    text = re.sub(r"@[A-Za-z]+\b", "", text)
    text = re.sub(r"(?i)(?<![A-Za-z0-9])xxx(?![A-Za-z0-9])", "", text)
    tokens: list[str] = []
    index = 0
    while index < len(text):
        match = LATIN_WORD.match(text, index)
        if match:
            tokens.append(match.group(0).casefold())
            index = match.end()
            continue
        if is_han(text[index]):
            tokens.append(text[index])
        index += 1
    return tokens


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    previous: list[tuple[int, int, int, int]] = [
        (index, 0, index, 0) for index in range(len(hypothesis) + 1)
    ]
    for ref_index, ref_token in enumerate(reference, start=1):
        row = [(ref_index, ref_index, 0, 0)]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            if ref_token == hyp_token:
                match = previous[hyp_index - 1]
                row.append((match[0], match[1], match[2], match[3]))
                continue
            deletion = previous[hyp_index]
            insertion = row[hyp_index - 1]
            substitution = previous[hyp_index - 1]
            candidates = [
                (deletion[0] + 1, deletion[1] + 1, deletion[2], deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2] + 1, insertion[3]),
                (substitution[0] + 1, substitution[1], substitution[2], substitution[3] + 1),
            ]
            row.append(min(candidates, key=lambda item: item[0]))
        previous = row
    distance, deletions, insertions, substitutions = previous[-1]
    return {
        "distance": distance, "deletions": deletions,
        "insertions": insertions, "substitutions": substitutions,
    }


def hypothesis_segments(
    run: dict[str, Any], speaker: str | None = None,
    exclude_control_segments: bool = False,
) -> list[dict[str, Any]]:
    segments = run["output"].get("segments")
    if segments is None:
        segments = run["output"]["result"].get("sentences", [])
    normalized = []
    for item in segments:
        item_speaker = item.get("speaker_id", item.get("speaker"))
        if speaker is not None and str(item_speaker) != speaker:
            continue
        item_text = item.get("text", "")
        if exclude_control_segments and re.fullmatch(r"\s*\[[^\]]+\]\s*", item_text):
            continue
        normalized.append({
            "start_s": float(item.get(
                "start_s", item.get("start_time", item.get("start_ms", 0) / 1000)
            )),
            "end_s": float(item.get(
                "end_s", item.get("end_time", item.get("end_ms", 0) / 1000)
            )),
            "text": item_text,
            "speaker": item_speaker,
        })
    return normalized


def reference_segments(reference: dict[str, Any]) -> tuple[list[dict[str, Any]], float, str]:
    if "utterances" in reference:  # SpiCE participant reference
        return (
            reference["utterances"],
            float(reference["source"]["duration_s"]),
            reference["reference_text"],
        )
    if "segments" in reference:  # prepared CantoMap reference
        segments = [{
            "start_s": item["start_ms"] / 1000,
            "end_s": item["end_ms"] / 1000,
            "text": item["text"]["characters_cer"],
        } for item in reference["segments"] if item["text"]["characters_cer"]]
        return (
            segments,
            float(reference["clip"]["duration_ms"]) / 1000,
            reference["references"]["chronological"]["characters_cer"],
        )
    raise ValueError("unsupported reference schema")


def reference_preprocessing(reference: dict[str, Any]) -> dict[str, Any]:
    if "segments" in reference:
        text_bearing = sum(
            bool(item["text"]["characters_cer"]) for item in reference["segments"]
        )
        return {
            "dataset": "CantoMap",
            "policy": reference["annotation_policy"]["cer_normalization"],
            "source_annotation_segments": len(reference["segments"]),
            "text_bearing_segments": text_bearing,
            "normalization_empty_segments": len(reference["segments"]) - text_bearing,
        }
    if "utterances" in reference:
        return {
            "dataset": "SpiCE",
            "policy": reference.get(
                "reference_scope",
                "Human-corrected participant utterance tier as frozen in the manifest",
            ),
            "source_utterances": len(reference["utterances"]),
        }
    return {"dataset": "unknown", "policy": "unknown"}


def text_in_window(items: list[dict[str, Any]], start: float, end: float) -> str:
    return " ".join(item["text"] for item in items
                    if item["end_s"] > start and item["start_s"] < end)


def score(reference_text: str, hypothesis_text: str) -> dict[str, Any]:
    reference = mixed_tokens(reference_text)
    hypothesis = mixed_tokens(hypothesis_text)
    counts = edit_counts(reference, hypothesis)
    return {
        "reference_tokens": len(reference),
        "hypothesis_tokens": len(hypothesis),
        **counts,
        "mixed_error_rate": counts["distance"] / len(reference) if reference else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-seconds", type=float, default=180.0)
    parser.add_argument(
        "--hypothesis-speaker",
        help=(
            "Score only one anonymous hypothesis speaker label. Use only when the "
            "reference is speaker-scoped and record that the mapping is oracle-selected."
        ),
    )
    parser.add_argument(
        "--include-control-segments",
        action="store_true",
        help="Include whole-segment labels such as [Silence] in transcript scoring",
    )
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text())
    run = json.loads(Path(args.run).read_text())
    hypotheses = hypothesis_segments(
        run, args.hypothesis_speaker,
        exclude_control_segments=not args.include_control_segments,
    )
    references, duration, complete_reference = reference_segments(reference)
    window = min(args.window_seconds, duration)
    starts = sorted({0.0, max(0.0, (duration - window) / 2), max(0.0, duration - window)})
    windows = []
    for start in starts:
        end = start + window
        windows.append({
            "start_s": start,
            "end_s": end,
            **score(text_in_window(references, start, end),
                    text_in_window(hypotheses, start, end)),
        })
    result = {
        "schema_version": 2,
        "metric": "orthography-sensitive native-script mixed-token error rate",
        "scorer_normalization_version": 2,
        "scorer_normalization": (
            "NFKC; one token per Han character or alphanumeric span with internal "
            "apostrophes or hyphens; casefold Latin text; punctuation is a separator; "
            "SpiCE language-origin tags and silenced xxx placeholders removed"
        ),
        "reference_preprocessing": reference_preprocessing(reference),
        "window_selection": (
            "Each window includes the complete text of every segment whose interval "
            "overlaps it; boundaries are not sample-exact crops."
        ),
        "epistemic_limit": (
            "Character/Latin-token edit distance measures transcript agreement only. "
            "It does not measure semantic equivalence, dialect normalization quality, "
            "interviewer transcription, diarization, or behavioral-analysis validity."
        ),
        "reference": str(Path(args.reference).resolve()),
        "run": str(Path(args.run).resolve()),
        "hypothesis_speaker_filter": args.hypothesis_speaker,
        "control_segments_excluded": not args.include_control_segments,
        "overall": score(complete_reference,
                         " ".join(item["text"] for item in hypotheses)),
        "windows": windows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["overall"]))


if __name__ == "__main__":
    main()
