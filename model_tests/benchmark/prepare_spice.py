#!/usr/bin/env python3
"""Extract frozen participant-reference intervals from a SpiCE TextGrid."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ITEM_RE = re.compile(r"(?ms)^    item \[(\d+)\]:\n(.*?)(?=^    item \[|\Z)")
INTERVAL_RE = re.compile(
    r'(?ms)^        intervals \[\d+\]:\n'
    r'            xmin = ([0-9.eE+-]+) *\n'
    r'            xmax = ([0-9.eE+-]+) *\n'
    r'            text = "(.*?)" *$'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tiers(path: Path) -> dict[str, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-16")
    tiers: dict[str, list[dict[str, Any]]] = {}
    for _, body in ITEM_RE.findall(text):
        name_match = re.search(r'^        name = "(.*?)" *$', body, re.MULTILINE)
        if not name_match:
            continue
        intervals = []
        for start, end, value in INTERVAL_RE.findall(body):
            intervals.append({
                "start_s": float(start),
                "end_s": float(end),
                "text": value.replace('""', '"'),
            })
        tiers[name_match.group(1)] = intervals
    return tiers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--textgrid", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, required=True,
                        help="Start in original recording seconds")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()

    textgrid = Path(args.textgrid).resolve()
    audio = Path(args.audio).resolve()
    output = Path(args.output).resolve()
    tiers = parse_tiers(textgrid)
    end = args.start + args.duration
    utterances = []
    for item in tiers["utterance"]:
        if not item["text"] or item["end_s"] <= args.start or item["start_s"] >= end:
            continue
        utterances.append({
            "start_s": max(item["start_s"], args.start) - args.start,
            "end_s": min(item["end_s"], end) - args.start,
            "speaker": "participant",
            "text": item["text"],
        })
    tasks = [item for item in tiers["task"]
             if item["end_s"] > args.start and item["start_s"] < end]
    manifest = {
        "schema_version": 1,
        "corpus": "SpiCE: Speech in Cantonese and English",
        "license": "CC BY 4.0",
        "source_url": args.source_url,
        "reference_scope": (
            "Human-corrected participant speech only. Interviewer speech is audible "
            "but intentionally absent from the corpus transcript."
        ),
        "source": {
            "audio_filename": audio.name,
            "audio_sha256": sha256(audio),
            "textgrid_filename": textgrid.name,
            "textgrid_sha256": sha256(textgrid),
            "start_s": args.start,
            "duration_s": args.duration,
            "tasks": tasks,
        },
        "utterances": utterances,
        "reference_text": " ".join(item["text"] for item in utterances),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "utterances": len(utterances),
        "reference_characters": len(manifest["reference_text"]),
    }))


if __name__ == "__main__":
    main()
