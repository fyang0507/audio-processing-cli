#!/usr/bin/env python3
"""Prepare a frozen CantoMap audio slice and timestamped reference JSON.

The CantoMap checkout and every generated artifact live under ignored
``model_tests/benchmark_data``. Only this parser and the slice manifest are
intended to be tracked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
import wave
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "model_tests/benchmark_data/cantomap"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "manifests/cantomap_yue_hk_37_38_d.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "model_tests/benchmark_data/prepared"
ANNOTATION_TOKEN = re.compile(r"&[A-Za-z]+[0-9]+")
UNKNOWN_TOKEN = re.compile(r"(?<![A-Za-z])xxx(?![A-Za-z])", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an ASR WAV and ELAN-derived reference for CantoMap."
    )
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to benchmark_data/prepared/<slice_id>.",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Parse and validate the ELAN reference without invoking ffmpeg.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate an existing prepared WAV.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("slice_id", "language", "corpus", "source", "clip", "tiers"):
        require(key in manifest, f"manifest is missing {key!r}")
    return manifest


def checkout_revision(corpus_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(corpus_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"not a Git checkout: {corpus_root}") from exc


def wav_metadata(path: Path) -> dict[str, int]:
    try:
        with wave.open(str(path), "rb") as source:
            frames = source.getnframes()
            rate = source.getframerate()
            return {
                "duration_ms": round(frames * 1000 / rate),
                "sample_rate_hz": rate,
                "channels": source.getnchannels(),
                "sample_width_bytes": source.getsampwidth(),
            }
    except (EOFError, wave.Error) as exc:
        pointer_hint = (
            " (is this still a Git LFS pointer?)"
            if path.stat().st_size < 1024
            else ""
        )
        raise ValueError(f"cannot read WAV {path}{pointer_hint}") from exc


def normalize_for_cer(text: str) -> str:
    """Create a conservative orthographic CER string.

    ELAN pause markers (#), ampersand annotation tokens, and unknown-speech
    placeholders are omitted. The raw fields retain every source symbol so
    filler/particle evaluation can be defined and adjudicated separately.
    """

    text = unicodedata.normalize("NFKC", text)
    text = ANNOTATION_TOKEN.sub("", text)
    text = UNKNOWN_TOKEN.sub("", text)
    text = text.replace("#", "")
    return "".join(
        character.casefold()
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def aligned_tier(
    document: ET.Element,
    time_slots: dict[str, int],
    tier_id: str,
) -> dict[tuple[int, int], dict[str, str]]:
    tier = next(
        (node for node in document.findall("./TIER") if node.get("TIER_ID") == tier_id),
        None,
    )
    require(tier is not None, f"ELAN tier {tier_id!r} not found")
    result: dict[tuple[int, int], dict[str, str]] = {}
    for annotation in tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION"):
        start_ref = annotation.get("TIME_SLOT_REF1")
        end_ref = annotation.get("TIME_SLOT_REF2")
        require(
            start_ref in time_slots and end_ref in time_slots,
            f"tier {tier_id!r} contains an unresolved time slot",
        )
        interval = (time_slots[start_ref], time_slots[end_ref])
        require(
            interval not in result,
            f"duplicate interval {interval} in tier {tier_id!r}",
        )
        result[interval] = {
            "annotation_id": annotation.get("ANNOTATION_ID", ""),
            "value": annotation.findtext("./ANNOTATION_VALUE", default="").strip(),
        }
    return result


def parse_segments(eaf_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    document = ET.parse(eaf_path).getroot()
    time_slots = {
        node.get("TIME_SLOT_ID", ""): int(node.get("TIME_VALUE", ""))
        for node in document.findall("./TIME_ORDER/TIME_SLOT")
        if node.get("TIME_SLOT_ID") and node.get("TIME_VALUE")
    }
    start_ms = manifest["clip"]["start_ms"]
    end_ms = manifest["clip"]["end_ms"]
    suffixes = manifest["tiers"]
    segments: list[dict[str, Any]] = []

    for speaker in suffixes["speakers"]:
        characters = aligned_tier(
            document, time_slots, speaker + suffixes["character_suffix"]
        )
        words = aligned_tier(document, time_slots, speaker + suffixes["word_suffix"])
        jyutping = aligned_tier(
            document, time_slots, speaker + suffixes["jyutping_suffix"]
        )
        require(
            characters.keys() == words.keys() == jyutping.keys(),
            f"{speaker} character, word, and Jyutping intervals do not match",
        )
        crossing = [
            interval
            for interval in characters
            if (interval[0] < start_ms < interval[1])
            or (interval[0] < end_ms < interval[1])
        ]
        require(not crossing, f"clip boundary cuts {speaker} annotations: {crossing}")

        for (source_start, source_end), annotation in characters.items():
            if source_start < start_ms or source_end > end_ms:
                continue
            require(
                source_start < source_end,
                f"invalid interval {(source_start, source_end)}",
            )
            segments.append(
                {
                    "source_annotation_id": annotation["annotation_id"],
                    "speaker": speaker,
                    "start_ms": source_start - start_ms,
                    "end_ms": source_end - start_ms,
                    "source_start_ms": source_start,
                    "source_end_ms": source_end,
                    "text": {
                        "characters_raw": annotation["value"],
                        "characters_segmented": words[(source_start, source_end)]["value"],
                        "characters_cer": normalize_for_cer(annotation["value"]),
                        "jyutping": jyutping[(source_start, source_end)]["value"],
                    },
                }
            )

    segments.sort(
        key=lambda item: (item["start_ms"], item["end_ms"], item["speaker"])
    )
    for index, segment in enumerate(segments, start=1):
        segment["segment_id"] = f"seg-{index:04d}"
    return segments


def overlap_regions(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for index, left in enumerate(segments):
        for right in segments[index + 1 :]:
            if right["start_ms"] >= left["end_ms"]:
                break
            start_ms = max(left["start_ms"], right["start_ms"])
            end_ms = min(left["end_ms"], right["end_ms"])
            if left["speaker"] != right["speaker"] and start_ms < end_ms:
                regions.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": end_ms - start_ms,
                        "speakers": sorted([left["speaker"], right["speaker"]]),
                        "segment_ids": [left["segment_id"], right["segment_id"]],
                    }
                )
    return regions


def build_references(
    segments: list[dict[str, Any]], speakers: list[str]
) -> dict[str, Any]:
    def concatenate(items: list[dict[str, Any]]) -> dict[str, str]:
        return {
            "characters_raw": "\n".join(item["text"]["characters_raw"] for item in items),
            "characters_cer": "".join(item["text"]["characters_cer"] for item in items),
            "jyutping": " ".join(item["text"]["jyutping"] for item in items),
        }

    return {
        "chronological": concatenate(segments),
        "by_speaker": {
            speaker: concatenate(
                [item for item in segments if item["speaker"] == speaker]
            )
            for speaker in speakers
        },
    }


def validate_selection(
    manifest: dict[str, Any],
    segments: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
) -> dict[str, Any]:
    speakers = manifest["tiers"]["speakers"]
    by_speaker = Counter(item["speaker"] for item in segments)
    transitions = sum(
        left["speaker"] != right["speaker"]
        for left, right in zip(segments, segments[1:])
    )
    counts = {
        "segments": len(segments),
        "segments_by_speaker": {speaker: by_speaker[speaker] for speaker in speakers},
        "speaker_transitions": transitions,
        "cross_speaker_overlap_pairs": len(overlaps),
        "cross_speaker_overlap_ms": sum(item["duration_ms"] for item in overlaps),
        "segments_with_empty_cer_reference": sum(
            not item["text"]["characters_cer"] for item in segments
        ),
    }
    expected = manifest["selection"]
    require(counts["segments"] == expected["expected_segments"], "segment count drift")
    require(
        counts["segments_by_speaker"] == expected["expected_segments_by_speaker"],
        "per-speaker segment count drift",
    )
    require(
        counts["speaker_transitions"] == expected["expected_speaker_transitions"],
        "speaker-transition count drift",
    )
    require(
        counts["cross_speaker_overlap_pairs"]
        == expected["expected_cross_speaker_overlap_pairs"],
        "cross-speaker overlap count drift",
    )
    return counts


def prepare_audio(
    source: Path,
    destination: Path,
    manifest: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        metadata = wav_metadata(destination)
    else:
        clip = manifest["clip"]
        preparation = manifest["preparation"]
        duration_ms = clip["end_ms"] - clip["start_ms"]
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}-",
            suffix=".wav",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{clip['start_ms'] / 1000:.3f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{duration_ms / 1000:.3f}",
                    "-map",
                    "0:a:0",
                    "-ac",
                    str(preparation["channels"]),
                    "-ar",
                    str(preparation["sample_rate_hz"]),
                    "-c:a",
                    preparation["codec"],
                    str(temporary_path),
                ],
                check=True,
            )
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        metadata = wav_metadata(destination)

    expected_duration = manifest["clip"]["end_ms"] - manifest["clip"]["start_ms"]
    require(abs(metadata["duration_ms"] - expected_duration) <= 1, "prepared duration drift")
    require(
        metadata["sample_rate_hz"] == manifest["preparation"]["sample_rate_hz"],
        "prepared sample-rate drift",
    )
    require(
        metadata["channels"] == manifest["preparation"]["channels"],
        "prepared channel-count drift",
    )
    return {"path": str(destination), "sha256": sha256(destination), **metadata}


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    corpus_root = args.corpus_root.resolve()
    manifest = read_manifest(manifest_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / manifest["slice_id"]).resolve()
    )
    source = manifest["source"]
    eaf_path = corpus_root / source["eaf"]
    audio_path = corpus_root / source["audio"]
    require(eaf_path.is_file(), f"missing EAF: {eaf_path}")
    require(audio_path.is_file(), f"missing audio: {audio_path}")
    require(
        checkout_revision(corpus_root) == manifest["corpus"]["revision"],
        "CantoMap checkout revision does not match the frozen manifest",
    )
    require(sha256(eaf_path) == source["eaf_sha256"], "source EAF checksum mismatch")
    require(sha256(audio_path) == source["audio_sha256"], "source audio checksum mismatch")
    source_audio = wav_metadata(audio_path)
    for field in ("duration_ms", "sample_rate_hz", "channels"):
        require(
            source_audio[field] == source[f"audio_{field}"],
            f"source audio {field} drift",
        )

    segments = parse_segments(eaf_path, manifest)
    overlaps = overlap_regions(segments)
    counts = validate_selection(manifest, segments, overlaps)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / "audio.wav"
    if args.reference_only:
        prepared_audio = (
            prepare_audio(audio_path, prepared_path, manifest, force=False)
            if prepared_path.exists()
            else None
        )
    else:
        prepared_audio = prepare_audio(
            audio_path, prepared_path, manifest, args.force
        )

    clip = manifest["clip"]
    speakers = manifest["tiers"]["speakers"]
    result = {
        "schema_version": 1,
        "slice_id": manifest["slice_id"],
        "language": manifest["language"],
        "source": {
            "corpus": manifest["corpus"]["name"],
            "repository": manifest["corpus"]["repository"],
            "revision": manifest["corpus"]["revision"],
            "license": manifest["corpus"]["license"],
            "eaf": source["eaf"],
            "audio": source["audio"],
            "eaf_sha256": source["eaf_sha256"],
            "audio_sha256": source["audio_sha256"],
        },
        "clip": {
            "source_start_ms": clip["start_ms"],
            "source_end_ms": clip["end_ms"],
            "duration_ms": clip["end_ms"] - clip["start_ms"],
            "prepared_audio": prepared_audio,
        },
        "annotation_policy": {
            "unit": "one source ELAN alignable annotation on an F/G character tier",
            "boundary_policy": clip["boundary_policy"],
            "time_origin": "segment times are milliseconds from prepared clip start",
            "cer_normalization": (
                "NFKC; remove whitespace, Unicode punctuation, # pause markers, "
                "&<letters><digits> corpus annotation tokens, and xxx unknown-speech tokens; "
                "casefold remaining Latin text"
            ),
            "epistemic_limit": (
                "ELAN annotation boundaries are utterance/activity units, not independently "
                "adjudicated conversational-turn boundaries"
            ),
        },
        "speakers": [
            {
                "id": speaker,
                "source_tiers": {
                    "characters": speaker + manifest["tiers"]["character_suffix"],
                    "segmented_characters": speaker + manifest["tiers"]["word_suffix"],
                    "jyutping": speaker + manifest["tiers"]["jyutping_suffix"],
                },
            }
            for speaker in speakers
        ],
        "segments": segments,
        "overlap_regions": overlaps,
        "references": build_references(segments, speakers),
        "counts": counts,
    }
    reference_path = output_dir / "reference.json"
    reference_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "slice_id": manifest["slice_id"],
                "reference": str(reference_path),
                "prepared_audio": prepared_audio,
                "counts": counts,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
