#!/usr/bin/env python3
"""Derive the minimal provisioned-environment partition from the packages' own pins.

VOCABULARY.md declares three provisioned environments as the floor and marks the joint
resolution of the three `torch` packages **unverified**. This probe answers that question
the only way it can be answered — by asking the resolver — and it answers the more general
question behind it: given each package's declared dependencies, what is the *smallest* set
of environments they can be grouped into?

It resolves every non-empty subset of the Python packages, so the result is a full
compatibility matrix rather than a single attempt. A matrix is what proves minimality: one
failed resolution shows that one grouping does not work, while the matrix shows that no
grouping works except the one reported.

Nothing is installed. `uv pip compile` resolves and writes a pinned requirement set; that is
enough to decide compatibility, and it costs under a second per group.

Two rules this probe exists to enforce:

- **The as-built environment is the artifact, not the upstream requirement file.** Every
  recorded figure in `model_tests/` came out of a specific `.venv`, and where an upstream
  file disagrees with it, the file is wrong about what ran. `REQUIREMENTS` below cites, per
  package, exactly where each constraint came from.
- **A dependency the code imports is a dependency, whether or not it is declared.** FireRed's
  LID stage imports `kaldi_native_fbank`, which its `pyproject.toml` does not list.

Usage:

    python model_tests/benchmark/run_env_partition_probe.py \
      --output model_tests/benchmark/results/2026-08-17-environment-partition.json \
      --raw-dir model_tests/benchmark_runs/env_partition
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Every constraint below is transcribed from a file or an installed distribution on this
# machine, and `citation` says which. A pin with no citation does not belong here.
REQUIREMENTS: dict[str, dict] = {
    "qwen3-asr-8bit": {
        "environment_as_built": "/tmp/mlxprobe-venv",
        "python_as_built": "3.13.9",
        "serves_packages": ["qwen3-asr-1.7b-8bit", "qwen3-asr-0.6b-8bit"],
        "citation": (
            "installed dist-info in /tmp/mlxprobe-venv/lib/python3.13/site-packages: "
            "mlx 0.32.0, mlx_audio 0.4.5, mlx_lm 0.31.3, transformers 5.12.1, "
            "huggingface_hub 1.27.0, sentencepiece 0.2.2 (the [stt] extra); no torch. "
            "The interpreter path is recorded in "
            "model_tests/benchmark/results/2026-08-16-qwen-capabilities.json."
        ),
        "requirements": [
            "mlx==0.32.0",
            "mlx-audio[stt]==0.4.5",
        ],
    },
    "firered-asr2s": {
        "environment_as_built": "model_tests/firered/FireRedASR2S/.venv",
        "python_as_built": "3.12.12",
        "serves_packages": ["firered-asr2s"],
        "citation": (
            "model_tests/firered/FireRedASR2S/pyproject.toml on a clean checkout at 4e7d9aa, "
            "which matches the as-built venv (torch 2.10.0, transformers 5.1.0, "
            "huggingface_hub 1.27.0, numpy 2.4.2). Its requirements.txt disagrees — "
            "torch==2.1.0+cu118, transformers==4.51.3 against a CUDA index — and cannot be "
            "what ran on Apple Silicon, so it is not used here. kaldi_native_fbank is added "
            "because fireredasr2s/fireredlid/data/feat.py:7 imports it and pyproject.toml "
            "does not declare it; 1.22.3 is the as-built version."
        ),
        "requirements": [
            "torch==2.10.0",
            "torchaudio==2.10.0",
            "transformers==5.1.0",
            "numpy==2.4.2",
            "cn2an==0.5.23",
            "kaldiio==2.18.1",
            "kaldi_native_fbank==1.22.3",
            "sentencepiece==0.2.1",
            "soundfile==0.13.1",
            "textgrid==1.6.1",
        ],
    },
    "qwen3-forcedaligner": {
        "environment_as_built": "model_tests/forced_aligner/.venv",
        "python_as_built": "3.12.12",
        "serves_packages": ["qwen3-forcedaligner"],
        "citation": (
            "model_tests/forced_aligner/test_single_speaker.py:4 imports Qwen3ForcedAligner "
            "from qwen_asr, and qwen_asr-0.0.6.dist-info/METADATA pins transformers==4.57.6 "
            "and accelerate==1.12.0."
        ),
        "requirements": [
            "qwen-asr==0.0.6",
            "torch",
            "soundfile",
        ],
    },
    "vibevoice-asr-7b": {
        "environment_as_built": "model_tests/vibevoice/VibeVoice/.venv",
        "python_as_built": "3.12.12",
        "serves_packages": ["vibevoice-asr-7b"],
        "citation": (
            "model_tests/vibevoice/VibeVoice/pyproject.toml [project].dependencies at "
            "checkout 94da20d, which caps transformers below 5.0.0. Server and UI extras "
            "(gradio, aiortc, uvicorn, fastapi, pydub, requests) are dropped: the recorded "
            "runner imports the model directly and never serves. The as-built venv resolved "
            "to transformers 4.57.6, torch 2.13.0, diffusers 0.39.0."
        ),
        "requirements": [
            "transformers>=4.51.3,<5.0.0",
            "torch",
            "accelerate",
            "diffusers",
            "librosa",
            "llvmlite>=0.40.0",
            "numba>=0.57.0",
            "numpy",
            "scipy",
            "tqdm",
            "ml-collections",
            "absl-py",
            "av",
        ],
    },
}

# Packages with no Python environment at all, recorded so the partition is complete.
NON_PYTHON = {
    "fluidaudio": "Swift toolchain build; no interpreter. FluidAudio 0.15.5 at 19600a48.",
    "speaker-diarization-coreml": "Core ML model package pulled alongside the Swift product.",
    "silero-vad": "Runs in the tool's own environment on onnxruntime; nothing to provision.",
}

PINS_OF_INTEREST = (
    "torch", "torchaudio", "transformers", "huggingface-hub", "tokenizers", "numpy",
    "mlx", "mlx-audio", "mlx-lm", "accelerate", "diffusers", "qwen-asr", "scipy",
    "librosa", "sentencepiece", "kaldi-native-fbank", "numba", "llvmlite",
)


def uv_version() -> str:
    out = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def compile_group(group: tuple[str, ...], python_version: str, raw_dir: Path) -> dict:
    """Resolve one candidate grouping. Returns the verdict and the resolver's own words."""
    lines: list[str] = []
    for package in group:
        lines.append(f"# {package}")
        lines.extend(REQUIREMENTS[package]["requirements"])
    body = "\n".join(lines) + "\n"

    slug = "+".join(group)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "requirements.in"
        source.write_text(body)
        target = Path(tmp) / "requirements.txt"
        proc = subprocess.run(
            ["uv", "pip", "compile", str(source), "-o", str(target),
             "--python-version", python_version, "--no-header"],
            capture_output=True, text=True,
        )
        resolved_text = target.read_text() if target.exists() else ""

    record: dict = {
        "packages": list(group),
        "python_version": python_version,
        "resolves": proc.returncode == 0,
        "input_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }

    if proc.returncode == 0:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw = raw_dir / f"{slug}.py{python_version}.txt"
        raw.write_text(resolved_text)
        pins = {}
        for line in resolved_text.splitlines():
            match = re.match(r"^([A-Za-z0-9._-]+)==([^\s;]+)", line)
            if match and match.group(1).lower() in PINS_OF_INTEREST:
                pins[match.group(1).lower()] = match.group(2)
        record["resolved_package_count"] = sum(
            1 for line in resolved_text.splitlines()
            if re.match(r"^[A-Za-z0-9._-]+==", line)
        )
        record["key_pins"] = dict(sorted(pins.items()))
        record["resolved_set"] = {
            "path": str(raw),
            "sha256": hashlib.sha256(resolved_text.encode()).hexdigest(),
        }
    else:
        # The resolver's message is the evidence. Keep it verbatim, not paraphrased.
        record["resolver_message"] = proc.stderr.strip()

    return record


def minimal_partitions(compatible: set[frozenset[str]],
                       packages: list[str]) -> list[list[list[str]]]:
    """Every smallest grouping in which each group resolves.

    Returns all of them, not one: minimal is weaker than unique, and a layout built on a
    minimum that had three equally small alternatives is a layout with an undocumented
    choice in it. Brute force over set partitions — four packages make this trivial, and it
    stays correct when a fifth arrives, which is the point of not hand-writing the answer.
    """
    for size in range(1, len(packages) + 1):
        found = [
            sorted((sorted(group) for group in partition), key=lambda group: group[0])
            for partition in _partitions(packages, size)
            if all(frozenset(group) in compatible for group in partition)
        ]
        if found:
            unique = {json.dumps(partition) for partition in found}
            return [json.loads(item) for item in sorted(unique)]
    raise AssertionError("every singleton resolves, so a partition always exists")


def _partitions(items: list[str], groups: int):
    """All ways to split `items` into exactly `groups` non-empty groups."""
    if groups == 1:
        yield [list(items)]
        return
    if len(items) < groups:
        return
    first, rest = items[0], items[1:]
    for smaller in _partitions(rest, groups - 1):
        yield [[first]] + smaller
    for same in _partitions(rest, groups):
        for index in range(len(same)):
            yield same[:index] + [[first] + same[index]] + same[index + 1:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--python-version", default="3.12",
                        help="interpreter the matrix is resolved against")
    args = parser.parse_args()

    if shutil.which("uv") is None:
        print("uv not on PATH; this probe resolves with uv and nothing else", file=sys.stderr)
        return 2

    packages = sorted(REQUIREMENTS)
    matrix: list[dict] = []
    compatible: set[frozenset[str]] = set()

    for size in range(1, len(packages) + 1):
        for group in itertools.combinations(packages, size):
            record = compile_group(group, args.python_version, args.raw_dir)
            matrix.append(record)
            if record["resolves"]:
                compatible.add(frozenset(group))
            verdict = "ok" if record["resolves"] else "CONFLICT"
            print(f"  {'+'.join(group):<70} {verdict}", file=sys.stderr)

    # Each package also resolved at the interpreter its recorded evidence actually used, so a
    # matrix run at one version cannot silently claim a pin set that never ran.
    as_built: list[dict] = []
    for package in packages:
        version = REQUIREMENTS[package]["python_as_built"].rsplit(".", 1)[0]
        if version == args.python_version:
            continue
        as_built.append(compile_group((package,), version, args.raw_dir))

    partitions = minimal_partitions(compatible, packages)

    # If every conflict is on one dependency, say so with a count rather than an impression:
    # the claim "this is a transformers problem" is only worth making if no conflict is about
    # anything else, and that is checkable.
    failures = [record for record in matrix if not record["resolves"]]
    mentioning = [record for record in failures
                  if "transformers" in record.get("resolver_message", "")]
    singleton_transformers = {
        record["packages"][0]: record["key_pins"].get("transformers")
        for record in matrix
        if record["resolves"] and len(record["packages"]) == 1
    }

    inputs = json.dumps(REQUIREMENTS, sort_keys=True).encode()
    document = {
        "probe": "environment-partition",
        "question": (
            "What is the smallest set of provisioned Python environments the transcription "
            "packages can be grouped into, given their declared dependencies?"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runner": "model_tests/benchmark/run_env_partition_probe.py",
        "requirements_sha256": hashlib.sha256(inputs).hexdigest(),
        "resolver": {
            "uv": uv_version(),
            "matrix_python_version": args.python_version,
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "installed_anything": False,
        },
        "requirements": REQUIREMENTS,
        "non_python_packages": NON_PYTHON,
        "matrix": matrix,
        "as_built_python_checks": as_built,
        "minimal_python_partition": partitions[0],
        "minimal_python_partition_is_unique": len(partitions) == 1,
        "minimal_python_partition_alternatives": partitions[1:],
        "conflict_axis": {
            "groups_tested": len(matrix),
            "groups_conflicting": len(failures),
            "conflicts_naming_transformers": len(mentioning),
            "resolved_transformers_per_package": singleton_transformers,
            "note": (
                "Every conflicting group's resolver message names transformers. No conflict "
                "here is about torch, numpy, or a platform wheel."
                if len(mentioning) == len(failures) else
                "At least one conflict is not about transformers; read the matrix before "
                "describing the split as a transformers problem."
            ),
        },
        "provisioned_environment_count": len(partitions[0]) + 1,  # + swift
        "provisioned_environment_note": (
            "Python environments from the partition, plus `swift`, which has no interpreter. "
            "`core` is the tool's own environment and is not provisioned."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"\nminimal partition: {partitions[0]}", file=sys.stderr)
    print(f"unique: {len(partitions) == 1}"
          + (f" (alternatives: {partitions[1:]})" if len(partitions) > 1 else ""),
          file=sys.stderr)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
