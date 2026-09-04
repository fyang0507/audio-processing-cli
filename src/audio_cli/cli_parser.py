"""Argument parser construction for the public ``audio`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from .profiles import PROFILES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio",
        description="Profile-driven, local-first audio utilities for agent workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Measure audio facts and optionally evaluate a profile.",
    )
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.add_argument("--profile", choices=sorted(PROFILES))
    inspect_parser.add_argument(
        "--vad-model", type=Path, help="Use a local Silero ONNX model."
    )
    inspect_parser.add_argument(
        "--report", type=Path, help="Also write the JSON inspection here."
    )
    inspect_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing report.",
    )

    enhance_parser = subparsers.add_parser(
        "enhance",
        help="Run the deterministic enhancement loop for a declared profile.",
    )
    enhance_parser.add_argument("input", type=Path, nargs="?")
    enhance_parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    enhance_parser.add_argument("-o", "--output", type=Path)
    enhance_parser.add_argument(
        "--skip",
        help=(
            "Comma-separated standard stage names to skip; all eligible stages are "
            "evaluated otherwise."
        ),
    )
    enhance_parser.add_argument("--adjustments", type=Path)
    enhance_parser.add_argument("--dry-run", action="store_true")
    enhance_parser.add_argument("--list-stages", action="store_true")
    enhance_parser.add_argument("--report", type=Path)
    enhance_parser.add_argument(
        "--vad-model", type=Path, help="Use a local Silero ONNX model."
    )
    enhance_parser.add_argument(
        "--allow-enhanced-input",
        action="store_true",
        help="Allow an explicitly marked enhanced render as input.",
    )
    enhance_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace an existing output/report only after the new render verifies "
            "successfully."
        ),
    )

    subparsers.add_parser(
        "doctor",
        help="Report tool, toolchain, platform, and provisioning state.",
    )

    packages_parser = subparsers.add_parser(
        "packages",
        help="Provision, verify, and remove model packages and their environments.",
    )
    package_commands = packages_parser.add_subparsers(
        dest="packages_command", required=True
    )

    package_commands.add_parser("list", help="What is provisioned, and what it occupies.")
    package_commands.add_parser("path", help="Resolved root and per-package locations.")

    pull_parser = package_commands.add_parser(
        "pull",
        help="Download weights, create environments, apply patches, build products.",
    )
    pull_parser.add_argument("packages", nargs="*", help="Package ids; or use --stack.")
    pull_parser.add_argument("--stack", help="Provision what this stack can use.")
    pull_parser.add_argument(
        "--want",
        help="Reserved until the planner lands, and refused until then rather than ignored.",
    )
    pull_parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-materialize the named packages even if the registry calls them ready.",
    )

    verify_parser = package_commands.add_parser(
        "verify",
        help="Re-check digests, environment locks, the pinned private API, and patches.",
    )
    verify_parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-sync a drifted environment from its lock.",
    )

    remove_parser = package_commands.add_parser(
        "remove", help="Remove packages; an environment goes when its last package does."
    )
    remove_parser.add_argument("packages", nargs="+")

    purge_parser = package_commands.add_parser(
        "purge", help="Remove everything this tool provisioned, from the registry."
    )
    purge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report reclaimable bytes and remove nothing.",
    )

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Inspect transcription capabilities and resolve an execution plan.",
    )
    transcribe_commands = transcribe_parser.add_subparsers(
        dest="transcribe_command", required=True
    )
    capabilities_parser = transcribe_commands.add_parser(
        "capabilities", help="Report what one stack can do with one input."
    )
    capabilities_parser.add_argument("--stack")
    capabilities_parser.add_argument("--input", type=Path)

    plan_parser = transcribe_commands.add_parser(
        "plan", help="Resolve requested capabilities to roles and packages."
    )
    plan_parser.add_argument("--stack")
    plan_parser.add_argument("--input", type=Path)
    plan_parser.add_argument("--want", help="Comma-separated capability names.")
    plan_parser.add_argument("--language")
    plan_parser.add_argument(
        "--vad", help="Pin the VAD backend for a requested vad capability."
    )
    plan_parser.add_argument(
        "--diarizer", help="Pin the diarizer backend when the request adds that role."
    )
    run_parser = transcribe_commands.add_parser(
        "run", help="Execute one resolved transcription request."
    )
    run_parser.add_argument("--stack")
    run_parser.add_argument("--input", type=Path)
    run_parser.add_argument("--want", help="Comma-separated capability names.")
    run_parser.add_argument("--language")
    run_parser.add_argument(
        "--vad", help="Pin the VAD backend for a requested vad capability."
    )
    run_parser.add_argument(
        "--diarizer", help="Pin the diarizer backend when the request adds that role."
    )
    run_parser.add_argument(
        "--range", dest="run_range", help="Process units intersecting START: or START:END."
    )
    run_parser.add_argument("--format", choices=("json", "md", "txt"), default="json")
    run_parser.add_argument("-o", "--output", type=Path)
    run_parser.add_argument(
        "--force", action="store_true", help="Replace an existing output or partial result."
    )

    export_parser = subparsers.add_parser(
        "export", help="Render one or more normalized transcripts for people or editors."
    )
    export_parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        required=True,
        help=(
            "Normalized result JSON; repeat in source-timeline order to merge "
            "continuations."
        ),
    )
    export_parser.add_argument(
        "--format", choices=("srt", "vtt", "md", "txt", "jsonl"), required=True
    )
    export_parser.add_argument("-o", "--output", type=Path)
    export_parser.add_argument(
        "--force", action="store_true", help="Replace an existing export destination."
    )
    return parser
