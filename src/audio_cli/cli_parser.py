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
    inspect_parser.add_argument("--vad-model", type=Path, help="Use a local Silero ONNX model.")
    inspect_parser.add_argument("--report", type=Path, help="Also write the JSON inspection here.")
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
    enhance_parser.add_argument(
        "-o", "--output", type=Path, help="Write enhanced media; incompatible with --dry-run."
    )
    enhance_parser.add_argument(
        "--skip",
        help=(
            "Comma-separated standard stage names to skip; all eligible stages are "
            "evaluated otherwise."
        ),
    )
    enhance_parser.add_argument(
        "--denoiser",
        choices=("stationary", "rnnoise"),
        default="stationary",
        help="Broadband method: stationary reference (default) or explicitly provisioned rnnoise-voice model; never falls back.",
    )
    enhance_parser.add_argument("--adjustments", type=Path)
    enhance_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate without rendering; incompatible with --output.",
    )
    enhance_parser.add_argument("--list-stages", action="store_true")
    enhance_parser.add_argument("--report", type=Path)
    enhance_parser.add_argument("--vad-model", type=Path, help="Use a local Silero ONNX model.")
    enhance_parser.add_argument(
        "--allow-enhanced-input",
        action="store_true",
        help="Allow an explicitly marked enhanced render as input.",
    )
    enhance_parser.add_argument(
        "--force",
        action="store_true",
        help=("Replace an existing output/report only after the new render verifies successfully."),
    )

    subparsers.add_parser(
        "doctor",
        help="Report tool, toolchain, platform, and provisioning state.",
    )

    report_parser = subparsers.add_parser("report", help="Navigate saved enhancement reports.")
    report_commands = report_parser.add_subparsers(dest="report_command", required=True)
    summary_parser = report_commands.add_parser(
        "summary",
        help="Project existing outcomes and measurement locations as JSON; no audio work.",
        description=(
            "Read a saved enhancement report and emit concise JSON with report_pointer "
            "locations (JSON Pointers). Applied stages and an empty unresolved list do not "
            "certify every goal or perceptual quality. Missing outcomes stay absent."
        ),
    )
    summary_parser.add_argument("input", type=Path, help="Saved enhancement report JSON.")

    packages_parser = subparsers.add_parser(
        "packages",
        help="Provision, verify, and remove model packages and their environments.",
    )
    package_commands = packages_parser.add_subparsers(dest="packages_command", required=True)

    package_commands.add_parser("list", help="What is provisioned, and what it occupies.")
    package_commands.add_parser("path", help="Resolved root and per-package locations.")

    pull_parser = package_commands.add_parser(
        "pull",
        help="Download weights, create environments, apply patches, build products.",
    )
    pull_parser.add_argument("packages", nargs="*", help="Package ids; or use --stack.")
    pull_parser.add_argument(
        "--stack", help="Provision what this stack can use; ids: audio transcribe stacks."
    )
    pull_parser.add_argument(
        "--want",
        help="Unsupported here; use transcribe plan --want and pull its missing package ids.",
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
        help="Discover stacks, inspect capabilities, plan, transcribe, and export saved results.",
    )
    transcribe_commands = transcribe_parser.add_subparsers(dest="transcribe_command", required=True)
    transcribe_commands.add_parser(
        "stacks", help="List declared stack ids and decision cues without media or provisioning."
    )
    capabilities_parser = transcribe_commands.add_parser(
        "capabilities",
        help="Report what one stack can do with one input.",
        usage="%(prog)s --stack STACK --input INPUT [-h]",
    )
    stack_help = "Required: explicit transcription stack; ids: audio transcribe stacks."
    capabilities_parser.add_argument("--stack", help=stack_help)
    capabilities_parser.add_argument("--input", type=Path, help="Required: original media path.")

    plan_parser = transcribe_commands.add_parser(
        "plan",
        help="Resolve requested capabilities to roles and packages.",
        usage="%(prog)s --stack STACK --input INPUT [options]",
    )
    plan_parser.add_argument("--stack", help=stack_help)
    plan_parser.add_argument("--input", type=Path, help="Required: original media path.")
    plan_parser.add_argument("--want", help="Comma-separated capability names.")
    plan_parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit generated sample_output; retain every decision and provisioning field.",
    )
    plan_parser.add_argument("--language")
    plan_parser.add_argument("--vad", help="Pin the VAD backend for a requested vad capability.")
    plan_parser.add_argument(
        "--diarizer", help="Pin the diarizer backend when the request adds that role."
    )
    run_parser = transcribe_commands.add_parser(
        "run",
        help="Execute one resolved transcription request.",
        usage="%(prog)s --stack STACK --input INPUT [options]",
        description=(
            "Host stage and elapsed-time progress goes to stderr. Raw backend stdout/stderr "
            "is retained at the announced paths, including on failure. Use --log-dir for "
            "durable storage; the default uses temporary files. Backend warnings are "
            "diagnostics, not recognition-quality verdicts."
        ),
    )
    run_parser.add_argument("--stack", help=stack_help)
    run_parser.add_argument("--input", type=Path, help="Required: original media path.")
    run_parser.add_argument("--want", help="Comma-separated capability names.")
    run_parser.add_argument("--language")
    run_parser.add_argument("--vad", help="Pin the VAD backend for a requested vad capability.")
    run_parser.add_argument(
        "--diarizer", help="Pin the diarizer backend when the request adds that role."
    )
    run_parser.add_argument(
        "--range", dest="run_range", help="Process units intersecting START: or START:END."
    )
    run_parser.add_argument("--format", choices=("json", "md", "txt"), default="json")
    run_parser.add_argument("-o", "--output", type=Path)
    run_parser.add_argument(
        "--log-dir",
        type=Path,
        help="Retain raw stage logs in unique run directories here; refuses if unusable.",
    )
    run_parser.add_argument(
        "--receipt",
        action="store_true",
        help="With --output and --format json, print a concise JSON receipt; saved JSON is unchanged. "
        "Partial runs report the saved partial path and still exit 4.",
    )
    run_parser.add_argument(
        "--force", action="store_true", help="Replace an existing output or partial result."
    )

    _add_export_arguments(
        transcribe_commands.add_parser(
            "export", help="Render saved result JSON offline, without media probing or models."
        )
    )
    _add_export_arguments(
        subparsers.add_parser("export", help="Compatibility alias for audio transcribe export.")
    )
    return parser


def _add_export_arguments(export_parser: argparse.ArgumentParser) -> None:
    export_parser.description = (
        "Render saved result JSON offline; no stack selection or model provisioning is needed. "
        "audio export is a compatibility alias for audio transcribe export."
    )
    export_parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        required=True,
        help=("Normalized result JSON; repeat in source-timeline order to merge continuations."),
    )
    export_parser.add_argument(
        "--format", choices=("srt", "vtt", "md", "txt", "jsonl"), required=True
    )
    export_parser.add_argument("-o", "--output", type=Path)
    export_parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include real segment or word time ranges in txt/md; refuse untimed text.",
    )
    export_parser.add_argument(
        "--provenance",
        action="store_true",
        help="Include saved source, stack, timing basis when present, and input coverage in txt/md.",
    )
    export_parser.add_argument(
        "--force", action="store_true", help="Replace an existing export destination."
    )
