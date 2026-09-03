from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adjustments import AdjustmentError, load_adjustments
from .environments import ManifestError
from .media import MediaError, media_summary, probe_media
from .packages import (
    Provisioner,
    ProvisioningError,
    doctor,
    list_report,
    load_registry,
    path_report,
    select,
)
from .pipeline import (
    EnhancementPipeline,
    PipelineError,
    inspect_source,
    validate_skips,
    write_report,
)
from .profiles import PROFILES, STAGE_ORDER, get_profile
from .transcribe import catalog as transcribe_catalog
from .transcribe import orchestrator as transcribe_orchestrator
from .transcribe import planner as transcribe_planner
from .transcribe import refusals as transcribe_refusals
from .transcribe import stacks as transcribe_stacks
from .transcribe.plan import serialize_plan
from .vad import SileroOnnxVad, VadError


def _parser() -> argparse.ArgumentParser:
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
        help="Comma-separated standard stage names to skip; all eligible stages are evaluated otherwise.",
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
        help="Replace an existing output/report only after the new render verifies successfully.",
    )

    subparsers.add_parser(
        "doctor",
        help="Report tool, toolchain, platform, and provisioning state.",
    )

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
        "--repair", action="store_true", help="Re-sync a drifted environment from its lock."
    )

    remove_parser = package_commands.add_parser(
        "remove", help="Remove packages; an environment goes when its last package does."
    )
    remove_parser.add_argument("packages", nargs="+")

    purge_parser = package_commands.add_parser(
        "purge", help="Remove everything this tool provisioned, from the registry."
    )
    purge_parser.add_argument(
        "--dry-run", action="store_true", help="Report reclaimable bytes and remove nothing."
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
    plan_parser.add_argument("--vad", help="Pin the VAD backend for a requested vad capability.")
    plan_parser.add_argument(
        "--diarizer", help="Pin the diarizer backend when the request adds that role."
    )
    run_parser = transcribe_commands.add_parser(
        "run", help="Execute one resolved Qwen transcription request."
    )
    run_parser.add_argument("--stack")
    run_parser.add_argument("--input", type=Path)
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
        "--force", action="store_true", help="Replace an existing output or partial result."
    )
    return parser


def _print_json(payload: object, *, stream=None) -> None:
    if stream is None:
        stream = sys.stdout
    json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
    stream.write("\n")


def _ensure_writable_target(path: Path | None, *, force: bool, label: str) -> None:
    if path is not None and path.exists() and not force:
        raise PipelineError(
            f"{label} already exists: {path}; pass --force to replace it"
        )


def _run_inspect(args: argparse.Namespace) -> int:
    # Checked before any work, the way `enhance` checks it: the analysis decodes the file and runs
    # speech detection over it, and refusing the destination afterwards throws all of that away.
    _ensure_writable_target(args.report, force=args.force, label="Report")
    profile = get_profile(args.profile) if args.profile else None
    detector = SileroOnnxVad(args.vad_model)
    report = inspect_source(args.input, profile=profile, detector=detector)
    if args.report:
        write_report(args.report, report)
    _print_json(report)
    return 0


def _run_enhance(args: argparse.Namespace) -> int:
    profile = get_profile(args.profile)
    if args.list_stages:
        _print_json(
            {
                "profile": profile.name,
                "profile_version": profile.version,
                "processing_order": list(STAGE_ORDER),
                "stages": [
                    {"name": stage, "eligible": profile.stage_enabled(stage)}
                    for stage in STAGE_ORDER
                ],
            }
        )
        return 0
    if args.input is None:
        raise PipelineError("INPUT is required unless --list-stages is used")
    if not args.dry_run and args.output is None:
        raise PipelineError("--output is required unless --dry-run is used")
    if args.dry_run and args.output is not None:
        raise PipelineError("--output is not used with --dry-run")

    _ensure_writable_target(args.output, force=args.force, label="Output")
    default_report = (
        Path(f"{args.output}.report.json") if args.output is not None else None
    )
    report_path = args.report or default_report
    _ensure_writable_target(report_path, force=args.force, label="Report")

    skipped = validate_skips(args.skip)
    probe = probe_media(args.input)
    summary = media_summary(args.input, probe)
    duration = float(summary["duration_seconds"])
    adjustments = load_adjustments(
        args.adjustments,
        duration=duration,
        nyquist_hz=24_000.0,
    )
    detector = SileroOnnxVad(args.vad_model)
    pipeline = EnhancementPipeline(
        profile,
        skipped_stages=skipped,
        adjustments=adjustments,
        detector=detector,
    )
    report = pipeline.run(
        args.input,
        output=args.output,
        dry_run=args.dry_run,
        allow_enhanced_input=args.allow_enhanced_input,
    )
    if report_path is not None:
        write_report(report_path, report)
    _print_json(report)
    return 0


def _run_doctor(_args: argparse.Namespace) -> int:
    _print_json(doctor())
    return 0


def _run_packages(args: argparse.Namespace) -> int:
    command = args.packages_command
    if command == "list":
        _print_json(list_report())
        return 0
    if command == "path":
        _print_json(path_report())
        return 0

    provisioner = Provisioner()
    if command == "pull":
        if args.want and not args.stack:
            raise ProvisioningError(
                "stack_required", "--want needs --stack: capabilities are resolved per stack",
                exit_code=2, fix="audio packages pull --stack <stack>",
            )
        if args.want:
            # Refused rather than ignored. Nothing downstream of here reads `--want`: resolving
            # capabilities to a package set is the planner's job (#12), and a stack still
            # provisions every package it can use. Accepting the flag silently would be the real
            # failure — a caller would believe it had narrowed a 17 GiB download it never touched.
            raise ProvisioningError(
                "want_not_implemented",
                "capabilities cannot narrow a pull yet: resolving them to packages is the "
                f"planner's job, so --stack {args.stack} provisions every package it can use",
                exit_code=2, field="--want", provided=args.want,
                fix=f"audio packages pull --stack {args.stack}",
            )
        selection = select(args.packages, stack=args.stack)
        _print_json(provisioner.pull(selection, repair=args.repair, stack=args.stack))
        return 0
    if command == "verify":
        report = provisioner.verify(repair=args.repair)
        _print_json(report)
        # A failed check is the whole point of verify, so it must not exit 0.
        return 3 if report["failed"] else 0
    if command == "remove":
        _print_json(provisioner.remove(args.packages))
        return 0
    if command == "purge":
        _print_json(provisioner.purge(dry_run=args.dry_run))
        return 0
    raise ProvisioningError("unknown_command", f"unknown packages command {command!r}",
                            exit_code=2)


def _run_transcribe(args: argparse.Namespace) -> int:
    command = args.transcribe_command
    wants = args.want if command in {"plan", "run"} else None
    request = transcribe_planner.resolve_request(
        stack_id=args.stack,
        input_path=args.input,
        wants=wants,
        language=getattr(args, "language", None),
        vad=getattr(args, "vad", None),
        diarizer=getattr(args, "diarizer", None),
    )
    if command == "run":
        if request.stack.id not in {"qwen-1.7b", "qwen-0.6b"}:
            issue = 22 if request.stack.id == "firered" else 23
            raise transcribe_refusals.stack_run_unavailable(request.stack.id, issue)
        try:
            run_range = transcribe_orchestrator.parse_range(args.run_range)
        except ValueError as exc:
            raise transcribe_refusals.range_invalid(
                request.input_path,
                request.stack.id,
                request.wants,
                str(args.run_range),
                str(exc),
                language=request.language,
                vad=request.vad,
                diarizer=request.diarizer,
            ) from exc
        transcribe_orchestrator.validate_output_targets(
            request,
            args.output,
            output_format=args.format,
            run_range=run_range,
            force=args.force,
        )
    else:
        run_range = None
    # Validation above is deliberately complete before this probe, and the registry is read
    # only after the probe.  No request refusal can be shadowed by provisioning state.
    metadata = transcribe_catalog.input_metadata(
        request.input_path, probe_media(request.input_path)
    )
    if command == "capabilities":
        _print_json(transcribe_catalog.build_catalog(request.stack, metadata))
        return 0
    if command == "plan":
        registry = load_registry()
        ready = {
            package_id
            for package_id, entry in registry.get("packages", {}).items()
            if entry.get("state") == "ready"
        }
        plan = transcribe_planner.build_plan(
            request, metadata, provisioned_packages=ready
        )
        _print_json(serialize_plan(plan))
        return 0
    if command == "run":
        product = transcribe_orchestrator.run(
            request,
            metadata,
            output=args.output,
            output_format=args.format,
            run_range=run_range,
            force=args.force,
        )
        if args.format == "json":
            _print_json(product.payload)
        else:
            sys.stdout.write(
                transcribe_orchestrator.render_human(product.payload, args.format)
            )
        return 0
    raise ValueError(f"unknown transcribe command {command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "enhance":
            return _run_enhance(args)
        if args.command == "doctor":
            return _run_doctor(args)
        if args.command == "packages":
            return _run_packages(args)
        if args.command == "transcribe":
            return _run_transcribe(args)
        parser.error(f"Unknown command: {args.command}")
    except transcribe_refusals.Refusal as exc:
        _print_json(exc.payload, stream=sys.stderr)
        return exc.exit_code
    except ProvisioningError as exc:
        _print_json({"error": exc.as_dict()}, stream=sys.stderr)
        return exc.exit_code
    except ManifestError as exc:
        _print_json(
            {"error": {"code": "manifest_invalid", "detail": str(exc)}}, stream=sys.stderr
        )
        return 2
    except transcribe_stacks.StackTableError as exc:
        _print_json(
            {"error": {"code": "stack_table_invalid", "detail": str(exc)}},
            stream=sys.stderr,
        )
        return 2
    except AdjustmentError as exc:
        _print_json(
            {"error": exc.as_dict()},
            stream=sys.stderr,
        )
        return 2
    except (MediaError, PipelineError, VadError, ValueError) as exc:
        _print_json(
            {"error": {"type": type(exc).__name__, "message": str(exc)}},
            stream=sys.stderr,
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
