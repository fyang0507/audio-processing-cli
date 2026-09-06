from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adjustments import AdjustmentError, load_adjustments
from .cli_parser import build_parser
from .command import Refusal
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
    verified_artifact,
)
from .pipeline import (
    DenoiserModel,
    EnhancementPipeline,
    PipelineError,
    inspect_source,
    summarize_report,
    validate_skips,
    write_report,
)
from .profiles import STAGE_ORDER, get_profile
from .transcribe import catalog as transcribe_catalog
from .transcribe import orchestrator as transcribe_orchestrator
from .transcribe import planner as transcribe_planner
from .transcribe import refusals as transcribe_refusals
from .transcribe import stacks as transcribe_stacks
from .transcribe.plan import serialize_plan
from .vad import SileroOnnxVad, VadError


def _parser() -> argparse.ArgumentParser:
    return build_parser()


def _print_json(payload: object, *, stream=None) -> None:
    if stream is None:
        stream = sys.stdout
    json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
    stream.write("\n")


def _ensure_writable_target(path: Path | None, *, force: bool, label: str) -> None:
    if path is not None and path.exists() and not force:
        raise PipelineError(f"{label} already exists: {path}; pass --force to replace it")


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
    default_report = Path(f"{args.output}.report.json") if args.output is not None else None
    report_path = args.report or default_report
    _ensure_writable_target(report_path, force=args.force, label="Report")

    skipped = validate_skips(args.skip)
    denoiser_model = None
    if args.denoiser == "rnnoise":
        if "environment-denoise" in skipped or not profile.stage_enabled("environment-denoise"):
            raise PipelineError("--denoiser rnnoise requires an enabled environment-denoise stage")
        model_path, provenance = verified_artifact("rnnoise-voice")
        denoiser_model = DenoiserModel(model_path, provenance["sha256"], provenance)
    probe = probe_media(args.input)
    summary = media_summary(args.input, probe)
    if "duration_seconds" not in summary:
        raise PipelineError("Input has no available probed duration for adjustment validation")
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
        denoiser_model=denoiser_model,
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

    if command == "pull":
        if args.want is not None and not args.stack:
            raise ProvisioningError(
                "stack_required",
                "--want needs --stack: capabilities are resolved per stack",
                exit_code=2,
                fix="use transcribe plan with the original --input, an explicit --stack, and --want; "
                "then run the plan next command for missing packages",
            )
        if args.want is not None:
            # A pull has no input-specific plan. Keep capability selection in transcribe plan
            # and explicit package selection here instead of broadening the requested download.
            raise ProvisioningError(
                "want_not_implemented",
                "--want belongs to transcribe plan; packages pull accepts explicit package ids "
                "or every package available to --stack",
                exit_code=2,
                field="--want",
                provided=args.want,
                fix="use transcribe plan with the original --input, --stack, and --want; "
                "then run the plan next command for missing packages",
            )
        selection = select(args.packages, stack=args.stack)
        provisioner = Provisioner()
        _print_json(provisioner.pull(selection, repair=args.repair, stack=args.stack))
        return 0
    provisioner = Provisioner()
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
    raise ProvisioningError("unknown_command", f"unknown packages command {command!r}", exit_code=2)


def _run_transcribe(args: argparse.Namespace) -> int:
    command = args.transcribe_command
    if command == "stacks":
        _print_json(transcribe_stacks.discovery())
        return 0
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
        try:
            run_range = transcribe_orchestrator.parse_range(args.run_range)
        except ValueError as exc:
            raise transcribe_refusals.range_invalid(str(args.run_range), str(exc)) from exc
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
        plan = transcribe_planner.build_plan(request, metadata, provisioned_packages=ready)
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
            sys.stdout.write(transcribe_orchestrator.render_human(product.payload, args.format))
        return 0
    raise ValueError(f"unknown transcribe command {command!r}")


def _run_export(args: argparse.Namespace) -> int:
    from .export import (
        IncompatibleResultsError,
        InvalidResultError,
        OutputExistsError,
        OutputWriteError,
        ReadableTimingRequiredError,
        TimestampsUnsupportedError,
        TimingRequiredError,
        UnsafeOutputError,
        export_documents,
    )
    from .export import refusals as export_refusals

    if args.force and args.output is None:
        raise export_refusals.output_required_for_force()

    try:
        product = export_documents(
            args.inputs,
            args.format,
            output=args.output,
            force=args.force,
            timestamps=args.timestamps,
        )
    except ReadableTimingRequiredError as exc:
        raise export_refusals.timing_required_for_timestamps(
            exc.input_path, exc.segment_id
        ) from exc
    except TimestampsUnsupportedError as exc:
        raise export_refusals.timestamps_unsupported_for_format(exc.output_format) from exc
    except TimingRequiredError as exc:
        roles = exc.plan.get("roles", {}) if isinstance(exc.plan, dict) else {}
        asr = roles.get("asr", {}) if isinstance(roles, dict) else {}
        config = asr.get("config", {}) if isinstance(asr, dict) else {}
        language = config.get("language") if isinstance(config, dict) else None
        vad_role = roles.get("vad", {}) if isinstance(roles, dict) else {}
        vad = None
        if (
            isinstance(vad_role, dict)
            and vad_role.get("selected_by") == "pin:--vad"
            and isinstance(vad_role.get("backend"), str)
        ):
            vad = vad_role["backend"]
        execution = exc.plan.get("execution", {}) if isinstance(exc.plan, dict) else {}
        selected_range = execution.get("range", {}) if isinstance(execution, dict) else {}
        requested_range = (
            selected_range.get("requested") if isinstance(selected_range, dict) else None
        )
        run_range = None
        if (
            isinstance(requested_range, (list, tuple))
            and len(requested_range) == 2
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in requested_range
            )
        ):
            run_range = f"{requested_range[0]}:{requested_range[1]}"
        raise export_refusals.timing_required_for_format(
            exc.input_path,
            args.format,
            exc.found,
            exc.stack,
            exc.wants,
            source_path=exc.source_path,
            language=language if isinstance(language, str) else None,
            vad=vad,
            run_range=run_range,
            word_timing_outcome=exc.word_timing_outcome,
        ) from exc
    except OutputExistsError as exc:
        raise export_refusals.output_exists(
            args.inputs,
            args.format,
            exc.output,
            replaceable=exc.replaceable,
            timestamps=args.timestamps,
        ) from exc
    except UnsafeOutputError as exc:
        raise export_refusals.output_is_canonical_input(exc.output, exc.protected) from exc
    except OutputWriteError as exc:
        raise export_refusals.output_path_invalid(exc.output, exc.output, exc.reason) from exc
    except InvalidResultError as exc:
        raise export_refusals.export_input_invalid(exc.input_path, exc.reason) from exc
    except IncompatibleResultsError as exc:
        raise export_refusals.export_inputs_incompatible(
            exc.input_paths, exc.reason, fix=exc.fix
        ) from exc

    if args.output is None:
        sys.stdout.write(product.content)
    else:
        _print_json(product.summary(args.output))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "enhance":
            return _run_enhance(args)
        if args.command == "report":
            _print_json(summarize_report(args.input))
            return 0
        if args.command == "doctor":
            return _run_doctor(args)
        if args.command == "packages":
            return _run_packages(args)
        if args.command == "transcribe":
            return _run_transcribe(args)
        if args.command == "export":
            return _run_export(args)
        parser.error(f"Unknown command: {args.command}")
    except Refusal as exc:
        _print_json(exc.payload, stream=sys.stderr)
        return exc.exit_code
    except ProvisioningError as exc:
        _print_json({"error": exc.as_dict()}, stream=sys.stderr)
        return exc.exit_code
    except ManifestError as exc:
        _print_json({"error": {"code": "manifest_invalid", "detail": str(exc)}}, stream=sys.stderr)
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
