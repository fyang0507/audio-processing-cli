"""Top-level enhancement orchestration and public pipeline helpers."""

from __future__ import annotations

from pathlib import Path

from ..adjustments import GainAdjustment
from ..dsp import apply_machine_region_corrections
from ..media import atomic_write_json, temporary_directory
from ..profiles import STAGE_ORDER, Profile
from ..vad import VadDetector
from .loudness import normalize_loudness
from .models import PipelineError
from .preparation import prepare_run
from .publication import publish_output
from .reporting import build_report
from .stages import process_stages


class EnhancementPipeline:
    def __init__(
        self,
        profile: Profile,
        *,
        skipped_stages: set[str] | None = None,
        adjustments: list[GainAdjustment] | None = None,
        detector: VadDetector | None = None,
    ) -> None:
        self.profile = profile
        self.skipped_stages = skipped_stages or set()
        self.adjustments = adjustments or []
        self.detector = detector

    def run(
        self,
        source: Path,
        *,
        output: Path | None,
        dry_run: bool,
        allow_enhanced_input: bool = False,
    ) -> dict[str, object]:
        prepared = prepare_run(
            self.profile,
            self.detector,
            source,
            output=output,
            dry_run=dry_run,
            allow_enhanced_input=allow_enhanced_input,
        )
        staged = process_stages(
            self.profile,
            self.skipped_stages,
            self.adjustments,
            prepared,
        )
        with temporary_directory() as temp_dir:
            loudness = normalize_loudness(
                self.profile,
                self.skipped_stages,
                prepared,
                staged,
                temp_dir,
                apply_region_corrections=apply_machine_region_corrections,
            )
            report = build_report(self.profile, prepared, staged, loudness)
            if dry_run:
                return report
            return publish_output(
                self.profile,
                self.skipped_stages,
                prepared,
                staged,
                loudness,
                temp_dir,
                report,
            )


def write_report(path: Path, report: dict[str, object]) -> None:
    atomic_write_json(path, report)


def validate_skips(raw: str | None) -> set[str]:
    if raw is None or not raw.strip():
        return set()
    items = [item.strip() for item in raw.split(",")]
    if any(not item for item in items):
        raise PipelineError("--skip must be a comma-separated list of stage names")
    unknown = sorted(set(items) - set(STAGE_ORDER))
    if unknown:
        raise PipelineError(
            f"Unknown stage(s) in --skip: {', '.join(unknown)}; "
            f"valid stages: {', '.join(STAGE_ORDER)}"
        )
    if len(items) != len(set(items)):
        raise PipelineError("--skip contains a duplicate stage name")
    return set(items)
