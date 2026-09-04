"""Public enhancement pipeline assembled from explicit processing phases."""

from __future__ import annotations

from pathlib import Path

from .adjustments import GainAdjustment
from .dsp import (  # re-export the established module surface
    SignalAnalysis,
    analyze_signal,
    apply_channel_balance,
    apply_environment_cleanup,
    apply_frequency_adjustments,
    apply_fullband_adjustments,
    apply_machine_region_corrections,
    apply_source_balance,
    apply_voice_enhancement,
    evaluate_profile,
    regional_measurements,
)
from .media import (  # re-export the established module surface
    atomic_write_json,
    decode_audio,
    encode_output,
    ffmpeg_version,
    is_enhanced_media,
    measure_loudness,
    media_summary,
    probe_media,
    render_loudness_normalized,
    require_runtime,
    temporary_directory,
    temporary_output_path,
    write_float_wav,
)
from .pipeline_loudness import normalize_loudness
from .pipeline_models import PipelineError
from .pipeline_preparation import _detect_speech, _vad_audio, inspect_source, prepare_run
from .pipeline_publication import publish_output
from .pipeline_reporting import (
    _canonical_hash,
    _program_observation,
    _region_manifest,
    _resolved_operations_hash,
    _round_loudness,
    build_report,
)
from .pipeline_stages import (
    _disabled_stage,
    _skipped_stage,
    _stage_result,
    process_stages,
)
from .profiles import PROFILES, STAGE_ORDER, Profile
from .vad import MODEL_SHA256, MODEL_URL, SileroOnnxVad, VadDetector


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
