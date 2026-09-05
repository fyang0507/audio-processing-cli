"""Stable report construction and resolved-operation identity."""

from __future__ import annotations

import hashlib
import json
import math

from .. import __version__
from ..dsp import SignalAnalysis, evaluate_profile
from ..media import ffmpeg_version
from ..profiles import STAGE_ORDER, Profile
from ..vad import MODEL_SHA256, MODEL_URL
from .models import LoudnessRun, PreparedRun, StageRun


def _round_loudness(data: dict[str, float]) -> dict[str, float | None]:
    return {
        key: round(value, 3) if math.isfinite(value) else None
        for key, value in sorted(data.items())
    }


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolved_operations_hash(
    profile: Profile,
    stages: list[dict[str, object]],
    adjustments: list[dict[str, object]],
    source_sha256: object,
) -> str:
    stage_resolutions: list[dict[str, object]] = []
    for stage in stages:
        raw_operations = stage.get("operations", [])
        operations: list[object] = []
        if isinstance(raw_operations, list):
            for operation in raw_operations:
                if isinstance(operation, dict):
                    operations.append(
                        {
                            key: value
                            for key, value in operation.items()
                            if key != "codec_peak_correction_db"
                        }
                    )
                else:
                    operations.append(operation)
        stage_resolutions.append(
            {
                "name": stage.get("name"),
                "status": stage.get("status"),
                "reason": stage.get("reason"),
                "operations": operations,
            }
        )
    return _canonical_hash(
        {
            "profile": profile.as_dict(),
            "stages": stage_resolutions,
            "adjustments": adjustments,
            "source_sha256": source_sha256,
        }
    )


def _program_observation(measurement: dict[str, float]) -> dict[str, object]:
    return {
        "observation_id": "program_001",
        "type": "program_loudness",
        "scope": {"time": "all", "frequency": "all"},
        "integrated_loudness_lufs": round(measurement["input_i"], 3),
        "loudness_range_lu": round(measurement["input_lra"], 3),
        "true_peak_dbtp": round(measurement["input_tp"], 3),
    }


def _region_manifest(analysis: SignalAnalysis) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for index, region in enumerate(analysis.speech_regions, 1):
        item = region.as_dict()
        item.update({"region_id": f"speech_{index:03d}", "kind": "speech"})
        manifest.append(item)
    for region in analysis.machine_regions:
        manifest.append(
            {
                "region_id": region.region_id,
                "kind": "non_speech_program",
                "start": round(region.start, 6),
                "end": round(region.end, 6),
                "overlaps_speech": region.overlaps_speech,
            }
        )
    return manifest


def build_report(
    profile: Profile,
    prepared: PreparedRun,
    staged: StageRun,
    loudness: LoudnessRun,
) -> dict[str, object]:
    """Build the dry-run report shared unchanged with the publication path."""
    report: dict[str, object] = {
        "schema_version": "1",
        "kind": "audio_enhancement_report",
        "engine": {
            "name": "audio-processing-cli",
            "version": __version__,
            "ffmpeg": ffmpeg_version(),
            "vad": {
                "model": prepared.detector.model_version,
                "model_sha256": MODEL_SHA256,
                "source": MODEL_URL,
            },
        },
        "profile": profile.as_dict(),
        "source": prepared.source_info,
        "processing_order": list(STAGE_ORDER),
        "regions": _region_manifest(prepared.analysis),
        "observations": [
            *prepared.analysis.observations,
            _program_observation(prepared.before_program),
        ],
        "rule_evaluations": evaluate_profile(prepared.analysis, profile, prepared.before_program),
        "stages": staged.stages,
        "adjustments": staged.resolved_adjustments,
        "measurements": {
            "before": {
                "program": _round_loudness(prepared.before_program),
                "regional": prepared.before_regional,
            },
            "predicted": {
                "program_before_normalization": _round_loudness(loudness.pre_program),
                "program": _round_loudness(loudness.simulated_program),
                "regional": loudness.simulated_regional,
            },
        },
        "final_peak_validation": {
            "status": "predicted_pass",
            "predicted_true_peak_dbtp": round(loudness.simulated_program["input_tp"], 3),
            "limit_true_peak_dbtp": loudness.simulated_peak_limit,
        },
        "timeline_preserved": True,
        "dry_run": prepared.dry_run,
        "rendered": False,
    }
    report["resolved_operations_sha256"] = _resolved_operations_hash(
        profile,
        staged.stages,
        staged.resolved_adjustments,
        prepared.source_info["sha256"],
    )
    return report
