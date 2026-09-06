"""Freeze the supported API retained by the enhancement-pipeline facade."""

from __future__ import annotations

import audio_cli.pipeline as pipeline

SUPPORTED_EXPORTS = {
    "DenoiserModel",
    "MODEL_SHA256",
    "MODEL_URL",
    "PROFILES",
    "STAGE_ORDER",
    "EnhancementPipeline",
    "GainAdjustment",
    "PipelineError",
    "Profile",
    "SignalAnalysis",
    "SileroOnnxVad",
    "VadDetector",
    "analyze_signal",
    "apply_channel_balance",
    "apply_environment_cleanup",
    "apply_frequency_adjustments",
    "apply_fullband_adjustments",
    "apply_machine_region_corrections",
    "apply_source_balance",
    "apply_voice_enhancement",
    "atomic_write_json",
    "build_report",
    "decode_audio",
    "encode_output",
    "evaluate_profile",
    "ffmpeg_version",
    "inspect_source",
    "is_enhanced_media",
    "measure_loudness",
    "media_summary",
    "normalize_loudness",
    "prepare_run",
    "probe_media",
    "process_stages",
    "publish_output",
    "regional_measurements",
    "render_loudness_normalized",
    "require_runtime",
    "temporary_directory",
    "temporary_output_path",
    "validate_skips",
    "write_float_wav",
    "write_report",
    "summarize_report",
}


def test_pipeline_facade_preserves_its_supported_project_api() -> None:
    assert set(pipeline.__all__) == SUPPORTED_EXPORTS
    assert len(pipeline.__all__) == len(SUPPORTED_EXPORTS)
    assert all(getattr(pipeline, name) is not None for name in SUPPORTED_EXPORTS)
