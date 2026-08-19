from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from scipy import signal

from . import __version__
from .adjustments import GainAdjustment
from .dsp import (
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
from .media import (
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
from .profiles import PROFILES, STAGE_ORDER, Profile
from .vad import MODEL_SHA256, MODEL_URL, SileroOnnxVad, VadDetector


class PipelineError(RuntimeError):
    pass


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


def _vad_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono at 16 kHz, never longer than the source timeline it will be timestamped against.

    `resample_poly` returns `ceil(n * up / down)` samples, so 1332160 samples at 48 kHz became
    444054 at 16 kHz where the exact figure is 444053.333. Two thirds of a sample sounds like
    nothing, and it is -- until a speech region runs to the end of the signal and is published
    ending at 27.753375 s in a 27.753333 s file. That is a bound the media does not have, and the
    treatment clamp that pulled it back to the real end then reported a *negative* extension,
    contradicting the guarantee that speech effects are fully engaged at the last detected sample.

    Truncating to the floor keeps the detector's timeline inside the source's. It gives up at most
    one 16 kHz sample of tail, which is 62.5 us and cannot carry speech; inventing a bound past
    the end of the file is the more expensive mistake.
    """
    mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)
    if sample_rate == 16_000:
        return mono
    divisor = math.gcd(sample_rate, 16_000)
    resampled = signal.resample_poly(
        mono, 16_000 // divisor, sample_rate // divisor
    ).astype(np.float32)
    return resampled[: mono.size * 16_000 // sample_rate]


def _detect_speech(
    audio: np.ndarray,
    sample_rate: int,
    profile: Profile,
    detector: VadDetector,
) -> list:
    vad_audio = _vad_audio(audio, sample_rate)
    return detector.detect(
        vad_audio,
        16_000,
        threshold=profile.vad_threshold,
        exit_threshold=profile.vad_exit_threshold,
        min_speech_ms=profile.vad_min_speech_ms,
        min_silence_ms=profile.vad_min_silence_ms,
        speech_pad_ms=profile.vad_speech_pad_ms,
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


def inspect_source(
    source: Path,
    *,
    profile: Profile | None,
    detector: VadDetector | None = None,
) -> dict[str, object]:
    require_runtime()
    if not source.is_file():
        raise PipelineError(f"Input media does not exist: {source}")
    policy = profile or PROFILES["product-demo"]
    detector = detector or SileroOnnxVad()
    probe = probe_media(source)
    audio, sample_rate = decode_audio(source)
    regions = _detect_speech(audio, sample_rate, policy, detector)
    analysis = analyze_signal(audio, sample_rate, regions, policy)
    program = measure_loudness(
        source,
        target_lufs=policy.target_lufs,
        target_lra=policy.target_lra_lu,
        target_true_peak=policy.target_true_peak_dbtp,
    )
    observations = [*analysis.observations, _program_observation(program)]
    payload: dict[str, object] = {
        "schema_version": "1",
        "kind": "audio_inspection",
        "engine": {
            "name": "audio-processing-cli",
            "version": __version__,
            "ffmpeg": ffmpeg_version(),
            "analysis_policy_version": "1",
            "vad": {
                "model": detector.model_version,
                "model_sha256": MODEL_SHA256,
                "source": MODEL_URL,
            },
        },
        "source": media_summary(source, probe),
        "observations": observations,
        "regions": _region_manifest(analysis),
        "measurements": {"program": _round_loudness(program)},
    }
    if profile is not None:
        payload["profile"] = profile.as_dict()
        payload["rule_evaluations"] = evaluate_profile(analysis, profile, program)
    return payload


def _stage_result(
    stage: str,
    profile: Profile,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        "name": stage,
        "profile_rule": f"{profile.name}.{stage}",
        **result,
    }


def _skipped_stage(stage: str, profile: Profile) -> dict[str, object]:
    return _stage_result(
        stage,
        profile,
        {"status": "skipped", "reason": "explicitly_skipped", "operations": []},
    )


def _disabled_stage(stage: str, profile: Profile) -> dict[str, object]:
    return _stage_result(
        stage,
        profile,
        {"status": "no_op", "reason": "disabled_by_profile", "operations": []},
    )


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
        require_runtime()
        if not source.is_file():
            raise PipelineError(f"Input media does not exist: {source}")
        if not dry_run and output is None:
            raise PipelineError("--output is required unless --dry-run is used")
        if output is not None and source.resolve() == output.resolve():
            raise PipelineError("Output must not overwrite the canonical input")

        probe = probe_media(source)
        if is_enhanced_media(probe) and not allow_enhanced_input:
            raise PipelineError(
                "Input is marked as an enhanced render. Start from the canonical original, "
                "or pass --allow-enhanced-input when this is intentional."
            )
        source_info = media_summary(source, probe)
        detector = self.detector or SileroOnnxVad()
        audio, sample_rate = decode_audio(source)
        speech_regions = _detect_speech(audio, sample_rate, self.profile, detector)
        analysis = analyze_signal(audio, sample_rate, speech_regions, self.profile)
        before_program = measure_loudness(
            source,
            target_lufs=self.profile.target_lufs,
            target_lra=self.profile.target_lra_lu,
            target_true_peak=self.profile.target_true_peak_dbtp,
        )
        before_regional = regional_measurements(audio, sample_rate, analysis)
        current = audio
        stages: list[dict[str, object]] = []
        resolved_adjustments: list[dict[str, object]] = []

        stage = "channel-balance"
        if stage in self.skipped_stages:
            stages.append(_skipped_stage(stage, self.profile))
        elif not self.profile.stage_enabled(stage):
            stages.append(_disabled_stage(stage, self.profile))
        else:
            current, result = apply_channel_balance(current, self.profile, analysis)
            stages.append(_stage_result(stage, self.profile, result))

        stage = "environment-denoise"
        if stage in self.skipped_stages:
            stages.append(_skipped_stage(stage, self.profile))
        elif not self.profile.stage_enabled(stage):
            stages.append(_disabled_stage(stage, self.profile))
        else:
            current, result = apply_environment_cleanup(
                current, sample_rate, self.profile, analysis
            )
            stages.append(_stage_result(stage, self.profile, result))

        current, frequency_adjustments = apply_frequency_adjustments(
            current,
            sample_rate,
            self.adjustments,
            analysis.duration_seconds,
            self.profile.region_fade_ms,
        )
        resolved_adjustments.extend(frequency_adjustments)

        stage = "voice-enhance"
        if stage in self.skipped_stages:
            stages.append(_skipped_stage(stage, self.profile))
        elif not self.profile.stage_enabled(stage):
            stages.append(_disabled_stage(stage, self.profile))
        else:
            current, result = apply_voice_enhancement(
                current, sample_rate, self.profile, analysis
            )
            stages.append(_stage_result(stage, self.profile, result))

        stage = "source-balance"
        if stage in self.skipped_stages:
            source_stage_report = _skipped_stage(stage, self.profile)
            stages.append(source_stage_report)
        elif not self.profile.stage_enabled(stage):
            source_stage_report = _disabled_stage(stage, self.profile)
            stages.append(source_stage_report)
        else:
            current, result = apply_source_balance(
                current, sample_rate, self.profile, analysis
            )
            source_stage_report = _stage_result(stage, self.profile, result)
            stages.append(source_stage_report)
        raw_abstained_regions = source_stage_report.get("abstained_regions", [])
        abstained_source_region_ids = (
            {str(region_id) for region_id in raw_abstained_regions}
            if isinstance(raw_abstained_regions, list)
            else set()
        )

        current, fullband_adjustments = apply_fullband_adjustments(
            current,
            sample_rate,
            self.adjustments,
            analysis.duration_seconds,
            self.profile.region_fade_ms,
        )
        resolved_adjustments.extend(fullband_adjustments)
        resolved_adjustments.sort(key=lambda item: str(item["adjustment_id"]))

        with temporary_directory() as temp_dir:
            pre_loudness_wav = temp_dir / "pre-loudness.wav"
            final_wav = temp_dir / "final.wav"
            write_float_wav(pre_loudness_wav, current, sample_rate)
            render_true_peak_target = (
                self.profile.target_true_peak_dbtp
                - self.profile.codec_true_peak_headroom_db
            )
            pre_program = measure_loudness(
                pre_loudness_wav,
                target_lufs=self.profile.target_lufs,
                target_lra=self.profile.target_lra_lu,
                target_true_peak=render_true_peak_target,
            )

            stage = "program-loudness"
            needs_normalization = (
                abs(pre_program["input_i"] - self.profile.target_lufs) > 0.5
                or pre_program["input_tp"] > self.profile.target_true_peak_dbtp
            )
            normalization_resolution: dict[str, object] | None = None
            program_operation: dict[str, object] | None = None
            if stage in self.skipped_stages:
                stages.append(_skipped_stage(stage, self.profile))
                if pre_program["input_tp"] > -0.1:
                    raise PipelineError(
                        "Final peak validation failed: program-loudness was skipped and the "
                        f"predicted true peak is {pre_program['input_tp']:.2f} dBTP"
                    )
                shutil.copyfile(pre_loudness_wav, final_wav)
            elif not self.profile.stage_enabled(stage):
                stages.append(_disabled_stage(stage, self.profile))
                shutil.copyfile(pre_loudness_wav, final_wav)
            elif needs_normalization:
                program_operation = {
                    "type": "ebu-r128-linear-gain-and-true-peak-limiter",
                    "target_lufs": self.profile.target_lufs,
                    "target_lra_lu": self.profile.target_lra_lu,
                    "target_true_peak_dbtp": self.profile.target_true_peak_dbtp,
                    "render_true_peak_dbtp": render_true_peak_target,
                    "codec_headroom_db": self.profile.codec_true_peak_headroom_db,
                    "measured_input": _round_loudness(pre_program),
                }
                stages.append(
                    _stage_result(
                        stage,
                        self.profile,
                        {
                            "status": "applied",
                            "reason": "program_outside_profile_target",
                            "operations": [program_operation],
                            "component_evaluations": [
                                {
                                    "component": "loudness-range",
                                    "status": (
                                        "no_op"
                                        if pre_program["input_lra"]
                                        <= self.profile.target_lra_lu
                                        else "abstained"
                                    ),
                                    "reason": (
                                        "inside_target"
                                        if pre_program["input_lra"]
                                        <= self.profile.target_lra_lu
                                        else "dynamic_lra_control_would_change_relative_region_balance"
                                    ),
                                    "measured_lra_lu": round(
                                        pre_program["input_lra"], 3
                                    ),
                                    "target_maximum_lra_lu": self.profile.target_lra_lu,
                                }
                            ],
                        },
                    )
                )
                normalization_resolution = render_loudness_normalized(
                    pre_loudness_wav,
                    final_wav,
                    target_lufs=self.profile.target_lufs,
                    target_lra=self.profile.target_lra_lu,
                    target_true_peak=render_true_peak_target,
                    measurement=pre_program,
                    sample_rate=sample_rate,
                )
                program_operation["resolution"] = normalization_resolution
            else:
                stages.append(
                    _stage_result(
                        stage,
                        self.profile,
                        {
                            "status": "no_op",
                            "reason": "program_inside_target",
                            "operations": [],
                        },
                    )
                )
                shutil.copyfile(pre_loudness_wav, final_wav)

            simulated_program = measure_loudness(
                final_wav,
                target_lufs=self.profile.target_lufs,
                target_lra=self.profile.target_lra_lu,
                target_true_peak=self.profile.target_true_peak_dbtp,
            )
            simulated_audio, simulated_sample_rate = decode_audio(final_wav)
            simulated_regional = regional_measurements(
                simulated_audio, simulated_sample_rate, analysis
            )
            source_verification_passes: list[dict[str, object]] = []
            if (
                source_stage_report["status"] == "applied"
                and needs_normalization
                and program_operation is not None
            ):
                for verification_pass in range(1, 4):
                    operations = source_stage_report.get("operations", [])
                    assert isinstance(operations, list)
                    operation_by_region = {
                        str(item["region_id"]): item
                        for item in operations
                        if isinstance(item, dict) and "region_id" in item
                    }
                    corrections: dict[str, float] = {}
                    for measured_region in simulated_regional["machine_regions"]:
                        region_id = str(measured_region["region_id"])
                        if region_id in abstained_source_region_ids:
                            continue
                        difference = float(measured_region["difference_from_speech_db"])
                        if (
                            self.profile.machine_relative_minimum_lu
                            <= difference
                            <= self.profile.machine_relative_maximum_lu
                        ):
                            continue
                        item = operation_by_region.get(region_id)
                        if item is None:
                            region = next(
                                region
                                for region in analysis.machine_regions
                                if region.region_id == region_id
                            )
                            item = {
                                "type": "regional-full-band-gain",
                                "region_id": region_id,
                                "scope": {
                                    "time": {
                                        "start": round(region.start, 6),
                                        "end": round(region.end, 6),
                                    },
                                    "frequency": "all",
                                },
                                "target_difference_db": self.profile.machine_relative_target_lu,
                                "resolved_gain_db": 0.0,
                                "boundary_fade_ms": self.profile.region_fade_ms,
                            }
                            operations.append(item)
                            operation_by_region[region_id] = item
                        previous_gain = float(item.get("resolved_gain_db", 0.0))
                        desired_delta = (
                            self.profile.machine_relative_target_lu - difference
                        )
                        resolved_total = float(
                            np.clip(
                                previous_gain + desired_delta,
                                -self.profile.machine_max_attenuation_db,
                                self.profile.machine_max_boost_db,
                            )
                        )
                        delta = resolved_total - previous_gain
                        if abs(delta) < 0.05:
                            continue
                        corrections[region_id] = delta
                        item["resolved_gain_db"] = round(resolved_total, 3)
                        item["downstream_compensation_db"] = round(
                            float(item.get("downstream_compensation_db", 0.0)) + delta,
                            3,
                        )
                    if not corrections:
                        break
                    current = apply_machine_region_corrections(
                        current,
                        sample_rate,
                        analysis,
                        corrections,
                        self.profile.region_fade_ms,
                    )
                    write_float_wav(pre_loudness_wav, current, sample_rate)
                    pre_program = measure_loudness(
                        pre_loudness_wav,
                        target_lufs=self.profile.target_lufs,
                        target_lra=self.profile.target_lra_lu,
                        target_true_peak=render_true_peak_target,
                    )
                    normalization_resolution = render_loudness_normalized(
                        pre_loudness_wav,
                        final_wav,
                        target_lufs=self.profile.target_lufs,
                        target_lra=self.profile.target_lra_lu,
                        target_true_peak=render_true_peak_target,
                        measurement=pre_program,
                        sample_rate=sample_rate,
                    )
                    program_operation["resolution"] = normalization_resolution
                    program_operation["measured_input_after_source_compensation"] = (
                        _round_loudness(pre_program)
                    )
                    simulated_program = measure_loudness(
                        final_wav,
                        target_lufs=self.profile.target_lufs,
                        target_lra=self.profile.target_lra_lu,
                        target_true_peak=self.profile.target_true_peak_dbtp,
                    )
                    simulated_audio, simulated_sample_rate = decode_audio(final_wav)
                    simulated_regional = regional_measurements(
                        simulated_audio, simulated_sample_rate, analysis
                    )
                    source_verification_passes.append(
                        {
                            "pass": verification_pass,
                            "corrections_db": {
                                key: round(value, 3)
                                for key, value in sorted(corrections.items())
                            },
                            "resulting_regions": simulated_regional["machine_regions"],
                        }
                    )
                source_stage_report["downstream_verification_passes"] = (
                    source_verification_passes
                )
            simulated_peak_limit = (
                -0.1
                if stage in self.skipped_stages
                else self.profile.target_true_peak_dbtp
            )
            if simulated_program["input_tp"] > simulated_peak_limit:
                raise PipelineError(
                    "Predicted true-peak verification failed: "
                    f"measured {simulated_program['input_tp']:.2f} dBTP, "
                    f"limit {simulated_peak_limit:.2f} dBTP"
                )
            report: dict[str, object] = {
                "schema_version": "1",
                "kind": "audio_enhancement_report",
                "engine": {
                    "name": "audio-processing-cli",
                    "version": __version__,
                    "ffmpeg": ffmpeg_version(),
                    "vad": {
                        "model": detector.model_version,
                        "model_sha256": MODEL_SHA256,
                        "source": MODEL_URL,
                    },
                },
                "profile": self.profile.as_dict(),
                "source": source_info,
                "processing_order": list(STAGE_ORDER),
                "regions": _region_manifest(analysis),
                "observations": [
                    *analysis.observations,
                    _program_observation(before_program),
                ],
                "rule_evaluations": evaluate_profile(
                    analysis, self.profile, before_program
                ),
                "stages": stages,
                "adjustments": resolved_adjustments,
                "measurements": {
                    "before": {
                        "program": _round_loudness(before_program),
                        "regional": before_regional,
                    },
                    "predicted": {
                        "program_before_normalization": _round_loudness(pre_program),
                        "program": _round_loudness(simulated_program),
                        "regional": simulated_regional,
                    },
                },
                "final_peak_validation": {
                    "status": "predicted_pass",
                    "predicted_true_peak_dbtp": round(simulated_program["input_tp"], 3),
                    "limit_true_peak_dbtp": simulated_peak_limit,
                },
                "timeline_preserved": True,
                "dry_run": dry_run,
                "rendered": False,
            }
            report["resolved_operations_sha256"] = _resolved_operations_hash(
                self.profile,
                stages,
                resolved_adjustments,
                source_info["sha256"],
            )
            if dry_run:
                return report

            assert output is not None
            with temporary_output_path(output) as encoded_temp:
                encode_output(
                    source,
                    final_wav,
                    encoded_temp,
                    original_sha256=str(source_info["sha256"]),
                    has_video=bool(probe.get("has_video")),
                )
                after_probe = probe_media(encoded_temp)
                after_program = measure_loudness(
                    encoded_temp,
                    target_lufs=self.profile.target_lufs,
                    target_lra=self.profile.target_lra_lu,
                    target_true_peak=self.profile.target_true_peak_dbtp,
                )
                true_peak_limit = (
                    self.profile.target_true_peak_dbtp
                    if stage not in self.skipped_stages
                    else -0.1
                )
                codec_peak_correction_db = 0.0
                codec_safe_wav = temp_dir / "codec-safe.wav"
                for _ in range(2):
                    if (
                        after_program["input_tp"] <= true_peak_limit
                        or stage in self.skipped_stages
                        or not self.profile.stage_enabled(stage)
                    ):
                        break
                    correction = true_peak_limit - after_program["input_tp"] - 0.1
                    codec_peak_correction_db += correction
                    safe_audio, safe_sample_rate = decode_audio(final_wav)
                    safe_audio = safe_audio * (
                        10.0 ** (codec_peak_correction_db / 20.0)
                    )
                    write_float_wav(codec_safe_wav, safe_audio, safe_sample_rate)
                    encode_output(
                        source,
                        codec_safe_wav,
                        encoded_temp,
                        original_sha256=str(source_info["sha256"]),
                        has_video=bool(probe.get("has_video")),
                    )
                    after_probe = probe_media(encoded_temp)
                    after_program = measure_loudness(
                        encoded_temp,
                        target_lufs=self.profile.target_lufs,
                        target_lra=self.profile.target_lra_lu,
                        target_true_peak=self.profile.target_true_peak_dbtp,
                    )
                if program_operation is not None:
                    program_operation["codec_peak_correction_db"] = round(
                        codec_peak_correction_db, 6
                    )
                after_audio, after_sample_rate = decode_audio(encoded_temp)
                after_regional = regional_measurements(
                    after_audio, after_sample_rate, analysis
                )
                if source_stage_report["status"] == "applied":
                    source_operations = source_stage_report.get("operations", [])
                    assert isinstance(source_operations, list)
                    source_operation_by_region = {
                        str(item["region_id"]): item
                        for item in source_operations
                        if isinstance(item, dict) and "region_id" in item
                    }
                    final_region_evaluations: list[dict[str, object]] = []
                    unbounded_failures: list[str] = []
                    for measured_region in after_regional["machine_regions"]:
                        region_id = str(measured_region["region_id"])
                        difference = float(measured_region["difference_from_speech_db"])
                        if region_id in abstained_source_region_ids:
                            final_region_evaluations.append(
                                {
                                    "region_id": region_id,
                                    "difference_from_speech_db": round(difference, 3),
                                    "status": "abstained_overlap",
                                }
                            )
                            continue
                        inside = (
                            self.profile.machine_relative_minimum_lu - 0.25
                            <= difference
                            <= self.profile.machine_relative_maximum_lu + 0.25
                        )
                        item = source_operation_by_region.get(region_id, {})
                        resolved_gain = float(item.get("resolved_gain_db", 0.0))
                        bounded = (
                            abs(resolved_gain - self.profile.machine_max_boost_db)
                            <= 0.05
                            or abs(
                                resolved_gain + self.profile.machine_max_attenuation_db
                            )
                            <= 0.05
                        )
                        status = (
                            "inside_target"
                            if inside
                            else "bounded_outside_target"
                            if bounded
                            else "outside_target"
                        )
                        if status == "outside_target":
                            unbounded_failures.append(region_id)
                        final_region_evaluations.append(
                            {
                                "region_id": region_id,
                                "difference_from_speech_db": round(difference, 3),
                                "status": status,
                            }
                        )
                    source_stage_report["final_region_evaluations"] = (
                        final_region_evaluations
                    )
                    if unbounded_failures:
                        raise PipelineError(
                            "Source-balance verification failed for unbounded regions: "
                            + ", ".join(unbounded_failures)
                        )
                output_info = media_summary(encoded_temp, after_probe)
                duration_delta_ms = 1000.0 * (
                    float(output_info["duration_seconds"])
                    - float(source_info["duration_seconds"])
                )
                timeline_ok = abs(duration_delta_ms) <= 50.0
                loudness_ok = (
                    stage in self.skipped_stages
                    or not self.profile.stage_enabled(stage)
                    or abs(after_program["input_i"] - self.profile.target_lufs) <= 0.6
                )
                peak_ok = after_program["input_tp"] <= true_peak_limit
                if not timeline_ok:
                    raise PipelineError(
                        f"Timeline verification failed: output duration changed by {duration_delta_ms:.1f} ms"
                    )
                if not loudness_ok:
                    raise PipelineError(
                        "Program loudness verification failed: "
                        f"measured {after_program['input_i']:.2f} LUFS, target {self.profile.target_lufs:.2f} LUFS"
                    )
                if not peak_ok:
                    raise PipelineError(
                        "True-peak verification failed: "
                        f"measured {after_program['input_tp']:.2f} dBTP, limit {true_peak_limit:.2f} dBTP"
                    )
                os.replace(encoded_temp, output)

            # Re-probe the final path so its path and hash describe the durable artifact.
            durable_probe = probe_media(output)
            durable_info = media_summary(output, durable_probe)
            report["output"] = durable_info
            report["measurements"]["after"] = {
                "program": _round_loudness(after_program),
                "regional": after_regional,
                "duration_delta_ms": round(duration_delta_ms, 3),
            }
            report["final_peak_validation"] = {
                "status": "pass",
                "measured_true_peak_dbtp": round(after_program["input_tp"], 3),
                "limit_true_peak_dbtp": round(true_peak_limit, 3),
            }
            report["timeline_preserved"] = timeline_ok
            report["dry_run"] = False
            report["rendered"] = True
            report["resolved_operations_sha256"] = _resolved_operations_hash(
                self.profile,
                stages,
                resolved_adjustments,
                source_info["sha256"],
            )
            return report


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
            f"Unknown stage(s) in --skip: {', '.join(unknown)}; valid stages: {', '.join(STAGE_ORDER)}"
        )
    if len(items) != len(set(items)):
        raise PipelineError("--skip contains a duplicate stage name")
    return set(items)
