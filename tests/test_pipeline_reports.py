"""Actual encoded measurements, raw diagnostics, and unresolved target reporting."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audio_cli.dsp import analyze_signal
from audio_cli.pipeline import publication, reporting
from audio_cli.pipeline.models import LoudnessRun, PreparedRun, StageRun
from audio_cli.pipeline.outcomes import actual_program, unresolved_outcomes
from audio_cli.profiles import PROFILES
from audio_cli.vad_contract import SpeechRegion

PROFILE = PROFILES["product-demo"]


def raw_program(lufs=-16.43, peak=-1.77, lra=17.5):
    # Actual emitted demo diagnostics: input_i=-16.43 versus output_i=-14.74.
    return {
        "input_i": lufs,
        "input_tp": peak,
        "input_lra": lra,
        "output_i": -14.74,
        "output_tp": -1.5,
        "output_lra": 24.3,
    }


def stage_state():
    source = {
        "name": "source-balance",
        "status": "applied",
        "reason": "non_overlapping_regions_balanced_to_speech_reference",
        "operations": [{"region_id": "machine_audio_001", "resolved_gain_db": 12}],
        "abstained_regions": ["machine_audio_002"],
    }
    stages = [
        {
            "name": "environment-denoise",
            "status": "applied",
            "operations": [{"type": "minimum-phase-highpass"}],
            "reason": "eligible_environmental_cleanup_resolved",
            "component_evaluations": [
                {
                    "component": "broadband-denoise",
                    "status": "abstained",
                    "reason": "no_reliable_noise_only_region",
                }
            ],
        },
        source,
        {
            "name": "program-loudness",
            "status": "applied",
            "reason": "program_outside_profile_target",
            "operations": [],
            "component_evaluations": [
                {
                    "component": "loudness-range",
                    "status": "abstained",
                    "reason": "dynamic_lra_control_would_change_relative_region_balance",
                    "measured_lra_lu": 18.5,
                }
            ],
        },
    ]
    return StageRun(
        np.zeros((32000, 1), dtype=np.float32), stages, [], source, {"machine_audio_002"}
    )


def measured_regions():
    return {
        "machine_regions": [
            {"region_id": "machine_audio_001", "difference_from_speech_db": -4.751},
            {"region_id": "machine_audio_002", "difference_from_speech_db": 9.0},
        ]
    }


def test_program_actual_names_measurements_and_omits_unavailable_values():
    assert actual_program(raw_program()) == {
        "integrated_loudness_lufs": -16.43,
        "loudness_range_lu": 17.5,
        "true_peak_dbtp": -1.77,
    }
    assert actual_program({"input_i": float("-inf"), "input_tp": -3.0}) == {"true_peak_dbtp": -3.0}
    assert actual_program({}) == {}


def test_unresolved_reports_applied_parent_abstained_child_and_final_bounded_regions():
    staged = stage_state()
    unresolved = unresolved_outcomes(
        PROFILE, staged, raw_program(), measured_regions(), measured_at="encoded_output"
    )
    assert unresolved == [
        {
            "stage": "environment-denoise",
            "component": "broadband-denoise",
            "status": "abstained",
            "reason": "no_reliable_noise_only_region",
        },
        {
            "stage": "source-balance",
            "region_id": "machine_audio_002",
            "status": "abstained_overlap",
            "reason": "speech_and_machine_audio_overlap",
        },
        {
            "stage": "program-loudness",
            "component": "loudness-range",
            "status": "abstained",
            "reason": "dynamic_lra_control_would_change_relative_region_balance",
            "measured_lra_lu": 17.5,
            "target_maximum_lra_lu": 11.0,
            "measured_at": "encoded_output",
        },
        {
            "stage": "source-balance",
            "region_id": "machine_audio_001",
            "status": "bounded_outside_target",
            "difference_from_speech_db": -4.751,
            "reason": "correction_bound_reached",
            "measured_at": "encoded_output",
            "target_difference_db": {"minimum": -4.0, "maximum": -2.0},
        },
    ]
    assert staged.stages[0]["status"] == "applied"


def test_lra_resolved_by_final_signal_is_not_reported_as_unmet():
    unresolved = unresolved_outcomes(
        PROFILE,
        stage_state(),
        raw_program(lra=10),
        measured_regions(),
        measured_at="encoded_output",
    )
    assert not any(item.get("component") == "loudness-range" for item in unresolved)


def test_disabled_and_skipped_stages_do_not_invent_unmet_targets():
    source = {
        "name": "source-balance",
        "status": "no_op",
        "reason": "disabled_by_profile",
        "operations": [],
    }
    staged = StageRun(
        np.zeros((1, 1)),
        [source, {"name": "program-loudness", "status": "skipped"}],
        [],
        source,
        set(),
    )
    assert (
        unresolved_outcomes(
            PROFILE, staged, raw_program(), measured_regions(), measured_at="encoded_output"
        )
        == []
    )


def report_inputs(tmp_path):
    staged = stage_state()
    analysis = analyze_signal(staged.current, 16000, [SpeechRegion(0.2, 1.8, 0.9, 1)], PROFILE)
    prepared = PreparedRun(
        tmp_path / "source.wav",
        tmp_path / "output.m4a",
        False,
        {"has_video": False},
        {"sha256": "original-digest", "duration_seconds": 2},
        SimpleNamespace(model_version="fake-vad"),
        staged.current,
        16000,
        analysis,
        raw_program(-35, -20),
        measured_regions(),
    )
    loudness = LoudnessRun(
        tmp_path / "final.wav",
        raw_program(-29, -10),
        raw_program(-16.2, -2.6),
        measured_regions(),
        {},
        -1.5,
    )
    return prepared, staged, loudness


def test_dry_report_preserves_raw_diagnostics_and_labels_fixed_source_regions(
    tmp_path, monkeypatch
):
    import json

    monkeypatch.setattr(reporting, "ffmpeg_version", lambda: "fake")
    prepared, staged, loudness = report_inputs(tmp_path)
    prepared.dry_run = True
    report = reporting.build_report(PROFILE, prepared, staged, loudness)
    assert report["schema_version"] == "1"
    assert "after" not in report["measurements"]
    assert (
        report["measurements"]["predicted"]["program_actual"]["integrated_loudness_lufs"] == -16.2
    )
    assert report["measurements"]["predicted"]["program"]["output_i"] == -14.74
    assert report["region_basis"] == {
        "timeline": "source",
        "detection": "original_source",
        "regional_measurements": "fixed_source_regions",
        "region_ids": "report_local",
    }
    assert report["unresolved"][-1]["measured_at"] == "predicted_pre_encode"
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_published_actual_values_use_last_post_codec_measurement(tmp_path, monkeypatch):
    prepared, staged, loudness = report_inputs(tmp_path)
    monkeypatch.setattr(reporting, "ffmpeg_version", lambda: "fake")
    report = reporting.build_report(PROFILE, prepared, staged, loudness)
    measurements = iter([raw_program(-16, -0.4, 18), raw_program(-16.3, -1.9, 17)])
    measured_paths = []

    def measure(path, **kwargs):
        measured_paths.append(path)
        return next(measurements)

    monkeypatch.setattr(publication, "measure_loudness", measure)
    monkeypatch.setattr(publication, "probe_media", lambda p: {})
    monkeypatch.setattr(
        publication,
        "media_summary",
        lambda p, probe: {"duration_seconds": 2, "path": str(p), "sha256": "encoded-digest"},
    )
    monkeypatch.setattr(publication, "decode_audio", lambda p: (staged.current, 16000))
    monkeypatch.setattr(publication, "write_float_wav", lambda *a: None)
    monkeypatch.setattr(publication, "regional_measurements", lambda *a: measured_regions())
    monkeypatch.setattr(
        publication,
        "encode_output",
        lambda source, wave, output, **kw: Path(output).write_bytes(b"synthetic encoded bytes"),
    )
    published = publication.publish_output(
        PROFILE, set(), prepared, staged, loudness, tmp_path, report
    )
    assert len(measured_paths) == 2
    assert prepared.output.read_bytes() == b"synthetic encoded bytes"
    actual = published["measurements"]["after"]["program_actual"]
    assert actual == {
        "integrated_loudness_lufs": -16.3,
        "true_peak_dbtp": -1.9,
        "loudness_range_lu": 17.0,
    }
    assert published["measurements"]["after"]["program"] == raw_program(-16.3, -1.9, 17)
    assert published["unresolved"][2]["measured_lra_lu"] == 17.0
    assert published["unresolved"][-1]["status"] == "bounded_outside_target"
    assert loudness.program_operation["codec_peak_correction_db"] < 0
    assert published["final_peak_validation"]["measured_true_peak_dbtp"] == actual["true_peak_dbtp"]
