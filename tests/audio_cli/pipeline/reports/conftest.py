"""Compact report shapes from the inspection and enhancement report owners."""

import json

import pytest

from audio_cli.pipeline.timing import timeline_verification


def source(digest):
    return {
        "path": "/historical/moved.wav",
        "sha256": digest * 64,
        "decoded_audio": {
            "duration_basis": "decoded_pcm",
            "duration_seconds": 10.0,
            "sample_count": 480000,
            "sample_rate_hz": 48000,
            "time_origin": "first_decoded_sample",
        },
    }


def region(identifier, start, end, kind="non_speech_program"):
    return {"region_id": identifier, "start": start, "end": end, "kind": kind}


@pytest.fixture
def enhancement():
    return {
        "kind": "audio_enhancement_report",
        "schema_version": "1",
        "source": source("a"),
        "output": source("b"),
        "profile": {"name": "product-demo", "version": "5"},
        "rendered": True,
        "dry_run": False,
        "timeline_preserved": True,
        "timeline_verification": timeline_verification(checked=True),
        "region_basis": {
            "timeline": "source",
            "detection": "original_source",
            "regional_measurements": "fixed_source_regions",
            "region_ids": "report_local",
        },
        "regions": [region("speech_001", 4, 6, "speech"), region("machine_001", 1, 3)],
        "stages": [
            {
                "name": "environment-denoise",
                "status": "applied",
                "component_evaluations": [
                    {
                        "component": "broadband-denoise",
                        "status": "applied",
                        "scope": {"time": "speech_regions", "frequency": "all"},
                        "speech_preservation": {"status": "abstained", "reason": "not_measured"},
                    }
                ],
            }
        ],
        "measurements": {
            "before": {"program_actual": {"integrated_loudness_lufs": -32}},
            "predicted": {"program_actual": {"integrated_loudness_lufs": -16}},
            "after": {
                "program": {"input_i": -16.1, "output_i": -12},
                "program_actual": {"integrated_loudness_lufs": -16.1},
                "regional": {
                    "speech_program_rms_dbfs": -20,
                    "machine_regions": [
                        {
                            "region_id": "machine_001",
                            "measured_rms_dbfs": -23,
                            "difference_from_speech_db": -3,
                        }
                    ],
                },
            },
        },
        "unresolved": [],
    }


@pytest.fixture
def inspection():
    return {
        "kind": "audio_inspection",
        "schema_version": "1",
        "source": source("b"),
        "region_basis": {
            "timeline": "source",
            "detection": "inspected_source",
            "regional_measurements": "fixed_source_regions",
            "region_ids": "report_local",
        },
        "regions": [region("speech_001", 3.8, 6.2, "speech"), region("machine_002", 1.5, 3)],
        "measurements": {
            "program": {"input_i": -16.1, "output_i": -12},
            "program_actual": {"integrated_loudness_lufs": -16.1},
        },
        "observations": [
            {
                "type": "speech_program_level",
                "region_id": "speech_program",
                "measured_rms_dbfs": -19,
                "scope": {"time": "speech_regions", "frequency": "all"},
            },
            {
                "type": "regional_level_difference",
                "region_id": "machine_002",
                "measured_rms_dbfs": -22,
                "reference_region": "speech_program",
                "difference_db": -3,
                "scope": {"time": {"start": 1.5, "end": 3}, "frequency": "all"},
            },
        ],
        "rule_evaluations": [
            {"observation_id": "level_002", "status": "inside_target", "observed_difference_db": -3}
        ],
    }


@pytest.fixture
def saved(tmp_path):
    def save(report, name="report.json"):
        path = tmp_path / name
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    return save


def resolve(report, pointer):
    value = report
    for key in pointer[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value
