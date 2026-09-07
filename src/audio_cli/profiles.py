from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import get_type_hints

STAGE_ORDER = (
    "channel-balance",
    "environment-denoise",
    "voice-enhance",
    "source-balance",
    "program-loudness",
)


@dataclass(frozen=True)
class Profile:
    name: str
    version: str
    channel_balance_enabled: bool
    channel_no_op_db: float
    channel_max_correction_db: float
    channel_correlation_minimum: float
    environment_denoise_enabled: bool
    highpass_hz: float
    subbass_ratio_threshold: float
    hum_excess_db_threshold: float
    voice_enhance_enabled: bool
    voice_target_rms_dbfs: float
    voice_max_gain_db: float
    voice_max_attenuation_db: float
    voice_presence_gain_db: float
    compressor_threshold_dbfs: float
    compressor_ratio: float
    source_balance_enabled: bool
    machine_relative_minimum_lu: float
    machine_relative_maximum_lu: float
    machine_relative_target_lu: float
    machine_max_boost_db: float
    machine_max_attenuation_db: float
    program_loudness_enabled: bool
    target_lufs: float
    target_lra_lu: float
    target_true_peak_dbtp: float
    codec_true_peak_headroom_db: float
    broadband_max_reduction_db: float = 6.0
    vad_threshold: float = 0.5
    vad_exit_threshold: float = 0.35
    vad_min_speech_ms: int = 100
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 120
    region_fade_ms: int = 40
    speech_transition_placement: str = "outside"
    voice_boundary_bridge_silence_ms: int = 400
    voice_boundary_guard_ms: int = 80
    voice_boundary_search_ms: int = 2000
    voice_boundary_noise_margin_db: float = 6.0
    voice_boundary_speech_margin_db: float = 12.0

    def stage_enabled(self, stage: str) -> bool:
        field = stage.replace("-", "_") + "_enabled"
        return bool(getattr(self, field))

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["processing_order"] = list(STAGE_ORDER)
        return data


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate field {key!r}")
        result[key] = value
    return result


def _validate_profile(data: object, name: str) -> Profile:
    if not isinstance(data, dict):
        raise ValueError("profile must be a JSON object")
    expected = {field.name for field in fields(Profile)}
    if missing := expected - data.keys():
        raise ValueError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown := data.keys() - expected:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
    for key, kind in get_type_hints(Profile).items():
        value = data[key]
        valid = type(value) is kind
        if kind is float:
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
        if not valid:
            raise ValueError(f"{key}: expected {kind.__name__}")
    if data["name"] != name or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name):
        raise ValueError("name must match the filename and use lowercase hyphenated words")
    if not re.fullmatch(r"[1-9][0-9]*", data["version"]):
        raise ValueError("version must be a positive integer string")
    if data["speech_transition_placement"] != "outside":
        raise ValueError("speech_transition_placement must be 'outside'")
    unit_fields = {
        "vad_threshold",
        "vad_exit_threshold",
        "channel_correlation_minimum",
        "subbass_ratio_threshold",
    }
    signed_fields = {
        "voice_target_rms_dbfs",
        "compressor_threshold_dbfs",
        "target_lufs",
        "target_true_peak_dbtp",
        "machine_relative_minimum_lu",
        "machine_relative_maximum_lu",
        "machine_relative_target_lu",
    }
    for key, value in data.items():
        if type(value) not in (int, float):
            continue
        if key in unit_fields and not 0 <= value <= 1:
            raise ValueError(f"{key}: must be between 0 and 1")
        if key not in signed_fields and value < 0:
            raise ValueError(f"{key}: must be nonnegative")
    if not 0 < data["highpass_hz"] < 24000:
        raise ValueError("highpass_hz: must be between 0 and the 48 kHz processing Nyquist limit")
    if data["compressor_ratio"] < 1:
        raise ValueError("compressor_ratio: must be at least 1")
    # The media loudness probe passes these values to FFmpeg loudnorm.
    for key, minimum, maximum in (
        ("target_lufs", -70, -5),
        ("target_lra_lu", 1, 50),
        ("target_true_peak_dbtp", -9, 0),
    ):
        if not minimum <= data[key] <= maximum:
            raise ValueError(f"{key}: must be between {minimum} and {maximum}")
    if data["target_true_peak_dbtp"] - data["codec_true_peak_headroom_db"] < -9:
        raise ValueError("true-peak target after codec headroom must be at least -9 dBTP")
    if data["vad_exit_threshold"] > data["vad_threshold"]:
        raise ValueError("vad_exit_threshold must not exceed vad_threshold")
    if not (
        data["machine_relative_minimum_lu"]
        <= data["machine_relative_target_lu"]
        <= data["machine_relative_maximum_lu"]
    ):
        raise ValueError("machine_relative_target_lu must fall within its minimum and maximum")
    for key in ("voice_target_rms_dbfs", "compressor_threshold_dbfs", "target_true_peak_dbtp"):
        if data[key] > 0:
            raise ValueError(f"{key}: must not exceed 0")
    return Profile(**data)


def _load_profiles(directory: Traversable) -> dict[str, Profile]:
    profiles: dict[str, Profile] = {}
    for resource in sorted(directory.iterdir(), key=lambda entry: entry.name):
        if not resource.is_file() or not resource.name.endswith(".json"):
            continue
        name = resource.name.removesuffix(".json")
        try:
            data = json.loads(
                resource.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
            )
            profiles[name] = _validate_profile(data, name)
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f"Invalid bundled profile {resource.name}: {exc}") from exc
    if not profiles:
        raise ValueError("No bundled profile JSON files found")
    return profiles


PROFILES: dict[str, Profile] = _load_profiles(files("audio_cli").joinpath("profile_configs"))


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}; choose one of: {choices}") from exc
