from __future__ import annotations

from dataclasses import asdict, dataclass

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


PROFILES: dict[str, Profile] = {
    "transcription": Profile(
        name="transcription",
        version="3",
        channel_balance_enabled=True,
        channel_no_op_db=1.5,
        channel_max_correction_db=6.0,
        channel_correlation_minimum=0.92,
        environment_denoise_enabled=True,
        highpass_hz=70.0,
        subbass_ratio_threshold=0.025,
        hum_excess_db_threshold=9.0,
        voice_enhance_enabled=True,
        voice_target_rms_dbfs=-28.0,
        voice_max_gain_db=20.0,
        voice_max_attenuation_db=4.0,
        voice_presence_gain_db=1.5,
        compressor_threshold_dbfs=-24.0,
        compressor_ratio=2.0,
        source_balance_enabled=False,
        machine_relative_minimum_lu=-4.0,
        machine_relative_maximum_lu=-2.0,
        machine_relative_target_lu=-3.0,
        machine_max_boost_db=0.0,
        machine_max_attenuation_db=0.0,
        program_loudness_enabled=True,
        target_lufs=-23.0,
        target_lra_lu=7.0,
        target_true_peak_dbtp=-3.0,
        codec_true_peak_headroom_db=1.0,
    ),
    "product-demo": Profile(
        name="product-demo",
        version="3",
        channel_balance_enabled=True,
        channel_no_op_db=1.5,
        channel_max_correction_db=6.0,
        channel_correlation_minimum=0.92,
        environment_denoise_enabled=True,
        highpass_hz=75.0,
        subbass_ratio_threshold=0.02,
        hum_excess_db_threshold=8.0,
        voice_enhance_enabled=True,
        voice_target_rms_dbfs=-30.0,
        voice_max_gain_db=24.0,
        voice_max_attenuation_db=4.0,
        voice_presence_gain_db=2.0,
        compressor_threshold_dbfs=-24.0,
        compressor_ratio=2.5,
        source_balance_enabled=True,
        machine_relative_minimum_lu=-4.0,
        machine_relative_maximum_lu=-2.0,
        machine_relative_target_lu=-3.0,
        machine_max_boost_db=12.0,
        machine_max_attenuation_db=30.0,
        program_loudness_enabled=True,
        target_lufs=-16.0,
        target_lra_lu=11.0,
        target_true_peak_dbtp=-1.5,
        codec_true_peak_headroom_db=1.0,
    ),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile {name!r}; choose one of: {choices}") from exc
