"""Public façade for deterministic audio analysis and transforms."""

from ..adjustments import GainAdjustment
from ..profiles import Profile
from ..vad_contract import SpeechRegion
from .analysis import analyze_signal, evaluate_profile
from .denoise.guide import calibrate_guide_input
from .denoise.processor import apply_guided_denoise
from .dynamics import apply_voice_enhancement
from .levels import EPSILON, amplitude_to_db, peak_dbfs, rms_dbfs
from .measurements import regional_measurements
from .regions import (
    MachineRegion,
    SignalAnalysis,
    resolve_speech_treatment_intervals,
    smooth_time_mask,
)
from .spectral import (
    apply_environment_cleanup,
    apply_frequency_adjustments,
    finish_environment_cleanup,
    prepare_environment_filters,
)
from .treatment import (
    apply_channel_balance,
    apply_fullband_adjustments,
    apply_machine_region_corrections,
    apply_source_balance,
)

__all__ = [
    "EPSILON",
    "GainAdjustment",
    "MachineRegion",
    "Profile",
    "SignalAnalysis",
    "SpeechRegion",
    "amplitude_to_db",
    "analyze_signal",
    "apply_channel_balance",
    "apply_environment_cleanup",
    "apply_frequency_adjustments",
    "apply_fullband_adjustments",
    "apply_guided_denoise",
    "apply_machine_region_corrections",
    "apply_source_balance",
    "apply_voice_enhancement",
    "calibrate_guide_input",
    "evaluate_profile",
    "finish_environment_cleanup",
    "peak_dbfs",
    "prepare_environment_filters",
    "regional_measurements",
    "resolve_speech_treatment_intervals",
    "rms_dbfs",
    "smooth_time_mask",
]
