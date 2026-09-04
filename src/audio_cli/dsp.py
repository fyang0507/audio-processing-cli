"""Public façade for deterministic audio analysis and transforms."""

from .adjustments import GainAdjustment
from .dsp_analysis import analyze_signal, evaluate_profile
from .dsp_levels import EPSILON, amplitude_to_db, peak_dbfs, rms_dbfs
from .dsp_regions import (
    MachineRegion,
    SignalAnalysis,
    resolve_speech_treatment_intervals,
    smooth_time_mask,
)
from .dsp_transforms import (
    apply_channel_balance,
    apply_environment_cleanup,
    apply_frequency_adjustments,
    apply_fullband_adjustments,
    apply_machine_region_corrections,
    apply_source_balance,
    apply_voice_enhancement,
    regional_measurements,
)
from .profiles import Profile
from .vad import SpeechRegion

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
    "apply_machine_region_corrections",
    "apply_source_balance",
    "apply_voice_enhancement",
    "evaluate_profile",
    "peak_dbfs",
    "regional_measurements",
    "resolve_speech_treatment_intervals",
    "rms_dbfs",
    "smooth_time_mask",
]
