"""Pure calibration of a denoising guide; never a gain on the delivered signal."""

from __future__ import annotations

import numpy as np

from ...vad_contract import SpeechRegion


def calibrate_guide_input(
    audio: np.ndarray,
    sample_rate: int,
    speech: list[SpeechRegion],
    *,
    target_rms_dbfs: float,
    peak_limit_dbfs: float,
    maximum_gain_db: float,
) -> tuple[np.ndarray | None, float, dict[str, object]]:
    """One continuous shared gain, measured over the original detected scopes."""
    mask = np.zeros(len(audio), dtype=bool)
    for region in speech:
        mask[
            max(0, round(region.start * sample_rate)) : min(
                len(audio), round(region.end * sample_rate)
            )
        ] = True
    speech_samples = audio[mask].astype(np.float64)
    if not speech_samples.size or not np.any(speech_samples):
        return None, 1.0, {"status": "abstained", "reason": "no_speech_energy"}
    level = float(10 * np.log10(np.mean(speech_samples**2)))
    peak = float(20 * np.log10(np.max(np.abs(audio.astype(np.float64)))))
    requested = target_rms_dbfs - level
    headroom = peak_limit_dbfs - peak
    gain_db = min(requested, headroom, maximum_gain_db)
    gain = 10 ** (gain_db / 20)
    # Explain the already resolved gain; these fields never feed processing.
    target_attained = gain_db == requested
    limiting_reasons = [
        reason
        for reason, bound in (("peak_headroom", headroom), ("maximum_gain", maximum_gain_db))
        if bound == gain_db and bound < requested
    ]
    return (
        (audio * gain).astype(np.float32),
        gain,
        {
            "status": "applied" if abs(gain_db) > 1e-9 else "no_op",
            "scope": "model_guide_only",
            "speech_reference": "supplied_regions",
            "input_speech_rms_dbfs": round(level, 6),
            "target_speech_rms_dbfs": target_rms_dbfs,
            "resolved_gain_db": round(gain_db, 6),
            "target_attained": target_attained,
            "limiting_reasons": limiting_reasons,
            "resolved_speech_rms_dbfs": round(level + gain_db, 6),
            "peak_limit_dbfs": peak_limit_dbfs,
            "maximum_gain_db": maximum_gain_db,
        },
    )
