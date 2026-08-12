from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AdjustmentError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_adjustment",
        field: str | None = None,
        provided: object | None = None,
        allowed: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.provided = _json_safe(provided)
        self.allowed = allowed

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": type(self).__name__,
            "code": self.code,
            "message": str(self),
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.provided is not None:
            payload["provided"] = self.provided
        if self.allowed is not None:
            payload["allowed"] = self.allowed
        return payload


def _json_safe(value: object) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class GainAdjustment:
    adjustment_id: str
    gain_db: float
    start: float
    end: float
    frequency_low_hz: float | None
    frequency_high_hz: float | None
    frequency_shape: str | None

    @property
    def is_full_band(self) -> bool:
        return self.frequency_low_hz is None

    @property
    def is_full_time(self) -> bool:
        return self.start == 0.0 and self.end == float("inf")

    @property
    def placement(self) -> str:
        if not self.is_full_band:
            return "before-voice-enhance"
        if not self.is_full_time:
            return "after-source-balance"
        return "before-program-loudness"

    def resolved_end(self, duration: float) -> float:
        return duration if self.end == float("inf") else self.end

    def as_dict(self, duration: float) -> dict[str, object]:
        frequency: str | dict[str, object]
        if self.is_full_band:
            frequency = "all"
        else:
            frequency = {
                "low_hz": self.frequency_low_hz,
                "high_hz": self.frequency_high_hz,
                "shape": self.frequency_shape,
            }
        time: str | dict[str, float]
        if self.is_full_time:
            time = "all"
        else:
            time = {"start": self.start, "end": self.resolved_end(duration)}
        return {
            "adjustment_id": self.adjustment_id,
            "type": "gain",
            "gain_db": self.gain_db,
            "scope": {"time": time, "frequency": frequency},
            "placement": self.placement,
        }


def _as_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdjustmentError(
            f"{field} must be a finite number",
            code="invalid_number",
            field=field,
            provided=value,
            allowed={"type": "finite_number"},
        )
    result = float(value)
    if not math.isfinite(result):
        raise AdjustmentError(
            f"{field} must be a finite number",
            code="non_finite_number",
            field=field,
            provided=result,
            allowed={"type": "finite_number"},
        )
    return result


def load_adjustments(
    path: Path | None,
    *,
    duration: float,
    nyquist_hz: float,
) -> list[GainAdjustment]:
    if path is None:
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AdjustmentError(f"Could not read adjustments file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AdjustmentError(f"Invalid adjustments JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"adjustments"}:
        raise AdjustmentError(
            "Adjustment file must contain exactly one top-level 'adjustments' array"
        )
    raw_items = payload["adjustments"]
    if not isinstance(raw_items, list):
        raise AdjustmentError("'adjustments' must be an array")

    result: list[GainAdjustment] = []
    for index, raw in enumerate(raw_items, 1):
        prefix = f"adjustments[{index - 1}]"
        if not isinstance(raw, dict) or set(raw) != {"type", "gain_db", "scope"}:
            raise AdjustmentError(
                f"{prefix} must contain exactly type, gain_db, and scope"
            )
        if raw["type"] != "gain":
            raise AdjustmentError(f"{prefix}.type must be 'gain'")
        gain_db = _as_number(raw["gain_db"], f"{prefix}.gain_db")
        if not -24.0 <= gain_db <= 12.0:
            raise AdjustmentError(
                f"{prefix}.gain_db must be between -24 and +12 dB",
                code="gain_out_of_range",
                field=f"{prefix}.gain_db",
                provided=gain_db,
                allowed={"minimum_db": -24.0, "maximum_db": 12.0},
            )
        scope = raw["scope"]
        if not isinstance(scope, dict) or set(scope) != {"time", "frequency"}:
            raise AdjustmentError(
                f"{prefix}.scope must contain exactly time and frequency"
            )

        raw_time = scope["time"]
        if raw_time == "all":
            start, end = 0.0, float("inf")
        elif isinstance(raw_time, dict) and set(raw_time) == {"start", "end"}:
            start = _as_number(raw_time["start"], f"{prefix}.scope.time.start")
            end = _as_number(raw_time["end"], f"{prefix}.scope.time.end")
            if start < 0 or end <= start or end > duration:
                raise AdjustmentError(
                    f"{prefix}.scope.time must satisfy 0 <= start < end <= {duration:.3f}",
                    code="scope_out_of_range",
                    field=f"{prefix}.scope.time",
                    provided={"start": start, "end": end},
                    allowed={
                        "minimum_start_seconds": 0.0,
                        "maximum_end_seconds": round(duration, 6),
                        "constraint": "start < end",
                    },
                )
        else:
            raise AdjustmentError(
                f"{prefix}.scope.time must be 'all' or an object with start/end"
            )

        raw_frequency = scope["frequency"]
        if raw_frequency == "all":
            low = high = None
            shape = None
        elif isinstance(raw_frequency, dict) and set(raw_frequency) == {
            "low_hz",
            "high_hz",
            "shape",
        }:
            low = _as_number(
                raw_frequency["low_hz"], f"{prefix}.scope.frequency.low_hz"
            )
            high = _as_number(
                raw_frequency["high_hz"], f"{prefix}.scope.frequency.high_hz"
            )
            shape = raw_frequency["shape"]
            if shape not in {"notch", "band"}:
                raise AdjustmentError(
                    f"{prefix}.scope.frequency.shape must be 'notch' or 'band'"
                )
            if low < 20 or high <= low or high >= nyquist_hz:
                raise AdjustmentError(
                    f"{prefix}.scope.frequency must satisfy 20 <= low_hz < high_hz < {nyquist_hz:g}",
                    code="scope_out_of_range",
                    field=f"{prefix}.scope.frequency",
                    provided={
                        "low_hz": low,
                        "high_hz": high,
                        "shape": shape,
                    },
                    allowed={
                        "minimum_low_hz": 20.0,
                        "maximum_high_hz_exclusive": round(nyquist_hz, 6),
                        "constraint": "low_hz < high_hz",
                    },
                )
        else:
            raise AdjustmentError(
                f"{prefix}.scope.frequency must be 'all' or an object with low_hz/high_hz/shape"
            )

        result.append(
            GainAdjustment(
                adjustment_id=f"adjustment_{index:03d}",
                gain_db=round(gain_db, 6),
                start=round(start, 6),
                end=end if end == float("inf") else round(end, 6),
                frequency_low_hz=low,
                frequency_high_hz=high,
                frequency_shape=shape,
            )
        )
    return result
