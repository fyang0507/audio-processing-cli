import json

import pytest

from audio_cli.adjustments import AdjustmentError, load_adjustments


def test_adjustments_resolve_placement_and_scope(tmp_path) -> None:
    path = tmp_path / "adjustments.json"
    path.write_text(
        json.dumps(
            {
                "adjustments": [
                    {
                        "type": "gain",
                        "gain_db": -8,
                        "scope": {
                            "time": "all",
                            "frequency": {
                                "low_hz": 55,
                                "high_hz": 65,
                                "shape": "notch",
                            },
                        },
                    },
                    {
                        "type": "gain",
                        "gain_db": 5,
                        "scope": {
                            "time": {"start": 2.0, "end": 3.0},
                            "frequency": "all",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    adjustments = load_adjustments(path, duration=5.0, nyquist_hz=24_000)
    assert [item.placement for item in adjustments] == [
        "before-voice-enhance",
        "after-source-balance",
    ]
    assert adjustments[0].adjustment_id == "adjustment_001"


@pytest.mark.parametrize("gain", [-24.1, 12.1])
def test_adjustments_reject_unsafe_gain(tmp_path, gain) -> None:
    path = tmp_path / "adjustments.json"
    path.write_text(
        json.dumps(
            {
                "adjustments": [
                    {
                        "type": "gain",
                        "gain_db": gain,
                        "scope": {"time": "all", "frequency": "all"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdjustmentError) as captured:
        load_adjustments(path, duration=5.0, nyquist_hz=24_000)
    assert captured.value.code == "gain_out_of_range"
    assert captured.value.field == "adjustments[0].gain_db"


@pytest.mark.parametrize(
    ("time_scope", "provided"),
    [
        ({"start": -0.1, "end": 2.0}, {"start": -0.1, "end": 2.0}),
        ({"start": 4.0, "end": 3.0}, {"start": 4.0, "end": 3.0}),
        ({"start": 4.0, "end": 5.0001}, {"start": 4.0, "end": 5.0001}),
        ({"start": 4.0, "end": 5.1}, {"start": 4.0, "end": 5.1}),
    ],
)
def test_adjustments_report_invalid_time_scope(tmp_path, time_scope, provided) -> None:
    path = tmp_path / "adjustments.json"
    path.write_text(
        json.dumps(
            {
                "adjustments": [
                    {
                        "type": "gain",
                        "gain_db": 3,
                        "scope": {"time": time_scope, "frequency": "all"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdjustmentError) as captured:
        load_adjustments(path, duration=5.0, nyquist_hz=24_000)
    error = captured.value.as_dict()
    assert error["code"] == "scope_out_of_range"
    assert error["field"] == "adjustments[0].scope.time"
    assert error["provided"] == provided
    assert error["allowed"] == {
        "minimum_start_seconds": 0.0,
        "maximum_end_seconds": 5.0,
        "constraint": "start < end",
    }


@pytest.mark.parametrize(
    "frequency_scope",
    [
        {"low_hz": 10, "high_hz": 100, "shape": "band"},
        {"low_hz": 1000, "high_hz": 900, "shape": "band"},
        {"low_hz": 1000, "high_hz": 24_000, "shape": "band"},
    ],
)
def test_adjustments_report_invalid_frequency_scope(tmp_path, frequency_scope) -> None:
    path = tmp_path / "adjustments.json"
    path.write_text(
        json.dumps(
            {
                "adjustments": [
                    {
                        "type": "gain",
                        "gain_db": -3,
                        "scope": {"time": "all", "frequency": frequency_scope},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdjustmentError) as captured:
        load_adjustments(path, duration=5.0, nyquist_hz=24_000)
    error = captured.value.as_dict()
    assert error["code"] == "scope_out_of_range"
    assert error["field"] == "adjustments[0].scope.frequency"
    assert error["provided"] == frequency_scope
    assert error["allowed"] == {
        "minimum_low_hz": 20.0,
        "maximum_high_hz_exclusive": 24_000,
        "constraint": "low_hz < high_hz",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time", float("nan")),
        ("frequency", float("inf")),
    ],
)
def test_adjustments_reject_non_finite_scope_values(tmp_path, field, value) -> None:
    time_scope = {"start": value, "end": 2.0} if field == "time" else "all"
    frequency_scope = (
        {"low_hz": 55, "high_hz": value, "shape": "notch"}
        if field == "frequency"
        else "all"
    )
    path = tmp_path / "adjustments.json"
    path.write_text(
        json.dumps(
            {
                "adjustments": [
                    {
                        "type": "gain",
                        "gain_db": 1,
                        "scope": {
                            "time": time_scope,
                            "frequency": frequency_scope,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdjustmentError) as captured:
        load_adjustments(path, duration=5.0, nyquist_hz=24_000)
    error = captured.value.as_dict()
    assert error["code"] == "non_finite_number"
    assert error["provided"] in {"NaN", "Infinity"}
