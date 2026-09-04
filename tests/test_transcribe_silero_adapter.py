from __future__ import annotations

import pytest

from audio_cli.transcribe.adapters import normalize_vad_regions


def test_vad_normalizer_publishes_only_owned_fields() -> None:
    assert normalize_vad_regions(
        [
            {
                "start": 0.1,
                "end": 0.9,
                "mean_probability": 0.8,
            }
        ]
    ) == [{"start": 0.1, "end": 0.9}]


@pytest.mark.parametrize(
    "invalid_bound",
    [False, "0.0", float("nan"), float("inf"), 10**400],
)
def test_vad_rejects_non_finite_or_coercible_bounds(
    invalid_bound: object,
) -> None:
    with pytest.raises(ValueError, match="region 0 start"):
        normalize_vad_regions([{"start": invalid_bound, "end": 1.0}])


@pytest.mark.parametrize(
    "regions",
    [
        [{"start": -0.1, "end": 0.5}],
        [{"start": 0.5, "end": 0.5}],
        [{"start": 0.6, "end": 0.5}],
        [{"start": 0.0, "end": 0.8}, {"start": 0.7, "end": 1.0}],
        [{"start": 1.0, "end": 1.5}, {"start": 0.0, "end": 0.5}],
        [{"start": 0.0, "end": 0.0000004}],
    ],
)
def test_vad_rejects_nonpositive_nonchronological_or_overlapping_regions(
    regions: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError):
        normalize_vad_regions(regions)
