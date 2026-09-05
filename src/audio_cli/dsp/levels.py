"""Scalar and array decibel measurements."""

from __future__ import annotations

import math

import numpy as np

EPSILON = 1e-12


def amplitude_to_db(value: float) -> float:
    return 20.0 * math.log10(max(abs(value), EPSILON))


def rms_dbfs(samples: np.ndarray) -> float:
    array = np.asarray(samples, dtype=np.float64)
    if array.size == 0:
        return -240.0
    return amplitude_to_db(float(np.sqrt(np.mean(np.square(array)) + EPSILON)))


def peak_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -240.0
    return amplitude_to_db(float(np.max(np.abs(samples))))
