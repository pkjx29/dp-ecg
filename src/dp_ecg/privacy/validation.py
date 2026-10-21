"""Shared argument checks for the privacy mechanisms.

Invalid privacy parameters must fail loudly: a silently mis-calibrated
mechanism still produces plausible-looking numbers, so the bug would be
invisible in the output.
"""

from __future__ import annotations

import math

import numpy as np


def check_epsilon(epsilon: float) -> None:
    if not (isinstance(epsilon, (int, float)) and math.isfinite(epsilon) and epsilon > 0):
        raise ValueError(f"epsilon must be a positive finite number, got {epsilon!r}")


def check_delta(delta: float) -> None:
    if not (isinstance(delta, (int, float)) and 0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")


def check_sensitivity(sensitivity: float) -> None:
    """Zero is allowed: a provably insensitive query needs no noise."""
    if not (isinstance(sensitivity, (int, float)) and math.isfinite(sensitivity) and sensitivity >= 0):
        raise ValueError(f"sensitivity must be finite and >= 0, got {sensitivity!r}")


def check_bounds(lower: float, upper: float) -> None:
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError(f"bounds must be finite, got [{lower}, {upper}]")
    if lower > upper:
        raise ValueError(f"lower bound {lower} exceeds upper bound {upper}")


def as_finite_array(values) -> np.ndarray:
    """Values entering a bounded query. NaN has no position inside [lo, hi],
    so it is rejected rather than clipped; +-inf is fine (clipping maps it to
    a bound)."""
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        raise ValueError("input contains NaN; drop or impute per record before the query")
    return arr
