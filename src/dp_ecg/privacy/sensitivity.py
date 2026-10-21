"""Global sensitivities of the bounded queries used in this project.

Adjacency: D and D' are neighbours if one is the other with ONE record added
or removed (unbounded DP). Opacus uses the same relation for DP-SGD, so
budgets from both parts compose.

Every bound below is a worst case over ALL datasets (global sensitivity),
which is what the mechanisms need. Local sensitivity (the change around the
actual dataset) can be smaller but is not safe to calibrate noise with,
because it is itself a function of the private data.
"""

from __future__ import annotations

import numpy as np

from .validation import check_bounds


def count_sensitivity() -> float:
    """Add/remove one record changes a count by exactly 1 (L1 = L2 = 1)."""
    return 1.0


def bounded_sum_sensitivity(lower: float, upper: float) -> float:
    """Sum of values clipped to [lower, upper]: the added or removed record
    contributes its own value, at most max(|lower|, |upper|) in size."""
    check_bounds(lower, upper)
    return float(max(abs(lower), abs(upper)))


def centered_sum_sensitivity(lower: float, upper: float) -> float:
    """Sum of (clip(x) - c) with public c = (lower + upper)/2: each term lies
    in [-(upper-lower)/2, (upper-lower)/2], so the sensitivity is half the
    range. This is never larger than bounded_sum_sensitivity."""
    check_bounds(lower, upper)
    return float((upper - lower) / 2.0)


def histogram_l1_sensitivity() -> float:
    """Fixed, public, disjoint bins: a record lands in at most one bin, so one
    count changes by 1. (Under replace-one adjacency it would be 2.)"""
    return 1.0


def vector_sum_sensitivities(lower, upper) -> tuple[float, float]:
    """(L1, L2) sensitivity of a per-coordinate sum of d-dimensional records
    with coordinate j clipped to [lower_j, upper_j]. With m_j the largest
    magnitude in coordinate j: L1 = sum(m), L2 = sqrt(sum(m^2)).
    For d equal ranges L2 = sqrt(d)*m but L1 = d*m."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape")
    for lo, hi in zip(lower.ravel(), upper.ravel()):
        check_bounds(lo, hi)
    m = np.maximum(np.abs(lower), np.abs(upper))
    return float(m.sum()), float(np.sqrt((m**2).sum()))
