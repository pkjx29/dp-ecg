"""Two-sided geometric ("discrete Laplace") mechanism for integer queries.

For an integer-valued query with sensitivity Δ, add Z with
    P(Z = z) = (1 − α)/(1 + α) · α^|z|,   α = exp(−ε/Δ).
For neighbouring answers c, c' the pmf ratio at output y is
    α^(|y−c| − |y−c'|) ≤ α^(−|c−c'|) ≤ e^ε.
Unlike continuous Laplace, its output space is the integers, so the privacy
inequality can be checked EXACTLY on every output (see verification.py), and
it has no floating-point leakage in the released value.
"""

from __future__ import annotations

import math

import numpy as np

from .validation import check_epsilon, check_sensitivity


def geometric_alpha(sensitivity: float, epsilon: float) -> float:
    check_sensitivity(sensitivity)
    check_epsilon(epsilon)
    if sensitivity == 0:
        return 0.0
    return math.exp(-epsilon / sensitivity)


def geometric_log_pmf(z, alpha: float) -> np.ndarray:
    z = np.abs(np.asarray(z))
    if alpha == 0:
        return np.where(z == 0, 0.0, -np.inf)
    return math.log((1 - alpha) / (1 + alpha)) + z * math.log(alpha)


def geometric_mechanism(value: int, sensitivity: float, epsilon: float,
                        rng: np.random.Generator) -> int:
    alpha = geometric_alpha(sensitivity, epsilon)
    if alpha == 0:
        return int(value)
    # Difference of two i.i.d. geometric variables on {0, 1, ...} is two-sided geometric.
    g1, g2 = rng.geometric(1 - alpha, size=2) - 1
    return int(value + g1 - g2)
