"""Gaussian mechanism, written from scratch in NumPy.

M(D) = f(D) + N(0, σ² I) for f with L2 sensitivity Δ2. Two calibrations:

* analytic (default): the smallest σ for which the EXACT privacy profile of
  the Gaussian mechanism (Balle & Wang, ICML 2018, Thm 8) satisfies
      δ(ε) = Φ(Δ/2σ − εσ/Δ) − e^ε Φ(−Δ/2σ − εσ/Δ) ≤ δ.
  δ(ε) is strictly decreasing in σ, so bisection finds σ to any precision.
  Valid for every ε > 0.
* classical (Dwork & Roth 2014, Thm A.1): σ = Δ·sqrt(2 ln(1.25/δ))/ε.
  A sufficient condition proved only for 0 < ε < 1; rejected outside that.
  Always at least as large as the analytic σ (tests check this).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from .validation import check_delta, check_epsilon, check_sensitivity


def gaussian_exact_delta(l2_sensitivity: float, sigma: float, epsilon: float) -> float:
    """Smallest δ for which N(0, σ²) noise on a Δ-sensitive query is (ε, δ)-DP."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if l2_sensitivity == 0:
        return 0.0
    a = l2_sensitivity / (2.0 * sigma)
    b = epsilon * sigma / l2_sensitivity
    # norm.cdf(-a-b) can underflow while e^ε overflows; use logs for the product.
    second = math.exp(epsilon + norm.logcdf(-a - b))
    return float(max(norm.cdf(a - b) - second, 0.0))


def gaussian_sigma_classical(l2_sensitivity: float, epsilon: float, delta: float) -> float:
    check_sensitivity(l2_sensitivity)
    check_epsilon(epsilon)
    check_delta(delta)
    if not epsilon < 1:
        raise ValueError("classical Gaussian calibration is only proved for 0 < epsilon < 1; "
                         "use calibration='analytic'")
    return l2_sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def gaussian_sigma_analytic(l2_sensitivity: float, epsilon: float, delta: float,
                            rel_tol: float = 1e-10) -> float:
    """Bisection on σ for the exact δ(ε) curve. Returns the upper end of the
    final bracket, so the returned σ always satisfies δ(ε) ≤ δ."""
    check_sensitivity(l2_sensitivity)
    check_epsilon(epsilon)
    check_delta(delta)
    if l2_sensitivity == 0:
        return 0.0
    lo, hi = 1e-12 * l2_sensitivity, l2_sensitivity
    while gaussian_exact_delta(l2_sensitivity, hi, epsilon) > delta:
        hi *= 2.0
    while (hi - lo) > rel_tol * hi:
        mid = 0.5 * (lo + hi)
        if gaussian_exact_delta(l2_sensitivity, mid, epsilon) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def gaussian_sigma(l2_sensitivity: float, epsilon: float, delta: float,
                   calibration: str = "analytic") -> float:
    if calibration == "analytic":
        return gaussian_sigma_analytic(l2_sensitivity, epsilon, delta)
    if calibration == "classical":
        return gaussian_sigma_classical(l2_sensitivity, epsilon, delta)
    raise ValueError(f"unknown calibration {calibration!r}")


def sample_gaussian(sigma: float, shape, rng: np.random.Generator) -> np.ndarray:
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return np.zeros(shape)
    return rng.normal(loc=0.0, scale=sigma, size=shape)


def gaussian_mechanism(value, l2_sensitivity: float, epsilon: float, delta: float,
                       rng: np.random.Generator, calibration: str = "analytic") -> np.ndarray:
    value = np.asarray(value, dtype=float)
    sigma = gaussian_sigma(l2_sensitivity, epsilon, delta, calibration)
    return value + sample_gaussian(sigma, value.shape, rng)
