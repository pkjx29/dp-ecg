"""Laplace and Gaussian mechanisms, written from scratch in NumPy.

Neighbouring datasets: throughout this module two datasets are neighbours if
one can be obtained from the other by ADDING OR REMOVING one record
("unbounded" DP). This is the same notion Opacus uses for DP-SGD, so the
budgets spent here and in training compose cleanly. See NOTES.md for the
derivations behind every formula below.

This is teaching / research code. Textbook floating-point Laplace sampling is
known to leak through the low-order bits of the output (Mironov, CCS 2012);
a production system should use a vetted library such as OpenDP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _check_eps(epsilon: float) -> None:
    if not (epsilon > 0 and math.isfinite(epsilon)):
        raise ValueError(f"epsilon must be a positive finite number, got {epsilon}")


def _check_sensitivity(sensitivity: float) -> None:
    if not (sensitivity > 0 and math.isfinite(sensitivity)):
        raise ValueError(f"sensitivity must be positive and finite, got {sensitivity}")


# --------------------------------------------------------------------------
# Noise scales
# --------------------------------------------------------------------------

def laplace_scale(l1_sensitivity: float, epsilon: float) -> float:
    """b = Δ1 / ε. Laplace(b) noise then gives ε-DP."""
    _check_sensitivity(l1_sensitivity)
    _check_eps(epsilon)
    return l1_sensitivity / epsilon


def gaussian_sigma(l2_sensitivity: float, epsilon: float, delta: float) -> float:
    """Classical Gaussian mechanism (Dwork & Roth 2014, Thm A.1).

    σ = Δ2 · sqrt(2 ln(1.25/δ)) / ε gives (ε, δ)-DP, but the proof only holds
    for ε < 1, so larger ε is rejected rather than silently mis-calibrated.
    """
    _check_sensitivity(l2_sensitivity)
    _check_eps(epsilon)
    if not 0 < epsilon < 1:
        raise ValueError("classical Gaussian calibration is only valid for 0 < epsilon < 1")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    return l2_sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def gaussian_exact_delta(l2_sensitivity: float, sigma: float, epsilon: float) -> float:
    """The smallest δ for which N(0, σ²) noise is (ε, δ)-DP (Balle & Wang 2018).

    δ(ε) = Φ(Δ/2σ − εσ/Δ) − e^ε · Φ(−Δ/2σ − εσ/Δ)

    Used in the tests to confirm that `gaussian_sigma` really does deliver
    the δ it promises (it is conservative, so the exact δ is smaller).
    """
    a = l2_sensitivity / (2.0 * sigma)
    b = epsilon * sigma / l2_sensitivity
    return float(norm.cdf(a - b) - math.exp(epsilon) * norm.cdf(-a - b))


# --------------------------------------------------------------------------
# Mechanisms
# --------------------------------------------------------------------------

def laplace_mechanism(value, l1_sensitivity: float, epsilon: float,
                      rng: np.random.Generator):
    """Return value + Laplace(Δ1/ε) noise (independent per coordinate)."""
    b = laplace_scale(l1_sensitivity, epsilon)
    value = np.asarray(value, dtype=float)
    return value + rng.laplace(loc=0.0, scale=b, size=value.shape)


def gaussian_mechanism(value, l2_sensitivity: float, epsilon: float, delta: float,
                       rng: np.random.Generator):
    """Return value + N(0, σ²) noise with σ from the classical calibration."""
    sigma = gaussian_sigma(l2_sensitivity, epsilon, delta)
    value = np.asarray(value, dtype=float)
    return value + rng.normal(loc=0.0, scale=sigma, size=value.shape)


# --------------------------------------------------------------------------
# Sensitivities of the queries used in this project
# --------------------------------------------------------------------------

def count_sensitivity() -> float:
    """Adding/removing one record changes a count by exactly 1."""
    return 1.0


def bounded_sum_sensitivity(lower: float, upper: float) -> float:
    """Sum of values clipped to [lower, upper].

    Add/remove one record changes the sum by that record's value, whose
    magnitude is at most max(|lower|, |upper|).
    """
    if lower > upper:
        raise ValueError("lower must be <= upper")
    return max(abs(lower), abs(upper))


def vector_sum_sensitivities(lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    """(L1, L2) sensitivity of a per-feature sum of a d-dimensional record.

    One record contributes the vector x with |x_j| <= m_j = max(|lo_j|, |hi_j|),
    so the sum moves by at most ||m||_1 in L1 and ||m||_2 in L2. For d features
    of equal range, L2 = sqrt(d) * m while L1 = d * m — this gap is why the
    Gaussian mechanism wins for high-dimensional queries.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if np.any(lower > upper):
        raise ValueError("lower must be <= upper elementwise")
    m = np.maximum(np.abs(lower), np.abs(upper))
    return float(m.sum()), float(np.sqrt((m ** 2).sum()))


# --------------------------------------------------------------------------
# DP statistics built from the mechanisms
# --------------------------------------------------------------------------

@dataclass
class PrivacyCost:
    epsilon: float
    delta: float = 0.0

    def __add__(self, other: "PrivacyCost") -> "PrivacyCost":
        # Basic sequential composition: budgets add up.
        return PrivacyCost(self.epsilon + other.epsilon, self.delta + other.delta)


def dp_count(n: int, epsilon: float, rng: np.random.Generator) -> tuple[float, PrivacyCost]:
    return float(laplace_mechanism(n, count_sensitivity(), epsilon, rng)), PrivacyCost(epsilon)


def dp_mean(x, lower: float, upper: float, epsilon: float, rng: np.random.Generator,
            mechanism: str = "laplace", delta: float = 0.0) -> tuple[float, PrivacyCost]:
    """DP mean as c + (noisy sum of (x - c)) / (noisy count), budget split 50/50.

    c = (lower + upper) / 2 is the public midpoint of the clipping range.
    Centering on it halves the sensitivity of the sum, from max(|lo|, |hi|)
    to (hi - lo) / 2, and makes the count noise multiply only (mean - c).
    Under add/remove neighbours n itself is private, so it is noised too
    (Laplace, pure DP); the sum uses the requested mechanism.
    """
    c = (lower + upper) / 2.0
    x = np.clip(np.asarray(x, dtype=float), lower, upper) - c
    eps_half = epsilon / 2.0
    noisy_n, cost_n = dp_count(len(x), eps_half, rng)
    sens = bounded_sum_sensitivity(lower - c, upper - c)
    if mechanism == "laplace":
        noisy_sum = float(laplace_mechanism(x.sum(), sens, eps_half, rng))
        cost_s = PrivacyCost(eps_half)
    elif mechanism == "gaussian":
        noisy_sum = float(gaussian_mechanism(x.sum(), sens, eps_half, delta, rng))
        cost_s = PrivacyCost(eps_half, delta)
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    noisy_n = max(noisy_n, 1.0)  # post-processing: avoid dividing by <= 0
    mean = float(np.clip(c + noisy_sum / noisy_n, lower, upper))
    return mean, cost_n + cost_s


def dp_histogram(x, bin_edges, epsilon: float, rng: np.random.Generator) -> tuple[np.ndarray, PrivacyCost]:
    """Laplace histogram. A record falls into exactly one bin, so adding or
    removing it changes one count by 1: L1 sensitivity is 1, not n_bins."""
    counts, _ = np.histogram(x, bins=bin_edges)
    noisy = laplace_mechanism(counts, 1.0, epsilon, rng)
    return np.clip(noisy, 0, None), PrivacyCost(epsilon)


def dp_standardizer(X: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                    epsilon: float, delta: float, rng: np.random.Generator):
    """DP per-feature mean and std for a d-dimensional feature matrix.

    Features are first clipped to fixed bounds [lo, hi] and mapped to [-1, 1]
    so every coordinate has the same sensitivity. Three Gaussian releases,
    each with ε/3 and δ/3:
      * count                     (L2 sensitivity 1)
      * vector sum of z           (L2 sensitivity sqrt(d), since |z_j| <= 1)
      * vector sum of z²          (L2 sensitivity sqrt(d), since 0 <= z_j² <= 1)
    Returns (mean, std, cost) in the ORIGINAL feature units.
    """
    X = np.asarray(X, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    d = X.shape[1]
    half_range = (upper - lower) / 2.0
    centre = (upper + lower) / 2.0
    Z = (np.clip(X, lower, upper) - centre) / half_range  # in [-1, 1]

    e, dl = epsilon / 3.0, delta / 3.0
    n = max(float(gaussian_mechanism(len(Z), 1.0, e, dl, rng)), 1.0)
    s1 = gaussian_mechanism(Z.sum(axis=0), math.sqrt(d), e, dl, rng)
    s2 = gaussian_mechanism((Z ** 2).sum(axis=0), math.sqrt(d), e, dl, rng)

    mean_z = np.clip(s1 / n, -1.0, 1.0)
    var_z = np.clip(s2 / n - mean_z ** 2, 1e-4, 1.0)  # post-processing keeps var valid
    mean = centre + half_range * mean_z
    std = half_range * np.sqrt(var_z)
    return mean, std, PrivacyCost(epsilon, delta)
