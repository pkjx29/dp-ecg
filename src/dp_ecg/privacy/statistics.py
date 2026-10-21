"""DP summary statistics built on the mechanisms, with budget tracking.

Every function takes a BudgetLedger and records what it spends, so a set of
releases can never silently use more than the declared total. Each returns
a Release carrying everything needed to audit it (query, bounds,
sensitivity, mechanism, ε, δ, noise scale) and the noisy value. Exact values
are never stored in a Release.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import sensitivity as S
from .accounting import BudgetLedger
from .gaussian import gaussian_sigma, sample_gaussian
from .laplace import laplace_scale, sample_laplace
from .validation import as_finite_array, check_bounds


@dataclass
class Release:
    name: str
    query: str
    mechanism: str
    epsilon: float
    delta: float
    sensitivity: float
    sensitivity_norm: str
    noise_scale: float          # Laplace b, or Gaussian σ
    lower: float | None
    upper: float | None
    value: float | list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _noise(mechanism: str, sens: float, eps: float, delta: float, shape, rng,
           calibration: str) -> tuple[np.ndarray, float, str]:
    if mechanism == "laplace":
        b = laplace_scale(sens, eps)
        return sample_laplace(b, shape, rng), b, "L1"
    if mechanism == "gaussian":
        sigma = gaussian_sigma(sens, eps, delta, calibration)
        return sample_gaussian(sigma, shape, rng), sigma, "L2"
    raise ValueError(f"unknown mechanism {mechanism!r}")


def dp_count(name: str, n: int, ledger: BudgetLedger, epsilon: float, rng: np.random.Generator,
             mechanism: str = "laplace", delta: float = 0.0,
             calibration: str = "analytic") -> Release:
    ledger.spend(name, epsilon, delta if mechanism == "gaussian" else 0.0)
    sens = S.count_sensitivity()
    noise, scale, norm = _noise(mechanism, sens, epsilon, delta, (), rng, calibration)
    # Post-processing: a count cannot be negative. Costs no privacy.
    return Release(name, "count", mechanism, epsilon, delta if mechanism == "gaussian" else 0.0,
                   sens, norm, scale, None, None, float(max(n + noise, 0.0)))


def dp_sum(name: str, x, lower: float, upper: float, ledger: BudgetLedger, epsilon: float,
           rng: np.random.Generator, mechanism: str = "laplace", delta: float = 0.0,
           calibration: str = "analytic") -> Release:
    check_bounds(lower, upper)
    x = np.clip(as_finite_array(x), lower, upper)
    ledger.spend(name, epsilon, delta if mechanism == "gaussian" else 0.0)
    sens = S.bounded_sum_sensitivity(lower, upper)
    noise, scale, norm = _noise(mechanism, sens, epsilon, delta, (), rng, calibration)
    return Release(name, "clipped sum", mechanism, epsilon, delta if mechanism == "gaussian" else 0.0,
                   sens, norm, scale, lower, upper, float(x.sum() + noise))


def dp_mean(name: str, x, lower: float, upper: float, ledger: BudgetLedger, epsilon: float,
            rng: np.random.Generator, mechanism: str = "laplace", delta: float = 0.0,
            count_fraction: float = 0.5, calibration: str = "analytic") -> Release:
    """mean = c + NoisySum(x − c) / NoisyCount, with c = (lower+upper)/2 public.

    Under add/remove adjacency n is private, so it is noised too. The budget
    is split count_fraction : (1 − count_fraction) between count and sum (and
    δ the same way for Gaussian). Empty input is allowed: the noisy count is
    floored at 1 and the result clipped to [lower, upper], both
    post-processing, so the answer is always defined.
    """
    check_bounds(lower, upper)
    if not 0 < count_fraction < 1:
        raise ValueError("count_fraction must be in (0, 1)")
    x = np.clip(as_finite_array(x), lower, upper)
    c = (lower + upper) / 2.0
    eps_n, eps_s = epsilon * count_fraction, epsilon * (1 - count_fraction)
    d = delta if mechanism == "gaussian" else 0.0
    d_n, d_s = d * count_fraction, d * (1 - count_fraction)

    ledger.spend(f"{name} [count part]", eps_n, d_n)
    ledger.spend(f"{name} [sum part]", eps_s, d_s)
    sens_n = S.count_sensitivity()
    sens_s = S.centered_sum_sensitivity(lower, upper)
    noise_n, scale_n, norm = _noise(mechanism, sens_n, eps_n, d_n, (), rng, calibration)
    noise_s, scale_s, _ = _noise(mechanism, sens_s, eps_s, d_s, (), rng, calibration)
    noisy_n = max(len(x) + float(noise_n), 1.0)
    mean = float(np.clip(c + ((x - c).sum() + float(noise_s)) / noisy_n, lower, upper))
    return Release(name, f"mean (NoisySum/NoisyCount, count share {count_fraction})", mechanism,
                   epsilon, d, sens_s, norm, scale_s, lower, upper, mean)


def dp_histogram(name: str, x, bin_edges, ledger: BudgetLedger, epsilon: float,
                 rng: np.random.Generator) -> Release:
    """Laplace histogram over fixed public bins; values outside the bins are
    not counted (a per-record rule, so the sensitivity argument still holds)."""
    x = as_finite_array(x)
    edges = np.asarray(bin_edges, dtype=float)
    ledger.spend(name, epsilon, 0.0)
    counts, _ = np.histogram(x, bins=edges)
    b = laplace_scale(S.histogram_l1_sensitivity(), epsilon)
    noisy = np.clip(counts + sample_laplace(b, counts.shape, rng), 0, None)
    return Release(name, f"histogram, {len(counts)} public bins", "laplace", epsilon, 0.0,
                   S.histogram_l1_sensitivity(), "L1", b, float(edges[0]), float(edges[-1]),
                   [float(v) for v in noisy])
