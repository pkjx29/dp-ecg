"""Tests for the from-scratch DP mechanisms.

Three kinds of check:
  1. Sensitivity: brute-force every add/remove neighbour and confirm the query
     never moves by more than the claimed sensitivity.
  2. Noise scale: the sampled noise has the variance the formula promises.
  3. Privacy: the output densities on neighbouring datasets satisfy the DP
     inequality (Laplace: log-ratio <= ε everywhere; Gaussian: exact δ(ε) is
     no larger than the δ we asked for).
"""

import math

import numpy as np
import pytest
from scipy.stats import laplace, norm

from dpecg import mechanisms as M

N_SAMPLES = 200_000


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------- sensitivity

def _max_change_over_neighbours(query, data):
    """Largest |q(D) - q(D')| over all D' formed by removing one record of D.
    (Adding a record is the same pair seen from the other side.)"""
    base = query(data)
    return max(np.abs(query(np.delete(data, i, axis=0)) - base).max() for i in range(len(data)))


@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-2.0, 5.0), (-3.0, -1.0)])
def test_bounded_sum_sensitivity_is_tight_upper_bound(rng, lo, hi):
    data = np.clip(rng.uniform(lo - 1, hi + 1, size=50), lo, hi)
    data[0] = hi if abs(hi) >= abs(lo) else lo  # include the extreme record
    observed = _max_change_over_neighbours(np.sum, data)
    claimed = M.bounded_sum_sensitivity(lo, hi)
    assert observed <= claimed + 1e-12
    assert observed == pytest.approx(claimed)  # tight: the extreme record hits it


def test_count_sensitivity(rng):
    data = rng.normal(size=30)
    assert _max_change_over_neighbours(lambda d: float(len(d)), data) == M.count_sensitivity()


def test_histogram_l1_sensitivity_is_one(rng):
    data = rng.normal(size=40)
    edges = np.linspace(-4, 4, 17)
    l1 = max(np.abs(np.histogram(np.delete(data, i), edges)[0] - np.histogram(data, edges)[0]).sum()
             for i in range(len(data)))
    assert l1 == 1


def test_vector_sum_sensitivities(rng):
    d = 9
    lo, hi = -np.ones(d), np.ones(d)
    data = rng.choice([-1.0, 1.0], size=(20, d))
    l1, l2 = M.vector_sum_sensitivities(lo, hi)
    assert l1 == pytest.approx(d) and l2 == pytest.approx(math.sqrt(d))
    base = data.sum(axis=0)
    diffs = [data.sum(axis=0) - np.delete(data, i, axis=0).sum(axis=0) for i in range(len(data))]
    assert max(np.abs(x).sum() for x in diffs) <= l1 + 1e-12
    assert max(np.linalg.norm(x) for x in diffs) <= l2 + 1e-12
    assert base.shape == (d,)


# ---------------------------------------------------------------- noise scale

@pytest.mark.parametrize("sens,eps", [(1.0, 0.1), (1.0, 1.0), (5.0, 2.0)])
def test_laplace_scale_and_empirical_variance(rng, sens, eps):
    b = M.laplace_scale(sens, eps)
    assert b == pytest.approx(sens / eps)
    noise = M.laplace_mechanism(np.zeros(N_SAMPLES), sens, eps, rng)
    assert noise.mean() == pytest.approx(0.0, abs=4 * b * math.sqrt(2 / N_SAMPLES))
    assert noise.var() == pytest.approx(2 * b ** 2, rel=0.03)  # Var[Lap(b)] = 2b²


@pytest.mark.parametrize("sens,eps,delta", [(1.0, 0.5, 1e-5), (3.0, 0.9, 1e-6)])
def test_gaussian_sigma_and_empirical_std(rng, sens, eps, delta):
    sigma = M.gaussian_sigma(sens, eps, delta)
    assert sigma == pytest.approx(sens * math.sqrt(2 * math.log(1.25 / delta)) / eps)
    noise = M.gaussian_mechanism(np.zeros(N_SAMPLES), sens, eps, delta, rng)
    assert noise.std() == pytest.approx(sigma, rel=0.01)


def test_noise_scales_inversely_with_epsilon():
    assert M.laplace_scale(1, 0.5) == pytest.approx(2 * M.laplace_scale(1, 1.0))
    assert M.gaussian_sigma(1, 0.25, 1e-5) == pytest.approx(2 * M.gaussian_sigma(1, 0.5, 1e-5))


def test_vector_output_shape_and_independence(rng):
    out = M.gaussian_mechanism(np.zeros((N_SAMPLES // 10, 3)), 1.0, 0.5, 1e-5, rng)
    assert out.shape == (N_SAMPLES // 10, 3)
    corr = np.corrcoef(out.T)
    assert np.abs(corr[np.triu_indices(3, 1)]).max() < 0.02


# ---------------------------------------------------------------- privacy

@pytest.mark.parametrize("eps", [0.1, 1.0, 3.0])
def test_laplace_privacy_loss_bounded_by_epsilon(eps):
    """For neighbouring answers f(D) and f(D') = f(D) + Δ, the log density
    ratio of the outputs must be <= ε at every output y."""
    sens = 2.0
    b = M.laplace_scale(sens, eps)
    y = np.linspace(-50, 50, 20_001)
    log_ratio = laplace.logpdf(y, loc=0.0, scale=b) - laplace.logpdf(y, loc=sens, scale=b)
    assert np.abs(log_ratio).max() <= eps + 1e-9
    assert np.abs(log_ratio).max() == pytest.approx(eps)  # and the bound is attained


@pytest.mark.parametrize("eps,delta", [(0.1, 1e-5), (0.5, 1e-5), (0.99, 1e-8)])
def test_gaussian_exact_delta_within_target(eps, delta):
    sigma = M.gaussian_sigma(1.0, eps, delta)
    exact = M.gaussian_exact_delta(1.0, sigma, eps)
    assert 0 <= exact <= delta


def test_gaussian_exact_delta_matches_numerical_integral():
    """Cross-check the closed form against direct numerical integration of
    δ = ∫ max(0, p(y) − e^ε q(y)) dy for p = N(0,σ²), q = N(Δ,σ²)."""
    sens, sigma, eps = 1.0, 1.5, 0.7
    y = np.linspace(-20, 20, 400_001)
    gap = norm.pdf(y, 0, sigma) - math.exp(eps) * norm.pdf(y, sens, sigma)
    numeric = np.trapezoid(np.clip(gap, 0, None), y)
    assert M.gaussian_exact_delta(sens, sigma, eps) == pytest.approx(numeric, rel=1e-4)


# ---------------------------------------------------------------- validation & stats

@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_bad_epsilon(bad):
    with pytest.raises(ValueError):
        M.laplace_scale(1.0, bad)


def test_classical_gaussian_rejects_large_epsilon():
    with pytest.raises(ValueError):
        M.gaussian_sigma(1.0, 1.5, 1e-5)


def test_seeded_rng_is_reproducible():
    a = M.laplace_mechanism(np.zeros(5), 1, 1, np.random.default_rng(42))
    b = M.laplace_mechanism(np.zeros(5), 1, 1, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_dp_mean_is_unbiased_enough_and_tracks_cost(rng):
    x = rng.uniform(40, 120, size=10_000)
    est, cost = M.dp_mean(x, 30, 200, 1.0, rng)
    assert cost.epsilon == pytest.approx(1.0) and cost.delta == 0.0
    assert est == pytest.approx(x.mean(), abs=1.0)


def test_dp_standardizer_recovers_moments(rng):
    X = rng.normal([1.0, -2.0, 50.0], [0.5, 1.0, 10.0], size=(200_000, 3))  # noise on E[z²] ~ σ/n, so n must be large
    lo, hi = np.array([-2, -8, 0.0]), np.array([4, 4, 100.0])
    mean, std, cost = M.dp_standardizer(X, lo, hi, 0.5, 1e-6, rng)
    np.testing.assert_allclose(mean, X.mean(0), rtol=0.05, atol=0.05)
    np.testing.assert_allclose(std, X.std(0), rtol=0.05)
    assert cost.epsilon == 0.5 and cost.delta == 1e-6


# ---------------------------------------------------------------- every neighbour of a small dataset

SMALL = np.array([62.0, 75.0, 58.0, 110.0, 91.0, 47.0, 180.0, 70.0])  # heart rates, bpm
LO, HI = 20.0, 250.0


def _neighbours(data, lo, hi):
    """All add/remove neighbours we can enumerate: remove each record, or add
    a record at the bounds, the midpoint, or a value outside the bounds
    (which clipping must neutralise)."""
    removed = [np.delete(data, i) for i in range(len(data))]
    added = [np.append(data, v) for v in (lo, hi, (lo + hi) / 2, hi + 1000.0, lo - 1000.0)]
    return removed + added


def _clipped_centered_sum(d):
    c = (LO + HI) / 2
    return float((np.clip(d, LO, HI) - c).sum())


@pytest.mark.parametrize("eps", [0.1, 1.0])
def test_laplace_dp_holds_on_every_neighbour(eps):
    """For each neighbour D', the worst-case log density ratio of the released
    sum is |f(D) - f(D')| / b, and must not exceed ε."""
    sens = M.bounded_sum_sensitivity(LO - (LO + HI) / 2, HI - (LO + HI) / 2)
    b = M.laplace_scale(sens, eps)
    y = np.linspace(-3000, 3000, 60_001)
    fD = _clipped_centered_sum(SMALL)
    worst = 0.0
    for nb in _neighbours(SMALL, LO, HI):
        fN = _clipped_centered_sum(nb)
        log_ratio = laplace.logpdf(y, fD, b) - laplace.logpdf(y, fN, b)
        worst = max(worst, np.abs(log_ratio).max())
    assert worst <= eps + 1e-9


@pytest.mark.parametrize("eps,delta", [(0.5, 1e-5), (0.9, 1e-6)])
def test_gaussian_dp_holds_on_every_neighbour(eps, delta):
    """For each neighbour, the exact δ at this ε for the actual shift
    |f(D) - f(D')| must be within the δ the mechanism promises."""
    sens = (HI - LO) / 2
    sigma = M.gaussian_sigma(sens, eps, delta)
    fD = _clipped_centered_sum(SMALL)
    for nb in _neighbours(SMALL, LO, HI):
        shift = abs(fD - _clipped_centered_sum(nb))
        if shift == 0:
            continue
        assert M.gaussian_exact_delta(shift, sigma, eps) <= delta


@pytest.mark.parametrize("eps", [0.1, 1.0])
def test_laplace_histogram_dp_holds_on_every_neighbour(eps):
    """Vector release: independent Laplace per bin, so the log ratio is the
    sum over bins of |Δcount_k| / b, which must stay within ε."""
    edges = np.linspace(LO, HI, 11)
    b = M.laplace_scale(1.0, eps)
    hD = np.histogram(SMALL, edges)[0]
    for nb in _neighbours(SMALL, LO, HI):
        hN = np.histogram(nb, edges)[0]  # same call as dp_histogram
        assert np.abs(hD - hN).sum() / b <= eps + 1e-12
