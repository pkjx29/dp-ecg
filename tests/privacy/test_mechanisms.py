"""Calibration, sampling and privacy-inequality tests for the mechanisms.

Distinction kept throughout: closed-form and exact checks test the privacy
inequality; sampled checks (variances, means) only test that the sampler
draws from the calibrated distribution and are labelled as diagnostics.
"""

import math

import numpy as np
import pytest
from scipy.stats import laplace, norm

from dp_ecg.privacy.discrete import geometric_alpha, geometric_log_pmf, geometric_mechanism
from dp_ecg.privacy.gaussian import (
    gaussian_exact_delta,
    gaussian_mechanism,
    gaussian_sigma,
    gaussian_sigma_analytic,
    gaussian_sigma_classical,
)
from dp_ecg.privacy.laplace import laplace_mechanism, laplace_scale

N = 200_000


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ------------------------------------------------------------ Laplace

@pytest.mark.parametrize("sens,eps", [(1.0, 0.1), (1.0, 1.0), (5.0, 2.0)])
def test_laplace_scale_formula(sens, eps):
    assert laplace_scale(sens, eps) == pytest.approx(sens / eps)


@pytest.mark.parametrize("sens,eps", [(1.0, 0.5), (5.0, 2.0)])
def test_laplace_sampler_diagnostic(rng, sens, eps):
    """Diagnostic, not a privacy test: Var[Lap(b)] = 2b²."""
    b = laplace_scale(sens, eps)
    noise = laplace_mechanism(np.zeros(N), sens, eps, rng)
    assert noise.var() == pytest.approx(2 * b**2, rel=0.03)
    assert noise.mean() == pytest.approx(0.0, abs=5 * b * math.sqrt(2 / N))


@pytest.mark.parametrize("eps", [0.1, 1.0, 3.0])
def test_laplace_log_ratio_bounded_and_tight(eps):
    """Closed-form density ratio on a fine grid: never above ε, and reached."""
    sens = 2.0
    b = laplace_scale(sens, eps)
    y = np.linspace(-60, 60, 120_001)
    lr = laplace.logpdf(y, 0.0, b) - laplace.logpdf(y, sens, b)
    assert np.abs(lr).max() <= eps + 1e-9
    assert np.abs(lr).max() == pytest.approx(eps)


def test_zero_sensitivity_adds_no_noise(rng):
    out = laplace_mechanism([3.0, 4.0], 0.0, 1.0, rng)
    np.testing.assert_array_equal(out, [3.0, 4.0])
    assert gaussian_sigma(0.0, 1.0, 1e-5) == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_bad_epsilon(bad):
    with pytest.raises(ValueError):
        laplace_scale(1.0, bad)
    with pytest.raises(ValueError):
        gaussian_sigma(1.0, bad, 1e-5)


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_rejects_bad_sensitivity(bad):
    with pytest.raises(ValueError):
        laplace_scale(bad, 1.0)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 2.0])
def test_rejects_bad_delta(bad):
    with pytest.raises(ValueError):
        gaussian_sigma(1.0, 0.5, bad)


def test_seeded_rng_reproducible():
    a = laplace_mechanism(np.zeros(5), 1, 1, np.random.default_rng(42))
    b = laplace_mechanism(np.zeros(5), 1, 1, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------ Gaussian

def test_exact_delta_matches_numerical_integral():
    """δ(ε) = ∫ max(0, p − e^ε q): closed form vs direct integration."""
    sens, sigma, eps = 1.0, 1.5, 0.7
    y = np.linspace(-20, 20, 400_001)
    gap = norm.pdf(y, 0, sigma) - math.exp(eps) * norm.pdf(y, sens, sigma)
    numeric = np.trapezoid(np.clip(gap, 0, None), y)
    assert gaussian_exact_delta(sens, sigma, eps) == pytest.approx(numeric, rel=1e-4)


@pytest.mark.parametrize("eps,delta", [(0.1, 1e-5), (0.5, 1e-6), (1.0, 1e-5), (4.0, 1e-5), (10.0, 1e-8)])
def test_analytic_sigma_meets_delta_and_is_tight(eps, delta):
    sigma = gaussian_sigma_analytic(1.0, eps, delta)
    assert gaussian_exact_delta(1.0, sigma, eps) <= delta
    # Tight: 1% less noise would break the guarantee.
    assert gaussian_exact_delta(1.0, 0.99 * sigma, eps) > delta


@pytest.mark.parametrize("eps,delta", [(0.1, 1e-5), (0.5, 1e-6), (0.99, 1e-8)])
def test_classical_sigma_is_valid_and_conservative(eps, delta):
    classical = gaussian_sigma_classical(1.0, eps, delta)
    assert classical == pytest.approx(math.sqrt(2 * math.log(1.25 / delta)) / eps)
    assert gaussian_exact_delta(1.0, classical, eps) <= delta
    assert classical >= gaussian_sigma_analytic(1.0, eps, delta)


def test_classical_rejects_large_epsilon():
    with pytest.raises(ValueError):
        gaussian_sigma_classical(1.0, 1.5, 1e-5)


def test_gaussian_sigma_scales_linearly_with_sensitivity():
    assert gaussian_sigma(3.0, 0.8, 1e-5) == pytest.approx(3 * gaussian_sigma(1.0, 0.8, 1e-5), rel=1e-8)


def test_gaussian_sampler_diagnostic(rng):
    sigma = gaussian_sigma(2.0, 0.5, 1e-5)
    noise = gaussian_mechanism(np.zeros(N), 2.0, 0.5, 1e-5, rng)
    assert noise.std() == pytest.approx(sigma, rel=0.01)


def test_vector_noise_is_independent(rng):
    out = gaussian_mechanism(np.zeros((N // 10, 3)), 1.0, 0.5, 1e-5, rng)
    corr = np.corrcoef(out.T)
    assert np.abs(corr[np.triu_indices(3, 1)]).max() < 0.02


# ------------------------------------------------------------ geometric

def test_geometric_pmf_sums_to_one():
    alpha = geometric_alpha(1.0, 0.7)
    z = np.arange(-2000, 2001)
    assert np.exp(geometric_log_pmf(z, alpha)).sum() == pytest.approx(1.0, abs=1e-12)


def test_geometric_sampler_matches_pmf(rng):
    """Diagnostic: empirical frequencies of the sampler vs the pmf."""
    alpha = geometric_alpha(1.0, 1.0)
    draws = np.array([geometric_mechanism(0, 1.0, 1.0, rng) for _ in range(40_000)])
    for z in (-2, -1, 0, 1, 2):
        assert (draws == z).mean() == pytest.approx(np.exp(geometric_log_pmf(z, alpha)), abs=0.01)
