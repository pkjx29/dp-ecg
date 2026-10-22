"""Fourier Perturbation Algorithm (FPA_k, Rastogi & Nath, SIGMOD 2010) and the
time-domain baselines it is compared with.

The query is a multi-channel series Q(D) of shape (L, N), here the SUM over
records of each record's 12-lead beat template, with L2 sensitivity Δ2
(each record's template is clipped to L2 norm <= Δ2 before summing).

FPA_k, per channel:
  1. F = DFT(x), orthonormal ("ortho") scaling.
  2. Keep F_0..F_{k-1}: the low frequencies, where a smooth signal keeps most
     of its energy.
  3. Write them as one real vector u = (F_0, √2·Re F_1.., √2·Im F_1..) of
     length 2k−1 (F_0 is real for a real signal).
  4. Add noise to u, undo the √2, rebuild the full spectrum (zeros above k,
     F_{N−j} = conj(F_j)) and invert. Everything after the noise is
     post-processing: free.

Sensitivity. For a real signal Parseval gives
    ||x||² = |F_0|² + 2·Σ_j |F_j|² (+ Nyquist term)  >=  ||u||²,
because each F_j (j >= 1) appears twice, as itself and as its mirror F_{N−j}.
So u is a contraction of x: ||u(D) − u(D')||_2 <= Δ2, over all L leads
together. Hence:
  * Laplace (pure ε-DP): ||·||_1 <= sqrt(L(2k−1))·||·||_2, b = sqrt(L(2k−1))·Δ2/ε.
  * Gaussian ((ε, δ)-DP): σ from the analytic calibration with Δ2 directly.
The √2 weighting matters: noising F_j directly would treat its energy as
counted once and add √2 more noise than needed.

Comparison with noise on all L·N time samples: Laplace there needs
b = sqrt(L·N)·Δ2/ε (LPA), so for Laplace, truncating to k coefficients cuts
the dimension inside the L1/L2 conversion as well as the number of noisy
values. Gaussian noise has no such conversion, so its gain from truncation is
smaller. FPA also adds a deterministic truncation error, which sets a best k.
"""

from __future__ import annotations

import math

import numpy as np

from .gaussian import gaussian_sigma
from .laplace import laplace_scale, sample_laplace
from .validation import check_epsilon

SQRT2 = math.sqrt(2.0)


def _check_k(k: int, n: int) -> None:
    # k <= ceil(n/2) keeps every kept index j >= 1 distinct from its mirror n − j
    # (and excludes the Nyquist term for even n), which the Parseval bound needs.
    if not 1 <= k <= (n + 1) // 2:
        raise ValueError(f"k must be in [1, {(n + 1) // 2}] for series of length {n}, got {k}")


def truncate_dft(x: np.ndarray, k: int) -> np.ndarray:
    """First k orthonormal DFT coefficients of each row of x (L, N) -> (L, k) complex."""
    _check_k(k, x.shape[-1])
    return np.fft.fft(x, axis=-1, norm="ortho")[..., :k]


def to_real_vector(coeffs: np.ndarray) -> np.ndarray:
    """(L, k) complex -> (L, 2k−1) real u with ||u||_2 <= ||x||_2 (see module docstring)."""
    return np.concatenate([coeffs[:, :1].real, SQRT2 * coeffs[:, 1:].real, SQRT2 * coeffs[:, 1:].imag], axis=1)


def from_real_vector(u: np.ndarray, k: int) -> np.ndarray:
    re = np.concatenate([u[:, :1], u[:, 1:k] / SQRT2], axis=1)
    im = np.concatenate([np.zeros_like(u[:, :1]), u[:, k:] / SQRT2], axis=1)
    return re + 1j * im


def reconstruct(coeffs: np.ndarray, n: int) -> np.ndarray:
    """Inverse of truncate_dft with the discarded coefficients set to zero and
    conjugate symmetry restored so the result is real."""
    L, k = coeffs.shape
    full = np.zeros((L, n), dtype=complex)
    full[:, 0] = coeffs[:, 0].real            # the DC term of a real signal is real
    full[:, 1:k] = coeffs[:, 1:k]
    full[:, n - k + 1:] = np.conj(coeffs[:, 1:k][:, ::-1])
    return np.fft.ifft(full, axis=-1, norm="ortho").real


def fpa_scale(l2_sensitivity: float, epsilon: float, k: int, n_channels: int) -> float:
    return laplace_scale(math.sqrt((2 * k - 1) * n_channels) * l2_sensitivity, epsilon)


def fpa_release(x: np.ndarray, l2_sensitivity: float, epsilon: float, k: int,
                rng: np.random.Generator) -> np.ndarray:
    """ε-DP release of a (L, N) series with L2 sensitivity Δ2, via FPA_k (Laplace)."""
    check_epsilon(epsilon)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    L, n = x.shape
    u = to_real_vector(truncate_dft(x, k))
    noisy = u + sample_laplace(fpa_scale(l2_sensitivity, epsilon, k, L), u.shape, rng)
    return reconstruct(from_real_vector(noisy, k), n)


def fpa_gaussian_release(x: np.ndarray, l2_sensitivity: float, epsilon: float, delta: float, k: int,
                         rng: np.random.Generator) -> np.ndarray:
    """(ε, δ)-DP variant: u has L2 sensitivity <= Δ2, so analytic Gaussian noise applies directly."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    L, n = x.shape
    u = to_real_vector(truncate_dft(x, k))
    noisy = u + rng.normal(0.0, gaussian_sigma(l2_sensitivity, epsilon, delta), u.shape)
    return reconstruct(from_real_vector(noisy, k), n)


def lpa_release(x: np.ndarray, l2_sensitivity: float, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """ε-DP baseline: Laplace on every time sample, L1 sensitivity <= sqrt(L·N)·Δ2."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    b = laplace_scale(math.sqrt(x.size) * l2_sensitivity, epsilon)
    return x + sample_laplace(b, x.shape, rng)


def gaussian_time_release(x: np.ndarray, l2_sensitivity: float, epsilon: float, delta: float,
                          rng: np.random.Generator) -> np.ndarray:
    """(ε, δ)-DP baseline: analytic Gaussian on every time sample (L2 sensitivity Δ2)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return x + rng.normal(0.0, gaussian_sigma(l2_sensitivity, epsilon, delta), x.shape)


def clip_rows_l2(templates: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale each record's (L, N) template so its Frobenius (L2) norm is <= max_norm."""
    flat = templates.reshape(len(templates), -1)
    norms = np.linalg.norm(flat, axis=1)
    factor = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return templates * factor[:, None, None]
