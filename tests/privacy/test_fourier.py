"""FPA_k: transform properties, sensitivity, calibration and the privacy bound."""

import math

import numpy as np
import pytest

from dp_ecg.privacy.fourier import (
    clip_rows_l2,
    fpa_release,
    fpa_scale,
    from_real_vector,
    lpa_release,
    reconstruct,
    to_real_vector,
    truncate_dft,
)


def test_full_truncation_round_trip_is_exact():
    # Odd length: k = (N+1)/2 coefficients plus their mirrors are the whole spectrum.
    x = np.random.default_rng(0).normal(size=(3, 71))
    assert np.allclose(reconstruct(truncate_dft(x, 36), 71), x)


def test_low_pass_keeps_smooth_signals():
    t = np.arange(70) / 70
    x = np.sin(2 * np.pi * 2 * t)[None]   # frequency 2 -> coefficient index 2
    assert np.allclose(reconstruct(truncate_dft(x, 3), 70), x, atol=1e-12)


def test_real_vector_is_a_contraction_and_invertible():
    """Parseval with conjugate symmetry: for ANY difference of L2 norm 1 between
    neighbouring series, u differs by at most 1 in L2 and sqrt(L(2k-1)) in L1.
    The bound is tight when all energy sits in the kept frequencies."""
    rng = np.random.default_rng(1)
    k, L, n = 6, 12, 70
    for _ in range(200):
        d = rng.normal(size=(L, n))
        d /= np.linalg.norm(d)
        u = to_real_vector(truncate_dft(d, k))
        assert np.linalg.norm(u) <= 1 + 1e-12
        assert np.abs(u).sum() <= math.sqrt(L * (2 * k - 1)) + 1e-9
    low = np.cos(2 * np.pi * 3 * np.arange(n) / n)[None]
    low /= np.linalg.norm(low)
    assert np.linalg.norm(to_real_vector(truncate_dft(low, k))) == pytest.approx(1.0)
    c = truncate_dft(rng.normal(size=(L, n)), k)
    expected = c.copy()
    expected[:, 0] = c[:, 0].real          # DC of a real signal is real; u drops its (zero) imaginary part
    assert np.allclose(from_real_vector(to_real_vector(c), k), expected)


def test_fpa_noise_scale():
    assert fpa_scale(2.0, 0.5, k=5, n_channels=12) == pytest.approx(math.sqrt(9 * 12) * 2.0 / 0.5)


def test_fpa_beats_lpa_on_smooth_signal():
    """Diagnostic of the paper's claim: for a smooth series, FPA_k with small k
    has far less error than adding noise to every sample."""
    rng = np.random.default_rng(2)
    t = np.arange(70) / 70
    x = np.tile(50 * np.sin(2 * np.pi * t), (12, 1))
    err = lambda f: np.mean([np.linalg.norm(f() - x) for _ in range(200)])  # noqa: E731
    assert err(lambda: fpa_release(x, 1.0, 1.0, 3, rng)) < 0.5 * err(lambda: lpa_release(x, 1.0, 1.0, rng))


def test_clip_rows_l2():
    t = np.ones((4, 2, 3))
    t[0] *= 10
    out = clip_rows_l2(t, 2.0)
    assert np.allclose(np.linalg.norm(out.reshape(4, -1), axis=1), [2.0, 2.0, 2.0, 2.0])


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        truncate_dft(np.zeros((1, 10)), 6)  # max k for N=10 is 5
