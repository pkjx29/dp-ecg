"""Laplace mechanism, written from scratch in NumPy.

Guarantee (Dwork & Roth 2014, Thm 3.6): if f has L1 sensitivity Δ1 under the
adjacency relation, then M(D) = f(D) + Lap(Δ1/ε) per coordinate is ε-DP.
Calibration (`laplace_scale`) and sampling (`sample_laplace`) are separate
so each can be tested on its own.

Caveat: textbook floating-point Laplace sampling leaks through the
low-order bits of the output (Mironov, CCS 2012). This is teaching code; a
deployment should use a vetted library (OpenDP) or a discrete mechanism.
"""

from __future__ import annotations

import numpy as np

from .validation import check_epsilon, check_sensitivity


def laplace_scale(l1_sensitivity: float, epsilon: float) -> float:
    """b = Δ1 / ε."""
    check_sensitivity(l1_sensitivity)
    check_epsilon(epsilon)
    return l1_sensitivity / epsilon


def sample_laplace(scale: float, shape, rng: np.random.Generator) -> np.ndarray:
    if scale < 0:
        raise ValueError("scale must be >= 0")
    if scale == 0:
        return np.zeros(shape)
    return rng.laplace(loc=0.0, scale=scale, size=shape)


def laplace_mechanism(value, l1_sensitivity: float, epsilon: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Release value + Lap(Δ1/ε) noise, independently per coordinate.
    A zero-sensitivity query is returned unchanged: it reveals nothing about
    any single record, so adding noise would only cost accuracy."""
    value = np.asarray(value, dtype=float)
    scale = laplace_scale(l1_sensitivity, epsilon)
    return value + sample_laplace(scale, value.shape, rng)
