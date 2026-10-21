"""Deterministic, per-record waveform preprocessing.

Nothing here is fitted to the data: the filter and the amplitude scale are
fixed constants. Each record is transformed using only itself, so
preprocessing costs no privacy budget and train/val/test are treated
identically by construction.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


class WaveformError(ValueError):
    pass


def validate_waveform(sig: np.ndarray, n_samples: int, n_leads: int) -> np.ndarray:
    """Return a float array of shape (n_samples, n_leads); raise on wrong shape."""
    sig = np.asarray(sig, dtype=float)
    if sig.shape != (n_samples, n_leads):
        raise WaveformError(f"expected shape ({n_samples}, {n_leads}), got {sig.shape}")
    return sig


def preprocess(sig: np.ndarray, fs: int, bandpass_hz=(0.5, 40.0), order: int = 4,
               scale_mv: float = 1.0, clip_mv: float = 5.0) -> tuple[np.ndarray, int]:
    """(samples, leads) in mV -> (leads, samples) float32, plus the number of
    non-finite values that were replaced by 0.

    * 0.5-40 Hz zero-phase Butterworth band-pass: removes baseline wander and
      mains/EMG noise without shifting wave timing.
    * divide by a fixed scale and clip to +-clip_mv: bounded inputs, no
      fitted statistics. Morphology and relative amplitudes are preserved.
    """
    n_bad = int((~np.isfinite(sig)).sum())
    sig = np.where(np.isfinite(sig), sig, 0.0)
    sos = butter(order, list(bandpass_hz), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, sig, axis=0)
    out = np.clip(filtered / scale_mv, -clip_mv, clip_mv)
    return out.T.astype(np.float32), n_bad
