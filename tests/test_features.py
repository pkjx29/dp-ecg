"""Feature extractor checked on a synthetic ECG with known ground truth."""

import numpy as np
import pytest

from dpecg.data import FS, LEADS
from dpecg.features import FEATURE_NAMES, clip_and_impute, extract


def synthetic_ecg(hr_bpm=75.0, r_mv=1.2, t_mv=0.3, seconds=10, seed=0):
    """Gaussian-bump beats: a narrow R wave and a broad T wave 250 ms later,
    with the same shape in every lead plus a little white noise."""
    t = np.arange(seconds * FS) / FS
    sig = np.zeros_like(t)
    for beat_t in np.arange(0.4, seconds - 0.5, 60.0 / hr_bpm):
        sig += r_mv * np.exp(-0.5 * ((t - beat_t) / 0.012) ** 2)
        sig += t_mv * np.exp(-0.5 * ((t - beat_t - 0.25) / 0.04) ** 2)
    sig += np.random.default_rng(seed).normal(0, 0.01, size=sig.shape)
    return np.tile(sig[:, None], (1, len(LEADS)))


def feat(x, name):
    return x[FEATURE_NAMES.index(name)]


@pytest.mark.parametrize("hr", [50.0, 75.0, 120.0])
def test_heart_rate_recovered(hr):
    x = extract(synthetic_ecg(hr_bpm=hr), age=50, sex=1)
    assert feat(x, "hr_bpm") == pytest.approx(hr, rel=0.03)
    assert feat(x, "sdnn_s") < 0.02  # perfectly regular rhythm


def test_amplitudes_have_correct_sign_and_order():
    x = extract(synthetic_ecg(r_mv=1.2, t_mv=0.3), age=50, sex=0)
    r, t = feat(x, "r_amp_II"), feat(x, "t_amp_II")
    assert 0.6 < r < 1.5          # band-pass shrinks a sharp peak a little
    assert 0.1 < t < r            # T upright and smaller than R
    assert feat(x, "s_amp_II") <= 0


def test_flat_line_gives_nans_that_impute_inside_bounds():
    x = extract(np.zeros((10 * FS, len(LEADS))), age=300, sex=1)
    assert np.isnan(feat(x, "hr_bpm"))
    assert feat(x, "age") == 90  # PTB-XL's 300 code is capped
    clean = clip_and_impute(x[None, :])
    assert not np.isnan(clean).any()
