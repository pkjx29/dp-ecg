import numpy as np
import pandas as pd
import pytest

from dp_ecg.data.dataset import ECGDataset
from dp_ecg.data.preprocessing import WaveformError, preprocess, validate_waveform
from dp_ecg.data.splits import PatientLeakageError, assign_splits, check_patient_disjoint


def test_preprocess_shape_dtype_and_bounds():
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 1, (1000, 12))
    sig[5, 3] = np.nan
    sig[10, 0] = 50.0  # artefact spike far outside the clip range
    out, n_bad = preprocess(sig, 100, clip_mv=5.0)
    assert out.shape == (12, 1000) and out.dtype == np.float32
    assert n_bad == 1 and np.isfinite(out).all() and np.abs(out).max() <= 5.0


def test_preprocess_removes_baseline_wander():
    t = np.arange(1000) / 100
    drift = np.tile((2.0 * np.sin(2 * np.pi * 0.05 * t))[:, None], (1, 12))
    out, _ = preprocess(drift, 100)
    assert np.abs(out[:, 100:-100]).max() < 0.05  # 0.05 Hz is below the 0.5 Hz cut-off


def test_validate_rejects_wrong_shape():
    with pytest.raises(WaveformError):
        validate_waveform(np.zeros((999, 12)), 1000, 12)


def test_splits_and_leakage_check():
    db = pd.DataFrame({"strat_fold": [1, 9, 10, 2]})
    assert list(assign_splits(db, {"train": [1, 2], "val": [9], "test": [10]})) == ["train", "val", "test", "train"]
    with pytest.raises(ValueError):
        assign_splits(db, {"train": [1], "val": [9], "test": [10]})
    assert check_patient_disjoint([1, 2, 3], ["train", "val", "test"]) == {"test": 1, "train": 1, "val": 1}
    with pytest.raises(PatientLeakageError):
        check_patient_disjoint([1, 1, 2], ["train", "test", "val"])


def test_dataset_returns_float32():
    w = np.zeros((4, 12, 1000), dtype=np.float16)
    ds = ECGDataset(w, np.array([0, 1, 0, 1]), np.array([1, 3]))
    x, y = ds[0]
    assert len(ds) == 2 and x.dtype.is_floating_point and x.shape == (12, 1000) and float(y) == 1.0
