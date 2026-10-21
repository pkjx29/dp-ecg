"""Opacus integration on a tiny synthetic problem with the project CNN."""

import numpy as np
import torch

from dp_ecg.config import load_config
from dp_ecg.data.dataset import ECGDataset
from dp_ecg.models.cnn import build_model
from dp_ecg.training.loop import PrivacySettings, train_model


def test_dp_training_reports_budget_within_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    w = rng.normal(0, 0.1, (200, 12, 1000)).astype(np.float16)
    w[y == 1, :, 100:120] += 1
    cfg = {"optimizer": "adam", "learning_rate": 0.002, "weight_decay": 0.0, "batch_size": 32, "epochs": 2}
    xv, yv = ECGDataset(w, y, np.arange(160, 200)).tensors()
    res = train_model(build_model(load_config("baseline")["model"]), ECGDataset(w, y, np.arange(160)),
                      xv, yv.numpy(), cfg, 0, PrivacySettings(2.0, 1e-5, 1.0))
    p = res.privacy
    assert np.isfinite(p["achieved_epsilon"]) and 0 < p["achieved_epsilon"] <= 2.0 * 1.01
    assert p["noise_multiplier"] > 0 and p["sample_rate"] == 32 / 160
    assert p["steps"] > 0 and len(res.history) == 2 and "epsilon_so_far" in res.history[-1]
    assert isinstance(res.model, torch.nn.Module) and not hasattr(res.model, "_module")
