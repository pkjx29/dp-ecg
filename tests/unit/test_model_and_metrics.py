import numpy as np
import pytest
import torch
from opacus.validators import ModuleValidator

from dp_ecg.config import load_config
from dp_ecg.models.cnn import ECGCNN, build_model
from dp_ecg.models.metrics import choose_threshold, classification_metrics


def test_cnn_output_shape_and_opacus_compatible():
    model = build_model(load_config("baseline")["model"])
    assert model(torch.zeros(3, 12, 1000)).shape == (3,)
    assert ModuleValidator.validate(model, strict=False) == []


def test_batchnorm_model_would_be_rejected():
    bad = torch.nn.Sequential(torch.nn.Conv1d(12, 8, 3), torch.nn.BatchNorm1d(8))
    assert ModuleValidator.validate(bad, strict=False)


def test_cnn_rejects_bad_group_config():
    with pytest.raises(ValueError):
        ECGCNN(channels=(30,), kernel_sizes=(3,), groupnorm_groups=8)


def test_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    thr = choose_threshold(y, s)
    m = classification_metrics(y, s, thr)
    assert m["auroc"] == 1.0 and m["balanced_accuracy"] == 1.0
    assert m["confusion_matrix"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 2}
    assert m["positive_prevalence"] == pytest.approx(0.4)


def test_configs_parse():
    for name in ["data", "baseline", "dp_sgd", "privacy_mechanisms", "attack"]:
        assert isinstance(load_config(name), dict)
    shares = [q["epsilon_share"] for q in load_config("privacy_mechanisms")["summary_statistics"]["queries"]]
    assert sum(shares) == pytest.approx(1.0)  # the release spends exactly its declared budget
