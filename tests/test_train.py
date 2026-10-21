"""DP-SGD wiring: the accountant's ε must not exceed the target, and both
paths must learn an easy synthetic problem."""

import numpy as np
import pytest

from dpecg.train import TrainConfig, evaluate, train


@pytest.fixture(scope="module")
def easy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y


def test_non_private_learns(easy_data):
    X, y = easy_data
    model, eps = train(X, y, TrainConfig(epochs=5, batch_size=128))
    assert eps is None
    assert evaluate(model, X, y)["accuracy"] > 0.9


def test_dp_sgd_respects_target_epsilon_and_learns(easy_data):
    X, y = easy_data
    model, eps = train(X, y, TrainConfig(epochs=5, batch_size=128), target_epsilon=2.0, delta=1e-5)
    assert 0 < eps <= 2.0 + 1e-3
    assert evaluate(model, X, y)["accuracy"] > 0.8
