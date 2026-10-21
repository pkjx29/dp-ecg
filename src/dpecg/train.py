"""Small MLP classifier, trained either normally or with DP-SGD (Opacus).

Both paths share the model, optimiser, batch size and number of epochs, so
the only difference between "baseline" and "DP" is what DP-SGD adds:
per-example gradient clipping to norm C, Gaussian noise N(0, σ²C²) added to
the summed gradient, and Poisson sampling of batches (each record joins a
batch independently with probability q = batch_size / n). The accountant
(RDP) turns (q, σ, steps) into the (ε, δ) that is reported.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    hidden: int = 64
    epochs: int = 30
    batch_size: int = 512        # expected batch size under Poisson sampling
    lr: float = 0.05
    momentum: float = 0.9
    max_grad_norm: float = 1.0   # C, the per-example clipping norm (DP only)
    seed: int = 0


def make_model(d: int, hidden: int) -> nn.Module:
    # No BatchNorm: it mixes examples within a batch, which breaks
    # per-example gradients and therefore DP-SGD.
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).squeeze(1).numpy()


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = predict_proba(model, X)
    return {"accuracy": accuracy_score(y, p >= 0.5), "auroc": roc_auc_score(y, p)}


def train(X_tr: np.ndarray, y_tr: np.ndarray, cfg: TrainConfig,
          target_epsilon: float | None = None, delta: float = 1e-5):
    """Train and return (model, epsilon_spent). epsilon_spent is None when
    target_epsilon is None (non-private baseline)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    ds = TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),
                       torch.as_tensor(y_tr, dtype=torch.float32))
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    model = make_model(X_tr.shape[1], cfg.hidden)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    engine = None
    if target_epsilon is not None:
        from opacus import PrivacyEngine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Opacus warns about the secure RNG being off
            engine = PrivacyEngine(accountant="rdp")
            model, opt, loader = engine.make_private_with_epsilon(
                module=model, optimizer=opt, data_loader=loader,
                target_epsilon=target_epsilon, target_delta=delta,
                epochs=cfg.epochs, max_grad_norm=cfg.max_grad_norm)

    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            if len(xb) == 0:  # Poisson sampling can draw an empty batch
                continue
            opt.zero_grad()
            loss_fn(model(xb).squeeze(1), yb).backward()
            opt.step()

    eps = engine.get_epsilon(delta) if engine is not None else None
    if engine is not None:
        model = model._module  # unwrap GradSampleModule for plain inference
    return model, eps
