"""Shared training loop for the non-private baseline and DP-SGD.

Both paths use the same model, data and selection rule. With `privacy` set,
Opacus wraps the model, optimiser and data loader:
  * Poisson sampling: each record joins a batch independently with
    probability q = batch_size / n (the accountant assumes exactly this);
  * per-example gradients clipped to L2 norm <= max_grad_norm (C);
  * Gaussian noise N(0, σ²C²) added to the summed gradient, σ chosen by
    Opacus so the RDP accountant reaches the target ε after all epochs.

Model selection: after each epoch the validation AUROC is computed and the
best epoch's weights are kept. Validation records are not training records,
so this choice does not consume the training set's privacy budget. The
released model is post-processing of the whole training run, so the
reported guarantee is the ε after the LAST epoch.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class PrivacySettings:
    target_epsilon: float
    delta: float
    max_grad_norm: float
    accountant: str = "rdp"
    poisson_sampling: bool = True
    secure_mode: bool = False


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    history: list[dict] = field(default_factory=list)
    privacy: dict | None = None
    seconds: float = 0.0


@torch.no_grad()
def predict_scores(model: nn.Module, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    """Sigmoid probabilities for a (N, leads, samples) tensor, in batches."""
    model.eval()
    out = [torch.sigmoid(model(x[i:i + batch_size].float())) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy() if out else np.zeros(0)


def _optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=0.9,
                               weight_decay=cfg["weight_decay"])
    raise ValueError(f"unknown optimizer {cfg['optimizer']!r}")


def train_model(model: nn.Module, train_ds: Dataset, x_val: torch.Tensor, y_val: np.ndarray,
                cfg: dict, seed: int, privacy: PrivacySettings | None = None) -> TrainResult:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen)
    optimizer = _optimizer(model, cfg)
    engine = None
    t0 = time.time()

    if privacy is not None:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator

        errors = ModuleValidator.validate(model, strict=False)
        if errors:  # fail loudly instead of silently "fixing" the architecture
            raise ValueError(f"model is not Opacus-compatible: {errors}")
        engine = PrivacyEngine(accountant=privacy.accountant, secure_mode=privacy.secure_mode)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Secure RNG turned off")
            model, optimizer, loader = engine.make_private_with_epsilon(
                module=model, optimizer=optimizer, data_loader=loader,
                target_epsilon=privacy.target_epsilon, target_delta=privacy.delta,
                epochs=cfg["epochs"], max_grad_norm=privacy.max_grad_norm,
                poisson_sampling=privacy.poisson_sampling)

    loss_fn = nn.BCEWithLogitsLoss()
    history, best_auc, best_state, best_epoch, steps = [], -1.0, None, -1, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            if len(xb) == 0:  # Poisson sampling can draw an empty batch
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(xb.float()), yb)
            loss.backward()
            optimizer.step()
            steps += 1
            total += loss.item() * len(xb)
            n += len(xb)
        val_auc = float(roc_auc_score(y_val, predict_scores(model, x_val)))
        row = {"epoch": epoch, "train_loss": total / max(n, 1), "val_auroc": val_auc}
        if engine is not None:
            row["epsilon_so_far"] = float(engine.get_epsilon(privacy.delta))
        history.append(row)
        log.info("epoch %d %s", epoch, {k: round(v, 4) for k, v in row.items() if k != "epoch"})
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            inner = model._module if engine is not None else model
            best_state = copy.deepcopy(inner.state_dict())

    final = model._module if engine is not None else model
    final.load_state_dict(best_state)
    info = None
    if engine is not None:
        info = {
            "requested_epsilon": privacy.target_epsilon,
            "achieved_epsilon": float(engine.get_epsilon(privacy.delta)),
            "delta": privacy.delta,
            "noise_multiplier": float(optimizer.noise_multiplier),
            "max_grad_norm": privacy.max_grad_norm,
            "sample_rate": float(loader.sample_rate),
            "expected_batch_size": cfg["batch_size"],
            "poisson_sampling": privacy.poisson_sampling,
            "steps": steps,
            "epochs": cfg["epochs"],
            "accountant": privacy.accountant,
            "secure_mode": privacy.secure_mode,
        }
    return TrainResult(final, best_epoch, history, info, time.time() - t0)
