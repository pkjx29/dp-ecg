"""Likelihood-ratio membership attack (LiRA, Carlini et al., S&P 2022), online variant.

Setup, per training configuration (non-private, or DP-SGD at a target ε):
  * pool = the 9,286 training-fold records.
  * K models are trained with the SAME recipe, each on an independent random
    half of the pool (every record is IN for about K/2 models, OUT for the rest).
  * For every model and every pool record, store the logit-scaled confidence on
    the true label, φ = log(p_y / (1 − p_y)). Carlini et al. show φ is close
    to Gaussian across models, unlike the raw loss.
Attack on model i and record x: using only the OTHER K−1 models, fit a
Gaussian to φ(x) over models where x was IN and one where x was OUT, and score
    Λ(x) = log N(φ_i(x); μ_in, σ_in²) − log N(φ_i(x); μ_out, σ_out²).
With few shadows, per-example variances are noisy, so (as the paper recommends
for small K) one global variance is used for IN and one for OUT. Every model
takes a turn as the target, and the resulting scores are pooled.

Ground truth is exact: model i's mask says which records it trained on, so
members and non-members come from the same distribution by construction.

Reported: AUROC and TPR at low FPR (0.1%, 1%). The low-FPR regime is where
LiRA is much stronger than average-case threshold attacks. For an
(ε, δ)-DP training run, any attack must satisfy TPR ≤ e^ε·FPR + δ, which
gives a hard ceiling to compare against.

Note: these models see half the training data, so the DP noise multiplier
for a given ε differs from the full-data runs. The attack audits the training
RECIPE at that ε; it does not attack the exact checkpoints in runs/.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-7


def logit_confidence(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """φ = log(p_y / (1 − p_y)) for a sigmoid output p and binary label y."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    p_true = np.where(np.asarray(y) == 1, p, 1 - p)
    return np.log(p_true) - np.log(1 - p_true)


def random_masks(n_models: int, n_records: int, rng: np.random.Generator) -> np.ndarray:
    """(K, N) boolean IN-masks, each record included independently with prob. 1/2."""
    return rng.random((n_models, n_records)) < 0.5


def lira_scores(phi: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-model-out online LiRA.

    phi, masks: (K, N). Returns (scores, is_member), each flattened over the
    K targets × N records. Records with no IN or no OUT shadow among the other
    models are skipped.
    """
    k, n = phi.shape
    scores, labels = [], []
    for i in range(k):
        others = np.arange(k) != i
        ph, mk = phi[others], masks[others]
        n_in, n_out = mk.sum(0), (~mk).sum(0)
        ok = (n_in > 0) & (n_out > 0)
        mu_in = np.where(mk, ph, 0).sum(0) / np.maximum(n_in, 1)
        mu_out = np.where(~mk, ph, 0).sum(0) / np.maximum(n_out, 1)
        # Global (shared) variances: more stable than per-record ones with K ≈ 16.
        sd_in = math.sqrt(np.mean(((ph - mu_in) ** 2)[mk & ok[None]]) + 1e-12)
        sd_out = math.sqrt(np.mean(((ph - mu_out) ** 2)[~mk & ok[None]]) + 1e-12)
        s = norm.logpdf(phi[i], mu_in, sd_in) - norm.logpdf(phi[i], mu_out, sd_out)
        scores.append(s[ok])
        labels.append(masks[i][ok].astype(int))
    return np.concatenate(scores), np.concatenate(labels)


def summarise(scores: np.ndarray, is_member: np.ndarray, epsilon: float | None, delta: float | None,
              fprs=(0.001, 0.01)) -> dict:
    fpr, tpr, _ = roc_curve(is_member, scores)
    out = {"auroc": float(roc_auc_score(is_member, scores)), "n_scores": int(len(scores)),
           "member_share": float(is_member.mean())}
    for f in fprs:
        out[f"tpr_at_fpr_{f:g}"] = float(np.interp(f, fpr, tpr))
        if epsilon is not None:
            # Hypothesis-testing bound for (ε, δ)-DP: TPR ≤ e^ε·FPR + δ.
            out[f"dp_bound_tpr_at_fpr_{f:g}"] = float(min(1.0, math.exp(epsilon) * f + delta))
    step = max(1, len(fpr) // 400)
    out["roc"] = {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}
    return out
