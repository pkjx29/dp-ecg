"""Black-box membership-inference attack (threshold attacks, Yeom et al. 2018).

Threat model: the attacker has a candidate ECG record, knows its true label,
and can query the model for its predicted probability. They guess "member"
when the model is unusually confident/correct on the record. Three scores
(higher = more likely a member):
  * loss            : −BCE(true label, p)       (uses the true label)
  * max_probability : max(p, 1 − p)             (label-free)
  * entropy         : −H(p)                     (label-free)

Protocol (no test-fold data sets a threshold):
  * calibration set: members = random half A of the training records,
    non-members = validation fold; the threshold maximising balanced
    accuracy here is fixed.
  * evaluation set:  members = the disjoint half B of the training records,
    subsampled to the size of the test fold; non-members = test fold.
Members and non-members are balanced in each set. Folds are patient-disjoint,
so no patient is both a member and a non-member.

An attack AUROC near 0.5 means the attack cannot tell members apart; it is
EMPIRICAL evidence, not a privacy guarantee, and a weak attack can miss
leakage a stronger attack (e.g. LiRA with shadow models) would find.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

EPS = 1e-7


def attack_scores(p: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return {"loss": -loss, "max_probability": np.maximum(p, 1 - p), "entropy": -entropy}


def split_members(train_idx: np.ndarray, n_cal: int, n_eval: int, rng: np.random.Generator):
    """Two disjoint random member sets of the requested sizes."""
    perm = rng.permutation(train_idx)
    if n_cal + n_eval > len(perm):
        raise ValueError("not enough training records for disjoint member sets")
    return perm[:n_cal], perm[n_cal:n_cal + n_eval]


def tpr_at_fpr(is_member: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(is_member, score)
    return float(np.interp(target_fpr, fpr, tpr))


def evaluate_attack(cal_member: np.ndarray, cal_score: np.ndarray, ev_member: np.ndarray,
                    ev_score: np.ndarray, fpr_targets, n_boot: int, rng: np.random.Generator) -> dict:
    fpr, tpr, thr = roc_curve(cal_member, cal_score)
    threshold = float(thr[int(np.argmax(tpr - fpr))])
    guess = (ev_score >= threshold).astype(int)
    auc = float(roc_auc_score(ev_member, ev_score))
    boots = []
    n = len(ev_member)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(ev_member[i])) == 2:
            boots.append(roc_auc_score(ev_member[i], ev_score[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "auroc": auc, "auroc_ci95": [float(lo), float(hi)],
        "threshold_from_calibration": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(ev_member, guess)),
        "precision": float(precision_score(ev_member, guess, zero_division=0)),
        "recall": float(recall_score(ev_member, guess, zero_division=0)),
        **{f"tpr_at_fpr_{t:g}": tpr_at_fpr(ev_member, ev_score, t) for t in fpr_targets},
        "n_members": int(ev_member.sum()), "n_non_members": int((1 - ev_member).sum()),
    }
