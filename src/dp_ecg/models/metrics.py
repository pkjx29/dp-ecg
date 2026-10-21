"""Classification metrics and validation-based threshold choice."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (Youden's J) on the given data.
    Called on the VALIDATION split only; the test split never sets it."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    best = int(np.argmax(tpr - fpr))
    return float(min(thr[best], 1.0))


def classification_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_prevalence": float(y_true.mean()),
        "n": int(len(y_true)),
    }
