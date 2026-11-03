"""LiRA scoring on simulated confidences with known ground truth."""

import numpy as np

from dp_ecg.attacks.lira import lira_scores, logit_confidence, random_masks, summarise


def test_logit_confidence_symmetry():
    assert np.allclose(logit_confidence(np.array([0.9]), np.array([1])),
                       logit_confidence(np.array([0.1]), np.array([0])))
    assert logit_confidence(np.array([0.5]), np.array([1]))[0] == 0.0


def _simulate(gap, k=16, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    masks = random_masks(k, n, rng)
    base = rng.normal(0, 1, n)                      # per-record difficulty
    phi = base + gap * masks + rng.normal(0, 0.3, (k, n))
    return phi, masks


def test_lira_detects_leakage_and_not_absent_leakage():
    leak = summarise(*lira_scores(*_simulate(gap=0.5)), epsilon=None, delta=None)
    none = summarise(*lira_scores(*_simulate(gap=0.0)), epsilon=None, delta=None)
    assert leak["auroc"] > 0.75
    assert abs(none["auroc"] - 0.5) < 0.02


def test_per_record_calibration_beats_global_threshold():
    """LiRA's point: comparing each record with its own IN/OUT distribution
    beats a single global threshold on the confidence."""
    from sklearn.metrics import roc_auc_score
    phi, masks = _simulate(gap=0.3)
    scores, member = lira_scores(phi, masks)
    lira_auc = roc_auc_score(member, scores)
    global_auc = roc_auc_score(masks.ravel(), phi.ravel())
    assert lira_auc > global_auc + 0.05


def test_dp_bound_reported():
    s = summarise(*lira_scores(*_simulate(gap=0.0)), epsilon=1.0, delta=1e-5)
    assert np.isclose(s["dp_bound_tpr_at_fpr_0.001"], np.e * 0.001 + 1e-5)
