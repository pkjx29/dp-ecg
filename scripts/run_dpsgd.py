"""Steps 1 and 3: non-private baselines, then DP-SGD across a grid of ε.

Privacy accounting for a DP model (all on the training folds 1-8):
  * feature standardisation: DP mean/std via the from-scratch Gaussian
    mechanism (mechanisms.dp_standardizer), cost (ε_pre, δ_pre)
  * DP-SGD training with Opacus, RDP accountant, cost (ε_sgd, δ_sgd)
  * total by basic sequential composition: (ε_pre + ε_sgd, δ_pre + δ_sgd)
The test fold is used only for evaluation. Clipping bounds come from
physiology (features.FEATURE_BOUNDS), not from the data.

Outputs: results/baselines.csv, results/dpsgd_results.csv,
         results/privacy_utility.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from dpecg import mechanisms as M
from dpecg.data import TEST_FOLDS, TRAIN_FOLDS
from dpecg.features import bounds_arrays, clip_and_impute
from dpecg.train import TrainConfig, evaluate, train

EPS_SGD_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DELTA_SGD = 1e-5     # n_train ≈ 17k, so δ is well below 1/n ≈ 6e-5
DELTA_PRE = 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eps-pre", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true", help="1 seed, 3 ε values (smoke test)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)

    d = np.load(args.features, allow_pickle=False)
    keep = d["y"] >= 0  # drop the 411 records without a diagnostic label
    X, y, fold = clip_and_impute(d["X"][keep]), d["y"][keep].astype(np.float32), d["fold"][keep]
    tr, te = np.isin(fold, TRAIN_FOLDS), np.isin(fold, TEST_FOLDS)
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    print(f"train n={len(y_tr)}  test n={len(y_te)}  features={X.shape[1]}  "
          f"abnormal share train={y_tr.mean():.3f} test={y_te.mean():.3f}")

    def standardise(mean, std):
        return ((X_tr - mean) / std).astype(np.float32), ((X_te - mean) / std).astype(np.float32)

    # Exact (non-private) standardisation for the baselines.
    Xs_tr, Xs_te = standardise(X_tr.mean(0), X_tr.std(0) + 1e-8)
    # DP standardisation for the DP models (one release, reused for all runs).
    lo, hi = bounds_arrays()
    dp_mean, dp_std, pre_cost = M.dp_standardizer(X_tr, lo, hi, args.eps_pre, DELTA_PRE,
                                                  np.random.default_rng(12345))
    Xd_tr, Xd_te = standardise(dp_mean, dp_std)

    seeds = range(1 if args.quick else args.seeds)
    eps_grid = [0.5, 2.0, 8.0] if args.quick else EPS_SGD_GRID
    cfg = TrainConfig()

    # ------------------------------------------------------------ baselines
    base_rows = [{"model": "majority class", "seed": 0,
                  "accuracy": accuracy_score(y_te, np.full_like(y_te, round(y_tr.mean()))),
                  "auroc": 0.5}]
    lr = LogisticRegression(max_iter=2000).fit(Xs_tr, y_tr)
    base_rows.append({"model": "logistic regression (sklearn)", "seed": 0,
                      "accuracy": accuracy_score(y_te, lr.predict(Xs_te)),
                      "auroc": roc_auc_score(y_te, lr.predict_proba(Xs_te)[:, 1])})
    gb = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
    base_rows.append({"model": "gradient boosting (sklearn)", "seed": 0,
                      "accuracy": accuracy_score(y_te, gb.predict(X_te)),
                      "auroc": roc_auc_score(y_te, gb.predict_proba(X_te)[:, 1])})
    for s in seeds:
        cfg.seed = s
        m, _ = train(Xs_tr, y_tr, cfg)
        base_rows.append({"model": "MLP, non-private", "seed": s, **evaluate(m, Xs_te, y_te)})
        m, _ = train(Xd_tr, y_tr, cfg)
        base_rows.append({"model": "MLP, non-private, DP standardisation", "seed": s,
                          **evaluate(m, Xd_te, y_te)})
    base = pd.DataFrame(base_rows)
    base.to_csv(out / "baselines.csv", index=False)
    print(base.groupby("model", sort=False)[["accuracy", "auroc"]].mean().round(4).to_string())

    # ------------------------------------------------------------ DP-SGD
    rows = []
    for eps in eps_grid:
        for s in seeds:
            cfg.seed = s
            t0 = time.time()
            m, eps_spent = train(Xd_tr, y_tr, cfg, target_epsilon=eps, delta=DELTA_SGD)
            r = {"target_eps_sgd": eps, "eps_sgd": eps_spent, "eps_pre": pre_cost.epsilon,
                 "eps_total": eps_spent + pre_cost.epsilon,
                 "delta_total": DELTA_SGD + pre_cost.delta, "seed": s,
                 **evaluate(m, Xd_te, y_te), "seconds": time.time() - t0}
            rows.append(r)
            print(f"ε_sgd={eps:<5} seed={s}  ε_total={r['eps_total']:.2f}  "
                  f"acc={r['accuracy']:.4f}  auroc={r['auroc']:.4f}  ({r['seconds']:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "dpsgd_results.csv", index=False)

    # ------------------------------------------------------------ plot
    agg = res.groupby("target_eps_sgd").agg(
        eps_total=("eps_total", "mean"), acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
        auc=("auroc", "mean"), auc_sd=("auroc", "std")).fillna(0)
    ref = base.groupby("model", sort=False)[["accuracy", "auroc"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, sd, metric in [(axes[0], "acc", "acc_sd", "accuracy"), (axes[1], "auc", "auc_sd", "auroc")]:
        ax.errorbar(agg.eps_total, agg[col], yerr=agg[sd], fmt="o-", color="#2a6fdb",
                    capsize=3, label=f"DP-SGD MLP (mean ± sd, {len(seeds)} seeds)")
        ax.axhline(ref.loc["MLP, non-private", metric], color="#333333", ls="--",
                   label="same MLP, non-private")
        if metric == "accuracy":
            ax.axhline(ref.loc["majority class", metric], color="#999999", ls=":",
                       label="majority class")
        ax.set_xscale("log")
        ax.set_xlabel(f"total ε  (DP-SGD + {pre_cost.epsilon:g} for DP standardisation),  "
                      f"δ = {DELTA_SGD + pre_cost.delta:.1e}", fontsize=8)
        ax.set_ylabel(f"test {metric} (fold 10)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Privacy–utility trade-off: PTB-XL normal vs abnormal ECG, hand-crafted features")
    fig.tight_layout()
    fig.savefig(out / "privacy_utility.png", dpi=150)
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
