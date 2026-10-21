"""Step 2: DP summary statistics over all 21,799 PTB-XL records.

Releases (each row of the table reports its own ε and δ):
  * record counts per diagnostic group  (Laplace, sensitivity 1)
  * means of heart rate, QRS width, Sokolow-Lyon voltage and age, over all
    records and per group                (Laplace and Gaussian, NoisySum/NoisyCount)
  * heart-rate histogram                 (Laplace, sensitivity 1)

Then the mean-HR release is repeated many times per ε and the observed error
is compared with the error the noise formula predicts. Agreement between the
two is an end-to-end check that sensitivity and noise scale are implemented
correctly.

This is a stand-alone release, separate from model training. Publishing both
would cost the sum of the two budgets for records in the training folds.

Outputs: results/dp_summary_stats.csv, results/dp_stats_error.{csv,png},
         results/dp_hr_histogram.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dpecg import mechanisms as M
from dpecg.features import FEATURE_BOUNDS, FEATURE_NAMES

HR_LO, HR_HI = FEATURE_BOUNDS["hr_bpm"]
DELTA = 1e-6
REPEATS = 2000


def theory_rmse_mean(n: int, mu: float, eps: float, mechanism: str) -> float:
    """Predicted RMSE of the NoisySum/NoisyCount mean (first-order in 1/n).

    With c the public midpoint, S = sum(x - c), noise η on S and ν on n:
        c + (S + η)/(n + ν) − mean  ≈  (η − (mean − c)·ν) / n
    so  RMSE² ≈ (Var η + (mean − c)² · Var ν) / n².
    The count term is not negligible here: mean HR ≈ 74 bpm is 61 bpm from
    the midpoint 135, so the count noise is magnified 61×. (Using the true
    mean in a theory curve is fine; this plot is analysis, not a release.)
    """
    c = (HR_LO + HR_HI) / 2.0
    sens = (HR_HI - HR_LO) / 2.0
    var_count = 2.0 * M.laplace_scale(1.0, eps / 2) ** 2      # count is always Laplace
    if mechanism == "laplace":
        var_sum = 2.0 * M.laplace_scale(sens, eps / 2) ** 2
    else:
        var_sum = M.gaussian_sigma(sens, eps / 2, DELTA) ** 2
    return math.sqrt(var_sum + (mu - c) ** 2 * var_count) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    d = np.load(args.features, allow_pickle=False)
    X, y = d["X"], d["y"]
    col = {n: X[:, FEATURE_NAMES.index(n)] for n in ["hr_bpm", "qrs_width_s", "sokolow_mv", "age"]}
    print(f"{len(y)} records")

    # ---- one release per statistic at ε = 1, true value alongside
    rows = []
    eps = 1.0
    groups = [("all records", np.ones_like(y, bool)), ("normal", y == 0),
              ("abnormal", y == 1), ("no diagnostic label", y == -1)]
    for gname, gmask in groups:
        n_dp, cost = M.dp_count(int(gmask.sum()), eps, rng)
        rows.append({"group": gname, "statistic": "count", "true": int(gmask.sum()), "dp": n_dp,
                     "mechanism": "laplace", "epsilon": cost.epsilon, "delta": 0.0})
    for gname, gmask in groups[:3]:
        for feat, v in col.items():
            lo, hi = FEATURE_BOUNDS[feat]
            # A record whose feature could not be measured (NaN; 11 records for
            # heart rate) is left out. The filter looks at that record alone,
            # so the add/remove sensitivity argument still holds.
            vals = v[gmask & ~np.isnan(v)]
            for mech, dl in [("laplace", 0.0), ("gaussian", DELTA)]:
                m, cost = M.dp_mean(vals, lo, hi, eps, rng, mechanism=mech, delta=dl)
                rows.append({"group": gname, "statistic": f"mean {feat}",
                             "true": float(np.clip(vals, lo, hi).mean()), "dp": m,
                             "mechanism": mech, "epsilon": cost.epsilon, "delta": cost.delta})
    table = pd.DataFrame(rows)
    table["abs_error"] = (table["dp"] - table["true"]).abs()
    table.to_csv(out / "dp_summary_stats.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    hr_all = col["hr_bpm"][~np.isnan(col["hr_bpm"])]

    # ---- empirical vs theoretical error of the mean-HR release
    eps_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.8]
    n = len(hr_all)
    true = float(np.clip(hr_all, HR_LO, HR_HI).mean())
    err_rows = []
    for mech in ["laplace", "gaussian"]:
        for e in eps_grid:
            est = np.array([M.dp_mean(hr_all, HR_LO, HR_HI, e, rng, mechanism=mech,
                                      delta=DELTA if mech == "gaussian" else 0.0)[0]
                            for _ in range(REPEATS)])
            err_rows.append({"mechanism": mech, "epsilon": e,
                             "empirical_rmse": float(np.sqrt(np.mean((est - true) ** 2))),
                             "theory_rmse": theory_rmse_mean(n, true, e, mech)})
    err = pd.DataFrame(err_rows)
    err.to_csv(out / "dp_stats_error.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for mech, colour in [("laplace", "#2a6fdb"), ("gaussian", "#d1495b")]:
        e = err[err.mechanism == mech]
        ax.plot(e.epsilon, e.theory_rmse, "-", color=colour, label=f"{mech}: theory")
        ax.plot(e.epsilon, e.empirical_rmse, "o", color=colour, label=f"{mech}: observed ({REPEATS} runs)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ε  (total for the mean: half to sum, half to count)")
    ax.set_ylabel("RMSE of DP mean heart rate (bpm)")
    ax.set_title(f"DP mean heart rate, all PTB-XL records (n={n})\nGaussian δ={DELTA:g}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dp_stats_error.png", dpi=150)

    # ---- DP histogram
    edges = np.arange(30, 181, 10)
    true_h, _ = np.histogram(hr_all, edges)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    centres = (edges[:-1] + edges[1:]) / 2
    ax.bar(centres - 3, true_h, width=3, color="#999999", label="true")
    for e, colour in [(0.01, "#d1495b"), (0.1, "#2a6fdb")]:
        h, _ = M.dp_histogram(hr_all, edges, e, rng)
        ax.bar(centres + (3 if e == 0.1 else 0), h, width=3, color=colour, alpha=0.8, label=f"Laplace ε={e}")
    ax.set_yscale("log")
    ax.set_xlabel("heart rate (bpm)"); ax.set_ylabel("records")
    ax.set_title("Heart-rate histogram: true vs DP")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "dp_hr_histogram.png", dpi=150)
    print(err.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
