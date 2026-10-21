"""DP summary statistics of PTB-XL under ONE declared budget.

The public release (artifacts/reports/dp_summary_statistics.json) contains
only noisy values plus the metadata needed to audit them, and the ledger
showing that the per-query budgets add up to the declared total. Exact
values are never written next to the release.

The error diagnostic is separate and internal: it repeats the mean-HR
release thousands of times (each repetition would be a new release, so it
is an analysis of the mechanism, not something to publish) and writes only
RMSEs, which shows observed error agreeing with the noise formula.
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import artifact_dir, load_config, resolve
from ..data.features import FEATURE_BOUNDS, FEATURE_NAMES
from ..privacy.accounting import BudgetLedger
from ..privacy.gaussian import gaussian_sigma
from ..privacy.laplace import laplace_scale
from ..privacy.statistics import dp_count, dp_histogram, dp_mean


def _feature(X: np.ndarray, name: str) -> np.ndarray:
    v = X[:, FEATURE_NAMES.index(name)]
    # A record whose feature could not be measured is dropped; the rule looks
    # at that record alone, so the add/remove sensitivity argument holds.
    return v[~np.isnan(v)]


def release_summary_statistics() -> dict:
    cfg = load_config("privacy_mechanisms")["summary_statistics"]
    X = np.load(resolve(load_config("data")["processed_dir"]) / "features.npz")["X"]
    rng = np.random.default_rng(cfg["seed"])
    ledger = BudgetLedger(cfg["total_epsilon"], cfg["total_delta"])
    releases = []
    for q in cfg["queries"]:
        eps = cfg["total_epsilon"] * q["epsilon_share"]
        delta = cfg["total_delta"] * q.get("delta_share", 0.0)
        if q["kind"] == "count":
            r = dp_count(q["name"], len(X), ledger, eps, rng, q["mechanism"], delta)
        elif q["kind"] == "mean":
            lo, hi = FEATURE_BOUNDS[q["feature"]]
            r = dp_mean(q["name"], _feature(X, q["feature"]), lo, hi, ledger, eps, rng, q["mechanism"], delta)
        elif q["kind"] == "histogram":
            r = dp_histogram(q["name"], _feature(X, q["feature"]), q["bin_edges"], ledger, eps, rng)
        else:
            raise ValueError(f"unknown query kind {q['kind']!r}")
        releases.append(r.to_dict())
    out = {"population": cfg["population"], "adjacency": "add/remove one ECG record",
           "budget": ledger.to_dict(), "releases": releases,
           "note": "noisy values only; clipping bounds are fixed physiological ranges chosen without the data"}
    (artifact_dir("reports") / "dp_summary_statistics.json").write_text(json.dumps(out, indent=2))
    return out


def error_diagnostic() -> pd.DataFrame:
    cfg = load_config("privacy_mechanisms")["summary_statistics"]["error_diagnostic"]
    X = np.load(resolve(load_config("data")["processed_dir"]) / "features.npz")["X"]
    x = _feature(X, cfg["feature"])
    lo, hi = FEATURE_BOUNDS[cfg["feature"]]
    c, n = (lo + hi) / 2, len(x)
    true = float(np.clip(x, lo, hi).mean())
    rng = np.random.default_rng(0)
    rows = []
    for mech in ("laplace", "gaussian"):
        delta = cfg["delta"] if mech == "gaussian" else 0.0
        for eps in cfg["epsilons"]:
            est = np.array([dp_mean("diag", x, lo, hi, BudgetLedger(eps, max(delta, 0.0)), eps, rng,
                                    mech, delta).value for _ in range(cfg["repeats"])])
            # First-order theory: (η − (mean − c)ν)/n, so the count noise ν is
            # magnified by (mean − c). Uses the exact mean: analysis, not a release.
            if mech == "laplace":
                var_s, var_n = 2 * laplace_scale((hi - lo) / 2, eps / 2) ** 2, 2 * laplace_scale(1, eps / 2) ** 2
            else:
                var_s = gaussian_sigma((hi - lo) / 2, eps / 2, delta / 2) ** 2
                var_n = gaussian_sigma(1.0, eps / 2, delta / 2) ** 2
            rows.append({"mechanism": mech, "epsilon": eps,
                         "observed_rmse": float(np.sqrt(np.mean((est - true) ** 2))),
                         "theory_rmse": math.sqrt(var_s + (true - c) ** 2 * var_n) / n})
    df = pd.DataFrame(rows)
    df.to_csv(artifact_dir("metrics") / "dp_mean_error_diagnostic.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for mech, colour in [("laplace", "#2a6fdb"), ("gaussian", "#d1495b")]:
        d = df[df.mechanism == mech]
        ax.plot(d.epsilon, d.theory_rmse, "-", color=colour, label=f"{mech}: theory")
        ax.plot(d.epsilon, d.observed_rmse, "o", color=colour, label=f"{mech}: observed ({cfg['repeats']} runs)")
    ax.set(xscale="log", yscale="log", xlabel="ε for the mean (half count, half sum)",
           ylabel="RMSE of DP mean heart rate (bpm)",
           title=f"DP mean heart rate, PTB-XL (n={n}); Gaussian δ={cfg['delta']:g}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(artifact_dir("figures") / "dp_mean_error.png", dpi=150)
    plt.close(fig)
    return df
