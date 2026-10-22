"""Figures and Markdown tables built ONLY from saved artifacts.

The non-private baseline has no ε, so on privacy plots it is a horizontal
reference line, never a point at a made-up ε.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter, NullLocator

from ..config import artifact_dir, load_config
from ..privacy.verification import verify_all

BLUE, GREY, RED = "#2a6fdb", "#444444", "#d1495b"


def _md_table(df: pd.DataFrame) -> str:
    """Markdown table without extra dependencies; floats to 4 decimals."""
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.4f}"
        return "" if v is None else str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _eps_ticks(ax, eps) -> None:
    """Label exactly the achieved ε values on a log axis, no minor-tick clutter."""
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(list(eps), [f"{v:.2f}" for v in eps])


def load_runs() -> pd.DataFrame:
    rows = []
    for p in sorted((artifact_dir("metrics") / "runs").glob("*.json")):
        if p.name.endswith(".error.json"):
            continue
        r = json.loads(p.read_text())
        priv = r["privacy"] or {}
        rows.append({"run_id": r["run_id"], "kind": r["kind"], "status": r["status"], "seed": r["seed"],
                     "requested_epsilon": priv.get("requested_epsilon"),
                     "achieved_epsilon": priv.get("achieved_epsilon"), "delta": priv.get("delta"),
                     "noise_multiplier": priv.get("noise_multiplier"), "steps": priv.get("steps"),
                     "sample_rate": priv.get("sample_rate"), "best_epoch": r["best_epoch"],
                     **{f"test_{k}": r["test"][k] for k in
                        ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc")},
                     "threshold": r["test"]["threshold"]})
    return pd.DataFrame(rows)


METRICS = ["test_accuracy", "test_balanced_accuracy", "test_precision", "test_recall", "test_f1", "test_auroc"]


def _config_label(kind: str, requested_epsilon) -> str:
    return "non-private" if kind == "baseline" else f"DP-SGD, target ε={requested_epsilon:g}"


def aggregate(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean and sample sd (ddof=1) over seeds for each configuration."""
    df = df.assign(config=[_config_label(k, e) for k, e in zip(df.kind, df.requested_epsilon)])
    g = df.groupby(["config", "kind"], sort=False, dropna=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    out["n_seeds"] = g.size()
    out = out.reset_index()
    order = out.kind.eq("dp_sgd").astype(int)  # baseline first, then by ε descending
    eps = out.config.str.extract(r"ε=([0-9.]+)")[0].astype(float).fillna(1e9)
    return out.assign(_o=order, _e=-eps).sort_values(["_o", "_e"]).drop(columns=["_o", "_e"])


def privacy_utility_plot(runs: pd.DataFrame) -> None:
    dp_runs = runs[runs.kind == "dp_sgd"]
    base = runs[runs.kind == "baseline"]
    if dp_runs.empty or base.empty:
        return
    agg = aggregate(dp_runs, ["achieved_epsilon", *METRICS]).sort_values("achieved_epsilon_mean")
    n = int(agg.n_seeds.min())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, m, label in zip(axes, ["test_auroc", "test_balanced_accuracy", "test_f1"],
                            ["AUROC", "balanced accuracy", "F1 (MI = positive)"]):
        ax.scatter(dp_runs.achieved_epsilon, dp_runs[m], s=12, color=BLUE, alpha=0.35, label="single seed")
        ax.errorbar(agg.achieved_epsilon_mean, agg[f"{m}_mean"], yerr=agg[f"{m}_std"], fmt="o-", color=BLUE,
                    capsize=3, label=f"DP-SGD CNN, mean ± sd ({n} seeds)")
        mu, sd = base[m].mean(), base[m].std()
        ax.axhspan(mu - sd, mu + sd, color=GREY, alpha=0.15)
        ax.axhline(mu, color=GREY, ls="--", label=f"same CNN, non-private, mean ± sd ({len(base)} seeds)")
        ax.set(xscale="log", xlabel=f"achieved ε (δ = {dp_runs.delta.iloc[0]:g})", ylabel=f"test {label}")
        _eps_ticks(ax, agg.achieved_epsilon_mean)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Privacy–utility trade-off: PTB-XL NORM vs MI, test fold 10")
    fig.tight_layout()
    fig.savefig(artifact_dir("figures") / "privacy_utility.png", dpi=150)
    plt.close(fig)


def membership_plots(runs: pd.DataFrame) -> pd.DataFrame | None:
    path = artifact_dir("metrics") / "membership_inference.json"
    if not path.exists():
        return None
    mia = json.loads(path.read_text())
    meta = runs.set_index("run_id")
    rows = [{"run_id": rid, "kind": r["kind"], "seed": int(meta.loc[rid, "seed"]),
             "requested_epsilon": meta.loc[rid, "requested_epsilon"], "achieved_epsilon": r["achieved_epsilon"],
             **{f"{a}_auroc": v["auroc"] for a, v in r["attacks"].items()},
             "loss_auroc_ci_low": r["attacks"]["loss"]["auroc_ci95"][0],
             "loss_auroc_ci_high": r["attacks"]["loss"]["auroc_ci95"][1],
             "loss_tpr_at_fpr_0.01": r["attacks"]["loss"]["tpr_at_fpr_0.01"],
             "loss_balanced_accuracy": r["attacks"]["loss"]["balanced_accuracy"]}
            for rid, r in mia["results"].items() if rid in meta.index]
    df = pd.DataFrame(rows)

    # ROC curves: seed 0 of each configuration, to keep the figure readable.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for rid, roc in mia["roc_loss_attack"].items():
        if not rid.endswith("_seed0"):
            continue
        ax.plot(roc["fpr"], roc["tpr"], color=GREY if rid.startswith("baseline") else None, lw=1.5, label=rid)
    ax.plot([0, 1], [0, 1], ":", color="#999999")
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title="Loss-threshold membership attack (seed 0 models)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(artifact_dir("figures") / "mia_roc.png", dpi=150)
    plt.close(fig)

    dp, base = df[df.kind == "dp_sgd"], df[df.kind == "baseline"]
    if not dp.empty and not base.empty:
        agg = aggregate(dp, ["achieved_epsilon", "loss_auroc"]).sort_values("achieved_epsilon_mean")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(dp.achieved_epsilon, dp.loss_auroc, s=12, color=BLUE, alpha=0.35, label="single seed")
        ax.errorbar(agg.achieved_epsilon_mean, agg.loss_auroc_mean, yerr=agg.loss_auroc_std, fmt="o-",
                    color=BLUE, capsize=3, label=f"DP-SGD CNN, mean ± sd ({int(agg.n_seeds.min())} seeds)")
        mu, sd = base.loss_auroc.mean(), base.loss_auroc.std()
        ax.axhspan(mu - sd, mu + sd, color=GREY, alpha=0.15)
        ax.axhline(mu, color=GREY, ls="--", label=f"non-private CNN, mean ± sd ({len(base)} seeds)")
        ax.axhline(0.5, color=RED, ls=":", label="chance")
        delta = load_config("dp_sgd")["delta"]
        ax.set(xscale="log", xlabel=f"achieved ε (δ = {delta:g})", ylabel="attack AUROC (loss score)",
               title="Membership inference vs privacy budget")
        _eps_ticks(ax, agg.achieved_epsilon_mean)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(artifact_dir("figures") / "mia_vs_epsilon.png", dpi=150)
        plt.close(fig)
    return df


def mechanism_verification_report() -> dict:
    cfg = load_config("privacy_mechanisms")["verification"]
    report = verify_all(cfg["epsilon"], cfg["delta"])
    (artifact_dir("reports") / "mechanism_verification.json").write_text(json.dumps(report, indent=2))
    lines = [f"# Mechanism verification (ε = {cfg['epsilon']}, δ = {cfg['delta']})", "",
             f"Universe: all multisets of size ≤ {report['max_size']} over {report['domain']}; "
             f"{report['n_pairs']} add/remove neighbour pairs.", "",
             "| check | passed |", "|---|---|"]
    lines += [f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} |" for c in report["checks"]]
    lines += ["", "These checks test the implementation on a toy universe. The DP guarantee for "
              "arbitrary data comes from the mechanism theorems and sensitivity proofs (KNOWLEDGE.md)."]
    (artifact_dir("reports") / "mechanism_verification.md").write_text("\n".join(lines) + "\n")
    return report


def _mean_sd(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """'mean ± sd' strings per configuration for the Markdown summary."""
    agg = aggregate(df, cols)
    out = agg[["config", "n_seeds"]].copy()
    for c in cols:
        out[c] = ["–" if pd.isna(m) else f"{m:.4f} ± {s:.4f}" if pd.notna(s) else f"{m:.4f}"
                  for m, s in zip(agg[f"{c}_mean"], agg[f"{c}_std"])]
    return out


def write_summary(runs: pd.DataFrame, mia: pd.DataFrame | None) -> str:
    per_run = ["run_id", "seed", "requested_epsilon", "achieved_epsilon", "noise_multiplier", "steps", "best_epoch",
               *METRICS, "threshold"]
    md = ["# Experiment summary (generated from artifacts/metrics)", "",
          "Task: PTB-XL NORM vs MI, record-level DP, test fold 10 (n = 1,168). "
          "Each seed is a full training run; sd is the sample standard deviation across seeds.", "",
          "## Classification, mean ± sd over seeds", "",
          _md_table(_mean_sd(runs, ["achieved_epsilon", "test_auroc", "test_balanced_accuracy", "test_f1",
                                    "test_precision", "test_recall"])), "",
          "## Classification, every run", "",
          _md_table(runs.sort_values(["kind", "requested_epsilon", "seed"])[per_run]), ""]
    if mia is not None:
        md += ["## Membership inference, mean ± sd over seeds (loss attack unless stated)", "",
               _md_table(_mean_sd(mia, ["loss_auroc", "max_probability_auroc", "loss_tpr_at_fpr_0.01",
                                        "loss_balanced_accuracy"])), "",
               "## Membership inference, every run (95% bootstrap CI of the loss-attack AUROC)", "",
               _md_table(mia.sort_values(["kind", "requested_epsilon", "seed"]).drop(columns=["kind"])), ""]
    text = "\n".join(md)
    (artifact_dir("reports") / "results_summary.md").write_text(text)
    return text


def generate_all() -> str:
    mechanism_verification_report()
    runs = load_runs()
    if runs.empty:
        return "no completed runs yet"
    runs.to_csv(artifact_dir("metrics") / "runs_table.csv", index=False)
    privacy_utility_plot(runs)
    return write_summary(runs, membership_plots(runs))
