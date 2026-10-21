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


def privacy_utility_plot(runs: pd.DataFrame) -> None:
    dp = runs[runs.kind == "dp_sgd"].sort_values("achieved_epsilon")
    base = runs[runs.kind == "baseline"]
    if dp.empty or base.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, m, label in zip(axes, ["test_auroc", "test_balanced_accuracy", "test_f1"],
                            ["AUROC", "balanced accuracy", "F1 (MI = positive)"]):
        ax.plot(dp.achieved_epsilon, dp[m], "o-", color=BLUE, label="DP-SGD CNN")
        ax.axhline(base[m].mean(), color=GREY, ls="--", label="same CNN, non-private (no ε)")
        ax.set(xscale="log", xlabel=f"achieved ε (δ = {dp.delta.iloc[0]:g})", ylabel=f"test {label}")
        ax.set_xticks(dp.achieved_epsilon.round(2), [f"{v:.2f}" for v in dp.achieved_epsilon])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Privacy–utility trade-off: PTB-XL NORM vs MI, test fold 10 (one seed per ε)")
    fig.tight_layout()
    fig.savefig(artifact_dir("figures") / "privacy_utility.png", dpi=150)
    plt.close(fig)


def membership_plots() -> pd.DataFrame | None:
    path = artifact_dir("metrics") / "membership_inference.json"
    if not path.exists():
        return None
    mia = json.loads(path.read_text())
    rows = [{"run_id": rid, "kind": r["kind"], "achieved_epsilon": r["achieved_epsilon"],
             **{f"{a}_auroc": v["auroc"] for a, v in r["attacks"].items()},
             "loss_auroc_ci_low": r["attacks"]["loss"]["auroc_ci95"][0],
             "loss_auroc_ci_high": r["attacks"]["loss"]["auroc_ci95"][1],
             "loss_tpr_at_fpr_0.01": r["attacks"]["loss"]["tpr_at_fpr_0.01"],
             "loss_balanced_accuracy": r["attacks"]["loss"]["balanced_accuracy"]}
            for rid, r in mia["results"].items()]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for rid, roc in mia["roc_loss_attack"].items():
        ax.plot(roc["fpr"], roc["tpr"], color=GREY if rid.startswith("baseline") else None, lw=1.5,
                label=rid)
    ax.plot([0, 1], [0, 1], ":", color="#999999")
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title="Loss-threshold membership attack (evaluation set)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(artifact_dir("figures") / "mia_roc.png", dpi=150)
    plt.close(fig)

    dp = df[df.kind == "dp_sgd"].sort_values("achieved_epsilon")
    base = df[df.kind == "baseline"]
    if not dp.empty and not base.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.errorbar(dp.achieved_epsilon, dp.loss_auroc,
                    yerr=[dp.loss_auroc - dp.loss_auroc_ci_low, dp.loss_auroc_ci_high - dp.loss_auroc],
                    fmt="o-", color=BLUE, capsize=3, label="DP-SGD CNN (95% bootstrap CI)")
        b = base.iloc[0]
        ax.axhspan(b.loss_auroc_ci_low, b.loss_auroc_ci_high, color=GREY, alpha=0.15)
        ax.axhline(b.loss_auroc, color=GREY, ls="--", label="non-private CNN (no ε)")
        ax.axhline(0.5, color=RED, ls=":", label="chance")
        ax.set(xscale="log", xlabel="achieved ε", ylabel="attack AUROC (loss score)",
               title="Membership inference vs privacy budget")
        ax.legend(fontsize=8)
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


def write_summary(runs: pd.DataFrame, mia: pd.DataFrame | None) -> str:
    cols = ["run_id", "requested_epsilon", "achieved_epsilon", "noise_multiplier", "steps", "best_epoch",
            "test_accuracy", "test_balanced_accuracy", "test_precision", "test_recall", "test_f1",
            "test_auroc", "threshold"]
    md = ["# Experiment summary (generated from artifacts/metrics)", "",
          "Task: PTB-XL NORM vs MI, record-level DP, test fold 10. One seed per configuration.", "",
          _md_table(runs[cols]), ""]
    if mia is not None:
        md += ["## Membership inference (evaluation set: training members vs test-fold non-members)", "",
               _md_table(mia.drop(columns=["kind"])), ""]
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
    return write_summary(runs, membership_plots())
