"""Gradio dashboard: shows saved results only; pending when absent.
No uploads are accepted, so no user ECG is ever stored or sent anywhere."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from ..config import artifact_dir

INTRO = """
# Differentially private ECG classification (research demo)

**Task.** Classify 10-second 12-lead ECGs from PTB-XL as normal (NORM) or myocardial infarction (MI).

**Differential privacy.** A model is (ε, δ)-DP if adding or removing any single training ECG changes the
probability of any outcome by at most a factor e^ε (plus a small δ). Smaller ε = stronger privacy.

**DP-SGD.** Ordinary training averages gradients. DP-SGD clips each example's gradient to norm C,
adds Gaussian noise, and samples batches randomly; Opacus' accountant converts this into ε for a given δ.

**Limits.** The guarantee is per ECG record. Some patients have several records, so it is not a
patient-level guarantee. Membership-inference results are empirical, not proofs.

**This is a research demonstration, not a medical device. Scores are not diagnoses.**
"""


def _table() -> pd.DataFrame | str:
    path = artifact_dir("metrics") / "runs_table.csv"
    if not path.exists():
        return "Pending: no experiment results have been generated yet (run `dp-ecg report`)."
    df = pd.read_csv(path)
    cols = ["run_id", "requested_epsilon", "achieved_epsilon", "test_auroc", "test_balanced_accuracy", "test_f1"]
    return df[cols].round(4)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DP-ECG") as demo:
        gr.Markdown(INTRO)
        table = _table()
        if isinstance(table, str):
            gr.Markdown(table)
        else:
            gr.Dataframe(table, label="Test-set results (generated from saved metrics)")
        for name, caption in [("privacy_utility.png", "Utility vs achieved ε"),
                              ("mia_vs_epsilon.png", "Membership-inference AUROC vs ε"),
                              ("mia_roc.png", "Attack ROC curves"),
                              ("dp_mean_error.png", "DP mean: observed vs theoretical error")]:
            path = artifact_dir("figures") / name
            if path.exists():
                gr.Image(str(path), label=caption)
            else:
                gr.Markdown(f"*{caption}: pending.*")
    return demo
