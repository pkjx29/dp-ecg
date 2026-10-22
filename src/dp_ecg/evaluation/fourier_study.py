"""Fourier-domain perturbation on PTB-XL: releasing average heartbeat templates.

Query, per class (NORM, MI): the mean over records of each record's 12-lead
median beat (12 leads x 70 samples = -250..+450 ms around R at 100 Hz).
Each record's template is clipped to L2 norm <= C before summing, so one
record moves the sum by at most C in L2 under add/remove adjacency. The mean
is NoisySum / NoisyCount with 10% of ε on the count (Laplace) and 90% on the
sum. NORM and MI records are disjoint, so the two class releases compose in
parallel: releasing both costs ε, not 2ε.

Three ways of noising the sum, all at the same ε:
  * FPA_k     Laplace on the first k DFT coefficients per lead     (ε-DP)
  * LPA       Laplace on all 840 time samples                      (ε-DP)
  * Gaussian  analytic Gaussian on all 840 time samples            ((ε, δ)-DP)

This is a mechanism STUDY: the error of each release is measured against the
exact mean template, over many repetitions. Choosing k by looking at that
error uses the data, so a real release would fix k in advance or pay for
choosing it.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import artifact_dir, load_config, resolve
from ..data.features import BASE, detect_r_peaks, median_beat
from ..data.metadata import LEADS
from ..logging_utils import get_logger
from ..privacy.fourier import (
    clip_rows_l2,
    fpa_gaussian_release,
    fpa_release,
    gaussian_time_release,
    lpa_release,
    reconstruct,
    truncate_dft,
)
from ..privacy.laplace import laplace_scale, sample_laplace

log = get_logger(__name__)

CLIP_NORM = 8.0          # chosen after inspecting 400 template norms (p90 = 7.4, p99 = 8.9); see docs
COUNT_SHARE = 0.1
DELTA = 1e-6
EPSILONS = [0.1, 0.3, 1.0, 3.0, 10.0]
KS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35]
REPEATS = 200
BLUE, RED, GREY, GREEN = "#2a6fdb", "#d1495b", "#444444", "#2e8b57"


def beat_templates(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 12, 70) median-beat templates with PR-segment baseline removed, plus
    a boolean mask of records where at least one beat was found."""
    out = np.zeros((len(waveforms), 12, 70))
    ok = np.zeros(len(waveforms), dtype=bool)
    for i in range(len(waveforms)):
        x = np.asarray(waveforms[i], dtype=float).T          # (1000, 12)
        beat = median_beat(x, detect_r_peaks(x))
        if beat is not None:
            out[i] = (beat - beat[BASE].mean(axis=0)).T
            ok[i] = True
    return out, ok


def _noisy_mean(sum_release, n: int, eps: float, rng) -> np.ndarray:
    noisy_n = max(n + float(sample_laplace(laplace_scale(1.0, COUNT_SHARE * eps), (), rng)), 1.0)
    return sum_release(eps * (1 - COUNT_SHARE)) / noisy_n


def run_fourier_study() -> dict:
    data_cfg = load_config("data")
    pdir = resolve(data_cfg["processed_dir"])
    cache = pdir / "templates.npz"
    manifest = pd.read_csv(pdir / "manifest.csv")
    if cache.exists():
        z = np.load(cache)
        templates, ok = z["templates"], z["ok"]
    else:
        log.info("Computing median-beat templates for %d records", len(manifest))
        templates, ok = beat_templates(np.load(pdir / "waveforms.npy", mmap_mode="r"))
        np.savez_compressed(cache, templates=templates, ok=ok)
    clipped = clip_rows_l2(templates, CLIP_NORM)
    norms = np.linalg.norm(templates.reshape(len(templates), -1), axis=1)

    rng = np.random.default_rng(2010)
    rows, examples = [], {}
    for cls, label in [("NORM", 0), ("MI", 1)]:
        sel = (manifest["label"].to_numpy() == label) & ok
        x = clipped[sel]
        n, total = len(x), x.sum(axis=0)
        true_mean = total / n
        ref = np.linalg.norm(true_mean)

        for eps in EPSILONS:
            def add(mech, release_fn, k=None, eps=eps, n=n, true_mean=true_mean, ref=ref, cls=cls):
                errs = [np.linalg.norm(_noisy_mean(release_fn, n, eps, rng) - true_mean) / ref
                        for _ in range(REPEATS)]
                rows.append({"class": cls, "n": n, "epsilon": eps, "mechanism": mech, "k": k,
                             "rel_error": float(np.mean(errs))})

            add("LPA", lambda e, t=total: lpa_release(t, CLIP_NORM, e, rng))
            add("Gaussian", lambda e, t=total: gaussian_time_release(t, CLIP_NORM, e, DELTA, rng))
            for k in KS:
                add("FPA", lambda e, t=total, k=k: fpa_release(t, CLIP_NORM, e, k, rng), k)
                add("FPA-Gaussian", lambda e, t=total, k=k: fpa_gaussian_release(t, CLIP_NORM, e, DELTA, k, rng), k)
        # Truncation-only error (no noise): the floor FPA_k can reach at a given k.
        for k in KS:
            rows.append({"class": cls, "n": n, "epsilon": float("inf"), "mechanism": "FPA (no noise)", "k": k,
                         "rel_error": float(np.linalg.norm(reconstruct(truncate_dft(true_mean, k), 70) - true_mean)
                                            / ref)})
        examples[cls] = (true_mean, n, total)
        log.info("Fourier study: %s done (n=%d)", cls, n)
    df = pd.DataFrame(rows)
    df.to_csv(artifact_dir("metrics") / "fourier_perturbation.csv", index=False)

    best = (df[df.mechanism == "FPA"].sort_values("rel_error").groupby(["class", "epsilon"]).first()
            .reset_index()[["class", "epsilon", "k", "rel_error"]])
    lpa = df[df.mechanism == "LPA"].set_index(["class", "epsilon"]).rel_error
    gau = df[df.mechanism == "Gaussian"].set_index(["class", "epsilon"]).rel_error
    best_g = (df[df.mechanism == "FPA-Gaussian"].sort_values("rel_error").groupby(["class", "epsilon"]).first()
              .rel_error)
    best["fpa_gaussian_best_rel_error"] = [best_g[(c, e)] for c, e in zip(best["class"], best.epsilon)]
    best["lpa_rel_error"] = [lpa[(c, e)] for c, e in zip(best["class"], best.epsilon)]
    best["gaussian_rel_error"] = [gau[(c, e)] for c, e in zip(best["class"], best.epsilon)]
    summary = {
        "query": "mean 12-lead median-beat template per class (12 x 70 samples, 100 Hz)",
        "clip_norm_l2": CLIP_NORM, "share_of_records_clipped": float((norms[ok] > CLIP_NORM).mean()),
        "records_without_beats": int((~ok).sum()), "count_share_of_epsilon": COUNT_SHARE,
        "gaussian_delta": DELTA, "repeats": REPEATS,
        "composition": "NORM and MI are disjoint: the two class releases compose in parallel",
        "best_k_per_epsilon": best.to_dict(orient="records"),
    }
    (artifact_dir("metrics") / "fourier_perturbation.json").write_text(json.dumps(summary, indent=2))
    _plots(df, examples, best, rng)
    return summary


def _plots(df: pd.DataFrame, examples: dict, best: pd.DataFrame, rng) -> None:
    colours = plt.cm.viridis(np.linspace(0.05, 0.85, len(EPSILONS)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, cls in zip(axes, ["NORM", "MI"]):
        d = df[df["class"] == cls]
        for eps, col in zip(EPSILONS, colours):
            f = d[(d.mechanism == "FPA") & (d.epsilon == eps)]
            ax.plot(f.k, f.rel_error, "o-", color=col, ms=3, label=f"FPA_k, ε={eps:g}")
            ax.axhline(d[(d.mechanism == "LPA") & (d.epsilon == eps)].rel_error.iloc[0], color=col, ls="--", lw=1)
            g = d[(d.mechanism == "FPA-Gaussian") & (d.epsilon == eps)]
            ax.plot(g.k, g.rel_error, "^:", color=col, ms=3, lw=1)
        t = d[d.mechanism == "FPA (no noise)"]
        ax.plot(t.k, t.rel_error, "k:", lw=1.5, label="truncation only (no noise)")
        ax.set(yscale="log", xlabel="k (DFT coefficients kept per lead)",
               title=f"{cls} mean beat (n = {int(d.n.iloc[0])})")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("relative L2 error of released mean")
    axes[0].legend(fontsize=7)
    fig.suptitle("Same colour = same ε.  Solid: FPA_k (Laplace).  Dashed: Laplace on every sample.  "
                 f"Dotted ▲: Fourier + Gaussian (δ={DELTA:g})")
    fig.tight_layout()
    fig.savefig(artifact_dir("figures") / "fourier_error_vs_k.png", dpi=150)
    plt.close(fig)

    # Example at ε = 1 for MI: the k shown is the best k from the study (data-dependent; see module docstring).
    true_mean, n, total = examples["MI"]
    k = int(best[(best["class"] == "MI") & (best.epsilon == 1.0)].k.iloc[0])
    t_ms = (np.arange(70) - 25) * 10
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, lead in zip(axes, ["II", "V2"]):
        j = LEADS.index(lead)
        ax.plot(t_ms, _noisy_mean(lambda e: lpa_release(total, CLIP_NORM, e, rng), n, 1.0, rng)[j],
                color=RED, alpha=0.6, lw=1, label="LPA (Laplace on samples)")
        ax.plot(t_ms, _noisy_mean(lambda e: gaussian_time_release(total, CLIP_NORM, e, DELTA, rng), n, 1.0, rng)[j],
                color=GREEN, alpha=0.8, lw=1, label=f"Gaussian on samples (δ={DELTA:g})")
        ax.plot(t_ms, _noisy_mean(lambda e: fpa_release(total, CLIP_NORM, e, k, rng), n, 1.0, rng)[j],
                color=BLUE, lw=2, label=f"FPA_{k}")
        ax.plot(t_ms, true_mean[j], color=GREY, ls="--", lw=1.5, label="exact mean (evaluation only)")
        ax.set(xlabel="ms from R peak", title=f"lead {lead}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mV")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"MI mean beat released at ε = 1 (n = {n}); one draw of each mechanism")
    fig.tight_layout()
    fig.savefig(artifact_dir("figures") / "fourier_example_mi.png", dpi=150)
    plt.close(fig)
