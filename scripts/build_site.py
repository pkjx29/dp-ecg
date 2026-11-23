"""Build the static results page (GitHub Pages) from saved artifacts only.

Standard library only, so the Pages workflow needs no PyTorch. It never reads
data/ or checkpoints: the page shows exactly what is committed under
artifacts/metrics, artifacts/reports and artifacts/figures.

    python scripts/build_site.py --out _site
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/pkjx29/dp-ecg"
DATA_FILES = [
    "metrics/runs_table.csv",
    "metrics/membership_inference.json",
    "metrics/lira.json",
    "metrics/fourier_perturbation.json",
    "reports/dp_summary_statistics.json",
    "reports/mechanism_verification.json",
    "reports/results_summary.md",
]
FIGURES = [
    ("privacy_utility.png", "Test AUROC against achieved ε. The dashed line is the non-private baseline."),
    ("mia_vs_epsilon.png", "Loss-threshold membership attack AUROC against ε."),
    ("lira_roc.png", "LiRA ROC curves (log scale) with the (ε, δ)-DP bound."),
    ("fourier_error_vs_k.png", "Fourier perturbation: relative error of the released mean beat against k."),
    ("dp_mean_error.png", "DP mean heart rate: observed error against the noise formula."),
]


def _load(rel: str, artifacts: Path):
    return json.loads((artifacts / rel).read_text())


def _mean_sd(xs: list[float], digits: int = 3) -> str:
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.{digits}f} ± {sd:.{digits}f}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _config_key(run: dict) -> tuple[int, float]:
    return (0, 0.0) if run["kind"] == "baseline" else (1, -run["privacy"]["requested_epsilon"])


def classification_and_mia(artifacts: Path) -> str:
    runs = [json.loads(p.read_text()) for p in sorted((artifacts / "metrics/runs").glob("*.json"))
            if not p.name.endswith(".error.json")]
    runs = [r for r in runs if r.get("status") == "completed" and not r.get("synthetic_smoke_test")]
    mia = _load("metrics/membership_inference.json", artifacts)["results"]
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(_config_key(r), []).append(r)
    rows = []
    for key in sorted(groups):
        rs = groups[key]
        if key[0] == 0:
            name, eps, nm = "CNN, non-private", "– (no guarantee)", "–"
        else:
            p = rs[0]["privacy"]
            name = f"CNN, DP-SGD, target ε = {p['requested_epsilon']:g}"
            eps, nm = f"{p['achieved_epsilon']:.2f}", f"{p['noise_multiplier']:.2f}"
        attacks = [mia[r["run_id"]]["attacks"]["loss"] for r in rs if r["run_id"] in mia]
        above = sum(a["auroc_ci95"][0] > 0.5 for a in attacks)
        rows.append([
            html.escape(name), eps, nm, str(len(rs)),
            f"<b>{_mean_sd([r['test']['auroc'] for r in rs])}</b>",
            _mean_sd([r["test"]["balanced_accuracy"] for r in rs]),
            _mean_sd([r["test"]["f1"] for r in rs]),
            _mean_sd([a["auroc"] for a in attacks]) if attacks else "–",
            f"{above} of {len(attacks)}" if attacks else "–",
        ])
    n_test = runs[0]["test"]["n"] if runs else 0
    return (
        f"<p>Test fold 10, n = {n_test:,}. Mean ± sample sd over seeds; every seed is a full training run. "
        "δ = 10⁻⁵ for DP-SGD (record-level). The last two columns are the loss-threshold membership attack: "
        "its AUROC and how many runs have a 95% bootstrap CI entirely above 0.5.</p>"
        + _table(["model", "achieved ε", "noise multiplier", "seeds", "test AUROC", "balanced acc.", "F1 (MI)",
                  "attack AUROC", "attack CI above 0.5"], rows)
    )


def lira(artifacts: Path) -> str:
    d = _load("metrics/lira.json", artifacts)
    order = ["baseline", "dp_eps8", "dp_eps4", "dp_eps2", "dp_eps1"]

    def pct(v):
        if v is None:
            return "–"
        v = 100 * v
        return f"{v:.0f}%" if v >= 10 else f"{v:.1f}%" if v >= 1 else f"{v:.2f}%"

    rows = []
    for k in order:
        r = d["results"].get(k)
        if r is None:
            continue
        name = "non-private" if r["kind"] == "baseline" else f"DP-SGD, ε = {r['target_epsilon']:g}"
        rows.append([html.escape(name), f"{r['auroc']:.3f}", pct(r["tpr_at_fpr_0.001"]),
                     pct(r.get("dp_bound_tpr_at_fpr_0.001")), pct(r["tpr_at_fpr_0.01"]),
                     pct(r.get("dp_bound_tpr_at_fpr_0.01"))])
    p = d["protocol"]
    return (
        f"<p>Online LiRA (Carlini et al. 2022): {p['n_models_per_config']} shadow models per configuration, each "
        f"trained on a random half of the {p['pool_size']:,} training records. The DP bound is the most any attack "
        "can reach under (ε, δ)-DP: TPR ≤ e<sup>ε</sup>·FPR + δ.</p>"
        + _table(["model", "LiRA AUROC", "TPR at FPR 0.1%", "DP bound", "TPR at FPR 1%", "DP bound"], rows)
    )


def fourier(artifacts: Path) -> str:
    d = _load("metrics/fourier_perturbation.json", artifacts)
    rows = [[html.escape(r["class"]), f"{r['epsilon']:g}", f"{r['lpa_rel_error']:.3g}",
             f"{r['rel_error']:.3g} (k = {int(r['k'])})", f"{r['gaussian_rel_error']:.3g}",
             f"{r['fpa_gaussian_best_rel_error']:.3g}"] for r in d["best_k_per_epsilon"]]
    return (
        "<p>Release of the mean 12-lead heartbeat per class (Rastogi &amp; Nath 2010). Relative L2 error, averaged "
        f"over {d['repeats']} draws; Gaussian variants use δ = {d['gaussian_delta']:g}. k was chosen by looking at "
        "the error, so a real release would fix it in advance.</p>"
        + _table(["class", "ε", "Laplace on samples (LPA)", "FPA_k, Laplace", "Gaussian on samples",
                  "FPA_k + Gaussian"], rows)
    )


def summary_stats(artifacts: Path) -> str:
    d = _load("reports/dp_summary_statistics.json", artifacts)
    rows = []
    for r in d["releases"]:
        v = r["value"]
        val = (f"{len(v)}-bin histogram" if isinstance(v, list) else f"{v:,.0f}" if abs(v) >= 1000
               else f"{v:.4g}")
        bounds = "–" if r.get("lower") is None else f"[{r['lower']:g}, {r['upper']:g}]"
        rows.append([html.escape(r["name"]), html.escape(r["mechanism"]), f"{r['epsilon']:g}", f"{r['delta']:g}",
                     bounds, val])
    b = d["budget"]
    return (
        f"<p>All records; adjacency: {html.escape(d['adjacency'])}. Total budget ε = {b['total_epsilon']:g}, "
        f"δ = {b['total_delta']:g} (basic composition), spent ε = {b['spent_epsilon']:g}. "
        "Values are the noisy releases; the exact values are not published.</p>"
        + _table(["release", "mechanism", "ε", "δ", "clipping bounds", "noisy value"], rows)
    )


def verification(artifacts: Path) -> str:
    d = _load("reports/mechanism_verification.json", artifacts)
    passed = sum(c["passed"] for c in d["checks"])
    return (
        f"<p>Every neighbouring pair of datasets (multisets of size ≤ {d['max_size']} over {d['domain']}, "
        f"{d['n_pairs']} add/remove pairs) is checked for sensitivity, the Laplace ε bound, the exact Gaussian δ "
        f"and the geometric mechanism: <b>{passed} of {len(d['checks'])} checks pass</b>. These checks guard the "
        "implementation; the guarantee for arbitrary data comes from the proofs.</p>"
    )


CSS = """
:root{--bg:#ffffff;--fg:#1d2330;--muted:#5b6475;--line:#dde1e8;--accent:#2a6fdb;--card:#f6f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e6e9ee;--muted:#9aa3b2;--line:#2b313b;
--accent:#6ea0ff;--card:#181c22}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}h1{font-size:1.9rem;margin:0 0 .3rem;line-height:1.2}
h2{font-size:1.25rem;margin:2.4rem 0 .6rem;padding-top:.6rem;border-top:1px solid var(--line)}
p{margin:.5rem 0}.muted{color:var(--muted)}a{color:var(--accent)}
.note{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:16px 0}
.scroll{overflow-x:auto;margin:.8rem 0}table{border-collapse:collapse;width:100%;font-size:.92rem;
font-variant-numeric:tabular-nums}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;
white-space:nowrap}th{color:var(--muted);font-weight:600}
figure{margin:1.2rem 0}figure img{max-width:100%;height:auto;background:#fff;border-radius:6px;
border:1px solid var(--line)}figcaption{color:var(--muted);font-size:.9rem}
ul.links{padding-left:1.1rem}
"""


def build(out: Path, artifacts: Path = ROOT / "artifacts", commit: str | None = None) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    (out / "data").mkdir()
    figs: dict[str, str] = {}
    for name, caption in FIGURES:
        src = artifacts / "figures" / name
        if src.exists():
            shutil.copy2(src, out / "figures" / name)
            figs[name] = (f'<figure><img src="figures/{name}" alt="{html.escape(caption)}" loading="lazy">'
                          f"<figcaption>{html.escape(caption)}</figcaption></figure>")
    links = []
    for rel in DATA_FILES:
        src = artifacts / rel
        if src.exists():
            shutil.copy2(src, out / "data" / src.name)
            links.append(f'<li><a href="data/{src.name}">{src.name}</a></li>')
    commit = commit or os.environ.get("GITHUB_SHA")
    source = (f'<a href="{REPO_URL}/tree/{commit}">{commit[:7]}</a>' if commit
              else f'<a href="{REPO_URL}">{REPO_URL.removeprefix("https://")}</a>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DP-ECG results</title>
<meta name="description" content="Differentially private ECG classification on PTB-XL: DP-SGD, membership attacks.">

<style>{CSS}</style></head>
<body><main>
<h1>Differentially private ECG classification</h1>
<p class="muted">PTB-XL, normal vs myocardial infarction · Laplace and Gaussian mechanisms from scratch ·
Opacus DP-SGD · membership-inference attacks</p>
<div class="note">Research code, not a medical device. Every number below is read from the saved results in the
repository; nothing here trains a model or touches patient data. Code, derivations and limitations:
<a href="{REPO_URL}">{REPO_URL.removeprefix("https://")}</a>.</div>

<h2>DP-SGD: accuracy cost and membership attack</h2>
{classification_and_mia(artifacts)}
{figs.get("privacy_utility.png", "")}
{figs.get("mia_vs_epsilon.png", "")}

<h2>Stronger attack: LiRA with shadow models</h2>
{lira(artifacts)}
{figs.get("lira_roc.png", "")}

<h2>Private summary statistics over 21,799 ECGs</h2>
{summary_stats(artifacts)}
{figs.get("dp_mean_error.png", "")}

<h2>Mechanism verification</h2>
{verification(artifacts)}

<h2>Fourier perturbation of ECG beats</h2>
{fourier(artifacts)}
{figs.get("fourier_error_vs_k.png", "")}

<h2>Raw results</h2>
<ul class="links">{"".join(links)}</ul>
<p class="muted">Built from commit {source}. Limits: the guarantee is per ECG record, not per patient; three seeds
and one test split; attack results are empirical, not proofs.</p>
</main></body></html>
"""
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / ".nojekyll").write_text("")
    return out / "index.html"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    args = ap.parse_args()
    print(build(args.out))


if __name__ == "__main__":
    main()
