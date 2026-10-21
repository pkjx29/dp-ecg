"""Orchestration: baseline, DP-SGD sweep, membership inference.

Each completed run writes artifacts/metrics/runs/<run_id>.json only after its
metrics have been computed; an existing file means the run is done and it
is skipped on re-runs (delete the file to redo it). Checkpoints go to
artifacts/checkpoints/ (git-ignored).
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .attacks.membership import attack_scores, evaluate_attack, split_members
from .config import artifact_dir, load_config, resolve
from .data.dataset import ECGDataset
from .logging_utils import get_logger
from .models.cnn import build_model
from .models.metrics import choose_threshold, classification_metrics
from .reproducibility import environment_info, seed_everything
from .training.loop import PrivacySettings, predict_scores, train_model

log = get_logger(__name__, artifact_dir("logs") / "experiments.jsonl")


def load_data(data_cfg: dict, synthetic: bool = False):
    """(waveforms, manifest). synthetic=True builds a tiny random dataset for
    smoke tests; its results are never PTB-XL results."""
    if synthetic:
        rng = np.random.default_rng(0)
        n = 240
        y = rng.integers(0, 2, n)
        w = rng.normal(0, 0.1, (n, data_cfg["n_leads"], data_cfg["n_samples"])).astype(np.float16)
        w[y == 1, :, 100:120] += 1.0  # a learnable bump
        split = np.array(["train"] * 160 + ["val"] * 40 + ["test"] * 40)
        return w, pd.DataFrame({"split": split, "label": y, "patient_id": np.arange(n)})
    pdir = resolve(data_cfg["processed_dir"])
    return np.load(pdir / "waveforms.npy", mmap_mode="r"), pd.read_csv(pdir / "manifest.csv")


def _split(manifest: pd.DataFrame, name: str) -> np.ndarray:
    return manifest.index[manifest["split"] == name].to_numpy()


def _run_one(run_id: str, waveforms, manifest, model_cfg, train_cfg, seed, privacy, out_dir: Path,
             synthetic: bool) -> dict:
    labels = manifest["label"].to_numpy()
    tr, va, te = (_split(manifest, s) for s in ("train", "val", "test"))
    seed_everything(seed)
    x_val, y_val = ECGDataset(waveforms, labels, va).tensors()
    x_te, y_te = ECGDataset(waveforms, labels, te).tensors()
    res = train_model(build_model(model_cfg), ECGDataset(waveforms, labels, tr), x_val, y_val.numpy(),
                      train_cfg, seed, privacy)
    val_scores = predict_scores(res.model, x_val)
    thr = choose_threshold(y_val.numpy(), val_scores)
    ckpt = artifact_dir("checkpoints") / f"{'smoke_' if synthetic else ''}{run_id}.pt"
    torch.save({"state_dict": res.model.state_dict(), "model_cfg": model_cfg}, ckpt)
    record = {
        "run_id": run_id, "status": "completed", "synthetic_smoke_test": synthetic,
        "kind": "dp_sgd" if privacy else "baseline", "seed": seed,
        "model_cfg": model_cfg, "train_cfg": train_cfg, "privacy": res.privacy,
        "best_epoch": res.best_epoch, "history": res.history, "train_seconds": round(res.seconds, 1),
        "n_train": int(len(tr)),
        "validation": classification_metrics(y_val.numpy(), val_scores, thr),
        "test": classification_metrics(y_te.numpy(), predict_scores(res.model, x_te), thr),
        "checkpoint": str(ckpt.relative_to(ckpt.parents[2])),
        "environment": environment_info(),
    }
    if privacy:
        tol = load_config("dp_sgd")["epsilon_tolerance"]
        record["privacy"]["within_target"] = bool(
            res.privacy["achieved_epsilon"] <= privacy.target_epsilon * (1 + tol))
        if not record["privacy"]["within_target"]:
            record["status"] = "flagged: achieved epsilon exceeds target"
    (out_dir / f"{run_id}.json").write_text(json.dumps(record, indent=2))
    return record


def run_training(synthetic: bool = False, only: str | None = None, epochs: int | None = None) -> list[dict]:
    data_cfg, base_cfg, dp_cfg = load_config("data"), load_config("baseline"), load_config("dp_sgd")
    waveforms, manifest = load_data(data_cfg, synthetic)
    out_dir = artifact_dir("metrics") / ("runs_smoke" if synthetic else "runs")
    out_dir.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    jobs = []
    if only in (None, "baseline"):
        jobs.append(("baseline_seed%d" % base_cfg["train"]["seed"], dict(base_cfg["train"]),
                     base_cfg["train"]["seed"], None))
    if only in (None, "dp"):
        for eps in dp_cfg["target_epsilons"]:
            for seed in dp_cfg["train"]["seeds"]:
                jobs.append((f"dp_eps{eps:g}_seed{seed}", dict(dp_cfg["train"]), seed,
                             PrivacySettings(eps, dp_cfg["delta"], dp_cfg["max_grad_norm"], dp_cfg["accountant"],
                                             dp_cfg["poisson_sampling"], dp_cfg["secure_mode"])))
    records = []
    for run_id, train_cfg, seed, privacy in jobs:
        if epochs is not None:
            train_cfg["epochs"] = epochs
        path = out_dir / f"{run_id}.json"
        if path.exists() and not synthetic:
            log.info("skip %s (already completed)", run_id)
            records.append(json.loads(path.read_text()))
            continue
        log.info("start %s", run_id)
        try:
            records.append(_run_one(run_id, waveforms, manifest, base_cfg["model"], train_cfg, seed,
                                    privacy, out_dir, synthetic))
        except Exception as exc:  # record the failure, keep the sweep going
            log.error("run %s failed: %s", run_id, exc)
            (out_dir / f"{run_id}.error.json").write_text(json.dumps(
                {"run_id": run_id, "status": "failed", "error": repr(exc),
                 "traceback": traceback.format_exc()}, indent=2))
    return records


def run_membership_inference(synthetic: bool = False) -> dict:
    data_cfg, atk_cfg = load_config("data"), load_config("attack")
    waveforms, manifest = load_data(data_cfg, synthetic)
    labels = manifest["label"].to_numpy()
    tr, va, te = (_split(manifest, s) for s in ("train", "val", "test"))
    rng = np.random.default_rng(atk_cfg["seed"])
    mem_cal, mem_ev = split_members(tr, len(va), len(te), rng)
    cal_idx, ev_idx = np.concatenate([mem_cal, va]), np.concatenate([mem_ev, te])
    cal_member = np.r_[np.ones(len(mem_cal)), np.zeros(len(va))].astype(int)
    ev_member = np.r_[np.ones(len(mem_ev)), np.zeros(len(te))].astype(int)
    x_cal, y_cal = ECGDataset(waveforms, labels, cal_idx).tensors()
    x_ev, y_ev = ECGDataset(waveforms, labels, ev_idx).tensors()

    runs_dir = artifact_dir("metrics") / ("runs_smoke" if synthetic else "runs")
    results, roc = {}, {}
    for path in sorted(runs_dir.glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        run = json.loads(path.read_text())
        ckpt = torch.load(artifact_dir("checkpoints") / Path(run["checkpoint"]).name, weights_only=False)
        model = build_model(ckpt["model_cfg"])
        model.load_state_dict(ckpt["state_dict"])
        s_cal = attack_scores(predict_scores(model, x_cal), y_cal.numpy())
        s_ev = attack_scores(predict_scores(model, x_ev), y_ev.numpy())
        results[run["run_id"]] = {
            "kind": run["kind"],
            "achieved_epsilon": (run["privacy"] or {}).get("achieved_epsilon"),
            "attacks": {name: evaluate_attack(cal_member, s_cal[name], ev_member, s_ev[name],
                                              atk_cfg["fpr_targets"], atk_cfg["bootstrap_resamples"], rng)
                        for name in atk_cfg["scores"]},
        }
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(ev_member, s_ev["loss"])
        roc[run["run_id"]] = {"fpr": fpr[:: max(1, len(fpr) // 300)].tolist(),
                              "tpr": tpr[:: max(1, len(tpr) // 300)].tolist()}
        log.info("MIA %s loss-attack AUROC %.4f", run["run_id"], results[run["run_id"]]["attacks"]["loss"]["auroc"])
    out = {"protocol": {"calibration": {"members": int(len(mem_cal)), "non_members": int(len(va)),
                                        "non_member_source": "validation fold"},
                        "evaluation": {"members": int(len(mem_ev)), "non_members": int(len(te)),
                                       "non_member_source": "test fold"},
                        "attacker_knowledge": "record, true label, model's predicted probability (black box)",
                        "seed": atk_cfg["seed"], "bootstrap_resamples": atk_cfg["bootstrap_resamples"],
                        "synthetic_smoke_test": synthetic},
           "results": results, "roc_loss_attack": roc}
    name = "membership_inference_smoke.json" if synthetic else "membership_inference.json"
    (artifact_dir("metrics") / name).write_text(json.dumps(out, indent=2))
    return out
