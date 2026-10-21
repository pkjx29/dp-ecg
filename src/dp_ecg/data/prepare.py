"""Build the cached inputs every experiment reads.

* Waveforms for the NORM vs MI task -> data/processed/waveforms.npy
  (float16, shape (N, 12, 1000), about 280 MB) plus an INTERNAL manifest
  with ecg_id / patient_id / split / label (data/ is git-ignored).
* Hand-crafted features for all 21,799 records -> data/processed/features.npz,
  used by the DP summary statistics.
* A PUBLIC data report (counts only, no identifiers) ->
  artifacts/reports/data_report.json.
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ..config import artifact_dir, resolve
from ..logging_utils import get_logger
from .features import FEATURE_NAMES, extract
from .metadata import LEADS, load_metadata, select_task
from .preprocessing import preprocess, validate_waveform
from .splits import assign_splits, check_patient_disjoint

log = get_logger(__name__)
_ROOT: Path
_CFG: dict


def _init(root: str, cfg: dict) -> None:
    global _ROOT, _CFG
    _ROOT, _CFG = Path(root), cfg


def _read(rel_path: str) -> np.ndarray:
    sig, fields = wfdb.rdsamp(str(_ROOT / rel_path))
    if [s.upper() for s in fields["sig_name"]] != [lead.upper() for lead in LEADS]:
        raise ValueError(f"{rel_path}: unexpected lead order {fields['sig_name']}")
    if fields["fs"] != _CFG["sampling_rate_hz"]:
        raise ValueError(f"{rel_path}: sampling rate {fields['fs']}")
    return validate_waveform(sig, _CFG["n_samples"], _CFG["n_leads"])


def _waveform_job(rel_path: str):
    p = _CFG["preprocessing"]
    return preprocess(_read(rel_path), _CFG["sampling_rate_hz"], p["bandpass_hz"], p["filter_order"],
                      p["amplitude_scale_mv"], p["clip_mv"])


def _feature_job(args):
    rel_path, age, sex = args
    return extract(_read(rel_path), age, sex)


def prepare_waveforms(cfg: dict) -> dict:
    root, out_dir = resolve(cfg["dataset_root"]), resolve(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    db = load_metadata(root)
    sel, exclusions = select_task(db, cfg["task"]["negative"], cfg["task"]["positive"])
    sel["split"] = assign_splits(sel, cfg["folds"])
    patients_per_split = check_patient_disjoint(sel["patient_id"], sel["split"])

    log.info("Preprocessing %d waveforms", len(sel))
    arr = np.lib.format.open_memmap(out_dir / "waveforms.npy", mode="w+", dtype=np.float16,
                                    shape=(len(sel), cfg["n_leads"], cfg["n_samples"]))
    n_bad = np.zeros(len(sel), dtype=int)
    with Pool(os.cpu_count(), initializer=_init, initargs=(str(root), cfg)) as pool:
        for i, (x, bad) in enumerate(pool.imap(_waveform_job, sel["filename_lr"], chunksize=64)):
            arr[i], n_bad[i] = x, bad
    arr.flush()
    if not np.isfinite(np.asarray(arr[:: max(1, len(sel) // 500)], dtype=np.float32)).all():
        raise ValueError("non-finite values in processed waveforms")

    manifest = pd.DataFrame({"row": np.arange(len(sel)), "ecg_id": sel.index, "patient_id": sel["patient_id"],
                             "strat_fold": sel["strat_fold"], "split": sel["split"], "label": sel["label"],
                             "n_nonfinite_replaced": n_bad})
    manifest.to_csv(out_dir / "manifest.csv", index=False)  # internal: contains patient ids

    report = {
        "dataset": "PTB-XL v1.0.3, 100 Hz", "task": f"{cfg['task']['negative']} (0) vs {cfg['task']['positive']} (1)",
        "selection": exclusions,
        "splits": {s: {"records": int((sel["split"] == s).sum()),
                       "patients": patients_per_split.get(s, 0),
                       "positive_share": round(float(sel.loc[sel["split"] == s, "label"].mean()), 4)}
                   for s in ["train", "val", "test"]},
        "patient_disjoint_across_splits": True,
        "records_with_nonfinite_values": int((n_bad > 0).sum()),
        "waveform_shape": [cfg["n_leads"], cfg["n_samples"]],
        "preprocessing": cfg["preprocessing"],
    }
    (artifact_dir("reports") / "data_report.json").write_text(json.dumps(report, indent=2))
    log.info("Selection: %s", exclusions)
    return report


def prepare_features(cfg: dict) -> Path:
    root, out_dir = resolve(cfg["dataset_root"]), resolve(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    db = load_metadata(root)
    jobs = list(zip(db["filename_lr"], db["age"].fillna(60.0), db["sex"].astype(float)))
    log.info("Extracting %d features from %d records", len(FEATURE_NAMES), len(jobs))
    with Pool(os.cpu_count(), initializer=_init, initargs=(str(root), cfg)) as pool:
        X = np.stack(pool.map(_feature_job, jobs, chunksize=64))
    path = out_dir / "features.npz"
    np.savez_compressed(path, X=X, ecg_id=db.index.to_numpy(), feature_names=np.array(FEATURE_NAMES))
    return path
