"""Extract features for every PTB-XL record (all 21,799) and cache them.
Records without a diagnostic label are stored with y = -1.

Usage: python scripts/build_features.py [--root data/raw/ptb-xl] [--out data/features.npz]
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import wfdb

from dpecg.data import LEADS, load_metadata
from dpecg.features import FEATURE_NAMES, extract

ROOT: Path


def _one(args):
    rel_path, age, sex = args
    sig, fields = wfdb.rdsamp(str(ROOT / rel_path))
    assert [s.upper() for s in fields["sig_name"]] == [l.upper() for l in LEADS], fields["sig_name"]
    return extract(sig, age, sex)


def _init(root):
    global ROOT
    ROOT = Path(root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/ptb-xl")
    ap.add_argument("--out", default="data/features.npz")
    args = ap.parse_args()

    db = load_metadata(args.root, labelled_only=False)
    jobs = list(zip(db["filename_lr"], db["age"].fillna(60.0), db["sex"].astype(float)))
    print(f"Extracting {len(FEATURE_NAMES)} features from {len(jobs)} records ...", flush=True)
    with Pool(os.cpu_count(), initializer=_init, initargs=(args.root,)) as pool:
        X = np.stack(pool.map(_one, jobs, chunksize=64))

    np.savez_compressed(
        args.out, X=X, y=db["label"].to_numpy(), ecg_id=db.index.to_numpy(),
        patient_id=db["patient_id"].to_numpy(), fold=db["strat_fold"].to_numpy(),
        feature_names=np.array(FEATURE_NAMES))
    nan_rate = np.isnan(X).mean(axis=0)
    print(f"Saved {X.shape} to {args.out}; labelled {(db['label'] >= 0).sum()}, "
          f"abnormal share among labelled {db.loc[db['label'] >= 0, 'label'].mean():.3f}; "
          f"worst NaN rate {nan_rate.max():.4f} ({FEATURE_NAMES[int(nan_rate.argmax())]})")


if __name__ == "__main__":
    main()
