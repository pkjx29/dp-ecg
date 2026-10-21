"""PTB-XL metadata, labels and splits.

Task: binary classification, NORMAL vs ABNORMAL 12-lead ECG.
  * label 0 = the record's only diagnostic superclass is NORM
  * label 1 = at least one of MI, STTC, CD, HYP is present
  * label -1 = no diagnostic superclass (411 of 21,799 records). These are
    kept for the dataset-wide DP summary statistics but never used to train
    or test a classifier.
The superclass mapping follows the dataset authors' own example code
(scp_statements.csv, column diagnostic_class).

Splits use the authors' patient-wise stratified folds, so no patient is in
two splits: folds 1-8 train, fold 9 reserved (shadow data for the
membership-inference step), fold 10 test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100  # Hz, the records100/ version
TRAIN_FOLDS = tuple(range(1, 9))
RESERVED_FOLDS = (9,)
TEST_FOLDS = (10,)


def load_metadata(root: str | Path, labelled_only: bool = True) -> pd.DataFrame:
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp[scp["diagnostic"] == 1.0]["diagnostic_class"]

    def superclasses(codes: dict) -> set[str]:
        return {diag[c] for c in codes if c in diag.index}

    db["superclasses"] = db["scp_codes"].apply(superclasses)
    db["label"] = db["superclasses"].apply(lambda s: -1 if not s else int(s != {"NORM"}))
    return db[db["label"] >= 0].copy() if labelled_only else db


def split_masks(db: pd.DataFrame) -> dict[str, np.ndarray]:
    f = db["strat_fold"].to_numpy()
    return {
        "train": np.isin(f, TRAIN_FOLDS),
        "reserved": np.isin(f, RESERVED_FOLDS),
        "test": np.isin(f, TEST_FOLDS),
    }
