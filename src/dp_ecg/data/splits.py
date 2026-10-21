"""Fold-based splits with an explicit patient-leakage check."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PatientLeakageError(RuntimeError):
    pass


def assign_splits(db: pd.DataFrame, folds: dict[str, list[int]]) -> pd.Series:
    """Map strat_fold to a split name ('train' / 'val' / 'test')."""
    lookup = {f: name for name, fs in folds.items() for f in fs}
    split = db["strat_fold"].map(lookup)
    if split.isna().any():
        raise ValueError(f"folds {sorted(db.loc[split.isna(), 'strat_fold'].unique())} have no split")
    return split


def check_patient_disjoint(patient_ids, splits) -> dict[str, int]:
    """Raise if any patient appears in more than one split."""
    frame = pd.DataFrame({"patient": np.asarray(patient_ids), "split": np.asarray(splits)})
    per_patient = frame.groupby("patient")["split"].nunique()
    leaked = per_patient[per_patient > 1]
    if len(leaked):
        raise PatientLeakageError(f"{len(leaked)} patients appear in more than one split, "
                                  f"e.g. {list(leaked.index[:3])}")
    return {name: int(n) for name, n in frame.groupby("split")["patient"].nunique().items()}
