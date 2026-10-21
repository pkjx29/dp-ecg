"""PTB-XL metadata, diagnostic superclasses and task labels.

Superclasses follow the dataset authors' own mapping (scp_statements.csv,
column diagnostic_class, rows with diagnostic == 1). Every SCP code listed
for a record counts, whatever its likelihood value, as in the authors'
example code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100


def load_metadata(root: str | Path) -> pd.DataFrame:
    """All records, indexed by ecg_id, with a `superclasses` frozenset column."""
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp[scp["diagnostic"] == 1.0]["diagnostic_class"]
    db["superclasses"] = db["scp_codes"].apply(
        lambda codes: frozenset(diag[c] for c in codes if c in diag.index))
    return db


def select_task(db: pd.DataFrame, negative: str = "NORM", positive: str = "MI"
                ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep records whose superclass set is exactly {negative} or {positive}.

    Returns (selected records with a 0/1 `label` column, exclusion counts).
    Records with both classes, other superclasses, or no diagnostic
    superclass are excluded: their label would be ambiguous for this task.
    """
    neg = db["superclasses"].apply(lambda s: s == frozenset({negative}))
    pos = db["superclasses"].apply(lambda s: s == frozenset({positive}))
    none = db["superclasses"].apply(len) == 0
    both = db["superclasses"].apply(lambda s: {negative, positive} <= s)
    sel = db[neg | pos].copy()
    sel["label"] = pos[neg | pos].astype(int)
    report = {
        "total_records": len(db),
        f"selected_{negative}": int(neg.sum()),
        f"selected_{positive}": int(pos.sum()),
        "excluded_no_diagnostic_superclass": int(none.sum()),
        f"excluded_both_{negative}_and_{positive}": int(both.sum()),
        "excluded_other_superclasses": int(len(db) - neg.sum() - pos.sum() - none.sum() - both.sum()),
    }
    return sel, report
