"""Label selection on a hand-made metadata table (no download needed)."""

import pandas as pd

from dp_ecg.data.metadata import load_metadata, select_task


def _write(tmp_path):
    pd.DataFrame({
        "ecg_id": [1, 2, 3, 4, 5], "patient_id": [10, 11, 12, 13, 14],
        "scp_codes": ["{'NORM': 100.0, 'SR': 0.0}", "{'IMI': 50.0}", "{'NORM': 80.0, 'IMI': 100.0}",
                      "{'SR': 0.0}", "{'LAFB': 100.0}"],
        "strat_fold": [1, 9, 10, 3, 2],
    }).to_csv(tmp_path / "ptbxl_database.csv", index=False)
    pd.DataFrame({"code": ["NORM", "IMI", "LAFB", "SR"], "diagnostic": [1.0, 1.0, 1.0, None],
                  "diagnostic_class": ["NORM", "MI", "CD", None]}
                 ).set_index("code").to_csv(tmp_path / "scp_statements.csv")


def test_select_norm_vs_mi(tmp_path):
    _write(tmp_path)
    db = load_metadata(tmp_path)
    assert db.loc[3, "superclasses"] == frozenset({"NORM", "MI"})
    sel, rep = select_task(db, "NORM", "MI")
    assert list(sel.index) == [1, 2] and list(sel["label"]) == [0, 1]
    assert rep["excluded_both_NORM_and_MI"] == 1
    assert rep["excluded_no_diagnostic_superclass"] == 1
    assert rep["excluded_other_superclasses"] == 1
    assert sum(v for k, v in rep.items() if k != "total_records") == rep["total_records"]
