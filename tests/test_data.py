"""Label construction on a tiny hand-made metadata table (no download needed)."""

import pandas as pd

from dpecg.data import load_metadata, split_masks


def test_labels_and_splits(tmp_path):
    pd.DataFrame({
        "ecg_id": [1, 2, 3, 4], "patient_id": [10, 11, 12, 13],
        "scp_codes": ["{'NORM': 100.0, 'SR': 0.0}", "{'IMI': 50.0, 'SR': 0.0}",
                      "{'NORM': 80.0, 'LAFB': 100.0}", "{'SR': 0.0}"],
        "strat_fold": [1, 9, 10, 3],
    }).to_csv(tmp_path / "ptbxl_database.csv", index=False)
    pd.DataFrame({
        "code": ["NORM", "IMI", "LAFB", "SR"], "diagnostic": [1.0, 1.0, 1.0, None],
        "diagnostic_class": ["NORM", "MI", "CD", None],
    }).set_index("code").to_csv(tmp_path / "scp_statements.csv")

    db = load_metadata(tmp_path)
    # record 4 has only a rhythm code (no diagnostic superclass) -> dropped
    assert list(db.index) == [1, 2, 3]
    # NORM only -> 0; MI -> 1; NORM + CD -> 1 (any abnormal superclass)
    assert list(db["label"]) == [0, 1, 1]
    m = split_masks(db)
    assert list(m["train"]) == [True, False, False]
    assert list(m["reserved"]) == [False, True, False]
    assert list(m["test"]) == [False, False, True]
