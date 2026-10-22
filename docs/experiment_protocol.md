# Experiment protocol

1. `make install`: create the virtual environment and install the pinned requirements.
2. `dp-ecg download`: download and verify. Re-running only fetches missing or corrupted files.
3. `dp-ecg prepare`: write `data/processed/` and `artifacts/reports/data_report.json`, and check patient
   disjointness.
4. `dp-ecg verify-mechanisms`: write `artifacts/reports/mechanism_verification.{json,md}`. Exits non-zero if any
   check fails.
5. `dp-ecg summary-stats`: write `artifacts/reports/dp_summary_statistics.json`, the internal error diagnostic CSV
   and its figure.
6. `dp-ecg train-baseline` and `dp-ecg train-dp`: one JSON per run in `artifacts/metrics/runs/`.
   - Failures are written as `<run>.error.json` and the sweep continues.
   - Existing run files are skipped, so delete one to re-run it.
7. `dp-ecg attack`: write `artifacts/metrics/membership_inference.json`.
8. `dp-ecg report`: write the figures, `runs_table.csv` and `results_summary.md`.

**Seeds.**
- training: 0 (`configs/baseline.yaml`, `dp_sgd.yaml`)
- attack split: 7
- summary-statistics noise: 2026
- toy data: 0

Results are repeatable on the same machine and library versions. Bit-identical results across hardware or versions
are not guaranteed.

**Seeds.** `baseline.yaml` and `dp_sgd.yaml` list `train.seeds: [0, 1, 2]`, giving 15 runs in total (about 2 minutes
per baseline run and 2.5–4 minutes per DP run on an 8-core CPU). Reports show mean ± sample sd across seeds.

**Recorded per run:**
- requested and achieved ε, δ, noise multiplier, C, sample rate, steps, epochs, accountant, secure mode
- seeds, config, library versions, duration, history, validation and test metrics, checkpoint name, status
