# data/ (git-ignored)

* `raw/ptb-xl/` — PTB-XL v1.0.3 metadata and `records100/` (about 600 MB), fetched by `dp-ecg download`
  from PhysioNet's open-data mirror and checked against PhysioNet's SHA256SUMS.txt.
* `processed/waveforms.npy` — float16 (11,601, 12, 1000) NORM/MI waveforms (about 280 MB).
* `processed/manifest.csv` — INTERNAL: ecg_id, patient_id, split, label. Never published.
* `processed/features.npz` — hand-crafted features for all 21,799 records (for the DP summary statistics).

PTB-XL is CC BY 4.0. Cite Wagner et al., Scientific Data 7, 154 (2020), doi:10.1038/s41597-020-0495-6.
