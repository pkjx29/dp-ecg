# Architecture

```mermaid
flowchart TB
  subgraph data [src/dp_ecg/data]
    DL[download.py<br/>mirror + SHA-256] --> MD[metadata.py<br/>superclasses, NORM vs MI]
    MD --> SP[splits.py<br/>folds, patient-leakage check]
    SP --> PP[preprocessing.py<br/>fixed band-pass, scale, clip]
    PP --> PR[prepare.py<br/>waveforms.npy, manifest, data_report]
    MD --> FE[features.py<br/>R peaks, median beats, spectra]
  end
  subgraph privacy [src/dp_ecg/privacy]
    SE[sensitivity.py] --> LA[laplace.py] & GA[gaussian.py<br/>analytic + classical] & GE[discrete.py]
    LA & GA --> ST[statistics.py] --> LE[accounting.py<br/>BudgetLedger]
    NB[neighbours.py] --> VE[verification.py]
  end
  PR --> TR[training/loop.py<br/>baseline or Opacus DP-SGD] --> EX[experiments.py]
  CNN[models/cnn.py] --> TR
  EX --> MIA[attacks/membership.py]
  FE --> SS[evaluation/summary_stats.py]
  ST --> SS
  EX & MIA & SS & VE --> RP[evaluation/reports.py] --> ART[(artifacts/)]
  ART --> API[app/api.py FastAPI] --> UI[app/ui.py Gradio]
```

* **Library vs orchestration.** Everything is importable from `dp_ecg`. `cli.py` and `experiments.py` only wire the
  modules together, and the `Makefile` calls the CLI.
* **Configuration.** One YAML file per concern in `configs/`; code reads settings only through `config.load_config`.
* **Artifacts.** Run records (JSON) are written only after metrics are computed. Figures and reports are rebuilt from
  them by `dp-ecg report`, and the API and dashboard read only these files.
* **Separation from training.** The web app never trains or downloads anything.
