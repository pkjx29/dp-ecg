# CV entry (claims only what this repo shows)

**Differentially Private ECG Classification** | Python, NumPy, PyTorch, Opacus | github.com/pkjx29/dp-ecg
- Implemented the Laplace and Gaussian (analytic calibration) mechanisms from scratch in NumPy, with tests that check the privacy bound on every neighbour of a small dataset; released budgeted DP summary statistics (total ε = 1) over 21,799 PTB-XL ECGs.
- Trained a 1D CNN for normal vs myocardial-infarction ECGs with Opacus DP-SGD: test AUROC 0.949 non-private vs 0.887 at ε = 8 and 0.867 at ε = 1 (δ = 10⁻⁵, record-level).
- Ran a membership-inference attack: AUROC 0.533 (95% CI 0.510–0.555) on the non-private model vs 0.502 (CI 0.479–0.526) at ε = 1. Next: a stronger shadow-model attack and Fourier-domain perturbation of ECG time series.
