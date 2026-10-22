# CV entry (claims only what this repo shows)

**Differentially Private ECG Classification** | Python, NumPy, PyTorch, Opacus | github.com/pkjx29/dp-ecg
- Implemented the Laplace and Gaussian (analytic calibration) mechanisms from scratch in NumPy, with tests that check the privacy bound on every neighbour of a small dataset; released budgeted DP summary statistics (total ε = 1) over 21,799 PTB-XL ECGs.
- Trained a 1D CNN for normal vs myocardial-infarction ECGs with Opacus DP-SGD; over 3 seeds, test AUROC fell from 0.942 ± 0.006 (non-private) to 0.887 ± 0.003 at ε = 8 and 0.864 ± 0.009 at ε = 1 (δ = 10⁻⁵, record-level).
- Ran a membership-inference attack: AUROC 0.535 ± 0.002 on the non-private model (every run's 95% CI above chance) vs 0.499 ± 0.004 at ε = 1 (every DP run's CI includes chance). Next: a stronger shadow-model attack and Fourier-domain perturbation of ECG time series.
