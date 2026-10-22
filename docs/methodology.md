# Methodology

Derivations are in [KNOWLEDGE.md](../KNOWLEDGE.md); this page states the experiment design.

1. **Mechanisms.**
   - Laplace: b = Δ₁/ε.
   - Gaussian: analytic calibration (smallest σ with the exact δ(ε) ≤ δ). The classical
     σ = Δ₂√(2 ln(1.25/δ))/ε is available for ε < 1 only.
   - Two-sided geometric for integer queries.
2. **Verification.** Exhaustive add/remove neighbour pairs over a toy universe, checking sensitivity, the Laplace
   closed-form ratio, the Gaussian exact δ, and the geometric pmf ratio on every output.
3. **Summary statistics.** Six releases over all 21,799 records with shares of a total (ε = 1, δ = 10⁻⁶) under
   basic composition. Means use NoisySum/NoisyCount centred on the public midpoint of fixed physiological bounds.
4. **Classification task.** NORM-only vs MI-only PTB-XL records, 100 Hz, 12 × 1000, with fixed preprocessing.
   Folds 1–8 / 9 / 10 for train / validation / test.
5. **Models.** The same GroupNorm 1D CNN (59k parameters) is used for both settings:
   - baseline: Adam, lr 10⁻³, batch 128, 12 epochs
   - DP-SGD: Adam, lr 2·10⁻³, Poisson batches of expected size 256, C = 1, 12 epochs, target ε ∈ {1, 2, 4, 8},
     δ = 10⁻⁵, RDP accountant
6. **Selection and threshold.** Keep the epoch with the best validation AUROC. Choose the decision threshold by
   maximising balanced accuracy on validation.
7. **Test metrics.** Accuracy, balanced accuracy, precision, recall, F1, AUROC, the confusion matrix and prevalence,
   all on fold 10.
8. **Membership inference.** Loss, max-probability and entropy threshold attacks.
   - Calibration: train half A vs fold 9.
   - Evaluation: train half B vs fold 10, balanced.
   - AUROC with a 1,000-sample bootstrap CI, balanced accuracy, precision and recall at the calibrated threshold,
     and TPR at FPR 1% and 10%.

For a binary model, max probability and entropy are monotone functions of each other, so their AUROCs are identical
by construction.
