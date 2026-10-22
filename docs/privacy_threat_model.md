# Privacy threat model

| item | this project |
|---|---|
| protected unit | one ECG record (add/remove adjacency) |
| not protected | a patient as a whole when they have several records (group privacy degrades ε) |
| DP-SGD guarantee | (ε, δ) from Opacus' RDP accountant for the full training run, δ = 10⁻⁵ |
| summary-statistics guarantee | (1.0, 10⁻⁶) for the six releases together, basic composition |
| assumed public | model architecture, hyperparameters, feature clipping bounds, preprocessing constants, fold assignment |
| attacker in the experiments | black-box access to predicted probabilities; knows the record and its true label |
| attacker the guarantee covers | any attacker, including one who knows every other record (that is what DP bounds) |

**What the guarantees do not cover:**
- releasing several models trained on the same data (their ε values add up)
- the exploratory runs in this repo's history
- the non-private baseline, which has no guarantee, so its checkpoint should not be published
- side channels from floating-point sampling or a non-cryptographic RNG

**Verified vs assumed.**
- Verified by tests: sensitivity bounds on a toy universe, the calibration formulas, budget enforcement, that the
  achieved ε does not exceed the target, and that the model passes Opacus validation.
- Assumed: the correctness of Opacus' accountant and per-sample gradient computation, and the mechanism theorems
  from the cited papers.
