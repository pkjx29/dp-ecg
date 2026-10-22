# KNOWLEDGE: the maths and the design decisions

This file is the interview companion to the code. Every formula in `src/dp_ecg/privacy/` can be derived from
what is written here, and every design choice has its reason next to it.

---

## 1. Differential privacy and adjacency

A randomised algorithm M is **(ε, δ)-differentially private** if, for every pair of neighbouring datasets
D, D′ and every set of outputs S,

    P[M(D) ∈ S] ≤ e^ε · P[M(D′) ∈ S] + δ.

**Adjacency used everywhere in this repo:** D′ is D with **one ECG record added or removed** (unbounded DP).
Opacus uses the same relation, so budgets from the hand-written mechanisms and from DP-SGD are about the
same protected unit. Under *replace-one* adjacency some sensitivities double; for example a histogram goes
from 1 to 2.

**Reading ε.** Whatever an observer sees is at most e^ε times more likely with your record than without it.
At ε = 1 that factor is 2.7, and at ε = 8 it is about 3,000. δ is a small probability of failing that bound
and should be well below 1/n. Here n_train = 9,286 and δ = 10⁻⁵.

**Record vs patient.** PTB-XL has 21,799 records from 18,869 patients. In the NORM vs MI training set
there are 9,286 records from 8,522 patients. The guarantee protects one *record*. A patient with k records
gets, by group privacy, roughly (kε, k·e^{(k−1)ε}·δ). Patient-level DP would need patient-level adjacency:
sample and clip per patient, for example by summing a patient's per-record gradients before clipping, and
account per patient. That extension is described in §12.

---

## 2. Sensitivity (global vs local)

    Δ₁ f = max over neighbours ‖f(D) − f(D′)‖₁       Δ₂ f = max over neighbours ‖f(D) − f(D′)‖₂

*Global* sensitivity is the worst case over all datasets, and it is what calibrates noise. *Local*
sensitivity is the change around the actual dataset. It can be much smaller, but it depends on the private
data, so calibrating noise with it leaks information. (Smooth sensitivity and propose-test-release are the
careful ways to exploit it. This project does not.)

| query (values clipped to [lo, hi]) | Δ | why |
|---|---|---|
| count | 1 | one record changes the count by exactly 1 |
| sum | max(\|lo\|, \|hi\|) | the record's own value appears or disappears |
| sum of (x − c), with c = (lo+hi)/2 public | (hi − lo)/2 | centering halves the noise for heart rate: 250 → 115 |
| histogram over fixed public bins | Δ₁ = 1 | a record sits in exactly one bin |
| vector sum, d coordinates in [−1, 1] | Δ₁ = d, Δ₂ = √d | why the Gaussian mechanism wins in high dimension |

**Clipping is what creates the bound**, and the bounds must not come from the data, or the bounds themselves
leak. Here they are fixed physiological ranges (`data/features.py: FEATURE_BOUNDS`).

Code: `privacy/sensitivity.py`. Verified on every neighbour pair of a toy universe in
`privacy/verification.py`, including a regression test that catches the classic bugs: `max(lo, hi)`
without `abs`, and the centred half-range used for an uncentred sum.

---

## 3. Laplace mechanism, giving ε-DP

    M(D) = f(D) + Lap(b),   b = Δ₁ / ε,   density (1/2b)·exp(−|z|/b)

Proof, for a 1-D output y:

    p_D(y) / p_D′(y) = exp((|y − f(D′)| − |y − f(D)|) / b) ≤ exp(|f(D) − f(D′)| / b) ≤ exp(Δ₁ / b) = e^ε

The middle step is the triangle inequality. Vectors work coordinate by coordinate with the L1 norm.
Var[Lap(b)] = 2b², so the noise std is √2·Δ₁/ε.

---

## 4. Gaussian mechanism, giving (ε, δ)-DP

    M(D) = f(D) + N(0, σ² I)

**Privacy loss.** For outputs p = N(0, σ²) and q = N(Δ, σ²) (the worst-case neighbour),

    L(y) = ln p(y)/q(y) = (Δ² − 2yΔ)/(2σ²),   and with y ~ p:   L ~ N(μ, 2μ),   μ = Δ²/(2σ²).

**Exact privacy profile** (Balle & Wang 2018). The smallest δ at a given ε is

    δ(ε) = Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ).

It is strictly decreasing in σ, so the **analytic calibration** (the default) finds the smallest σ with
δ(ε) ≤ δ by bisection. It is valid for every ε > 0. The **classical calibration**
σ = Δ·√(2 ln(1.25/δ))/ε comes from a looser tail bound and is proved only for ε < 1. The code refuses
ε ≥ 1 for it. Tests confirm that classical σ ≥ analytic σ, that both satisfy the exact δ, and that the
analytic σ is tight: 1% less noise breaks the guarantee. At the same (ε, δ) the analytic σ is always smaller, so the released statistics are less noisy.

**Why Gaussian at all?** It scales with Δ₂, not Δ₁. For d equal-range coordinates that is √d against d.
It also composes cleanly in Rényi DP, which DP-SGD relies on.

---

## 5. Exact checks with a discrete mechanism

For integer queries the **two-sided geometric mechanism** adds Z with P(Z = z) ∝ α^|z|, α = e^{−ε/Δ}. Its
output space is the integers. For neighbouring answers c, c′ the pmf ratio is constant outside
[min(c, c′) − 1, max(c, c′) + 1], so checking that window checks *every* output. On the toy universe this
gives a complete check of the ε-DP inequality, which a continuous mechanism cannot offer.

**What the toy-universe checks prove.** They show that the implementation matches the theorems on that
universe. They do not prove DP for arbitrary data; the theorems plus the sensitivity proofs do that.
Monte Carlo variance checks only test the samplers and are labelled as diagnostics in the tests.

---

## 6. Composition and the summary-statistics budget

Basic sequential composition: (ε₁, δ₁) followed by (ε₂, δ₂) on the same data gives (ε₁+ε₂, δ₁+δ₂).
`privacy/accounting.py: BudgetLedger` records every release and refuses any release that would exceed the
declared total. The PTB-XL summary release (`configs/privacy_mechanisms.yaml`) spends a total of
ε = 1.0 and δ = 10⁻⁶ across six queries:
- a count
- four means (two Laplace, two Gaussian)
- a 15-bin histogram

**DP mean = NoisySum/NoisyCount.** Under add/remove adjacency n is private, so it is noised too. To first
order,

    mean_DP − mean ≈ (η − (mean − c)·ν) / n,   RMSE² ≈ (Var η + (mean − c)²·Var ν) / n².

I first left out the count term, and the error plot showed the gap: observed error was about 12% above the
prediction. The term matters because the mean heart rate (74 bpm) is 61 bpm away from the midpoint c = 135,
so the count noise is magnified 61 times. With both terms, theory and observation agree
(`artifacts/figures/dp_mean_error.png`).

**Post-processing is free.** Clamping a noisy count at 0 or flooring a denominator at 1 uses only DP
outputs.

---

## 7. DP-SGD (Abadi et al. 2016) via Opacus

One step:
1. **Poisson sampling.** Each record joins the batch with probability q = B/n (B = 256, n = 9,286, so
   q ≈ 0.028).
2. **Per-example gradient clipping.** gᵢ ← gᵢ·min(1, C/‖gᵢ‖), with C = 1. One record now moves the summed
   gradient by at most C in L2.
3. **Gaussian noise.** Add N(0, σ²C²I) to the sum and divide by B. σ is the *noise multiplier*.
4. **Optimiser step.** This is post-processing (Adam here).

**Accounting.** Each step is a *subsampled* Gaussian mechanism. Because a record is probably not in a given
batch, each step costs less than an unsampled release would. The RDP accountant adds the per-step costs in
Rényi space over T = epochs/q steps, then converts to (ε, δ). `make_private_with_epsilon` searches for σ so
that the target ε is reached after all epochs. The achieved ε is then *recomputed* and stored separately from
the requested ε, and a run whose achieved ε exceeds the target by more than 1% is flagged.

**Why GroupNorm, not BatchNorm.** BatchNorm mixes statistics across the batch, so one record's gradient
depends on the others and per-example clipping no longer bounds its influence. `ModuleValidator.validate`
runs before every private training run and fails loudly. The code does not silently "fix" the model.

**Model selection.** The best epoch by validation AUROC is kept. Validation records are not training records,
so the choice does not touch the training set's budget. The released weights are post-processing of the
whole run, so the reported guarantee is ε after the *last* epoch, which is conservative.

**What the accountant does NOT cover:** hyperparameter tuning on private data (I fixed the hyperparameters
before the sweep and did not tune them per ε), repeated runs on the same data, and any other release such as
the summary statistics. Each of those needs its own accounting.

**Secure RNG.** Opacus' `secure_mode=True` needs `torchcsprng`, which has no build for the installed
PyTorch, so the runs use the standard PRNG. That is fine for research and not for deployment.

---

## 8. Data, task and preprocessing

* **Data.** PTB-XL v1.0.3, 100 Hz records, 10 s, 12 leads, giving a (12, 1000) tensor.
* **Labels.** Diagnostic superclasses come from `scp_statements.csv`. A record is kept only if its
  superclass set is exactly {NORM} (label 0) or exactly {MI} (label 1): 9,069 NORM and 2,532 MI. Excluded:
  411 with no diagnostic superclass, 1 with both NORM and MI, and 9,786 with other superclasses.
* **Splits.** Official folds: 1–8 train, 9 validation, 10 test. The code checks that no patient appears in
  two splits and raises an error otherwise.
* **Preprocessing.** A 0.5–40 Hz zero-phase Butterworth band-pass removes baseline wander and noise without
  shifting wave timing. Each record is then divided by a fixed 1 mV and clipped to ±5 mV. **Nothing is fitted
  to the data**, so preprocessing has no privacy cost and cannot leak test statistics into training. (An
  earlier version of this project fitted DP feature standardisation. Fixed scaling is simpler and costs no
  budget.)
* **Imbalance.** About 22% of records are MI. The decision threshold maximises balanced accuracy on the
  *validation* fold, and the test results report it.

## 9. 1D CNN

Four blocks of Conv1d (stride 2) → GroupNorm(8) → ReLU, with channels 32/64/64/128 and kernels 7/5/5/3.
Each block halves the time axis (1000 → 63), followed by global average pooling, dropout and a linear
head, 59k parameters. Small models suffer less from DP noise, because the noise is added to every
parameter's gradient.

---

## 10. Membership inference

**Threat model.** The attacker has a candidate record and its true label, and can query the model's
probability (black box). The attack scores, where higher means "member":
- loss: −BCE
- max probability
- negative entropy

**Protocol.** Thresholds are fitted on a calibration set: one random half of the training records against
the validation fold. They are evaluated on an evaluation set: the disjoint other half against the test fold,
balanced. The test fold never sets a threshold. Reported for each run:
- AUROC with a 95% bootstrap CI
- balanced accuracy, precision and recall at the calibrated threshold
- TPR at FPR 1% and 10%

**Interpretation.** DP bounds any attacker's advantage. For ε-DP, TPR ≤ e^ε·FPR + δ. A weak attack scoring
near 0.5 does not prove privacy, and a stronger attack (LiRA, shadow models) could do better. The attack
AUC is not expected to fall monotonically with ε, because the differences are small and runs vary.

---

## 11. Common bugs and their fixes

| bug | symptom | fix |
|---|---|---|
| noise calibrated with local sensitivity or data-derived bounds | "DP" output that leaks | fixed public bounds, global sensitivity |
| `max(lo, hi)` instead of `max(abs(lo), abs(hi))` | too little noise when lo < 0 | regression test |
| classical Gaussian used at ε ≥ 1 | invalid guarantee | refuse it; use analytic |
| each statistic gets the full ε | total budget understated | BudgetLedger |
| BatchNorm under DP-SGD | invalid per-sample gradients | GroupNorm + ModuleValidator |
| reporting the target ε as achieved | mismatch with the accountant | store both; flag if exceeded |
| Poisson sampling off but RDP assumes it | ε wrong | keep `poisson_sampling: true` |
| tuning on the test fold | optimistic results | validation fold only |

**Debugging Opacus.** If `ModuleValidator.validate` lists errors, replace the layer (BatchNorm becomes
GroupNorm; LSTM becomes Opacus' DPLSTM). If ε explodes, check the sample rate and steps in the run JSON.
If accuracy collapses, lower σ (higher ε), increase the batch size, or reduce the model size.

## 12. Extending to patient-level DP

Group records by patient, sample *patients* with Poisson sampling, compute each patient's gradient as the sum
over their records, clip that per-patient gradient to C, then add noise and account with
q = B_patients / n_patients. Opacus does not do this out of the box. It needs a custom data loader and
per-group gradient aggregation before clipping.

## 13. Reproducing everything

See README §"Reproduce". Seeds are fixed (`reproducibility.py`). Bit-identical results across machines and
library versions are not guaranteed.

## References

Dwork & Roth 2014 · Balle & Wang, ICML 2018 · Abadi et al., CCS 2016 · Mironov, CSF 2017 and CCS 2012 ·
Yeom et al., CSF 2018 · Carlini et al., S&P 2022 (LiRA) · Yousefpour et al. 2021 (Opacus) ·
Papernot & Steinke, ICLR 2022 · Wagner et al., *Scientific Data* 2020 (PTB-XL) · Rastogi & Nath, SIGMOD 2010.
