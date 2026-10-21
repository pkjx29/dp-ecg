# NOTES: the maths behind this repo, in plain language

These notes are written so every formula in `src/dpecg/mechanisms.py` and
every number in the results can be derived on a whiteboard. Code references
point to the function that implements each idea.

---

## 1. What differential privacy promises

A randomised algorithm M is **(ε, δ)-differentially private** if for every
pair of *neighbouring* datasets D, D′ and every set of outputs S

    P[M(D) ∈ S]  ≤  e^ε · P[M(D′) ∈ S]  +  δ

**Neighbours** in this repo: D′ is D with one record **added or removed**
("unbounded DP"). Opacus uses the same definition, so budgets spent on
statistics and on training can be added together (§6).

What it means: whatever an attacker sees, it is at most e^ε times more
likely when your record is in the data than when it is not (plus a small
slack δ). With ε = 1, e^ε ≈ 2.7. δ should be much smaller than 1/n, because
a mechanism that publishes one random record outright satisfies δ = 1/n.

One record here is one ECG. PTB-XL has 21,799 records from 18,869 patients,
so a few patients appear more than once. Record-level DP for a patient with
k records gives (kε, k·e^{(k−1)ε}·δ) by *group privacy*. That is why the
README says "record-level" and not "patient-level".

---

## 2. Sensitivity: how much can one record move the answer?

For a query f that returns numbers,

    Δ1 = max over neighbours ‖f(D) − f(D′)‖₁      (L1 sensitivity)
    Δ2 = max over neighbours ‖f(D) − f(D′)‖₂      (L2 sensitivity)

| Query | Sensitivity | Why |
|---|---|---|
| count | 1 | one record changes the count by exactly 1 |
| sum of values clipped to [lo, hi] | max(\|lo\|, \|hi\|) | the record's own value is added or removed |
| same, after subtracting public c = (lo+hi)/2 | (hi − lo)/2 | values now lie in ±(hi−lo)/2. **Centering halves the noise** for, e.g., heart rate in [20, 250]: 250 → 115 |
| histogram (fixed public bins) | Δ1 = 1 | a record sits in exactly one bin, so only one count changes. The sensitivity is not the number of bins |
| vector sum of d features each in [−1, 1] | Δ1 = d, Δ2 = √d | the record contributes a vector whose entries are at most 1 in size |

**Clipping is what creates the bound.** Without it the sum of heart rates has
unbounded sensitivity (one corrupted record could read 10⁶ bpm). The bounds
must be fixed **without looking at the data**. If they came from the data's
own min/max, the bounds themselves would leak. Here they come from physiology
(`features.FEATURE_BOUNDS`).

Code: `count_sensitivity`, `bounded_sum_sensitivity`, `vector_sum_sensitivities`.
Tests: `tests/test_mechanisms.py` brute-forces every neighbour of a small dataset
and checks the claimed value is never exceeded and is actually reached.

---

## 3. The Laplace mechanism, giving ε-DP

    M(D) = f(D) + Lap(b),   b = Δ1 / ε,   Lap density p(z) = (1/2b)·exp(−|z|/b)

**Proof (one line of algebra).** Take a 1-D output y. The ratio of densities on
neighbours is

    p_D(y) / p_D′(y) = exp( (|y − f(D′)| − |y − f(D)|) / b )
                     ≤ exp( |f(D) − f(D′)| / b )          (triangle inequality)
                     ≤ exp( Δ1 / b ) = e^ε.

The same holds coordinate by coordinate for vectors, with Δ1 the L1 norm. That
is why Laplace pairs with L1 sensitivity. Integrating over any set S gives the
definition with δ = 0.

**Noise size.** Var[Lap(b)] = 2b², so std = √2·Δ1/ε, and E|noise| = b. The
noise falls as 1/ε, and halving ε doubles it.

Test `test_laplace_privacy_loss_bounded_by_epsilon` evaluates the log-ratio
on a grid and checks it never exceeds ε and reaches ε where the bound is tight.

---

## 4. The Gaussian mechanism, giving (ε, δ)-DP

    M(D) = f(D) + N(0, σ² I),   σ = Δ2 · √(2 ln(1.25/δ)) / ε     (valid for 0 < ε < 1)

**Privacy loss random variable.** Let p = N(0, σ²) (output on D) and
q = N(Δ, σ²) (output on D′, the worst case). For an output y,

    L(y) = ln p(y)/q(y) = ((y − Δ)² − y²) / (2σ²) = (Δ² − 2yΔ) / (2σ²).

L is linear in y, and y ~ N(0, σ²), so L is Gaussian too:

    E[L] = Δ²/(2σ²) =: μ,    Var[L] = (Δ/σ)² = 2μ.

The mechanism is (ε, δ)-DP when the bad event "L > ε" is rare enough. Bounding
that Gaussian tail and solving for σ gives the classical calibration above.
The 1.25 constant and the ε < 1 limit both come from the slack in that tail
bound (Dwork & Roth 2014, Thm A.1). Because the proof needs ε < 1, the code
raises an error instead of silently returning a wrong σ.

**Exact δ.** Balle & Wang (2018) give the tight answer:

    δ(ε) = Φ( Δ/(2σ) − εσ/Δ ) − e^ε · Φ( −Δ/(2σ) − εσ/Δ ).

`gaussian_exact_delta` implements it. The tests check (a) that the closed form
matches a direct numerical integral of max(0, p − e^ε q), and (b) that the
classical σ yields an exact δ ≤ the δ requested, i.e. the classical formula
is conservative.

**Why bother with Gaussian when Laplace gives pure DP?**
1. *High dimensions.* Laplace noise scales with Δ1 and Gaussian with Δ2. For d
   features each in [−1, 1], Δ1 = d but Δ2 = √d. With d = 75 that is 75 against
   about 8.7, so the Gaussian adds much less noise per coordinate.
2. *Composition.* Gaussian noise composes cleanly under Rényi DP, and DP-SGD
   relies on that (§7).

---

## 5. DP statistics used in this repo

**DP mean = NoisySum / NoisyCount** (`dp_mean`). Under add/remove neighbours, n
itself is private, so both parts are noised, each with ε/2:

    mean_DP = c + (Σ(xᵢ − c) + noise_sum) / (n + noise_count)

Write η for the sum noise and ν for the count noise. To first order in 1/n,

    mean_DP − mean ≈ (η − (mean − c)·ν) / n
    RMSE² ≈ ( Var η + (mean − c)² · Var ν ) / n²

The count noise is **not** negligible when the true mean is far from the
midpoint c. I first assumed it was, and the plot exposed the mistake: observed
error ran about 12% above my prediction. Mean heart rate is about 74 bpm and
c = 135, so the count noise is magnified 61 times. With both terms included,
theory and observation agree. For heart rate on all 21,799 records with
ε = 1 (Laplace), the RMSE is about 0.017 bpm. The error falls as 1/(εn):
more data means less noise per person. `scripts/dp_summary_stats.py`
repeats the release 2,000 times per ε and plots observed against predicted
RMSE (`results/dp_stats_error.png`). If the dots sit on the lines, the
sensitivity and the noise scale are both implemented correctly.

**Post-processing is free.** Clipping the noisy mean back into [lo, hi], or
flooring a noisy variance at a small positive number, uses only the DP output
and not the data, so it costs no privacy.

**DP standardiser** (`dp_standardizer`). To feed features to a neural network
we need each feature's mean and std, and computing them exactly would leak.
Steps:
1. Clip each feature to its physiological bounds and map it to z ∈ [−1, 1].
2. Release three things with the Gaussian mechanism, each with ε/3 and δ/3:
   the count (Δ2 = 1), the vector Σz (Δ2 = √d), and the vector Σz² (Δ2 = √d).
3. mean_z = Σz/n and var_z = Σz²/n − mean_z² (the second-moment identity),
   then map back to the original units.

This has a weak spot: features whose spread is small next to their bounds get
noisy variance estimates. That is the cost of choosing bounds without seeing
the data, and README reports how much accuracy it costs.

---

## 6. Composition

* **Basic sequential composition.** Running an (ε₁, δ₁) and an (ε₂, δ₂)
  mechanism on the same data is (ε₁+ε₂, δ₁+δ₂)-DP. Each DP model's total
  budget is therefore ε_standardiser + ε_DP-SGD.
* **Parallel composition.** Mechanisms run on *disjoint* subsets cost only the
  maximum ε. Example: the per-class counts in the summary table use disjoint
  records. The table still reports each release's own ε and does not claim the
  saving.
* Basic composition is loose for many steps. DP-SGD runs about 1,000 noisy steps
  and would be hopeless under basic composition, which is why it uses an accountant.

---

## 7. DP-SGD (Abadi et al., 2016) via Opacus

One training step:
1. **Poisson sampling.** Each record joins the batch independently with
   probability q = B/n (B = 512, n ≈ 17k, so q ≈ 0.03).
2. **Per-example gradients**, each **clipped** to L2 norm ≤ C (C = 1):
   gᵢ ← gᵢ · min(1, C/‖gᵢ‖). Now one record changes the *summed* gradient by
   at most C in L2. That is the sensitivity Δ2 = C.
3. **Gaussian mechanism.** Add N(0, σ²C²I) to the sum and divide by B. σ is
   the "noise multiplier".
4. Take an ordinary optimiser step. It is post-processing, so it is free.

**Accounting.** Each step is a subsampled Gaussian mechanism. Subsampling
*amplifies* privacy: a record that is probably not in the batch leaks less.
The RDP (Rényi DP) accountant adds up the per-step cost over T = epochs·n/B
steps in Rényi space, where Gaussians compose additively, and converts the
result to (ε, δ) at the end. `make_private_with_epsilon` searches for the σ
that lands on the target ε for the given q, T and δ.

**Why the model has no BatchNorm.** BatchNorm mixes statistics across the
batch, so a single example's gradient depends on the other examples, and
per-example clipping no longer bounds one record's influence.

**What utility costs.** Clipping biases gradients. The noise std per
coordinate of the averaged gradient is σC/B, so bigger batches and fewer steps
help, which is why features are standardised before training. Small ε means
large σ, which means noisier gradients and lower accuracy. That is the curve
in `results/privacy_utility.png`.

---

## 8. Honest caveats (be ready to discuss these)

1. **Floating-point attacks.** Textbook Laplace sampling on doubles leaks
   through the gaps between representable numbers (Mironov 2012). This code is
   for learning. Real releases should use OpenDP or Tumult, which use the
   snapping or discrete mechanisms.
2. **Hyperparameter tuning leaks.** Learning rate, epochs, C and batch size
   were fixed by hand before the DP sweep and were not tuned per ε on the test
   fold. Any tuning on private data has a privacy cost that is not counted
   here (Papernot & Steinke 2022).
3. **Bounds.** The feature bounds come from physiology. I sanity-checked them
   against the data's clipping rate, which a strict analysis would have to
   either pay for or avoid.
4. **Record vs patient.** See §1. The guarantee is per ECG record.
5. **Opacus's RNG.** Opacus warns that `secure_mode=False` uses a
   non-cryptographic RNG. That is acceptable for research but not for deployment.

---

## References

* C. Dwork, A. Roth. *The Algorithmic Foundations of Differential Privacy*, 2014. (Laplace, Gaussian, composition)
* B. Balle, Y.-X. Wang. *Improving the Gaussian Mechanism for Differential Privacy*, ICML 2018. (exact δ)
* M. Abadi et al. *Deep Learning with Differential Privacy*, CCS 2016. (DP-SGD, moments accountant)
* I. Mironov. *Rényi Differential Privacy*, CSF 2017; *On Significance of the Least Significant Bits for DP*, CCS 2012.
* A. Yousefpour et al. *Opacus: User-Friendly Differential Privacy Library in PyTorch*, 2021.
* N. Papernot, T. Steinke. *Hyperparameter Tuning with Rényi Differential Privacy*, ICLR 2022.
* P. Wagner et al. *PTB-XL, a large publicly available electrocardiography dataset*, Scientific Data 2020.
* V. Rastogi, S. Nath. *Differentially Private Aggregation of Distributed Time-Series with Transformation and Encryption*, SIGMOD 2010. (planned stretch goal)
