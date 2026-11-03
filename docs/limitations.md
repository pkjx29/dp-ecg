# Limitations

**Scientific**
* Three seeds per configuration measure training randomness, but there is only one train/test split, so test-set
  sampling uncertainty is not included in the sd.
* Two membership attacks were run: a threshold attack and online LiRA with 16 shadow models per configuration.
  LiRA's shadow models train on half the data (weaker than the full-data target models), and its scores are not
  independent, so no CI is reported. A stronger attacker (more shadow models, augmentations, white-box access) may
  find more. Near-chance results are evidence, not proof; the guarantee is the accountant's ε.
* DP models are poorly calibrated (validation thresholds below 0.01), so probabilities should not be read as risks.
* The task excludes mixed-diagnosis records, and PTB-XL comes from a single centre.
* Feature clipping bounds come from physiology, but I looked at the data's clipping rate while developing the feature
  pipeline. A strict analysis would fix the bounds before seeing any data.

**Privacy**
* The guarantee is record-level, not patient-level.
* Exploratory runs during development (an earlier feature-MLP sweep, visible in the git history) touched the same
  training folds. They are not part of the accounted release.
* Opacus `secure_mode` is off because `torchcsprng` is unavailable for the installed PyTorch. Floating-point Laplace
  sampling is vulnerable to the attack in Mironov (2012). Use OpenDP or Tumult for real releases.

**Engineering**
* The Docker image has not been built (no Docker daemon on the development machine).
* Nothing is deployed.
* Training is CPU-only and was tested on macOS arm64. CI runs on Ubuntu with CPU PyTorch.
