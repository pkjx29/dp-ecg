# Limitations

**Scientific**
* One seed per configuration, so variance is not measured. The differences between ε = 2 and ε = 4 are within
  noise.
* The membership attack is a simple threshold attack. Near-chance AUROC is weak evidence, and stronger attacks
  (LiRA, shadow models) may find more.
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
