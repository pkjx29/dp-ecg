# Experiment summary (generated from artifacts/metrics)

Task: PTB-XL NORM vs MI, record-level DP, test fold 10. One seed per configuration.

| run_id | requested_epsilon | achieved_epsilon | noise_multiplier | steps | best_epoch | test_accuracy | test_balanced_accuracy | test_precision | test_recall | test_f1 | test_auroc | threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_seed0 |  |  |  |  | 11 | 0.8639 | 0.8707 | 0.6366 | 0.8828 | 0.7398 | 0.9487 | 0.1944 |
| dp_eps1_seed0 | 1.0000 | 0.9985 | 2.5000 | 444.0000 | 12 | 0.7688 | 0.7803 | 0.4835 | 0.8008 | 0.6029 | 0.8667 | 0.0081 |
| dp_eps2_seed0 | 2.0000 | 1.9910 | 1.4941 | 444.0000 | 11 | 0.7834 | 0.7939 | 0.5036 | 0.8125 | 0.6218 | 0.8746 | 0.0046 |
| dp_eps4_seed0 | 4.0000 | 3.9979 | 1.0083 | 444.0000 | 11 | 0.7594 | 0.7897 | 0.4726 | 0.8438 | 0.6059 | 0.8831 | 0.0027 |
| dp_eps8_seed0 | 8.0000 | 7.9901 | 0.7507 | 444.0000 | 11 | 0.8014 | 0.8026 | 0.5309 | 0.8047 | 0.6398 | 0.8866 | 0.0047 |

## Membership inference (evaluation set: training members vs test-fold non-members)

| run_id | achieved_epsilon | loss_auroc | max_probability_auroc | entropy_auroc | loss_auroc_ci_low | loss_auroc_ci_high | loss_tpr_at_fpr_0.01 | loss_balanced_accuracy |
|---|---|---|---|---|---|---|---|---|
| baseline_seed0 |  | 0.5334 | 0.5278 | 0.5278 | 0.5099 | 0.5551 | 0.0171 | 0.5180 |
| dp_eps1_seed0 | 0.9985 | 0.5019 | 0.4961 | 0.4961 | 0.4794 | 0.5262 | 0.0120 | 0.5017 |
| dp_eps2_seed0 | 1.9910 | 0.5055 | 0.5002 | 0.5002 | 0.4822 | 0.5281 | 0.0094 | 0.5004 |
| dp_eps4_seed0 | 3.9979 | 0.5147 | 0.5073 | 0.5073 | 0.4897 | 0.5392 | 0.0137 | 0.5000 |
| dp_eps8_seed0 | 7.9901 | 0.5210 | 0.5122 | 0.5122 | 0.4979 | 0.5444 | 0.0197 | 0.5034 |
