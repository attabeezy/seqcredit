# Exploratory repeated-permutation control plan

This plan addresses the above-chance GRU and logistic-regression negative controls in the accepted synthetic-data comparison. It is specified after the original test results were inspected and is exploratory, not a preregistration or a confirmatory test.

## Fixed inputs and comparison

Use the accepted dataset, its frozen inputs and generator, the original borrower splits and all eligible histories. Verify their recorded hashes before and after execution. Fit preprocessing and class weights on inner-fit borrowers only. Use the same T4 backend, environment and deterministic settings for all new fits; preserve the original results separately. Model initialization and minibatch seed remain 42. Do not adjust settings or seeds based on results.

## Conditions

For each of ten permutation seeds 1051 through 1060, permute only the inner-fit binary target with NumPy's `default_rng(seed).permutation`, preserving its class counts. Seed 1051 is the original control. Use the same permutation for each model and stopping policy. Keep inner-stop, validation and test labels unchanged for scoring.

1. Fit the original balanced logistic regression with the permuted inner-fit labels; it has no early stopping.
2. Fit the original GRU with the permuted inner-fit labels and original genuine-label inner-stop policy, patience 5, minimum delta 0.0001 and a maximum of 20 epochs.
3. Fit the same GRU with the same initialization, batches, weights and permuted inner-fit labels for exactly one epoch, without consulting any genuine inner-stop labels. One epoch was the restored epoch of the historical control, so this is a retrospective sensitivity choice, not an independently selected stopping policy.

Also fit one genuine-label headline for each GRU stopping policy and one genuine-label logistic regression on the same backend, using identical splits and initialization. The original genuine-label policy is retained for the primary comparison; the one-epoch headline provides context for the stopping sensitivity. No test labels may influence fitting or stopping. Save borrower-aligned predictions, exact permutation identities, training histories, source and environment hashes, fitted-model artifacts, and failure states for every condition.

## Analysis and interpretation

Report every seed's held-out average precision (primary) and ROC-AUC (secondary), their distributions, and the original seed's position. For each seed report the paired difference between GRU stopping policies; report the one-epoch and original-policy genuine-label headlines separately. Compare the ten permutation results descriptively with the historical control and the constant-prevalence ranking reference. A distribution across ten permutations is a limited diagnostic, not a precise null distribution or proof of no leakage. Do not select or discard seeds based on outcomes; failures remain visible and are investigated.

The original paired GRU-minus-logistic-regression test interval remains conditional on those fitted models. These controls do not measure model-initialization or training-seed variability, establish superiority, validate the simulator against real borrowers, or eliminate all leakage paths. Genuine-label early stopping is a supervisory path in condition 2; any difference from condition 3 is a stopping-policy sensitivity, not an isolated causal estimate. Report unresolved results and do not make an unqualified confirmatory claim from the already inspected test set.
