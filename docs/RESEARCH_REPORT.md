# Static and temporal credit-risk models on the frozen synthetic sample

## Research question and scope

The project follows four steps: set parameters from the evidence already extracted from Analysis Notebooks A/B/C, identify necessary assumptions, generate and check one synthetic sample, then train static and sequence models on shared borrower splits and analyze their results. This report records the training and analysis stages of that workflow.

The comparison asks which model performs better at future nonrepayment prediction, by how much, and what the differences mean within the simulated sample. The order-shuffle and label-permutation analyses support interpretation of the recorded results; neither a GRU advantage nor a target score is required.

This is a controlled simulation study. The records are generated, not observed or anonymized Telecel customer transactions. The user accepted the parameter set and generated sample on 24 October 2026; acceptance concerns those simulation assumptions and sample checks, not model performance or deployment suitability.

Average precision (AP) is the primary metric specified by the frozen contract; ROC-AUC is secondary. Brier score, log loss and fixed-0.5 threshold metrics provide separate information about probability quality and classification. No minimum score or required winner determines acceptance.

The current headline workflow includes validation-fitted probability calibration. Its verified follow-up reduces test Brier scores to 0.0390 (logistic regression), 0.0392 (LightGBM) and 0.0384 (GRU), with unchanged discrimination. The [calibration section](#validation-fitted-calibration-follow-up) records the full results and limitations. Earlier tables below retain the original raw comparison and control results, with probability scores explicitly labelled as uncalibrated.

## Data and fixed design

The accepted sample contains 25,625 synthetic borrowers, 3,921,503 event rows and 3,855,443 background events. Every borrower has one designated index loan. Histories cover the 90 days strictly before the index; the primary label is absence of simulated principal or interest repayment during the following 60 days. Labels are derived from the recorded future lifecycle events, not copied from a simulator-private state. Future events, the index-loan amount, identifiers and latent risk are excluded from model features.

Borrower activity is heterogeneous: lambda is sampled from Gamma(shape=2, scale=74), and background count is 2 + Poisson(lambda), giving expected mean 150. The accepted sample averages 150.4563 events, with range 2–929. Every eligible event is retained. Static models use 18 aggregate features; the GRU uses 12 event-level features. Sequences are packed with offsets and lengths, then right-padded within each batch with explicit masks.

The amount level and continuous time trend share a private latent risk factor with the outcome process, with independent borrower-level, trend and transaction noise. These relationships are explicit simulation assumptions. Histories are not sorted by labels or risk, and no outcome-dependent payload permutation encodes a required winner. Historical aggregate evidence supports outcome reference counts and mapped transaction-type shares; it does not establish the assumed borrower-level amount, activity, timing or history/outcome relationships. The [parameter review](PROJECT.md#parameter-review-against-notebooks-abc) documents that distinction. No private notebook material was revisited for this analysis.

The single accepted source is `data/yara_telecel_dump.csv.gz`, 92,503,821 bytes, SHA-256 `6ec8bfc5a9e5a78f284624c4ed355aff163f42490cf2a49bf2142bdb5b30a068`. The [acceptance record](../src/configs/accepted_dataset.json) binds this checksum to the three frozen parameter inputs, their lock, the accepted audit and generator-source hashes. Neither parameters nor dataset bytes were changed for this comparison.

Borrowers are stratified on the realized three-class label into train/validation/test sizes 15,375/5,125/5,125. An additional stratified inner split leaves 13,068 borrowers for fitting and 2,307 for early stopping. Scalers and balanced class weights are fitted only on inner-fit borrowers. The outer validation and test sets neither fit preprocessing nor stop training. The final models are not refitted on train plus validation. Test data contain 210 nonrepayments (4.0976%).

## Models, interventions and uncertainty

All three models use fixed settings and identical borrower partitions. Logistic regression uses balanced class weights, maximum 1,000 iterations and seed 42. LightGBM uses up to 500 trees, learning rate 0.03, 31 leaves, balanced weights, two CPU threads and seed 42, with early stopping on the inner-stop set. The GRU has 64 hidden units, a sigmoid output, Adam learning rate 0.0005, batch size 64, maximum 20 epochs, patience 5 and minimum validation-loss improvement 0.0001. GRU training uses balanced sample weights; early stopping monitors unweighted inner-stop log loss and restores the best weights. Its seed is 42 and TensorFlow deterministic operations are enabled.

Three complete experiments are compared:

1. Headline: original histories and labels.
2. Training-label permutation: permute only inner-fit labels with the fixed named random stream. Preserve label prevalence, features and all evaluation labels. Early stopping still uses genuine inner-stop labels, as specified by the fixed policy.
3. Order shuffle: independently permute each borrower's payload columns across that borrower's fixed timestamps/gaps, in both training and evaluation, and refit the same models. This preserves every event payload and static input but disrupts their association with chronological position. It is a separate training experiment, not just a test-time perturbation.

The headline and both controls run in the same Google Colab T4 environment through WSL and google-colab-cli. A prior Windows CPU headline and an interrupted CPU label-control run are retained separately as reference evidence. The user requested Colab before GPU results; backend choice was not selected by scores, and CPU/GPU results are not pooled. The GPU execution plan was written after that CPU headline had been observed. This is therefore not a formally preregistered study; the data contract and model settings remained fixed, and the backend-switch chronology is preserved.

Uncertainty uses 1,000 paired borrower bootstrap draws with seed 42 from the locked test predictions. Each draw uses identical sampled borrower indices across all models and interventions. Reported 95% intervals are percentile intervals; paired differences are calculated within each draw, rather than inferred from overlap of separate model intervals. These are nominal, conditional intervals for the fitted models and this synthetic population. They exclude retraining, random-permutation, alternative generator and real-population uncertainty; there is no familywise error adjustment. An interval spanning zero does not establish a clear paired difference, and a negative-control fit is not a permutation-test null distribution.

## Integrity verification

Each experiment independently rereads the accepted events, derives labels and borrower splits, rebuilds preprocessing from inner-fit borrowers, reconstructs transformed inputs and intervention rules, and recalculates metrics. Saved model reloads must reproduce their predictions. SHA-256 manifests bind recorded artifacts to the executed source and accepted data. Engineering validation passing does not by itself establish an effect or resolve every negative-control concern.

The recovered results are also checked against the identical frozen local file. Only the remote absolute dataset path is translated; recorded experiment files are left untouched. A second ranking calculation independently reproduces the AP/ROC-AUC point estimates, bootstrap intervals and paired model/order contrasts from the downloaded predictions. The evidence record is [research_integrity.json](../artifacts/research_integrity.json). At comparison time the project test suite passed 47 tests, including historical extraction/mapping checks subsequently retired during workflow cleanup.

## Held-out results

Values below are point estimates with nominal conditional 95% paired borrower bootstrap intervals. All 1,000 draws were usable; no single-class draws were discarded. AP is average precision, not trapezoidal area under the precision–recall curve.

| Headline model | Average precision (primary) | ROC-AUC (secondary) |
| --- | --- | --- |
| Logistic regression | 0.0811 [0.0646, 0.1086] | 0.6358 [0.5961, 0.6766] |
| LightGBM | 0.0614 [0.0470, 0.0880] | 0.5742 [0.5346, 0.6133] |
| GRU | 0.0978 [0.0766, 0.1322] | 0.6787 [0.6403, 0.7173] |

The GRU has the highest point estimates. Its AP is approximately 2.39 times test prevalence (0.0410), a descriptive ranking comparison rather than a calibrated risk ratio. AP differences, rather than this ratio, determine the paired comparisons below.

| Paired difference | Average precision | ROC-AUC |
| --- | --- | --- |
| GRU − logistic regression | +0.0167 [−0.0017, +0.0382] | +0.0428 [+0.0120, +0.0721] |
| GRU − LightGBM | +0.0364 [+0.0145, +0.0641] | +0.1045 [+0.0617, +0.1452] |
| LightGBM − logistic regression | −0.0197 [−0.0387, −0.0019] | −0.0616 [−0.0985, −0.0235] |

The primary-metric GRU versus logistic-regression interval spans zero. Consequently, this analysis does **not** establish a clear AP advantage over the stronger static comparator. The GRU–LightGBM AP contrast and both GRU–static ROC-AUC contrasts have intervals above zero under this conditional analysis. These comparisons describe these fixed fits; they do not establish a general ordering of the algorithms.

![Held-out ROC and precision–recall curves](../artifacts/figures/ranking.png)

## Temporal-order intervention

Order-shuffled GRU AP is 0.0620 [0.0513, 0.0773], and ROC-AUC is 0.6246 [0.5869, 0.6605]. The paired headline-minus-shuffle differences are:

| Contrast | Point difference | 95% interval |
| --- | --- | --- |
| GRU AP, original minus shuffled order | +0.0359 | [+0.0174, +0.0636] |
| GRU ROC-AUC, original minus shuffled order | +0.0540 | [+0.0239, +0.0837] |

Both static models' test probabilities are exactly unchanged by this intervention. Thus the experiment preserves their information and computation while reducing GRU discrimination. This supports a contribution from payload–chronology association in this simulation and fitted training procedure.

The intervention also changes the GRU's training trajectory. Headline and label-permutation runs complete six epochs and restore the first epoch; the order-shuffle run completes eight and restores the third. The contrast compares two refitted pipelines under the same stopping policy, rather than a pure change to the predictions of one fixed network. A single refit and shuffle cannot separate all training variation from the representation effect, and the negative-control concern below limits the strength of the mechanistic claim.

![Paired model and temporal-order differences](../artifacts/figures/paired_differences.png)

## Negative-control finding and its limits

| Training-label permutation | Average precision | ROC-AUC |
| --- | --- | --- |
| Logistic regression | 0.0698 [0.0548, 0.0946] | 0.5941 [0.5534, 0.6362] |
| LightGBM | 0.0420 [0.0350, 0.0557] | 0.4894 [0.4497, 0.5309] |
| GRU | 0.0590 [0.0487, 0.0759] | 0.6024 [0.5652, 0.6414] |

LightGBM is near the constant-ranking reference. Logistic regression and GRU remain above ROC-AUC 0.5, with conditional intervals excluding 0.5. Their AP-minus-constant-reference intervals also exclude zero. These results cannot be dismissed as merely tiny deviations around chance. The logistic-regression headline-minus-label-permutation contrast is itself inconclusive on both ranking metrics. GRU's corresponding AP difference is +0.0388 [+0.0225, +0.0645].

The saved training labels are independently checked against the exact specified inner-fit permutation, and evaluated against unchanged genuine test labels. Features and predictions are checked separately. This rules out an incorrectly applied permutation in those recorded inputs; it is not a general proof of absence of leakage. The [negative-control audit](../artifacts/negative_control_audit.json) preserves label-overlap, feature-correlation and logistic-coefficient diagnostics without additional model fits.

The audit finds 533 positives before and after permutation among 13,068 inner-fit borrowers. There are 28 overlapping positive labels, against expected overlap 21.74 under a uniform permutation; original/permuted label correlation is 0.0122. Small feature-label associations remain after permutation, including correlation +0.0242 with mean amount, and headline/permuted logistic test scores correlate +0.3076. These observations are compatible with a finite permutation retaining predictive direction, but do not establish that explanation or provide a calibrated permutation test.

Two limitations matter. First, one finite random permutation can accidentally align fitted coefficients with predictive feature directions that generalize to genuine held-out outcomes. A test-borrower bootstrap conditions on that fitted direction; it does not quantify randomness over label permutations. Second, LightGBM and GRU still use genuine inner-stop labels, so this control retains an explicit supervision path through early stopping. Logistic regression has no early stopping, so that path does not explain its result. Chance alignment and retained selection supervision are plausible explanations, not established resolutions of the above-chance control.

Accordingly, the negative control is **not clean evidence that all unintended signal has been excluded**. The model ranking and order-intervention results are reproducible conditional findings, while stronger claims about the source of the improvement remain provisional. No new permutation, model setting or seed was selected to obtain a cleaner result.

## Original uncalibrated probability quality and fixed-threshold behavior

The constant reference predicts the inner-fit prevalence, 0.0407867, for every test borrower. It does not use test prevalence to choose its prediction. Lower Brier score and log loss are better.

| Headline model/reference | Brier score | Log loss |
| --- | --- | --- |
| Logistic regression | 0.2230 [0.2194, 0.2266] | 0.6457 [0.6373, 0.6546] |
| LightGBM | 0.0689 [0.0647, 0.0730] | 0.2617 [0.2482, 0.2743] |
| GRU | 0.1538 [0.1500, 0.1574] | 0.4801 [0.4715, 0.4879] |
| Constant inner-fit prevalence | 0.0393 [0.0343, 0.0443] | 0.1710 [0.1538, 0.1883] |

All models perform worse than the constant reference on both proper probability scores. Balanced training weights change the optimization target and can contribute to poor prevalence-scale probabilities. These raw outputs should therefore not be presented as validated absolute nonrepayment risks. The scores alone do not separate calibration error from discrimination and other probability-quality limitations. No test-set calibration or threshold optimization was performed.

| Fixed threshold 0.5 | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Logistic regression | 0.0699 | 0.5429 | 0.1238 |
| LightGBM | 0.0811 | 0.0571 | 0.0670 |
| GRU | 0.0831 | 0.4000 | 0.1376 |

These are descriptive threshold results. The threshold was fixed before evaluation and has no demonstrated operational cost or credit-policy justification.

## Computation and reproducibility

Execution uses a Tesla T4 with TensorFlow 2.21.0, Keras 3.15.0, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.8.0, LightGBM 4.6.0, joblib 1.5.3 and Python 3.13.16. The [environment record](../artifacts/colab_environment.json) includes deterministic-operation flags and the GPU preflight. The remote Git identifier denotes an ephemeral source-only execution snapshot, not the original repository history. No private archive, credentials or local environment was uploaded.

| Experiment | Training/evaluation pipeline seconds | GRU fit seconds | Peak process RAM (GiB) |
| --- | --- | --- | --- |
| Headline | 115.93 | 30.59 | 1.543 |
| Label permutation | 115.79 | 30.58 | 1.545 |
| Order shuffle | 125.72 | 39.22 | 1.543 |

These pipeline times exclude subsequent independent validation and bootstrap calculations. Peak RAM is the process's measured host-memory maximum; it is not peak GPU memory or a hard runtime memory limit. Each experiment stores approximately 62.4 MB before its manifest. The complete recovered archive is 187,422,720 bytes, verified across all 45 transport parts and as a whole. Its SHA-256 is `e1b29673ce64ce6c77a64ce7a3af464ab79f154451737e84604b80ca93e07d9f`.

After the recovered artifact manifests and accepted dataset were verified locally, the task-owned Colab session was terminated. The CLI then confirmed that no active sessions remained. Local reporting and checks do not require a running remote runtime.

The source-data workflow remains `python dataset.py`, then `python train.py`. The frozen comparison's orchestration and analysis implementation is preserved in [evaluate_comparison.py](../artifacts/evaluate_comparison.py), with its checksum recorded in the results. `src/scripts/research_report.py` revalidates that historical comparison and regenerates its research figures without fitting models; its source checks require the matching historical training implementation, rather than the revised calibration-integrated source. Figure rendering additionally uses the installed Matplotlib, whose version and output hashes are recorded in [research_figure_manifest.json](../artifacts/research_figure_manifest.json). PNG and editable SVG versions are preserved.

Evidence files:

Superseded CPU runs and completed-session logs were subsequently moved to bulk-clean/ with their original relative paths. Colab execution helpers remain active in src/colab/. Current Colab experiment directories and the research evidence below remain under artifacts/.

- [Machine-readable comparison](../artifacts/comparison_results.json), [execution plan](../artifacts/comparison_plan.json), [compact numerical report](../artifacts/comparison_report.md).
- [Local integrity verification](../artifacts/research_integrity.json), [negative-control diagnostics](../artifacts/negative_control_audit.json), [transport recovery receipt](../artifacts/colab_recovery.json).
- [Headline run](../artifacts/20261024T234056Z-52a5354f/manifest.json), [label-permutation run](../artifacts/20261024T234056Z-52a5354f-label-permutation/manifest.json), [order-shuffle run](../artifacts/20261024T234056Z-52a5354f-order-shuffle/manifest.json). Each directory includes predictions, metrics, validation, model inputs, fitted models and source/input checksums.

## Validation-fitted calibration follow-up

On 25 October 2026, a fresh Colab headline run reproduced the accepted event file byte for byte and all original validation/test predictions exactly. A separate sigmoid-calibration analysis then used that run's 5,125 validation borrowers (209 nonrepayments). Base models and the accepted dataset were unchanged. Each model used an unweighted, unpenalized logistic sigmoid on clipped logit scores, with five-fold out-of-fold validation assessment and final coefficients fitted on all validation borrowers. No calibration method or threshold was selected from test results.

| Model | Test Brier, raw → calibrated | Test log loss, raw → calibrated | Test ECE, raw → calibrated |
| --- | --- | --- | --- |
| Logistic regression | 0.223041 → 0.039005 | 0.645747 → 0.166359 | 0.414960 → 0.000962 |
| LightGBM | 0.068903 → 0.039171 | 0.261750 → 0.169693 | 0.113673 → 0.001037 |
| GRU | 0.153776 → 0.038437 | 0.480131 → 0.162220 | 0.311260 → 0.002392 |

All models improved Brier, log loss and ECE in out-of-fold validation as well. All six nominal 95% paired calibrated-minus-raw test Brier/log-loss intervals were below zero (1,000 draws, seed 42). The constant inner-training-prevalence reference has test Brier 0.039297 and log loss 0.171033: gains over this baseline are much smaller than gains over raw probabilities, and no interval establishing superiority to that baseline was computed. The positive fitted slopes preserved test AP and ROC-AUC exactly. Mean calibrated probabilities were approximately 4.15%, close to the 4.10% test event rate.

ECE uses ten equal-width bins; most calibrated predictions occupy the lowest bin, so the small aggregate errors do not establish uniformly accurate tail probabilities. At the unchanged threshold of 0.5, logistic regression flags two false positives; LightGBM and GRU flag none. Threshold selection remains a separate decision. These are exploratory results from a previously inspected synthetic test set; the bootstrap conditions on fixed base models and calibrators. Calibration does not resolve the negative-control concern or validate the assumed data distribution.

The [calibration report](../artifacts/fresh-colab-20261025/calibration/RESULTS.md) includes reliability plots, coefficients, prediction files and verification evidence. Normal `train.py` headline runs now produce calibration outputs automatically, include them in the run manifest, independently reconstruct them during validation, and show calibrated probability scores in the console. Raw comparison/control files remain available with their original schema. This adds a documented post-hoc probability step without rewriting the frozen generator contract or historical experiment artifacts.

## Research conclusion and next work

For this accepted simulated population and these fixed fits, GRU has the strongest point discrimination and loses AP/ROC-AUC when payload chronology is shuffled. Its primary-metric superiority over logistic regression is not established by the paired interval. The above-chance label-permutation results prevent a clean negative-control conclusion, and poor proper probability scores prevent treating the raw model outputs as validated risk probabilities. This is qualified evidence of temporal information in a specified simulator, not proof of a generally superior architecture or evidence of deployment performance.

The calibration follow-up substantially corrects the raw probability scale and modestly improves point probability scores over the constant reference, while preserving ranking. It does not strengthen the original temporal-order or negative-control claims. The four-step study can be reported with these limitations; they do not automatically require restarting it or launching more experiments. Additional follow-up work should use development data for selection and preserve the current results. The inspected test set cannot support an unqualified new confirmatory claim after tuning. Claims about an observed population would require appropriately governed observed data and external evaluation. The accepted simulator parameters and dataset stay frozen throughout this report.
