# Direction for the comparative research paper

Discussion note, 25 October 2026. This records the intended paper framing and an opinion on the reference paper's methodological approach. It does not change the accepted experiment or authorize additional experiments.

## Reference and overall opinion

Reference: *A Comparative Analysis of Few-Shot Learning Models for Text-Based Diagnosis of Crop Pests and Diseases in Ghana*, Nana Benyin E. N Essel-Biney and Isaac K. Y Aboah. Local source: `C:/Users/Dell/Documents/KNUST/YARA/88_A_Comparative_Analysis_of_F.pdf`. The methodological discussion is in Section 4, pages 3-4; the interpretation and limitations are in Sections 5-8.

This is a useful model for the structure and argument of our finished paper. It presents a complete research story: a practical problem, documented data preparation, comparison of three approaches, evaluation from several perspectives, and an explanation of what the findings mean.

One particularly useful feature is its willingness to report that the preferred approach did not achieve the highest classification accuracy. Our conclusions should likewise follow the evidence rather than require a particular architecture to win.

## How the structure maps to our study

| Element | Our study |
| --- | --- |
| Research question | How do static and temporal models compare for simulated nonrepayment prediction? |
| Data and provenance | Notebook-supported characteristics, explicit simulation assumptions, and a frozen accepted sample |
| Comparative methodology | Logistic regression, LightGBM and GRU using identical borrower partitions and eligible histories |
| Evaluation | Average precision, ROC-AUC, Brier score, log loss, ECE and paired uncertainty |
| Supporting analyses | Temporal-order and label-permutation controls; raw versus validation-calibrated probabilities |
| Discussion | Where temporal information helps, what calibration corrects, and what remains unresolved |

The paper should explain the full workflow clearly: establish parameters and their provenance, generate and check one sample, compare models fairly, and interpret the results with uncertainty and limitations. Existing checks and results should support this narrative without turning the paper into an inventory of implementation details.

## Methodological lessons to adopt

- **Separate different kinds of performance.** The reference distinguishes classification, ranking, calibration and representation quality. For our task, good ranking, accurate probabilities and useful decisions are separate achievements. Calibration provides concrete evidence of that distinction.
- **Use meaningful baselines.** Static models are substantive competitors. The constant inner-training-prevalence predictor is also necessary context for probability-quality results.
- **Document fair comparisons.** Explain shared borrower splits and histories, inner-training preprocessing and class weights, inner early stopping, and validation-only calibration. Test outcomes must not select generator assumptions or model settings.
- **Report unexpected and qualified findings.** Moderate scores, inconclusive differences and unresolved controls belong in the results and discussion.
- **Connect results to the research question.** Each metric or supporting analysis should answer a specific question rather than merely enlarge the results table.

## Methodological cautions in the reference

The reference should guide presentation without being treated as an unquestionable methodological standard.

Its few-shot evaluation covers 66 classes, whereas the full-data evaluation covers 100. Differences across these regimes therefore cannot be attributed solely to training-data quantity: the evaluated task and test population also change. Our comparisons should keep the evaluated population and target consistent when attributing differences to a modelling choice.

Some claims about semantic robustness describe plausible explanations or expected deployment benefits rather than directly tested performance under paraphrase or distribution shift. Our paper should distinguish measured findings from proposed explanations and future capabilities.

The reference also acknowledges single-seed evaluation. Our paired borrower bootstrap quantifies uncertainty conditional on the fitted models and this sample; it does not replace repeated-training or generator-seed uncertainty. That limitation should remain explicit rather than automatically trigger more experiments.

## The essential difference in data evidence

The reference's underlying records are observed plant-clinic consultations, although its natural-language descriptions are partly constructed. Our evaluated records are synthetic, and important borrower-level distributions and history-outcome relationships are specified by assumption.

Our study can support a credible controlled comparison, but its title, abstract, methodology and conclusions must make that boundary clear. Notebook-derived outcome proportions and mapped transaction shares do not establish the entire synthetic joint distribution. Model performance and successful calibration do not validate those assumptions against real customers.

The accepted sample remains fixed because it defines the agreed experimental setting. Disappointing performance is not a reason to change it. Any future empirical recalibration of the generator would constitute a separately justified revision requiring renewed acceptance.

## Proposed central argument

> Temporal modelling offers stronger point discrimination in this controlled simulation, while validation-based calibration substantially improves probability estimates. The gains and their uncertainty depend on the comparator, and neither result alone establishes real-world lending performance.

This argument must retain the qualifications in the [research report](RESEARCH_REPORT.md): the GRU's average-precision advantage over logistic regression is not established by the paired interval; the negative-control concern remains unresolved; and calibrated probability-score gains over the constant reference are modest. Calibration is an exploratory follow-up because test results had already been inspected.

The goal is a clear question, fair comparison, reproducible evidence and conclusions proportional to that evidence. A dramatic winner or a deployment claim is not required for a substantive paper.

## Relationship to current project documents

- [Project methodology](PROJECT.md): data provenance, accepted assumptions, splits, model fitting and calibration workflow.
- [Research report](RESEARCH_REPORT.md): numerical findings, supporting controls, calibration results and limitations.
- This note: intended paper structure and interpretation of the reference as a writing and methodological example.

This direction does not require importing the reference's few-shot design, adding model families, changing the data distribution, or building a deployment system. Those would be separate scope decisions.
