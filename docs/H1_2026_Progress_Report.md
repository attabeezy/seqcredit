# Comprehensive Progress Report: Sequential Credit Risk Modeling (Jan – Apr 2026)

**Project:** Sequential Deep Learning for Credit Risk Modeling in Data-Constrained Environments  
**Period:** January 1, 2026 – April 11, 2026  
**Status:** Framework complete; Preliminary benchmark results finalized.

---

## 1. Executive Summary: The Path to a Rigorous Framework
Over the past four months, the project has evolved from a basic data generation script into a comprehensive research framework. We navigated through critical data integrity challenges and model performance bottlenecks to establish a benchmark that demonstrates the high value of behavioral feature engineering over traditional sequential modeling. Our central finding is that well-engineered static features capture the primary default signal in the Ghanaian mobile money context, significantly outperforming standard sequential architectures.

## 2. Detailed Progression Timeline

### Phase 1: Foundation & Data Synthesis (January – February)
*   **The Problem:** Existing credit risk datasets are either private or lack the temporal granularity of mobile money (MoMo).
*   **The Progression:**
    *   **Modular Architecture (Jan 25):** Restructured the repository from a collection of scripts into a modular package (`src/seqcredit_model`).
    *   **Generator Rewrite (Jan 28):** Re-engineered the `CalibratedMoMoDataGenerator` from a global generator to a per-user simulator, enabling the creation of individualized transaction histories and loan archetypes for 10,000 users.
    *   **Baseline Validation (Feb 17):** Established initial benchmarks using Logistic Regression and a basic LSTM to verify that transaction sequences could be processed end-to-end.

### Phase 2: Debugging, Expansion & The "Imbalance Pivot" (March)
*   **The Problem:** Early models showed poor performance (AUC ~0.53), and label assignments were inconsistent.
*   **The Progression:**
    *   **Label Integrity Fix (Mar 11):** Identified and resolved a critical bug where models were training on outdated summary files. Standardized on `user_labels.csv` as the definitive ground truth.
    *   **Model Diversification (Mar 17):** Expanded the suite from 2 to 6 models, adding **Random Forest, XGBoost, LightGBM, and a Hybrid LSTM.**
    *   **Imbalance Correction (Mar 21):** Applied `class_weight='balanced'` and `scale_pos_weight` to address the 11% default rate. This led to a major performance jump, with static model AUCs rising from **~0.53 to ~0.88**.

### Phase 3: Interpretability, Calibration & Trust (Late March – Early April)
*   **The Problem:** High AUC alone is insufficient for credit risk; we needed to know *why* models predict default and if their probabilities are reliable.
*   **The Progression:**
    *   **XAI Integration (Mar 18):** Implemented SHAP analysis and Permutation Importance. Identified that **Behavioral Diversity** and **Loan History** are the primary drivers of risk.
    *   **Quantified Ablation (Apr 9):** Completed feature-group ablation studies. Dropping `behavioural_diversity` caused the largest AUC drop (-0.09), confirming its role as the dominant signal.
    *   **Calibration Layer (Mar 30):** Introduced reliability diagrams. Found that while the Hybrid LSTM is discriminative, Tree Ensembles (RF/XGB) provide more reliable probability scores for operational use.

### Phase 4: Scientific Rigor & Benchmark Automation (April)
*   **The Problem:** Results needed statistical validation to support publication claims.
*   **The Progression:**
    *   **Benchmark Automation (Apr 2):** Transitioned to dedicated evaluation scripts (`run_cv_benchmark.py`) using **5-Fold Stratified Cross-Validation**.
    *   **Statistical Significance (Apr 7):** Integrated bootstrap significance testing to calculate 95% Confidence Intervals, ensuring our model comparisons are statistically robust.
    *   **Conclusion on Sequences:** Standard sequential models (LSTMs) consistently underperformed compared to well-engineered static features, suggesting that aggregated behavior is more predictive of default than raw transaction order in this benchmark.

## 3. Technical Evolution: Codebase Maturity
| Feature | January State | April State |
| :--- | :--- | :--- |
| **Data Source** | Raw CSVs | Calibrated simulator with 10k users and 5 archetypes. |
| **Model Suite** | 2 models (LR, LSTM) | 6 models (including Hybrid LSTM). |
| **Validation** | Single-split (train/test) | 5-Fold CV + Bootstrap Significance Tests. |
| **Explainability** | None | SHAP, Surrogate Trees, and Feature Group Ablation. |
| **Performance** | AUC ~0.53 (Random) | AUC ~0.88 (Tree Ensembles) / ~0.85 (Hybrid LSTM). |

## 4. Key Performance Benchmarks (Finalized)
| Model | Target: Default (`y_default`) | Target: Late/Bad (`y_bad`) |
| :--- | :--- | :--- |
| **Random Forest** | **0.884 ± 0.014** | **0.727 ± 0.017** |
| **Logistic Regression**| 0.914 ± 0.007 | 0.725 ± 0.021 |
| **Hybrid LSTM** | 0.850 ± 0.033 | 0.705 ± 0.018 |
| **Standalone LSTM** | 0.523 ± 0.041 | 0.533 ± 0.015 |

**Core Conclusion:** Aggregated behavioral features (specifically diversity and history) capture the relevant predictive signal; the temporal order extracted by standard LSTMs adds negligible value for default prediction in this environment.

## 5. Strategic Next Steps
1.  **Manuscript Finalization (Apr 15):** Finalize the abstract for Deep Learning Indaba 2026 highlighting the "Static vs. Sequence" findings.
2.  **External Validation (Paper B):** Prepare the framework for ingestion of real Telecel Ghana transaction logs to verify if these synthetic findings hold in production.
3.  **Attention Mechanisms:** Future work will investigate if advanced attention-based sequential models can extract signals that LSTMs cannot.
