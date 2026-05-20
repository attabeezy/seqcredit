# Research Direction: Sequential Credit Risk Modeling

This document formalizes the research strategy, architectural claims, and the scientific trajectory of the `seqcredit_model` project. It serves as a bridge between the foundational findings in Paper A and the empirical scaling of Paper B.

---

## 1. Internal Foundations & Empirical References

The current research direction is grounded in the following validated artifacts:
- **Baseline Results:** Standard sequential architectures (LSTM) yield near-random performance (`AUC-ROC: 0.523`) on this benchmark [Ref: `data/cv_results_y_default.csv`].
- **Feature Importance:** SHAP and permutation importance indicate that transaction volume is secondary to behavioral entropy [Ref: `notebooks/analysis.ipynb`].
- **Ablation Evidence:** The removal of `behavioural_diversity` features results in a ~13.5% drop in AUC, the largest single-group delta observed [Ref: `data/ablation_features.csv`].

---

## 2. Proposed Architectural Novelty: Behavioral Diversity (BD)

Building on the empirical evidence above, this project proposes the **Behavioral Diversity (BD) Framework** as a specialized architecture for credit risk in data-constrained environments.

### 2.1 The Scientific Claim
The BD Framework posits that in emerging fintech markets, **Recipient Entropy** and **Temporal Regularity** are more robust predictors of default than the **Sequential Amount Patterns** traditionally targeted by deep learning models.

### 2.2 Core Contributions (Proposed)
- **Entropy vs. Sequence:** A paradigm shift from learning *when* events occur to modeling *how* users interact with a diverse ecosystem of providers [Ref: `src/seqcredit_model/pipeline.py`].
- **Calibration-First Risk:** Prioritizing **Probability Calibration (ECE)** over raw discrimination, addressing the "trust gap" in automated loan pricing [Ref: `docs/PROJECT.md` Section 6].
- **Sparsity Resilience:** Proving the framework's ability to maintain high discriminative power in "thin-file" contexts (histories < 15 transactions).

---

## 3. The Research Pipeline: From Framework to National Scale

The project follows a two-stage validation pipeline designed to establish both methodology and empirical truth.

### Stage I: The Methodology (Paper A - Indaba 2026)
*   **Research Goal:** Establish the **Ghanaian MoMo Benchmark** as a privacy-safe, high-fidelity proxy for African fintech research.
*   **Key Inquiry:** Can a calibrated synthetic dataset [Ref: `src/synthetic_params.json`] successfully motivate the "Static > Sequence" hypothesis?
*   **Scientific Milestone:** The formalization of the **Temporal Transaction Feature Engine** as a reusable open-source contribution.

### Stage II: National Validation (Paper B - Telecel Ghana)
*   **Research Goal:** Evaluate the **Fidelity Gap** between synthetic frameworks and national-scale telco logs (374M transactions).
*   **Key Inquiry:** How do real-world biases—specifically **Truncation Bias** and **Cold-Start Borrowers**—impact the robustness of the BD Framework?
*   **Scientific Milestone:** The first national-scale proof of the "Behavioral Entropy" lead, targeting a state-of-the-art `AUC-ROC: 0.94` [Ref: `docs/PROJECT.md` Section 10].

---

## 4. Research Frontiers & Open Questions

Future efforts are directed toward resolving the following scientific uncertainties:

### 4.1 The Sequential Bottleneck
*   **Inquiry:** Is the "sequence collapse" (0.52 AUC) an artifact of the LSTM architecture or an inherent property of MoMo data entropy?
*   **Proposed Investigation:** Implementing a **Transformer/Attention Baseline** to verify if self-attention can extract signals that recurrent units cannot.

### 4.2 Real-World Temporal Correction
*   **Inquiry:** How can models distinguish between "true default" and "observation window truncation" in high-growth portfolios?
*   **Proposed Investigation:** Development of the **`min_obs_days` filter** as an architectural necessity for telco-scale logs.

### 4.3 Generalizability & Inclusivity
*   **Inquiry:** Can behavioral entropy serve as a "Proxy History" for the **Zero-History borrower** (documented in 463 Telecel cases)?
*   **Proposed Investigation:** A targeted study on the "Inclusion Frontier" to measure model performance on users with no prior credit or transaction history.

---

## 5. Strategic Context

| Research Dimension | Stage I (Methodology) | Stage II (Empirical) |
| :--- | :--- | :--- |
| **Data Scope** | Calibrated Synthetic (10k Users) | **National Telco Logs (374M Txns)** |
| **Evidence Base** | Paper A / Indaba 2026 | Paper B / Telecel Ghana |
| **Focus** | Tool Design & Hypothesis | **Scalability & Domain Adaptation** |
| **Target Metric** | 0.88 AUC-ROC | **0.94 AUC-ROC (SOTA)** |

