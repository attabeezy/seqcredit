# From Black-Box to Publishable: XAI + Reliable ML for Synthetic Credit Risk Sequences

## Research goal (one unifying question)
**RQ:** *In a data-constrained, fully synthetic mobile money setting, can sequential credit risk models outperform strong static baselines **and** produce outputs that are interpretable, auditable, and reliable enough for decision-making?*

This project is not only about maximizing AUC; it is about building a complete **risk modeling framework**: prediction + explanation + probability reliability + (carefully bounded) actionability.

---

## Outcomes / targets (grounded in this repo’s labels)
The dataset defines `credit_risk_label` as:
- `-1`: non-borrower (excluded from training)
- `0`: good (repaid on time)
- `1`: late (≥1 loan repaid after 30-day term)
- `2`: default (≥1 loan not repaid)

**Primary task (already implemented):**
- **Default prediction:** `y_default = 1[label==2] else 0`

**Recommended secondary task (improves publishability):**
- **Bad outcome prediction:** `y_bad = 1[label in {1,2}] else 0`  
This makes “late payers” a first-class outcome instead of hiding them inside “non-default.”

---

## The five ideas as one pipeline (What → Why → Trust → Rules → Action)

### 1) Benchmark performance (the “What”)
Train and compare:
- Static models: Logistic Regression, Random Forest, XGBoost/LightGBM
- Sequential models: LSTM
- Hybrid model: Hybrid LSTM (sequence + static)

Report discrimination metrics:
- ROC-AUC, PR-AUC, F1, precision/recall  
Do this for both `y_default` and (recommended) `y_bad`.

**Paper contribution:** establishes whether sequential information adds value over static baselines under data constraints.

---

### 2) Feature contributions / XAI (the “Why”)
Goal: identify *which behavioral signals* drive risk predictions.

Implement:
- Permutation importance (model-agnostic baseline)
- SHAP for tree models (best-practice global + local explanations)
- Logistic regression coefficients (directional baseline)

**Deliverables:**
- Top-k feature importance tables (default vs bad outcome)
- SHAP summary plot for best tree model
- Short interpretation: what features matter for “late” vs “default”

**Paper contribution:** turns the work from “I built a model” into “I learned which behaviors the model relies on (under the generator).”

---

### 3) Surrogate decision tree (auditable global rules)
Goal: provide an interpretable approximation of your best black-box model.

Procedure:
1. Train the best-performing teacher model (often HybridLSTM or LGBM).
2. Compute teacher probabilities `p_teacher`.
3. Train a shallow decision tree to predict `p_teacher` from input features.
4. Measure **fidelity vs depth** (interpretability–accuracy tradeoff).

**Deliverables:**
- Fidelity curve (depth vs MAE/R² to teacher)
- A small tree plot + extracted human-readable rules

**Paper contribution:** creates an auditable “global explanation” that reviewers and practitioners can inspect.

---

### 4) Model calibration (the “Trust”)
Goal: ensure predicted probabilities correspond to real frequencies.

Implement:
- Reliability diagrams (calibration curves)
- Brier score (and optionally ECE)
- Post-hoc calibration: Platt scaling / isotonic regression (calibration split only)

**Deliverables:**
- Before/after calibration plots for the best models
- Table: AUC alongside Brier/ECE

**Paper contribution:** shows the model is not just good at ranking but also suitable for threshold-based decisions.

---

### 5) Causal inference (the “Action,” with strict boundaries)
Causal inference is about *interventions*, not just prediction.

**With synthetic-only data, this is publishable only if you do one of:**
- **Conservative route (recommended if you don’t change the generator):**
  - Include a clear statement: “We do not claim causal effects; XAI explains model behavior, not real-world causation.”
- **Stronger route (requires generator extension):**
  - Add an explicit simulated “treatment” (e.g., fee subsidy, limit change, reminders) and encode its effect in the generator.
  - Then test whether causal estimators recover the known effect.

**Paper contribution:** either (a) scientific rigor through clear limits, or (b) a causal simulation study if you extend the generator.

---

## Unifying claim (what the paper becomes)
This research is a **complete credit risk modeling framework** for data-constrained sequential data:

1. **Performance:** sequential vs static vs hybrid benchmarking  
2. **Explanation:** feature attribution (SHAP/permutation)  
3. **Auditability:** surrogate decision tree rules + fidelity  
4. **Reliability:** calibration and probability trustworthiness  
5. **Action boundary:** explicit predictive vs causal distinction (or causal simulation extension)

---

## Minimal “publishable” deliverables checklist
- [x] Results tables for both tasks (`y_default`, `y_bad`) across all models
- [x] SHAP summary + global importance table (default vs bad outcome)
- [x] Surrogate tree rule set + fidelity vs depth plot
- [x] Calibration curves + Brier score (before/after calibration)
- [x] Clear limitations: synthetic-only + predictive-not-causal
