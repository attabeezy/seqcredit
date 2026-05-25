# Research Direction

This note is intentionally directional. It is not the canonical source for current branch facts; use `docs/PROJECT.md` for that.

## Current Starting Point

The branch already contains:

- a synthetic benchmark pipeline
- a Databricks real-data pipeline
- static, recurrent, hybrid, and transformer-family model implementations

The tracked result artifacts currently support the synthetic benchmark more clearly than they support the real-data path.

## Near-Term Direction

### 1. Reconcile code and tracked benchmark artifacts

The live code surface now includes `GRUModel` and transformer classes, while checked-in benchmark artifacts still reflect an older `HybridLSTM` configuration. A near-term research task is to decide which lineup becomes authoritative and regenerate the tracked benchmark outputs accordingly.

### 2. Keep the real-data pipeline leakage-aware

The strongest architectural idea in the real-data code is the prediction-point split:

- features before the index loan
- labels after the index loan

That design should remain the baseline for any further real-data experiments.

### 3. Treat behavioral diversity as a core hypothesis, not a settled universal law

The checked-in synthetic ablation artifacts strongly support `behavioural_diversity` and `loan_history` as the top groups. Future work should test whether that ranking survives:

- regenerated synthetic benchmarks with the current model lineup
- real-data tracked benchmarks written to repo-tracked artifacts

### 4. Improve the tracked result layer

The repo would benefit from a cleaner result contract:

- one authoritative CSV set for synthetic CV
- one authoritative CSV set for real-data CV
- explicit manifests for model lineup, target definitions, and runtime context

## Working Principle

The next useful research step is not to add more narrative. It is to tighten the connection between:

- current code
- tracked outputs
- documentation claims
