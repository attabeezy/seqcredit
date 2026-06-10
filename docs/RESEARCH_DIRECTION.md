# Research Direction

This note is intentionally lightweight. Current benchmark numbers, ablation
findings, and the active real-data research narrative live in `docs/research.md`.

## Direction

The branch contains both a synthetic benchmark pipeline and a Databricks
real-data pipeline. The current research direction is led by the real-data
workflow.

Near-term work should focus on:

- keeping the real-data pipeline leakage-aware, with features before the index
  loan and labels after it
- validating the recurrent-model direction across reruns and runtime variation
- improving recurrent-model calibration
- deciding whether real-data notebook outputs should become tracked artifacts
  with a manifest
- fixing Notebook C into a clean single rerun notebook for the A+B workflow

## Documentation Rule

Do not duplicate hardcoded benchmark tables or ablation numbers here. Link to
`docs/research.md` for current research facts.
