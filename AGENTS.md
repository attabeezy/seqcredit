# Agent guidance

## Agreed research scope

The user's objective is a simple four-step study:

1. Set generation parameters from the evidence already extracted from Analysis Notebooks A/B/C. Clearly identify assumptions where the evidence is incomplete.
2. Generate one synthetic sample from those parameters and check it against them.
3. Train the static and sequence models on that same sample and borrower splits.
4. Analyze their results: which performs better, by how much, and what the findings mean, including uncertainty and relevant limitations.

This is the governing workflow. Use the existing calibration files as the evidence reference; do not routinely reopen the notebooks. The sample is a synthetic reconstruction of supported characteristics, not recovery of the original records. Let the models compete without forcing a winner or target score.

Keep work proportional to this objective. Address concrete problems directly. Reuse existing checks and outputs; do not add helper scripts, audit layers, reports or repeated runs without a specific need. Report unresolved limitations plainly. Additional controls or follow-up studies are supporting analyses when needed, not automatic prerequisites that restart or indefinitely extend the four-step study. Do not treat a documented limitation as user authorization to launch a new research program. Keep progress updates and final explanations concise.

## Implementation guidance

The user-facing workflow is dataset.py followed by train.py, both with no arguments. Default seed is 42. Do not reintroduce profile/scenario selection or per-user/dataset directory trees. There is one source dataset, data/yara_telecel_dump.csv.gz, containing all generated event rows. Run outputs belong in artifacts/. Tests use temporary datasets.

Use Python 3.13 and .venv/Scripts/python.exe. Dependencies are managed through pyproject.toml and uv.lock. Read README.md and docs/PROJECT.md. Run .venv/Scripts/python.exe -m pytest -q; smoke/pilot integration entrypoints live in tests/.

Generated records must be identified as simulated, never observed or anonymized customer data. A temporal advantage is a hypothesis; do not sort histories by risk, leak outcomes, tune from test metrics or force a required winner/AUC. Freeze data assumptions before running the competition. Keep identical shared histories and borrower splits across models; fit scalers/class weights on inner-train only.

Freeze exactly experiment_design.json, docs/calibration/ledger.json and docs/calibration/category_mapping.json through input_freeze.py. Verify embedded provenance and source-file hashes. Invalid data fails closed; never silently fall back to old caches.

archive/ contains private historical/calibration evidence. Do not delete it, execute its apps, restore dependencies, or publish real samples. Existing extracted evidence is retained in docs/calibration/*.json. Notebook extraction/mapping builders, their tests and the historical synthetic_params.json inventory are retired under bulk-clean/ with their original relative paths. Do not restore them to the normal workflow.

No feature cache or hard runtime budget enforcement is implemented. Keep limitations explicit. Public functions should have type hints. Test meaningful invariants.

The user owns validation of the parameter set and generated sample against it. Use the current parameter/configuration and existing calibration files as the working reference. Do not revisit or re-extract archived notebooks unless explicitly requested. Keep engineering integrity checks separate from scientific parameter validation; never claim user validation has occurred without evidence.

Borrower activity follows the configured shifted Gamma-Poisson distribution, expected mean 150, independent of risk/outcomes. Both model types use all eligible pre-index events; do not reintroduce a sequence cap. Keep real steps packed with offsets/lengths and construct explicit right-padding masks per batch.

The user accepted the current parameter set and sample on 2026-10-24 and explicitly requested freezing both. src/configs/accepted_dataset.json is the acceptance record. Preserve its exact dataset, three input hashes and generator-source hashes during routine work. Never refresh the parameter lock or replace the accepted dataset to make a check pass. Changes require explicit new user authorization, a new audit and renewed acceptance. Main loading/training verifies the frozen data checksum; isolated temporary test workloads remain allowed.

WSL Ubuntu plus the installed google-colab-cli is the primary execution route for most project code. Keep src/colab/ setup, transfer, launch, monitoring and recovery helpers active alongside the CLI wrapper, including required transfer inputs under artifacts/. Do not classify a Colab helper as unused merely because dataset.py or train.py does not import it or its previous session has ended. Check the complete remote execution path before retiring infrastructure. Transfer only required source and the accepted synthetic dataset, verify exact hashes on the remote runtime, and run headline and controls on the same backend. Keep CPU reference runs separate; never choose a backend from its test scores. Recover and verify artifacts locally, then stop only the task-owned Colab session. Never upload private archive material, environments, Git history or credentials.

bulk-clean/ is the retirement folder, excluded from Git and normal tests. Obsolete extraction tools, completed-session logs and superseded CPU runs are preserved there. Keep src/colab/, dataset.py, train.py, their supporting modules, the reporting code, active tests, frozen inputs/data and current Colab research results in the active project. Move retired files with their relative paths instead of creating replacement scripts.
