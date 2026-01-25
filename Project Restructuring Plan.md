# Project Restructuring Plan
Comprehensive reorganization of the sequential-crm-for-dce project for improved maintainability and clarity.
## Overview
Reorganize the project structure to follow best practices for research projects with clear separation of data, source code, documentation, and notebooks. All internal references and imports will be updated to reflect the new structure.
## Current Structure Issues
* Notebooks referenced at root but located in notebooks/ folder
* Documentation scattered across experiments/ subdirectories
* Data files not separated by type (real vs synthetic)
* Python scripts buried in experiments/ subdirectories
* Inconsistent folder naming conventions
## Proposed New Structure
```warp-runnable-command
sequential-crm-for-dce/
├── .git/
├── .virtual_documents/
├── data/
│   ├── real/
│   │   ├── transactions.xlsx - Table 1.csv
│   │   ├── transactions.xlsx - Table 5.csv
│   │   ├── engineered_features_real_data.csv
│   │   └── user_level_summary.csv
│   └── synthetic/
│       ├── synthetic_momo_calibrated.csv
│       ├── synthetic_user_profiles.csv
│       └── real_data_calibration.json
├── docs/
│   ├── COMPLETE_ANALYSIS.md
│   └── SYNTHETIC_DATA_GUIDE.md
├── notebooks/
│   ├── credit_risk_prediction_v1a.ipynb
│   ├── credit_risk_prediction_v1b.ipynb
│   ├── credit_risk_prediction_v1c.ipynb
│   ├── ctgan_syn_data_gen.ipynb
│   └── syn_data_gen.ipynb
├── src/
│   ├── feature_engineering/
│   │   └── real_temporal_feature_engineering.py
│   └── data_generation/
│       └── calibrated_synthetic_generator.py
├── .gitignore
├── LICENSE
├── README.md
└── SESSION_LOG.md
```
## Phase 1: Create New Directory Structure
### Tasks
* Create data/real/ directory
* Create data/synthetic/ directory
* Create src/feature_engineering/ directory
* Create src/data_generation/ directory
## Phase 2: Move Data Files
### Real Data Files
* Move data/transactions.xlsx - Table 1.csv → data/real/
* Move data/transactions.xlsx - Table 5.csv → data/real/
* Move experiments/feature_engineering/engineered_features_real_data.csv → data/real/
* Move experiments/feature_engineering/user_level_summary.csv → data/real/
### Synthetic Data Files
* Move data/synthetic-momo-data.csv → data/synthetic/
* Move experiments/multi-user/synthetic_momo_calibrated.csv → data/synthetic/
* Move experiments/multi-user/synthetic_user_profiles.csv → data/synthetic/
* Move experiments/multi-user/real_data_calibration.json → data/synthetic/
## Phase 3: Move Documentation Files
### Documentation
* Move experiments/feature_engineering/COMPLETE_ANALYSIS.md → docs/
* Move experiments/multi-user/SYNTHETIC_DATA_GUIDE.md → docs/
## Phase 4: Move Source Code
### Python Scripts
* Move experiments/feature_engineering/real_temporal_feature_engineering.py → src/feature_engineering/
* Move experiments/multi-user/calibrated_synthetic_generator.py → src/data_generation/
## Phase 5: Rename Notebook for Consistency
### Notebook Naming
* Rename notebooks/syn-data-gen.ipynb → notebooks/syn_data_gen.ipynb (for snake_case consistency)
## Phase 6: Update Internal References
### README.md Updates
* Update Colab badge links from root paths to notebooks/ paths
    * Before: `/blob/main/credit_risk_prediction_v1a.ipynb`
    * After: `/blob/main/notebooks/credit_risk_prediction_v1a.ipynb`
* Update repository structure section to reflect new organization
* Update syn-data-gen.ipynb references to syn_data_gen.ipynb
### SESSION_LOG.md Updates
* Update file path references in Section 9 (Technical Environment)
* Update file structure diagram to match new structure
* Update deliverables file paths in Section 6
* Update quick start commands in Section 19 with new paths
### SYNTHETIC_DATA_GUIDE.md Updates
* Update file path references in "Files Generated" section
* Update code examples to use new import paths:
    * `from src.feature_engineering.real_temporal_feature_engineering import ...`
* Update file loading examples with new data paths
### COMPLETE_ANALYSIS.md Updates
* Update dataset file path references
* Update code import examples to reflect new src/ structure
* Update file structure references
### Python Script Updates
#### src/feature_engineering/real_temporal_feature_engineering.py
* Update any hardcoded data paths to use relative paths
* Add docstring noting new location and import path
#### src/data_generation/calibrated_synthetic_generator.py
* Update calibration file loading path: `data/synthetic/real_data_calibration.json`
* Update output file paths to write to data/synthetic/
* Update import statements if any cross-references exist
### Notebook Updates
#### All Notebooks (v1a, v1b, v1c, ctgan_syn_data_gen, syn_data_gen)
* Update data file paths:
    * Raw data: `../data/real/`
    * Synthetic data: `../data/synthetic/`
* Update import statements:
    * `from src.feature_engineering.real_temporal_feature_engineering import ...`
    * `from src.data_generation.calibrated_synthetic_generator import ...`
* Add sys.path modifications if needed:
```python
import sys
sys.path.append('../src')
```
## Phase 7: Clean Up Empty Directories
### Directory Removal
* Remove experiments/feature_engineering/ (if empty)
* Remove experiments/multi-user/ (if empty)
* Remove experiments/ (if empty)
## Phase 8: Update .gitignore
### Additions
* Ensure data/real/*.csv patterns are appropriate
* Ensure data/synthetic/*.csv patterns are appropriate
* Add src/**pycache**/ patterns
* Add src/*/**pycache**/ patterns
## Phase 9: Create src/**init**.py Files
### Python Package Structure
* Create src/**init**.py (empty or with package info)
* Create src/feature_engineering/**init**.py
* Create src/data_generation/**init**.py
This makes src a proper Python package for cleaner imports
## Phase 10: Verification & Testing
### Validation Steps
* Verify all file references in documentation point to valid paths
* Test import statements in Python scripts
* Verify notebooks can load data from new paths
* Run git status to ensure all moves are tracked
* Check that no broken links exist in README.md
* Verify SESSION_LOG.md file tree matches actual structure
## Implementation Notes
### Git Operations
* Use `git mv` for all file moves to preserve history
* Commit changes in logical groups:
    1. Directory creation
    2. Data file moves
    3. Documentation moves
    4. Source code moves
    5. Reference updates
    6. Cleanup
### Backup Recommendation
* Ensure current state is committed before starting
* Consider creating a backup branch: `git checkout -b pre-restructure-backup`
### Testing After Restructure
* Run sample notebook cells to verify data loading
* Test Python imports from new src/ structure
* Verify all documentation links work
* Check that Colab badges resolve correctly
## Benefits of New Structure
* **Clear separation of concerns**: data, code, docs, notebooks
* **Professional structure**: follows Python project conventions
* **Easier navigation**: logical grouping of related files
* **Better imports**: src/ package structure enables clean imports
* **Scalability**: easy to add new modules or data sources
* **Consistency**: snake_case throughout (folders and files)
* **Documentation consolidation**: all docs in one place
* **Data organization**: clear separation of real vs synthetic data
## Risk Mitigation
* All file moves tracked via git to preserve history
* Systematic updates of references prevent broken links
* Phase-by-phase approach allows rollback at any stage
* Verification phase catches any missed references
