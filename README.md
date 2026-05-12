# P-SmokeNet: Temporal Fusion Network for Surgical Smoke Onset Prediction in MIS*
By Fathima Afrah Imran
20232278 | w2084807

This folder contains the implementation, preprocessing pipeline, experimental artifacts, and demo interface developed for the project.

## Note on omitted large files
Some very large files are **not included in this folder** due to size constraints.

These omitted files include:
- the full **Cholec80 dataset** (**80+ GB**) 
- the full precomputed **embedding cache** (**30+ GB**)
- some large generated outputs such as extended experiment outputs produced during repeated runs.

All omitted artifacts are **reproducible from the provided notebooks**:
- `preprocessing.ipynb`  
  Used for dataset preparation, label processing, clip construction, and feature cache generation steps.

- `main.ipynb`  
  Used for model training, ablation experiments, evaluation, and result generation.


## Project structure
### Root files
- `main.ipynb`  
- `preprocessing.ipynb`  

## Folder overview
### `artifacts/`
Stores generated outputs and experiment products.

#### `artifacts/common/`
General shared metadata and inventories:
- `video_inventory.csv`
- `smoke_segment_inventory.csv`
- `feature_cache_summary.csv`

#### `artifacts/embeddings/`
Stores precomputed spatial embeddings used by the model.  
Only a **small sample subset** is included here for backend use.

#### `artifacts/features/physics/`
Stores precomputed physics-guided features.  
Only a **small sample subset** is included here for backend use.

#### `artifacts/figures/`
Contains plots and visual outputs used during preprocessing, analysis, and evaluation, including:
- dataset distributions
- target balance plots
- label timelines
- alignment visualizations
- prediction curves

#### `artifacts/indices/`
Contains generated clip indices, labels, splits, and config files for different forecasting window settings, including:
- `clips_index.csv`
- `label_summary.csv`
- `splits.json`
- `config.json`
- per-video `.npz` label files

#### `artifacts/models/`
Saved trained model checkpoints for:
- horizon experiments
- backbone comparisons
- temporal architecture comparisons
- ablation studies

#### `artifacts/results/`
Contains experiment outputs including:
- validation and test predictions
- PR curves
- threshold search outputs
- event-level evaluation
- training history
- summaries for each model run
- comparison CSVs for ablations and experiments

#### `artifacts/logs/`
Reserved for run logs.

#### `artifacts/temp/`
Temporary intermediate files.

### `backend/`
Backend service for the demo system.

Contents include:
- `main.py` – application entry point
- `modeling.py` – model loading/inference logic
- `repository.py` – data access helpers
- `schemas.py` – request/response schemas
- `config.py` – configuration
- `requirements.txt` – backend dependencies
- `demo_videos/` – small sample demo videos included for interface demonstration

This folder supports the interactive prototype shown in the UI.

### `frontend/`
Frontend for the web-based demonstration interface.

Contents include:
- `index.html`
- `styles.css`
- `js/` modules for:
  - API integration
  - charts
  - formatting
  - routing
  - state management
  - UI rendering
  - page logic (`demo.js`, `technical.js`, `videos.js`)

This is the presentation layer for demonstrating predictions and outputs.

### `datasets/`
Contains dataset-related metadata included in the submission.

#### `datasets/Smoke_Cholec80/`
- smoke annotation mappings
- per-video CSV labels
- dataset index files
- readme/mapping files