# Trencher Predictive Modeling

A reproducible TinyMLOps pipeline for predicting subsea cable trencher load peaks, developed for the study *"Explainable TinyMLOps for Safety-Critical Cyber-Physical Systems"* (submitted to IEEE TSE).

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Configuration

All pipeline parameters are defined in `config.yaml`. Editing this file reconfigures features and the model without touching code. Key sections:

- **paths**: input/output directories
- **importer**: OSIsoft PI and PostgreSQL connection parameters
- **cleaner**: KP column name and prefix stripping
- **constructor**: soil layers, instruments, file filters
- **peak_definer**: rolling-window sizes and σ multipliers for load-peak labelling
- **model**: feature aggregation schema (`X_agg`, `y_agg`) and model hyperparameters

Credentials are read from `local_config.json` (not versioned). Create it at the repository root:

```json
{
  "osisoft_pi_username": "...",
  "osisoft_pi_password": "...",
  "D398": {
    "user": "...",
    "password": "...",
    "host": "...",
    "port": 5432,
    "database": "..."
  }
}
```

## Pipeline Steps

### 1. Data Import

Requires access to the OSIsoft PI historian and/or the PostgreSQL database.

```bash
trencher-import --source all
# or: python -m trencher.scripts.run_import --source pi
```

### 2. Data Cleaning

Finds the longest monotonic KP segment per cable and strips sensor-name prefixes.

```bash
trencher-pipeline --skip-construct --skip-aggregate
```

### 3. Feature Construction

Merges sensor data with soil-layer data, computes layer-proportion and boulder-percentage features, and labels load peaks.

```bash
trencher-pipeline --skip-clean --skip-aggregate
```

### 4. ML Pipeline

Trains L1-regularised logistic regression with interaction features using rolling temporal cross-validation.

```bash
trencher-pipeline --skip-clean --skip-construct --cable-count 100
```

### 5. Analysis

Generates descriptive plots, correlation analysis, RQ2 boxplots, and South Fork validation figures.

```bash
trencher-analysis --analysis all
```

## Data Availability

The sensor and soil-layer data are proprietary to DEME Group. The code is released to ensure full reproducibility of the pipeline logic, feature engineering, and modelling methodology. Researchers with access to analogous operational data can apply the pipeline by adapting `config.yaml`.

## Project Structure

```
trencherThesisPythonFilesExtract/
├── config.yaml              # Pipeline configuration (edit this)
├── requirements.txt
├── pyproject.toml
├── src/trencher/
│   ├── config.py            # Configuration loader
│   ├── data/
│   │   ├── importer.py      # PI & PostgreSQL import
│   │   ├── cleaner.py       # Monotonic KP trimming
│   │   ├── constructor.py   # Soil-layer proportions + peak labelling
│   │   └── aggregator.py    # Rolling-window aggregation
│   ├── model/
│   │   ├── pipeline.py      # ML pipeline (Pipeline class)
│   │   └── peak_definer.py  # Multi-window load-peak labelling
│   ├── analysis/
│   │   ├── descriptive.py   # Descriptive and case-study plots
│   │   ├── correlation.py   # Per-cable correlation analysis
│   │   ├── rq2.py           # RQ2 boxplots (peak vs no-peak)
│   │   └── southfork.py     # South Fork validation plots
│   └── scripts/             # CLI entry points
├── obsolete/                # Original flat scripts (preserved for reference)
└── README.md
```
