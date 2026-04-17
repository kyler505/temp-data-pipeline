# Temp Data Pipeline

A data pipeline for processing temperature data from NOAA (truth) and Open-Meteo (forecasts), with an integrated evaluation framework for assessing prediction accuracy.

## Quick Start

### 1. Installation

Create/activate a virtual environment, then install:

```bash
pip install -e ".[eval]"
```

### 2. Build canonical datasets (fetch + clean + aggregate + forecasts + features)

This is the recommended entrypoint for data fetching and feature engineering.

```bash
tempdata data \
  --station KLGA \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --timezone America/New_York
```

This writes cached, canonical datasets under `data/` by default. On HPRC, pass
`--data-dir "$TEMP_DATA_ROOT/data"` to write to scratch.

### 3. Run training / experimentation (creates a run folder)

```bash
tempdata train --station KLGA --start 2020-01-01 --end 2024-12-31
```

### 4. Run evaluation / reporting (creates a run folder)

```bash
tempdata eval --station KLGA --start 2020-01-01 --end 2024-12-31
```

See `docs/CLI.md` for details and `docs/HPRC_SETUP.md` for running on HPRC.

## Live Forecasting

Daily temperature forecasts for KLGA (LaGuardia) using Open-Meteo + a tuned stacked ensemble model.

```bash
# Run forecast + accuracy report
python scripts/live_forecast.py

# Accuracy report only
python scripts/live_forecast.py --report

# Use ridge instead of stacked
python scripts/live_forecast.py --model-type ridge
```

The live forecast defaults to the **stacked ensemble** model, which combines Ridge, XGBoost, LightGBM, and CatBoost via out-of-fold stacking with a Ridge meta-learner. Hyperparameters were tuned via Grace HPC runs.

**Default stacked config:**
- Meta alpha: 0.01, stacking splits: 5
- XGBoost: lr=0.05, max_depth=3
- LightGBM: lr=0.05, num_leaves=15, n_estimators=300
- CatBoost: lr=0.05, iterations=300

**Live feature set:**
`[tmax_pred_f, lead_hours, month, bias_7d, bias_14d, bias_30d, rmse_14d, rmse_30d, sigma_lead]`

Predictions are logged to `runs/live/predictions.jsonl` and scored once actuals become available.

## Model Types

| Model | Description |
|-------|-------------|
| `ridge` | Ridge regression bias correction |
| `xgboost` | Gradient boosted trees |
| `lightgbm` | LightGBM regression |
| `catboost` | CatBoost regression |
| `stacked` | OOF-stacked ensemble (Ridge + XGB + LGBM + CatBoost) |
| `passthrough` | Raw forecast (baseline) |
| `persistence` | Yesterday's observed max |
| `knn` | K-nearest neighbors |
| `ensemble` | Equal-weight Ridge + XGBoost |

## Tuning & Feature Engineering

```bash
# Feature ablation (LOO, forward selection, group ablation)
python -m tempdata.ablate --model stacked

# Hyperparameter tuning (focused grid on stacked ensemble)
python -m tempdata.tune
```

Results are saved to `runs/ablate/` and `runs/tune/` respectively.

## Documentation

Detailed guides are available in the `docs/` directory:

1.  **[Data Acquisition](docs/1.%20data-acquisition.md)**: Fetching NOAA, ERA5, and Open-Meteo data.
2.  **[Dataset Creation](docs/2.%20dataset-creation.md)**: Building analysis-ready temperature datasets.
3.  **[Evaluation Framework](docs/3.%20evaluation.md)**: Running and configuring evaluation experiments.
4.  **[Developer Guide](docs/4.%20developer-guide.md)**: Codebase structure, testing, and extension.

For Colab users, see the **[Colab Setup Guide](docs/COLAB_SETUP.md)**.
