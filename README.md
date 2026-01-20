# Temp Data Pipeline

A data pipeline for processing temperature data from NOAA with integrated temperature evaluation framework.

## Quick Start

### Data Pipeline
The hourly fetcher uses the public NOAA Global Hourly (ISD) CSV dataset, so no API token is required. Station metadata is stored in `stations/stations.csv` and must include the USAF/WBAN identifiers.

```bash
python scripts/fetch_noaa_hourly.py --station KLGA --start 2024-01-01 --end 2024-02-01
```

### Temperature Evaluation Framework
Evaluate daily Tmax prediction quality with detailed metrics and diagnostics:

```bash
# Install dependencies
pip install -e ".[eval]"

# Run evaluation from config
python scripts/eval_daily_tmax.py --config configs/eval_klga_v1.json

# Or with command line options
python scripts/eval_daily_tmax.py --station KLGA --start 2020-01-01 --end 2024-12-31
```

#### Evaluation Outputs
Each run creates `runs/<run_id>/` with:
- `config.json`: Frozen configuration
- `meta.json`: Run metadata (git hash, timestamp)
- `predictions.parquet`: Model predictions
- `residuals.parquet`: Residual analysis
- `metrics.json`: MAE, RMSE, bias, calibration
- `slices.json`: Metrics by month/season/lead time

## Documentation

- **[Colab Setup Guide](docs/COLAB_SETUP.md)** - Complete guide for running the pipeline in Google Colab with Google Drive integration for persistent data storage.
