# Temp Data Pipeline

A data pipeline for processing temperature data from NOAA with integrated backtesting framework.

## Quick Start

### Data Pipeline
The hourly fetcher uses the public NOAA Global Hourly (ISD) CSV dataset, so no API token is required. Station metadata is stored in `stations/stations.csv` and must include the USAF/WBAN identifiers.

```bash
python scripts/fetch_noaa_hourly.py --station KLGA --start 2024-01-01 --end 2024-02-01
```

### Backtesting Framework
Run end-to-end temperature prediction and trading backtests:

```bash
# Install backtest dependencies
pip install -e ".[backtest]"

# Run a basic backtest
python scripts/backtest_daily_tmax.py --station KLGA --start 2020-01-01 --end 2024-12-31
```

## Documentation

- **[Colab Setup Guide](docs/COLAB_SETUP.md)** - Complete guide for running the pipeline in Google Colab with Google Drive integration for persistent data storage.
