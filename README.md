# Temp Data Pipeline

A data pipeline for processing temperature data from NOAA (truth) and Open-Meteo (forecasts), with an integrated evaluation framework for assessing prediction accuracy.

## Quick Start

### 1. Installation

```bash
pip install -e ".[eval]"
```

### 2. Fetch Data

```bash
# Get ground truth (NOAA)
python scripts/fetch_noaa_hourly.py --station KLGA --start 2020-01-01 --end 2024-12-31

# Get forecasts (Open-Meteo)
python scripts/fetch_openmeteo_daily_forecast.py --station KLGA --forecast-days 14
```

### 3. Build Datasets

```bash
# Create daily max temperature from hourly observations
python scripts/build_daily_tmax.py --station KLGA --timezone America/New_York
```

### 4. Run Evaluation

```bash
# Evaluate forecast accuracy
python scripts/eval_daily_tmax.py --station KLGA --start 2020-01-01 --end 2024-12-31
```

## Documentation

Detailed guides are available in the `docs/` directory:

1.  **[Data Acquisition](docs/1.%20data-acquisition.md)**: Fetching NOAA, ERA5, and Open-Meteo data.
2.  **[Dataset Creation](docs/2.%20dataset-creation.md)**: Building analysis-ready temperature datasets.
3.  **[Evaluation Framework](docs/3.%20evaluation.md)**: Running and configuring evaluation experiments.
4.  **[Developer Guide](docs/4.%20developer-guide.md)**: Codebase structure, testing, and extension.

For Colab users, see the **[Colab Setup Guide](docs/COLAB_SETUP.md)**.
