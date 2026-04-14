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

## Documentation

Detailed guides are available in the `docs/` directory:

1.  **[Data Acquisition](docs/1.%20data-acquisition.md)**: Fetching NOAA, ERA5, and Open-Meteo data.
2.  **[Dataset Creation](docs/2.%20dataset-creation.md)**: Building analysis-ready temperature datasets.
3.  **[Evaluation Framework](docs/3.%20evaluation.md)**: Running and configuring evaluation experiments.
4.  **[Developer Guide](docs/4.%20developer-guide.md)**: Codebase structure, testing, and extension.

For Colab users, see the **[Colab Setup Guide](docs/COLAB_SETUP.md)**.
