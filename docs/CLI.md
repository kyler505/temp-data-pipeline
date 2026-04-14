# CLI usage (HPRC + local)

This project now exposes a small, opinionated CLI intended to make it clear what you should run.

After installing the package (recommended: in a venv):

```bash
pip install -e .
```

You can run:

## 1) Build datasets (data fetching/cleaning/feature engineering)

```bash
tempdata data \
  --station KLGA \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --timezone America/New_York \
  --data-dir "$TEMP_DATA_ROOT/data"
```

This produces the **canonical cached datasets** under `--data-dir`:

- Clean hourly truth: `clean/hourly_obs/<station>/*.parquet`
- Daily truth: `clean/daily_tmax/<station>/*.parquet`
- Forecasts (Open-Meteo): `clean/forecasts/openmeteo/<station>/*.parquet`
- Training feature table: `train/daily_tmax/<station>/train_daily_tmax.parquet`

Forecasts are **optional by default**.

- If Open-Meteo historical forecasts are available for the requested range, they will be fetched/cached and the training feature table will be built.
- If forecasts are not available, the command will continue without forecasts and will skip building the forecast-dependent feature table.
- If you want strict forecast coverage enforcement, pass `--require-forecast`.

## 2) Train / experiment (creates a run folder)

```bash
tempdata train --config configs/eval_klga_v1.json --data-dir "$TEMP_DATA_ROOT/data"
```

## 3) Evaluate / report (creates a run folder)

```bash
tempdata eval --config configs/eval_klga_v1.json --data-dir "$TEMP_DATA_ROOT/data"
```

For the repaired baseline path without a config file:

```bash
tempdata eval \
  --station KLGA \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --model-type ridge \
  --data-dir "$TEMP_DATA_ROOT/data"
```

Notes:
- `train` and `eval` currently share the same underlying evaluation runner and will each generate a new run directory unless you provide `--run-id`.
- On HPRC, prefer placing `--data-dir` under `$TEMP_DATA_ROOT` (scratch) for performance.
- On HPRC, `scripts/hprc/05_baseline_eval.slurm` is the recommended first verification job.
