# HPRC Environment Setup

This repo runs well on TAMU Grace as a CPU-first Slurm workflow. Start with the repaired KLGA baseline eval path before attempting larger model sweeps.

## Recommended layout

- Keep the repo clone in `$SCRATCH`, not `$HOME`
- Keep large parquet data under `$TEMP_DATA_ROOT/data`
- Keep the virtualenv in `$HOME/envs/temp-pipeline` or another persistent location

Example:

```bash
cd $SCRATCH
git clone <your-repo-url> temp-data-pipeline
cd temp-data-pipeline
export TEMP_DATA_ROOT=/scratch/user/$USER/temp-data
```

## Bootstrap the environment

The fastest setup path is:

```bash
cd $SCRATCH/temp-data-pipeline
bash scripts/hprc/00_bootstrap_env.sh
source ~/envs/temp-pipeline/bin/activate
export TEMP_DATA_ROOT=/scratch/user/$USER/temp-data
```

If you want to do it manually instead:

```bash
module purge
module load python/3.10
python -m venv ~/envs/temp-pipeline
source ~/envs/temp-pipeline/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install scipy pytest
mkdir -p "$TEMP_DATA_ROOT/data"
```

## First run: baseline evaluation

Use the baseline Slurm script first. It exercises the repaired eval path on existing canonical parquet data.

```bash
sbatch scripts/hprc/05_baseline_eval.slurm
```

You can override variables at submit time:

```bash
MODEL_TYPE=passthrough START=2020-01-01 END=2024-12-31 sbatch scripts/hprc/05_baseline_eval.slurm
```

Monitor jobs with:

```bash
squeue --me
```

Logs land in `logs/`, and run artifacts land in `runs/<run_id>/` under the repo clone.

## Baseline sweep

To submit the three baseline models as separate Slurm jobs:

```bash
bash scripts/hprc/06_baseline_sweep.sh
```

This submits `passthrough`, `persistence`, and `ridge` jobs using `scripts/hprc/05_baseline_eval.slurm`.

You can override the date range or station at submission time:

```bash
STATION=KLGA START=2020-01-01 END=2024-12-31 bash scripts/hprc/06_baseline_sweep.sh
```

## Syncing runs and data back to your laptop

From your local machine, use the rsync helper:

```bash
NETID=<your_netid> bash scripts/hprc/07_sync_grace.sh pull-runs
NETID=<your_netid> bash scripts/hprc/07_sync_grace.sh pull-data
```

Useful actions:

- `pull-runs` - copy `runs/` from Grace to your local repo
- `pull-data` - copy selected parquet data from Grace to your local repo
- `push-configs` - send HPRC configs/scripts/docs up to Grace
- `push-repo` - sync the repo source code up to Grace without `data/` or `runs/`

## Data-building workflow

If you need to rebuild data/features on Grace, use:

```bash
sbatch scripts/hprc/02_process_data.slurm
```

That command writes canonical data under `$TEMP_DATA_ROOT/data`, including:

- `clean/hourly_obs/<station>/*.parquet`
- `clean/daily_tmax/<station>/*.parquet`
- `clean/forecasts/openmeteo/<station>/*.parquet`
- `train/daily_tmax/<station>/train_daily_tmax.parquet`

## ERA5 note

ERA5 is not part of the recommended restart path. Get the KLGA Open-Meteo + NOAA baseline green first. If you later need ERA5 fetching, add the CDS API setup separately via `~/.cdsapirc`.
