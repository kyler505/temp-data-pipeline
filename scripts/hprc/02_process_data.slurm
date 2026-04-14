#!/bin/bash
# scripts/hprc/02_process_data.slurm
# Submit with: sbatch scripts/hprc/02_process_data.slurm

#SBATCH --job-name=tempdata-data
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tempdata_data_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# 1. Environment Setup
module purge
module load python/3.10

# Activate venv
source ~/envs/temp-pipeline/bin/activate

# Check Config
if [[ -z "$TEMP_DATA_ROOT" ]]; then
    export TEMP_DATA_ROOT=/scratch/user/$USER/temp-data
    echo "TEMP_DATA_ROOT was not set, defaulting to: $TEMP_DATA_ROOT"
fi

mkdir -p logs

echo "Job started on $(hostname) at $(date)"
echo "Using TEMP_DATA_ROOT: $TEMP_DATA_ROOT"

# 2. Pipeline Execution
STATION=${STATION:-"KLGA"}
START=${START:-"2020-01-01"}
END=${END:-"2024-12-31"}
TIMEZONE=${TIMEZONE:-"America/New_York"}

cd "$SLURM_SUBMIT_DIR"

# NOTE: We set --data-dir under TEMP_DATA_ROOT so canonical datasets are cached on scratch.

tempdata data \
  --station "$STATION" \
  --start "$START" \
  --end "$END" \
  --timezone "$TIMEZONE" \
  --data-dir "$TEMP_DATA_ROOT/data"

echo "Job finished at $(date)"
