#!/bin/bash
# scripts/hprc/03_train.slurm
# Submit with: sbatch scripts/hprc/03_train.slurm

#SBATCH --job-name=tempdata-train
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tempdata_train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module purge
module load python/3.10

source ~/envs/temp-pipeline/bin/activate

if [[ -z "$TEMP_DATA_ROOT" ]]; then
    export TEMP_DATA_ROOT=/scratch/user/$USER/temp-data
    echo "TEMP_DATA_ROOT was not set, defaulting to: $TEMP_DATA_ROOT"
fi

mkdir -p logs

STATION=${STATION:-"KLGA"}
START=${START:-"2020-01-01"}
END=${END:-"2024-12-31"}
TIMEZONE=${TIMEZONE:-"America/New_York"}
CONFIG=${CONFIG:-""}

cd "$SLURM_SUBMIT_DIR"

# Recommended workflow: run `tempdata data` first to populate cached canonical datasets.
# Then train using either a config or explicit args.

if [[ -n "$CONFIG" ]]; then
  tempdata train --config "$CONFIG" --data-dir "$TEMP_DATA_ROOT/data"
else
  tempdata train \
    --station "$STATION" \
    --start "$START" \
    --end "$END" \
    --data-dir "$TEMP_DATA_ROOT/data"
fi

