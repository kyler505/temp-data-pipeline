#!/bin/bash
# scripts/hprc/04_eval.slurm
# Submit with: sbatch scripts/hprc/04_eval.slurm

#SBATCH --job-name=tempdata-eval
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tempdata_eval_%j.out
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
CONFIG=${CONFIG:-""}
RUN_ID=${RUN_ID:-""}

cd "$SLURM_SUBMIT_DIR"

# Evaluate using either a config or explicit args.
# If RUN_ID is provided, results are written to that run directory.
# Otherwise, a new run id is generated (current behavior).

EXTRA_RUN_ID_ARGS=""
if [[ -n "$RUN_ID" ]]; then
  EXTRA_RUN_ID_ARGS="--run-id $RUN_ID"
fi

if [[ -n "$CONFIG" ]]; then
  tempdata eval --config "$CONFIG" --data-dir "$TEMP_DATA_ROOT/data" $EXTRA_RUN_ID_ARGS
else
  tempdata eval \
    --station "$STATION" \
    --start "$START" \
    --end "$END" \
    --data-dir "$TEMP_DATA_ROOT/data" \
    $EXTRA_RUN_ID_ARGS
fi

