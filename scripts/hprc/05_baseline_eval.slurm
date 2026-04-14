#!/bin/bash
# scripts/hprc/05_baseline_eval.slurm
# Submit with: sbatch scripts/hprc/05_baseline_eval.slurm

#SBATCH --job-name=tmax-baseline
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/tmax_baseline_%j.out

set -euo pipefail

module purge
module load python/3.10

source ~/envs/temp-pipeline/bin/activate

export TEMP_DATA_ROOT=${TEMP_DATA_ROOT:-"/scratch/user/$USER/temp-data"}

STATION=${STATION:-"KLGA"}
START=${START:-"2020-01-01"}
END=${END:-"2024-12-31"}
MODEL_TYPE=${MODEL_TYPE:-"ridge"}
RUN_ID=${RUN_ID:-"hprc_${MODEL_TYPE}_${SLURM_JOB_ID}"}

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Running baseline eval on $(hostname)"
echo "Station: $STATION"
echo "Date range: $START to $END"
echo "Model: $MODEL_TYPE"
echo "TEMP_DATA_ROOT: $TEMP_DATA_ROOT"

PYTHONPATH=src python -m tempdata.cli eval \
  --station "$STATION" \
  --start "$START" \
  --end "$END" \
  --model-type "$MODEL_TYPE" \
  --data-dir "$TEMP_DATA_ROOT/data" \
  --run-id "$RUN_ID"

echo "Run complete: runs/$RUN_ID"
