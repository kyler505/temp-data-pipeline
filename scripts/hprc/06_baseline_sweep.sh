#!/bin/bash
# scripts/hprc/06_baseline_sweep.sh
# Submit three baseline jobs (passthrough, persistence, ridge).
# Usage: bash scripts/hprc/06_baseline_sweep.sh

set -euo pipefail

MODELS=(passthrough persistence ridge)
STATION=${STATION:-"KLGA"}
START=${START:-"2020-01-01"}
END=${END:-"2024-12-31"}

for MODEL_TYPE in "${MODELS[@]}"; do
  echo "Submitting ${MODEL_TYPE} baseline job..."
  STATION="$STATION" \
  START="$START" \
  END="$END" \
  MODEL_TYPE="$MODEL_TYPE" \
  sbatch scripts/hprc/05_baseline_eval.slurm
done
