#!/bin/bash
# scripts/hprc/00_bootstrap_env.sh
# Run on a Grace login node after cloning the repo.

set -euo pipefail

PYTHON_MODULE=${PYTHON_MODULE:-"python/3.10"}
VENV_DIR=${VENV_DIR:-"$HOME/envs/temp-pipeline"}
TEMP_DATA_ROOT=${TEMP_DATA_ROOT:-"/scratch/user/$USER/temp-data"}

module purge
module load "$PYTHON_MODULE"

mkdir -p "$(dirname "$VENV_DIR")"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install scipy pytest

mkdir -p "$TEMP_DATA_ROOT/data"
mkdir -p logs

cat <<EOF
Bootstrap complete.

Next steps:
1. source "$VENV_DIR/bin/activate"
2. export TEMP_DATA_ROOT="$TEMP_DATA_ROOT"
3. sbatch scripts/hprc/05_baseline_eval.slurm
EOF
