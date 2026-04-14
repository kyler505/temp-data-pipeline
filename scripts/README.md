# Scripts status

This repository now uses a **package CLI** as the recommended interface.

## Recommended entrypoints

Use the `tempdata` CLI:

- `tempdata data` – fetch/clean/build canonical datasets (+features)
- `tempdata train` – training/experimentation (creates `runs/<run_id>/`)
- `tempdata eval` – evaluation/reporting (creates `runs/<run_id>/`)

See `docs/CLI.md`.

## HPRC scripts

Use `scripts/hprc/*.slurm` to submit jobs that call the CLI.

- `scripts/hprc/00_bootstrap_env.sh` - create/update the Grace venv and install required Python packages
- `scripts/hprc/02_process_data.slurm` - build canonical parquet datasets under `$TEMP_DATA_ROOT/data`
- `scripts/hprc/05_baseline_eval.slurm` - run the repaired KLGA baseline eval path on Grace
- `scripts/hprc/06_baseline_sweep.sh` - submit passthrough, persistence, and ridge as separate Slurm jobs
- `scripts/hprc/07_sync_grace.sh` - rsync helper for pulling runs/data or pushing repo/config updates

## Legacy + dev scripts

To reduce ambiguity, older one-off script entrypoints have been moved to:

- `scripts/legacy/` – older CLI wrappers kept for reference/backwards compatibility
- `scripts/dev/` – ad-hoc utilities and patch scripts
