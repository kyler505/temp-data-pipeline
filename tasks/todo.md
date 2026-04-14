# Restart plan research

## Goal
- Determine the safest way to restart the daily max temperature prediction pipeline.
- Identify what already works, what is broken, and what to refactor, remove, or add.

## Acceptance criteria
- Produce a codebase-based restart recommendation with concrete file references.
- Identify the current operational path and the broken path.
- Provide a prioritized roadmap, refactor list, and verification strategy.

## Checklist
- [x] Review repository structure and existing datasets
- [x] Review evaluation, feature, and CLI modules
- [x] Review docs and test coverage for drift
- [x] Collect specialist findings for architecture and restart strategy
- [x] Synthesize findings into a practical restart plan
- [x] Summarize verification story and next steps

## Working notes
- The repo contains a usable KLGA dataset in `data/`; this enables restart without rebuilding ingestion first.
- `src/tempdata/cli.py` advertises the main workflow but its eval/train wiring appears drifted from `src/tempdata/eval/config.py` and `src/tempdata/eval/runner.py`.
- `scripts/legacy/eval_daily_tmax.py` looks closer to the currently working evaluation path.
- Data layout under `data/` differs from some path assumptions in the CLI.

## Results
- The fastest reliable restart path is to use the existing KLGA datasets and repair the evaluation entrypoint before touching new modeling work.
- `scripts/legacy/eval_daily_tmax.py` is closer to the currently working operational path than `src/tempdata/cli.py`.
- The repo already contains evidence of a successful multi-model run under `runs/20260120_183019/`, with Ridge materially outperforming Persistence on held-out data.
- Primary debt is interface drift: CLI/config/runner/path contracts no longer match one another.
- Repaired `tempdata eval` to build a valid `EvalConfig`, load canonical parquet inputs, and call `run_evaluation` with real dataframes.
- Added a shared packaged loader in `src/tempdata/eval/data.py` and reused it from `scripts/legacy/eval_daily_tmax.py`.
- Added focused regression tests for canonical input loading, config-file loading, and datetime/date normalization in the eval path.
- Verified a real KLGA smoke run with `passthrough` now completes and writes artifacts to `runs/smoke_passthrough/`.
