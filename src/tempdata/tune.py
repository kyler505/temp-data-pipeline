"""Hyperparameter tuning for stacked ensemble on KLGA temperature forecast.

Usage:
    python -m tempdata.tune                          # defaults
    python -m tempdata.tune --data-path <path>       # custom data
    python -m tempdata.tune --n-folds 3 --quick      # fast smoke test

Searches over:
  - meta_alpha (ridge regularization on stacking meta-learner)
  - stacking_splits (number of OOF folds)
  - per-base-model hyperparams (lightgbm, xgboost, catboost)

Evaluates each candidate via time-series CV on the enriched training table,
writes summary.json with per-trial metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Candidate grid
# ---------------------------------------------------------------------------

QUICK_GRID: dict[str, list] = {
    "meta_alpha": [0.1, 1.0],
    "stacking_splits": [3, 5],
    "lgb_learning_rate": [0.05],
    "lgb_num_leaves": [31],
    "lgb_n_estimators": [300],
    "xgb_learning_rate": [0.05],
    "xgb_max_depth": [3, 5],
    "cat_learning_rate": [0.05],
    "cat_iterations": [300],
}

FULL_GRID: dict[str, list] = {
    "meta_alpha": [0.01, 0.1, 1.0, 10.0],
    "stacking_splits": [3, 5],
    "lgb_learning_rate": [0.01, 0.05, 0.1],
    "lgb_num_leaves": [15, 31, 63],
    "lgb_n_estimators": [200, 300],
    "xgb_learning_rate": [0.01, 0.05, 0.1],
    "xgb_max_depth": [3, 5],
    "cat_learning_rate": [0.01, 0.05],
    "cat_iterations": [300, 500],
}


def expand_grid(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a dict-of-lists grid into a list of dicts."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load the enriched training table."""
    df = pd.read_parquet(data_path)
    df["target_date_local"] = pd.to_datetime(df["target_date_local"])
    df = df.sort_values("target_date_local").reset_index(drop=True)
    return df


def time_series_cv_split(
    df: pd.DataFrame, n_folds: int = 5
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate time-series CV folds (no shuffle, expanding train window)."""
    n = len(df)
    fold_size = n // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        train_end = fold_size * i
        val_end = min(train_end + fold_size, n)
        if val_end <= train_end:
            continue
        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        if len(train_df) < 30 or len(val_df) < 10:
            continue
        folds.append((train_df, val_df))
    return folds


# ---------------------------------------------------------------------------
# Model construction from trial config
# ---------------------------------------------------------------------------

def build_stacked_from_trial(
    trial: dict[str, Any],
    features: list[str],
):
    """Build a StackedEnsembleForecaster from a trial config dict."""
    from tempdata.eval.models import (
        RidgeForecaster,
        XGBoostForecaster,
        LightGBMForecaster,
        CatBoostForecaster,
        StackedEnsembleForecaster,
    )

    alpha = 1.0  # ridge base model alpha (not meta)
    meta_alpha = trial["meta_alpha"]
    n_splits = trial["stacking_splits"]

    lgb_params = {
        "learning_rate": trial["lgb_learning_rate"],
        "num_leaves": trial["lgb_num_leaves"],
        "n_estimators": trial["lgb_n_estimators"],
    }
    xgb_params = {
        "learning_rate": trial["xgb_learning_rate"],
        "max_depth": trial["xgb_max_depth"],
    }
    cat_params = {
        "learning_rate": trial["cat_learning_rate"],
        "iterations": trial["cat_iterations"],
    }

    factories = [
        ("ridge", lambda: RidgeForecaster(alpha=alpha, features=features)),
        ("xgboost", lambda: XGBoostForecaster(features=features, hyperparams=xgb_params)),
        ("lightgbm", lambda: LightGBMForecaster(features=features, hyperparams=lgb_params)),
        ("catboost", lambda: CatBoostForecaster(features=features, hyperparams=cat_params)),
    ]

    return StackedEnsembleForecaster(
        base_model_factories=factories,
        meta_alpha=meta_alpha,
        n_splits=n_splits,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_trial(
    trial: dict[str, Any],
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one trial across all CV folds, return metrics dict."""
    fold_maes = []
    fold_rmses = []
    fold_times = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        t0 = time.time()
        try:
            model = build_stacked_from_trial(trial, features)
            model.fit(train_df)
            preds = model.predict_mu(val_df)
            actuals = val_df["tmax_actual_f"].values
            errors = np.abs(preds - actuals)
            fold_maes.append(float(np.mean(errors)))
            fold_rmses.append(float(np.sqrt(np.mean(errors ** 2))))
        except Exception as e:
            fold_maes.append(np.nan)
            fold_rmses.append(np.nan)
            if verbose:
                print(f"  Fold {fold_idx} failed: {e}")
        fold_times.append(time.time() - t0)

    valid_maes = [m for m in fold_maes if np.isfinite(m)]
    valid_rmses = [r for r in fold_rmses if np.isfinite(r)]

    return {
        "trial": trial,
        "mean_mae": float(np.mean(valid_maes)) if valid_maes else np.nan,
        "mean_rmse": float(np.mean(valid_rmses)) if valid_rmses else np.nan,
        "std_mae": float(np.std(valid_maes)) if valid_maes else np.nan,
        "n_valid_folds": len(valid_maes),
        "total_time_s": round(sum(fold_times), 2),
        "per_fold_mae": fold_maes,
        "per_fold_rmse": fold_rmses,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tune stacked ensemble hyperparameters")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/train/daily_tmax/KLGA/train_daily_tmax_enriched.parquet"),
        help="Path to enriched training parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tune"), help="Output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--quick", action="store_true", help="Use quick (small) grid")
    parser.add_argument("--max-trials", type=int, default=None, help="Cap number of trials")
    parser.add_argument("--verbose", action="store_true", help="Print per-fold details")
    args = parser.parse_args()

    # Features to use (full enriched set)
    from tempdata.features.enriched_features import get_enriched_feature_names
    base_features = ["tmax_pred_f", "sin_doy", "cos_doy", "bias_7d", "bias_14d", "lead_hours"]
    enriched = get_enriched_feature_names()
    features = base_features + enriched

    # Load data
    print(f"Loading data from {args.data_path}")
    df = load_training_data(args.data_path)
    # Filter to columns that exist
    features = [f for f in features if f in df.columns]
    print(f"Data: {len(df)} rows, {len(features)} features")
    print(f"Date range: {df['target_date_local'].min().date()} to {df['target_date_local'].max().date()}")

    # Generate folds
    folds = time_series_cv_split(df, n_folds=args.n_folds)
    print(f"CV folds: {len(folds)}")
    for i, (tr, va) in enumerate(folds):
        print(f"  Fold {i}: train={len(tr)} ({tr['target_date_local'].min().date()}..{tr['target_date_local'].max().date()}), val={len(va)} ({va['target_date_local'].min().date()}..{va['target_date_local'].max().date()})")

    # Build grid
    grid = QUICK_GRID if args.quick else FULL_GRID
    trials = expand_grid(grid)
    if args.max_trials:
        trials = trials[: args.max_trials]
    print(f"\nTrials: {len(trials)}")
    print()

    # Run sweep
    results = []
    for i, trial in enumerate(trials):
        print(f"[{i+1}/{len(trials)}] {trial}", end=" ... ", flush=True)
        res = evaluate_trial(trial, folds, features, verbose=args.verbose)
        results.append(res)
        mae = res["mean_mae"]
        rmse = res["mean_rmse"]
        print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  ({res['total_time_s']:.1f}s)")

    # Sort by MAE
    results.sort(key=lambda r: r["mean_mae"] if np.isfinite(r["mean_mae"]) else 1e9)

    # Summary
    print(f"\n{'='*60}")
    print("TOP 5 TRIALS (by MAE)")
    print(f"{'='*60}")
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1}: MAE={r['mean_mae']:.3f}  RMSE={r['mean_rmse']:.3f}  config={r['trial']}")

    # Write output
    run_id = date.today().isoformat() + "_tune"
    out_dir = args.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary = {
        "run_id": run_id,
        "data_path": str(args.data_path),
        "n_folds": args.n_folds,
        "n_trials": len(trials),
        "grid_type": "quick" if args.quick else "full",
        "features": features,
        "best_trial": results[0] if results else None,
        "all_results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults written to {summary_path}")

    if results:
        best = results[0]
        print(f"\nBest config: {best['trial']}")
        print(f"Best MAE:    {best['mean_mae']:.3f} ± {best['std_mae']:.3f}")
        print(f"Best RMSE:   {best['mean_rmse']:.3f}")


if __name__ == "__main__":
    main()
