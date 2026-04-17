#!/usr/bin/env python3
"""CLI for training temperature models.

Usage:
    python scripts/train_model.py --station KLGA --model-type xgboost --tune
"""

import argparse
import sys
from pathlib import Path
from datetime import date

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from tempdata.eval.config import EvalConfig, SplitConfig, ModelConfig
from tempdata.eval.runner import run_evaluation
from tempdata.config import data_root

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temperature models.")
    parser.add_argument("--station", required=True, help="Station ID")
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--model-type", required=True, choices=["xgboost", "ridge", "persistence", "knn", "lightgbm", "catboost", "stacked"], help="Model type")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (XGBoost only)")
    parser.add_argument("--run-name", help="Custom run name")
    return parser.parse_args()

def load_data_simple(station, start_date, end_date):
    """Simplified data loader using standard paths."""
    # This logic matches scripts/eval_daily_tmax.py roughly but simplified for brevity
    # In production, we might want to just call the robust load_data from eval_daily_tmax
    from scripts.eval_daily_tmax import load_data
    # We mock args structure to reuse load_data
    class MockArgs:
        def __init__(self):
            self.station = station
            self.start = str(start_date)
            self.end = str(end_date)
            self.forecast_file = None
            self.truth_file = None

    return load_data(MockArgs())

def main():
    args = parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    print(f"Loading data for {args.station}...")
    forecast_df, truth_df = load_data_simple(args.station, start_date, end_date)

    # Configure Model
    hyperparams = {}
    if args.model_type == "xgboost":
        hyperparams = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "early_stopping_rounds": 50
        }
        if args.tune:
            print("Hyperparameter tuning enabled (Grid Search placeholder)...")
            # Implement grid search logic here if needed, or update hyperparams
            # For now we stick to defaults or allow manual overrides via args if we added them

    config = EvalConfig(
        run_name=args.run_name or f"train_{args.model_type}_{args.station}",
        station_ids=[args.station],
        start_date_local=start_date,
        end_date_local=end_date,
        split=SplitConfig(type="static", train_frac=0.7, val_frac=0.15, test_frac=0.15),
        model=ModelConfig(type=args.model_type, hyperparams=hyperparams)
    )

    print(f"Training {args.model_type}...")
    result = run_evaluation(
        config=config,
        forecast_df=forecast_df,
        truth_df=truth_df,
        verbose=True
    )

    print(f"Training complete. Results saved to {result.run_path}")
    print(f"Test MAE: {result.metrics.forecast.mae:.2f}")

if __name__ == "__main__":
    main()
