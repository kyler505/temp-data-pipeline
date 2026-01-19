#!/usr/bin/env python3
"""CLI for running daily Tmax backtests.

Usage:
    python scripts/backtest_daily_tmax.py \
        --station KLGA \
        --start 2020-01-01 \
        --end 2025-08-26 \
        --lead-hours 28,29 \
        --run-id my_run
        
For full options:
    python scripts/backtest_daily_tmax.py --help
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tempdata.backtest.bins import BinSet
from tempdata.backtest.calibration import create_uncertainty_model
from tempdata.backtest.config import BacktestConfig, generate_run_id
from tempdata.backtest.data import load_backtest_data, print_data_summary
from tempdata.backtest.metrics import compute_all_metrics, print_metrics_summary
from tempdata.backtest.models import create_forecaster
from tempdata.backtest.pricing import create_price_provider
from tempdata.backtest.report import (
    create_run_dir,
    write_all_artifacts,
    write_summary_report,
)
from tempdata.backtest.simulator import run_backtest
from tempdata.backtest.strategy import create_strategy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run daily Tmax backtest with trading simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python scripts/backtest_daily_tmax.py --station KLGA --start 2020-01-01 --end 2024-12-31

  # With specific lead hours
  python scripts/backtest_daily_tmax.py --station KLGA --start 2020-01-01 --end 2024-12-31 --lead-hours 28,29

  # Custom configuration
  python scripts/backtest_daily_tmax.py --station KLGA --start 2020-01-01 --end 2024-12-31 \\
      --edge-min 0.05 --initial-bankroll 50000
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--station",
        required=True,
        help="Station ID (e.g., KLGA)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    
    # Data arguments
    parser.add_argument(
        "--forecast-file",
        help="Path to forecast parquet file",
    )
    parser.add_argument(
        "--truth-file",
        help="Path to truth (daily_tmax) parquet file",
    )
    parser.add_argument(
        "--feature-file",
        help="Path to pre-built feature parquet file (optional)",
    )
    parser.add_argument(
        "--lead-hours",
        help="Comma-separated list of lead hours to include (e.g., 28,29)",
    )
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=18,
        help="Minimum coverage hours for truth (default: 18)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model-type",
        choices=["ridge", "passthrough"],
        default="ridge",
        help="Forecast model type (default: ridge)",
    )
    parser.add_argument(
        "--model-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization parameter (default: 1.0)",
    )
    
    # Sigma arguments
    parser.add_argument(
        "--sigma-type",
        choices=["global", "bucketed", "rolling"],
        default="bucketed",
        help="Uncertainty model type (default: bucketed)",
    )
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=1.0,
        help="Minimum sigma value (default: 1.0)",
    )
    
    # Trading arguments
    parser.add_argument(
        "--edge-min",
        type=float,
        default=0.03,
        help="Minimum edge to trade (default: 0.03)",
    )
    parser.add_argument(
        "--max-per-market",
        type=float,
        default=0.02,
        help="Max bankroll fraction per market (default: 0.02)",
    )
    parser.add_argument(
        "--max-total",
        type=float,
        default=0.25,
        help="Max total exposure fraction (default: 0.25)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.01,
        help="Execution slippage (default: 0.01)",
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=10000.0,
        help="Initial bankroll (default: 10000)",
    )
    
    # Price arguments
    parser.add_argument(
        "--price-type",
        choices=["synthetic", "fixed", "adversarial"],
        default="synthetic",
        help="Price provider type (default: synthetic)",
    )
    parser.add_argument(
        "--price-noise",
        type=float,
        default=0.05,
        help="Price noise std (default: 0.05)",
    )
    
    # Output arguments
    parser.add_argument(
        "--run-id",
        help="Run identifier (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--output-dir",
        help="Base directory for run outputs (default: runs/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    # Splits
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Training data fraction (default: 0.70)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation data fraction (default: 0.15)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load forecast and truth data from files or default locations."""
    from tempdata.config import data_root
    
    # Load forecasts
    if args.forecast_file:
        forecast_df = pd.read_parquet(args.forecast_file)
    else:
        # Try default location
        forecast_dir = data_root() / "clean" / "forecasts" / "openmeteo" / args.station
        if not forecast_dir.exists():
            forecast_dir = data_root() / "raw" / "forecasts" / "openmeteo" / args.station
        
        if forecast_dir.exists():
            parquet_files = list(forecast_dir.glob("*.parquet"))
            if parquet_files:
                forecast_df = pd.concat([pd.read_parquet(f) for f in parquet_files])
            else:
                raise FileNotFoundError(f"No parquet files in {forecast_dir}")
        else:
            raise FileNotFoundError(
                f"Forecast directory not found: {forecast_dir}\n"
                "Use --forecast-file to specify a file path"
            )
    
    # Load truth
    if args.truth_file:
        truth_df = pd.read_parquet(args.truth_file)
    else:
        # Try default location
        truth_dir = data_root() / "clean" / "daily_tmax" / args.station
        if not truth_dir.exists():
            truth_dir = data_root() / "daily_tmax" / args.station
        
        if truth_dir.exists():
            parquet_files = list(truth_dir.glob("*.parquet"))
            if parquet_files:
                truth_df = pd.concat([pd.read_parquet(f) for f in parquet_files])
            else:
                raise FileNotFoundError(f"No parquet files in {truth_dir}")
        else:
            raise FileNotFoundError(
                f"Truth directory not found: {truth_dir}\n"
                "Use --truth-file to specify a file path"
            )
    
    return forecast_df, truth_df


def main() -> int:
    """Main entry point."""
    args = parse_args()
    verbose = not args.quiet
    
    # Parse dates
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    
    # Parse lead hours
    lead_hours = None
    if args.lead_hours:
        lead_hours = [int(x.strip()) for x in args.lead_hours.split(",")]
    
    # Generate run ID
    run_id = args.run_id or generate_run_id()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DAILY TMAX BACKTEST: {run_id}")
        print(f"{'='*60}")
        print(f"Station: {args.station}")
        print(f"Date range: {start_date} to {end_date}")
        if lead_hours:
            print(f"Lead hours: {lead_hours}")
        print()
    
    # Create configuration
    config = BacktestConfig(
        station_ids=[args.station],
        start_date_local=start_date,
        end_date_local=end_date,
        min_coverage_hours=args.min_coverage,
        lead_hours_allowed=lead_hours,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        model_type=args.model_type,
        model_alpha=args.model_alpha,
        sigma_type=args.sigma_type,
        sigma_floor=args.sigma_floor,
        price_type=args.price_type,
        price_noise=args.price_noise,
        edge_min=args.edge_min,
        max_per_market_pct=args.max_per_market,
        max_total_pct=args.max_total,
        slippage=args.slippage,
        initial_bankroll=args.initial_bankroll,
        random_seed=args.seed,
    )
    
    # Load data
    if verbose:
        print("[backtest] Loading data...")
    
    try:
        forecast_df, truth_df = load_data(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Load pre-built features if provided
    feature_df = None
    if args.feature_file:
        feature_df = pd.read_parquet(args.feature_file)
    
    # Load and prepare data
    dataset = load_backtest_data(
        config=config,
        forecast_df=forecast_df,
        truth_df=truth_df,
        feature_df=feature_df,
    )
    
    if verbose:
        print_data_summary(dataset)
    
    # Create components
    forecaster = create_forecaster(
        model_type=config.model_type,
        alpha=config.model_alpha,
        features=config.model_features,
    )
    
    uncertainty_model = create_uncertainty_model(
        sigma_type=config.sigma_type,
        sigma_buckets=config.sigma_buckets,
        sigma_floor=config.sigma_floor,
    )
    
    strategy = create_strategy(
        strategy_type="edge",
        edge_min=config.edge_min,
        max_per_market_pct=config.max_per_market_pct,
        max_total_pct=config.max_total_pct,
    )
    
    price_provider = create_price_provider(
        price_type=config.price_type,
        noise=config.price_noise,
        random_seed=config.random_seed,
    )
    
    bins = BinSet(config.bins_f)
    
    # Run backtest
    trades_df, daily_df, predictions_df = run_backtest(
        df_train=dataset.train,
        df_test=dataset.test,
        forecaster=forecaster,
        uncertainty_model=uncertainty_model,
        strategy=strategy,
        price_provider=price_provider,
        bins=bins,
        slippage=config.slippage,
        initial_bankroll=config.initial_bankroll,
        verbose=verbose,
    )
    
    # Compute metrics
    metrics = compute_all_metrics(
        predictions_df=predictions_df,
        trades_df=trades_df,
        daily_df=daily_df,
        initial_bankroll=config.initial_bankroll,
        df_full=dataset.test,
    )
    
    if verbose:
        print_metrics_summary(metrics)
    
    # Write artifacts
    output_dir = Path(args.output_dir) if args.output_dir else None
    artifacts = write_all_artifacts(
        config=config,
        metrics=metrics,
        trades_df=trades_df,
        daily_df=daily_df,
        predictions_df=predictions_df,
        run_id=run_id,
        base_path=output_dir,
    )
    
    # Write summary report
    run_dir = artifacts["config"].parent
    write_summary_report(config, metrics, run_id, run_dir)
    
    if verbose:
        print(f"\nRun complete: {run_id}")
        print(f"Artifacts: {run_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
