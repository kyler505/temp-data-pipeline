#!/usr/bin/env python3
"""CLI for running daily Tmax temperature evaluations.

Usage:
    python scripts/eval_daily_tmax.py --config configs/eval_klga_v1.json

    python scripts/eval_daily_tmax.py \
        --station KLGA \
        --start 2020-01-01 \
        --end 2025-08-26 \
        --run-id my_eval_run

For full options:
    python scripts/eval_daily_tmax.py --help
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from tempdata.eval.config import EvalConfig, SplitConfig, ModelConfig, UncertaintyConfig, generate_run_id
from tempdata.eval.runner import run_evaluation


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run daily Tmax temperature evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from config file
  python scripts/eval_daily_tmax.py --config configs/eval_klga_v1.json

  # Run with command line options
  python scripts/eval_daily_tmax.py --station KLGA --start 2020-01-01 --end 2024-12-31

  # Custom model settings
  python scripts/eval_daily_tmax.py --station KLGA --start 2020-01-01 --end 2024-12-31 \\
      --model-type ridge --model-alpha 0.5
        """,
    )

    # Config file (takes precedence)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file (overrides other arguments)",
    )

    # Required if no config file
    parser.add_argument(
        "--station",
        help="Station ID (e.g., KLGA)",
    )
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD)",
    )

    # Data arguments
    parser.add_argument(
        "--forecast-file",
        type=Path,
        help="Path to forecast parquet file",
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        help="Path to truth (daily_tmax) parquet file",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
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

    # Sigma/uncertainty arguments
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

    # Split arguments
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

    # Output arguments
    parser.add_argument(
        "--run-id",
        help="Run identifier (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--run-name",
        help="Human-readable run name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Base directory for run outputs (default: runs/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def load_data(args: argparse.Namespace, config: EvalConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load forecast and truth data from files or default locations."""
    from tempdata.config import data_root

    station = args.station
    if station is None and config is not None and config.station_ids:
        station = config.station_ids[0]

    if station is None and not (args.forecast_file and args.truth_file):
        raise ValueError("Station ID required (either via --station or config file)")

    # Load forecasts
    if args.forecast_file:
        forecast_df = pd.read_parquet(args.forecast_file)
    else:
        # Try default location
        forecast_dir = data_root() / "clean" / "forecasts" / "openmeteo" / station
        if not forecast_dir.exists():
            forecast_dir = data_root() / "raw" / "forecasts" / "openmeteo" / station

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
        truth_dir = data_root() / "clean" / "daily_tmax" / station
        if not truth_dir.exists():
            truth_dir = data_root() / "daily_tmax" / station

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

    # Load config from file or create from arguments
    if args.config:
        config = EvalConfig.load(args.config)
        # Allow overrides from command line
        if args.station:
            config.station_ids = [args.station]
        if args.start:
            config.start_date_local = date.fromisoformat(args.start)
        if args.end:
            config.end_date_local = date.fromisoformat(args.end)
    else:
        # Require station, start, end if no config file
        if not all([args.station, args.start, args.end]):
            print(
                "Error: --station, --start, and --end are required "
                "unless --config is provided",
                file=sys.stderr,
            )
            return 1

        # Parse lead hours
        lead_hours = None
        if args.lead_hours:
            lead_hours = [int(x.strip()) for x in args.lead_hours.split(",")]

        # Create config
        config = EvalConfig(
            run_name=args.run_name or f"eval_{args.station}",
            station_ids=[args.station],
            start_date_local=date.fromisoformat(args.start),
            end_date_local=date.fromisoformat(args.end),
            min_coverage_hours=args.min_coverage,
            lead_hours_allowed=lead_hours,
            split=SplitConfig(
                type="static",
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                test_frac=1.0 - args.train_frac - args.val_frac,
            ),
            model=ModelConfig(
                type=args.model_type,
                alpha=args.model_alpha,
            ),
            uncertainty=UncertaintyConfig(
                type=args.sigma_type,
                sigma_floor=args.sigma_floor,
            ),
            random_seed=args.seed,
        )

    # Generate run ID
    run_id = args.run_id or generate_run_id()

    # Load data
    if verbose:
        print("[eval] Loading data...")

    try:
        forecast_df, truth_df = load_data(args, config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Load pre-built features if provided
    feature_df = None
    if args.feature_file:
        feature_df = pd.read_parquet(args.feature_file)

    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        from tempdata.config import project_root
        output_dir = project_root() / "runs"

    # Run evaluation
    result = run_evaluation(
        config=config,
        forecast_df=forecast_df,
        truth_df=truth_df,
        feature_df=feature_df,
        run_id=run_id,
        output_dir=output_dir,
        verbose=verbose,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
