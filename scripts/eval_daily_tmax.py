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
        choices=["ridge", "passthrough", "persistence", "knn"],
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
    """Load forecast and truth data from files or default locations.

    Implements hybrid data loading:
    - Truth: ISD (before 2025-08-29) + GHCNh (from 2025-08-29 onward)
    - Forecast: ERA5 (before 2016) + Open-Meteo (2016 onward)
    - Automatic end-date buffering for ERA5T latency (~5 days)
    """
    from datetime import timedelta
    from tempdata.config import data_root

    # Transition dates
    ISD_CUTOFF = date(2025, 8, 29)
    OPENMETEO_START = date(2016, 1, 1)
    ERA5T_LATENCY_DAYS = 5

    station = args.station
    if station is None and config is not None and config.station_ids:
        station = config.station_ids[0]

    if station is None and not (args.forecast_file and args.truth_file):
        raise ValueError("Station ID required (either via --station or config file)")

    # Get date range from config
    start_date = config.start_date_local if config else date.fromisoformat(args.start)
    end_date = config.end_date_local if config else date.fromisoformat(args.end)

    # --- Apply end-date buffering for data latency ---
    today = date.today()
    max_truth_date = today - timedelta(days=ERA5T_LATENCY_DAYS)
    if end_date > max_truth_date:
        print(f"[eval] Buffering end_date from {end_date} to {max_truth_date} for data latency")
        end_date = max_truth_date
        if config:
            config.end_date_local = end_date

    # --- Load Forecasts (Deep Historical: ERA5, Standard: Open-Meteo) ---
    if args.forecast_file:
        forecast_df = pd.read_parquet(args.forecast_file)
    else:
        forecast_dfs = []

        # ERA5 for pre-2016 dates
        if start_date < OPENMETEO_START:
            era5_end = min(end_date, OPENMETEO_START)
            era5_dir = data_root() / "raw" / "era5" / station
            if era5_dir.exists():
                era5_files = list(era5_dir.glob("*.parquet"))
                if era5_files:
                    era5_df = pd.concat([pd.read_parquet(f) for f in era5_files])
                    # Filter to date range
                    era5_df["_date"] = pd.to_datetime(era5_df["ts_utc"]).dt.date
                    era5_df = era5_df[(era5_df["_date"] >= start_date) & (era5_df["_date"] < era5_end)]
                    era5_df = era5_df.drop(columns=["_date"])
                    forecast_dfs.append(era5_df)
                    print(f"[eval] Loaded {len(era5_df)} ERA5 rows for pre-2016 period")
            else:
                print(f"[eval] WARNING: ERA5 dir not found: {era5_dir}. Run fetch_era5_hourly.py first.")

        # Open-Meteo for 2016+
        if end_date >= OPENMETEO_START:
            om_start = max(start_date, OPENMETEO_START)
            forecast_dir = data_root() / "clean" / "forecasts" / "openmeteo" / station
            if not forecast_dir.exists():
                forecast_dir = data_root() / "raw" / "forecasts" / "openmeteo" / station

            if forecast_dir.exists():
                om_files = list(forecast_dir.glob("*.parquet"))
                if om_files:
                    om_df = pd.concat([pd.read_parquet(f) for f in om_files])
                    # Filter to date range
                    if "target_date_local" in om_df.columns:
                        om_df["_date"] = pd.to_datetime(om_df["target_date_local"]).dt.date
                    else:
                        om_df["_date"] = pd.to_datetime(om_df["issue_time_utc"]).dt.date
                    om_df = om_df[(om_df["_date"] >= om_start) & (om_df["_date"] <= end_date)]
                    om_df = om_df.drop(columns=["_date"])
                    forecast_dfs.append(om_df)
                    print(f"[eval] Loaded {len(om_df)} Open-Meteo rows for 2016+ period")
            else:
                print(f"[eval] WARNING: Open-Meteo dir not found: {forecast_dir}")

        if not forecast_dfs:
            raise FileNotFoundError(
                f"No forecast data found for station {station}\n"
                "Use --forecast-file to specify a file path"
            )

        forecast_df = pd.concat(forecast_dfs, ignore_index=True)

    # --- Load Truth (Legacy: ISD, Modern: GHCNh) ---
    if args.truth_file:
        truth_df = pd.read_parquet(args.truth_file)
    else:
        truth_dfs = []

        # ISD for pre-2025-08-29
        if start_date < ISD_CUTOFF:
            isd_end = min(end_date, ISD_CUTOFF - timedelta(days=1))
            isd_dir = data_root() / "clean" / "daily_tmax" / station
            if not isd_dir.exists():
                isd_dir = data_root() / "daily_tmax" / station

            if isd_dir.exists():
                # Look for ISD-sourced files (pattern: isd_*.parquet or *.parquet)
                isd_files = list(isd_dir.glob("*.parquet"))
                if isd_files:
                    isd_df = pd.concat([pd.read_parquet(f) for f in isd_files])
                    if "date_local" in isd_df.columns:
                        isd_df["_date"] = pd.to_datetime(isd_df["date_local"]).dt.date
                    elif "target_date_local" in isd_df.columns:
                        isd_df["_date"] = pd.to_datetime(isd_df["target_date_local"]).dt.date
                    else:
                        isd_df["_date"] = start_date  # fallback
                    isd_df = isd_df[(isd_df["_date"] >= start_date) & (isd_df["_date"] <= isd_end)]
                    isd_df = isd_df.drop(columns=["_date"])
                    truth_dfs.append(isd_df)
                    print(f"[eval] Loaded {len(isd_df)} ISD truth rows for pre-Aug-2025")

        # GHCNh for 2025-08-29+
        if end_date >= ISD_CUTOFF:
            ghcnh_start = max(start_date, ISD_CUTOFF)
            ghcnh_dir = data_root() / "clean" / "daily_tmax" / station
            if not ghcnh_dir.exists():
                ghcnh_dir = data_root() / "daily_tmax" / station

            if ghcnh_dir.exists():
                # Look for GHCNh-sourced files
                ghcnh_files = list(ghcnh_dir.glob("ghcnh_*.parquet"))
                if not ghcnh_files:
                    # Fallback: use all parquet files if no ghcnh_ prefix
                    ghcnh_files = list(ghcnh_dir.glob("*.parquet"))

                if ghcnh_files:
                    ghcnh_df = pd.concat([pd.read_parquet(f) for f in ghcnh_files])
                    if "date_local" in ghcnh_df.columns:
                        ghcnh_df["_date"] = pd.to_datetime(ghcnh_df["date_local"]).dt.date
                    elif "target_date_local" in ghcnh_df.columns:
                        ghcnh_df["_date"] = pd.to_datetime(ghcnh_df["target_date_local"]).dt.date
                    else:
                        ghcnh_df["_date"] = ghcnh_start
                    ghcnh_df = ghcnh_df[(ghcnh_df["_date"] >= ghcnh_start) & (ghcnh_df["_date"] <= end_date)]
                    ghcnh_df = ghcnh_df.drop(columns=["_date"])
                    truth_dfs.append(ghcnh_df)
                    print(f"[eval] Loaded {len(ghcnh_df)} GHCNh truth rows for Aug-2025+")

        if not truth_dfs:
            raise FileNotFoundError(
                f"No truth data found for station {station}\n"
                "Use --truth-file to specify a file path"
            )

        truth_df = pd.concat(truth_dfs, ignore_index=True)

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
